"""SuperTrack and SuperMimic for manipulation demonstration tracking (tasks.robomimic_tracking), as hand-object tracking has
them (algos.hand_algos): both fit the world model on k-step windows of simulator transitions and train the policy through
imagined rollouts of the frozen world model from replayed states. The world model is the generalized-coordinate model of the
scene (worldmodels.mjcf.MjcfWorldModel) over one control step: its layout's timestep is CONTROL_DT, and SuperTrack's
accelerations advance the position with the mean velocity of the control step's physics steps (``mean_velocity_lead``). The
systems options are the hand task's (algos.hand_fast), with a collection step of this task's (``RobomimicCollector``).

SuperTrack (``RobomimicSuperTrack``): the acceleration world model; the policy minimizes, over the imagined window, the L1
errors of what the task tracks (``TERMS``: the keypoints, the arms' joints, the grippers' fingers, the tracked objects'
positions and rotations), each term weighted once to 1 at calibration (supertrack.LossBalancer) or, with
``--st_task_weights``, by the task's reward weight over its kernel's width (the fingers, which the reward does not track,
weigh nothing); plus the failure penalty on the task's soft termination (``--st_fail_w``) and the regularizer on the actions
(``--st_reg``). ``--st_imag_noise`` adds Gaussian noise to the imagined actions. No reward, no critic.

SuperMimic (``RobomimicSuperMimic``; Liao et al., CoRL 2026): the increment world model; the policy maximizes the imagined
tracking return with a soft survival weight and a critic beyond the horizon. ``--bptt_decay lam`` scales the gradient through
the imagined state by lam at every step."""

from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..policies.critic import Critic
from ..policies.robomimic_policy import RobomimicPolicy
from ..tasks.hdmi import relative_rot, rot_angle
from ..tasks.robomimic_tracking import CONTROL_DT, RobomimicTask, unflatten_state
from ..worldmodels.mjcf import MjcfLayout, MjcfWorldModel, mean_velocity_lead
from .common import set_requires_grad
from .hand_fast import Collector, finite_step, optimizer, survive
from .supertrack import LossBalancer

WINDOW_KEYS = ("s", "s1", "u", "done")
ACC_KEYS = ("r", "steps", "done", "term", "ep_len", "object_err", "success")  # the log's sums over a row's collection steps


def replay_fields(task: RobomimicTask) -> dict:
    """A transition: the state, the position targets ``u``, the next state, the action before this step's, the frame."""
    return {
        "s": (task.state_dim, torch.float32),
        "s1": (task.state_dim, torch.float32),
        "u": (task.nu, torch.float32),
        "a_prev": (task.nu, torch.float32),
        "f": ((), torch.long),
        "done": ((), torch.bool),
    }


def world_model(model, args, kind: str, substeps: int) -> MjcfWorldModel:
    """The scene's world model over one control step (``model``: the compiled scene with its servos)."""
    lay = dataclasses.replace(MjcfLayout.from_model(model), dt=CONTROL_DT)
    lead = {"pos_lead": mean_velocity_lead(model, substeps)} if kind == "accel" else {}
    return MjcfWorldModel(lay, hidden=tuple(args.wm_hidden), act=args.wm_act, layer_norm=not args.wm_no_ln, kind=kind, **lead)


class RobomimicSuperMimic:
    kind = "increment"  # the world model's parameterization

    def __init__(self, args, model, task: RobomimicTask, dev: str, substeps: int):
        self.args, self.task, self.dev, self.nq = args, task, dev, task.nq
        self.goal_frames, self.horizon = args.goal_frames, args.horizon
        self.bptt_decay = float(getattr(args, "bptt_decay", 1.0))
        if not 0 < self.bptt_decay <= 1:
            raise ValueError(f"--bptt_decay must be in (0, 1], got {self.bptt_decay}")
        goal_dim = task.goal_frame_dim * self.goal_frames
        self.wm = world_model(model, args, self.kind, substeps).to(dev)
        self.pi = RobomimicPolicy(task.obs_dim, goal_dim, task.nu).to(dev)
        self.critic = Critic(task.obs_dim + goal_dim).to(dev)
        self.opt_wm = optimizer(args, dev, self.wm, "wm")
        self.opt_pi = optimizer(args, dev, self.pi, "pi")
        self.opt_v = optimizer(args, dev, self.critic, "critic")
        self.no_sync = bool(getattr(args, "no_sync", False))
        self.fk = task.fk  # the collection's forward kinematics, run eagerly; --compile_eager compiles them (RobomimicCollector)
        self.wm_rollout = torch.compile(self._wm_rollout)
        self.imag_step = torch.compile(self._imag_step)
        self.start_features = torch.compile(self._start_features)

    @property
    def optimizers(self):
        return self.opt_wm, self.opt_pi, self.opt_v

    def state_dict(self) -> dict:
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "critic": self.critic.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict()}  # fmt: skip

    def load_state_dict(self, ck: dict):
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"]), self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"]), self.opt_v.load_state_dict(ck["opt_v"])

    def unflatten(self, x: torch.Tensor):
        return unflatten_state(x, self.nq)

    # ------------------------------------------------------------------ collection
    @torch.no_grad()
    def act(self, S, a_prev: torch.Tensor, f: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        """The policy's action in the simulator, with Gaussian exploration noise; fits the policy's input normalizer."""
        fk = self.fk(S[0])
        x = self.task.obs(S, fk, a_prev), self.task.goal(S, fk, f, self.goal_frames)
        if noise > 0:
            self.pi.norm.update(torch.cat(x, -1))
        a = self.pi(*x)
        return (a + noise * torch.randn_like(a)).clamp(-1, 1) if noise > 0 else a

    @torch.no_grad()
    def update_wm_norms(self, S, S1, u: torch.Tensor):
        self.wm.update_norms(S[0], S[1], u, S1[0], S1[1])

    # ------------------------------------------------------------------ world model
    def _wm_rollout(self, s0: torch.Tensor, u: torch.Tensor, s1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The k-step rollout from ``s0`` (B, D) under ``u`` (B, k, nu) against ``s1`` (B, k, D): the normalized squared
        error summed over the valid steps, mean over windows."""
        S, loss = self.unflatten(s0), 0.0
        for j in range(u.shape[1]):
            S = self.wm.step(*S, u[:, j])
            e = self.wm.error(*S, *self.unflatten(s1[:, j])).pow(2).mean(-1)
            loss = loss + (e * mask[:, j]).sum() / s0.shape[0]
        return loss

    def wm_update(self, buffer: ReplayBuffer) -> torch.Tensor:
        """``--wm_batches`` steps on k-step windows; returns the last step's loss."""
        args = self.args
        for _ in range(args.wm_batches):
            w = buffer.sample_windows(args.wm_batch, args.wm_window)
            loss = self.wm_rollout(w["s"][:, 0], w["u"], w["s1"], w["mask"])
            self.opt_wm.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
            self.opt_wm.step()
        return loss.detach()

    # ------------------------------------------------------------------ policy and critic
    def _start_features(self, S, f: torch.Tensor, a_prev: torch.Tensor):
        fk = self.task.fk(S[0])
        return fk, self.task.critic_features(S, fk, f, a_prev, self.goal_frames)

    def _imag_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, x: torch.Tensor):
        """One imagined step from the state ``S`` with the features ``x`` (the observation and the goal there, the policy's
        input and the critic's): the policy's action, the world model's next state and its forward kinematics, the reward,
        the termination probability and the features there."""
        task = self.task
        a = self.pi.from_features(x)
        S = self.wm.step(*S, task.setpoint(a, f))
        if self.bptt_decay != 1.0:  # --bptt_decay: the gradient through the imagined state, scaled at every step
            S = tuple(self.bptt_decay * x + (1 - self.bptt_decay) * x.detach() for x in S)
        fk, f = task.fk(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, task.reward(S, fk, f, a, a_prev), task.soft_termination(S, fk, f), task.critic_features(S, fk, f, a, self.goal_frames)

    def _starts(self, buffer: ReplayBuffer):
        age, env = buffer.sample_start_points(self.args.pi_batch)
        return self.unflatten(buffer.at("s", age, env)), buffer.at("f", age, env), buffer.at("a_prev", age, env)

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states, then the critic's on their lambda-returns. Returns
        (policy loss, value loss, imagined reward as a share of the maximum, gradient norm, imagined survival)."""
        args, horizon, B = self.args, self.horizon, self.args.pi_batch
        S, f, a_prev = self._starts(buffer)
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        fk, x = self.start_features(S, f, a_prev)
        X, rews, dones = [x], [], []
        for _ in range(horizon):
            S, fk, f, a_prev, r, d, x = self.imag_step(S, fk, f, a_prev, x)
            rews.append(r), dones.append(d), X.append(x)
        r, d = torch.stack(rews), torch.stack(dones)
        alive = survive(d, self.no_sync)  # c_{t+1}: cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, B, device=self.dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(horizon, device=self.dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0) + args.gamma**horizon * alive[-1] * self.critic(X[-1])
        loss = -J.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        finite_step(self.opt_pi, self.pi.parameters(), grad_norm, self.no_sync)  # the step, unless the gradient is non-finite
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)

        X = torch.stack(X).detach()  # (horizon + 1, B, D)
        r, d, c_t = r.detach(), d.detach(), c_t.detach()
        self.critic.norm.update(X)
        with torch.no_grad():
            V = self.critic(X)
            R, targets = V[-1], []
            for t in range(horizon - 1, -1, -1):
                R = r[t] + args.gamma * (1 - d[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1]).reshape(-1)
        inputs = X[:-1].reshape(-1, X.shape[-1])
        minibatch = inputs.shape[0] // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(inputs.shape[0], device=self.dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * minibatch : (i + 1) * minibatch]
                v_loss = (self.critic(inputs[idx]) - targets[idx]).pow(2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
        imag_r = (c_t * r).sum(0).mean() / (horizon * self.task.max_step_reward)
        return loss.detach(), v_loss.detach(), imag_r, grad_norm.detach(), alive[-1].mean().detach()


class RobomimicSuperTrack(RobomimicSuperMimic):
    kind = "accel"
    TERMS = ("keypoint_pos", "joint_pos", "finger_pos", "object_pos", "object_ori")

    def __init__(self, args, model, task: RobomimicTask, dev: str, substeps: int):
        super().__init__(args, model, task, dev, substeps)
        if self.bptt_decay != 1.0:
            raise ValueError("--bptt_decay is SuperMimic's option here")
        self.critic = self.opt_v = None
        self.bal = LossBalancer(len(self.TERMS)).to(dev)
        self.fail_w, self.reg, self.imag_noise = float(args.st_fail_w), float(args.st_reg), float(args.st_imag_noise)
        if self.fail_w < 0 or self.reg < 0 or self.imag_noise < 0:
            raise ValueError(f"--st_fail_w >= 0, --st_reg >= 0, --st_imag_noise >= 0, got {self.fail_w}, {self.reg}, {self.imag_noise}")
        self.step_worth = float(len(self.TERMS))  # a step's worth of loss, the failure penalty's unit: every term at its calibrated 1
        if getattr(args, "st_task_weights", False):  # the task's weights and widths in the calibration's place
            c = task.cfg
            per_unit = (c.w_keypoint / c.keypoint_std, c.w_joint / c.joint_std, 0.0, c.w_object_pos / c.object_pos_std,
                        c.w_object_ori / c.object_ori_std)  # fmt: skip
            self.bal.w.copy_(torch.tensor(per_unit, device=dev) / self.horizon)
            self.bal.scale.copy_(torch.tensor([c.keypoint_std, c.joint_std, 0.01, c.object_pos_std, c.object_ori_std]))
            self.bal.ready.fill_(True)
            self.step_worth = c.w_keypoint + c.w_joint + c.w_object_pos + c.w_object_ori  # every error at its kernel's width
        self.st_step = torch.compile(self._st_step)

    @property
    def optimizers(self):
        return self.opt_wm, self.opt_pi

    def state_dict(self) -> dict:
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "bal": self.bal.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict()}  # fmt: skip

    def load_state_dict(self, ck: dict):
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"]), self.bal.load_state_dict(ck["bal"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"])

    def _errors(self, S, fk, f: torch.Tensor) -> torch.Tensor:
        """(B, terms) L1 errors of the state ``S`` against the reference's frame ``f``."""
        task, ref = self.task, self.task.ref
        qpos, _ = S
        p, R, _, _ = task._objects(S)
        dq = qpos[..., task.act_q] - ref.ctrl[f]
        return torch.stack(((fk - ref.keypoints[f]).abs().sum(-1).mean(-1), dq[..., task.arm].abs().mean(-1), dq[..., task.fingers].abs().mean(-1),
                            (p - ref.obj_p[f]).abs().sum(-1).mean(-1), rot_angle(relative_rot(ref.obj_R[f], R)).mean(-1)), -1)  # fmt: skip

    def _st_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, noise: torch.Tensor):
        """One imagined step: the policy's action (with ``noise`` on it in the rollout), the world model's next state and its
        forward kinematics, the errors against the frame it ends at, the termination probability, the reward."""
        task = self.task
        a = self.pi(task.obs(S, fk, a_prev), task.goal(S, fk, f, self.goal_frames))
        S = self.wm.step(*S, task.setpoint(a + noise, f))
        fk, f = task.fk(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, self._errors(S, fk, f), task.soft_termination(S, fk, f), task.reward(S, fk, f, a, a_prev)

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states. Returns (policy loss, NaN, imagined reward as a share
        of the maximum, gradient norm, imagined survival), the layout of SuperMimic's."""
        horizon, B = self.horizon, self.args.pi_batch
        S, f, a_prev = self._starts(buffer)
        set_requires_grad(self.wm, False)
        fk = self.task.fk(S[0])
        errs, dones, acts, rews = [], [], [], []
        for _ in range(horizon):
            noise = self.imag_noise * torch.randn_like(a_prev) if self.imag_noise else torch.zeros_like(a_prev)
            S, fk, f, a_prev, e, d, r = self.st_step(S, fk, f, a_prev, noise)
            errs.append(e), dones.append(d), acts.append(a_prev), rews.append(r)
        E, d = torch.stack(errs), torch.stack(dones)  # (H, B, terms), (H, B)
        terms = E.sum(0).mean(0)
        if not self.bal.calibrated:
            self.bal.calibrate(terms.detach().clamp(min=1e-6), torch.ones_like(terms))
        loss = (self.bal.w * terms).sum()
        alive = survive(d, self.no_sync)  # cumprod(1 - d, 0)
        if self.fail_w:  # every imagined step after a failure costs st_fail_w steps' worth of loss
            loss = loss + self.fail_w * (self.step_worth / horizon) * (1 - alive).sum(0).mean()
        A = torch.stack(acts)
        loss = loss + self.reg * ((A**2).mean() + A.abs().mean())
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        finite_step(self.opt_pi, self.pi.parameters(), grad_norm, self.no_sync)  # the step, unless the gradient is non-finite
        set_requires_grad(self.wm, True)
        with torch.no_grad():
            c_t = torch.cat((torch.ones(1, B, device=self.dev), alive[:-1]), 0)
            imag_r = (c_t * torch.stack(rews)).sum(0).mean() / (horizon * self.task.max_step_reward)
        return loss.detach(), torch.full((), float("nan"), device=self.dev), imag_r, grad_norm.detach(), alive[-1].mean().detach()


ROBOMIMIC_ALGOS = {"supertrack": RobomimicSuperTrack, "supermimic": RobomimicSuperMimic}


def collect_step(sim, task, algo, buffer, args, a_prev: torch.Tensor, ep_len: torch.Tensor, acc: dict, step, add, fk):
    """One collection step on the loop's state kept in fixed tensors (the previous action, the episode lengths, the log's
    sums ``acc``), without a host read: ``step(u)`` the simulator's control step with its resets, ``add(transition)`` the
    replay's write, ``fk`` the forward kinematics of the next state's reward and success."""
    s, f = sim.state(), sim.f.clone()
    S = unflatten_state(s, task.nq)
    a = algo.act(S, a_prev, f, noise=args.action_noise)
    u = task.setpoint(a, f)
    s1, term, done = step(u)
    S1, f1 = unflatten_state(s1, task.nq), task.ref.next_frame(f)
    fk1 = fk(S1[0])
    algo.update_wm_norms(S, S1, u)
    add({"s": s, "s1": s1, "u": u, "a_prev": a_prev, "f": f, "done": done})
    ep_len.add_(1)
    acc["r"] += task.reward(S1, fk1, f1, a, a_prev).sum() / task.max_step_reward
    acc["object_err"] += task.errors(S1, fk1, f1)["object_pos"].mean(-1).sum()
    acc["success"] += (task.success(S1) & (f1 == task.ref.frames - 1)).sum()  # at the reference's last frame
    acc["steps"] += sim.N
    acc["done"] += done.sum()
    acc["term"] += term.sum()
    acc["ep_len"] += torch.where(done, ep_len, 0.0).sum()
    ep_len.masked_fill_(done, 0.0)
    a_prev.copy_(torch.where(done.unsqueeze(-1), 0.0, a))


class RobomimicCollector(Collector):
    """The collection step for --no_sync and --collect_graph: hand_fast.Collector's streams and capture around this task's
    step (``collect_step`` with RobomimicSim.step_nosync); with --compile_eager the step's forward kinematics as one compiled
    function, the agent's too."""

    def __init__(self, sim, task, algo, buffer, args):
        self.sim, self.task, self.algo, self.buffer, self.args = sim, task, algo, buffer, args
        self.graphed = bool(args.collect_graph)
        if self.graphed and buffer.size_t is None:
            raise ValueError("--collect_graph writes at the replay's device-side head (ReplayBuffer.use_device_ptr(device_size=True))")
        self.a_prev = torch.zeros(sim.N, task.nu, device=sim.device)
        self.ep_len = torch.zeros(sim.N, device=sim.device)
        self.acc = {k: torch.zeros((), device=sim.device) for k in ACC_KEYS}
        self.fk = task.fk
        if args.compile_eager:
            self.fk = algo.fk = torch.compile(task.fk)
        self.calls, self.stream, self.graph, self.record = 0, None, None, None

    @torch.no_grad()
    def core(self, physics=None):
        add = self.buffer.add_device if self.graphed else self.buffer.add
        collect_step(self.sim, self.task, self.algo, self.buffer, self.args, self.a_prev, self.ep_len, self.acc,
                     lambda u: self.sim.step_nosync(u, physics, self.fk), add, self.fk)  # fmt: skip
