"""SuperTrack and SuperMimic for quadrotor trajectory tracking (tasks.quadrotor_tracking), as hand-object tracking has them
(algos.hand_algos): both fit the world model on k-step windows of simulator transitions and train the policy through imagined
rollouts of the frozen world model from replayed states. The systems options are the hand task's (algos.hand_fast), with a
collection step of this task's (``QuadCollector``).

SuperTrack (``QuadSuperTrack``): the acceleration world model; the policy minimizes, over the imagined window, the L1 errors
of what the task tracks (``TERMS``: position, rotation, linear velocity, body rates), each term weighted once to 1 at
calibration (supertrack.LossBalancer) or, with ``--st_task_weights``, by the task's reward weight over its kernel's width;
plus the failure penalty on the task's soft termination (``--st_fail_w``) and the regularizer on the actions (``--st_reg``),
which are offsets about the reference's command. ``--st_imag_noise`` adds Gaussian noise to the imagined actions. No reward,
no critic.

SuperMimic (``QuadSuperMimic``; Liao et al., CoRL 2026): the increment world model; the policy maximizes the imagined tracking
return with a soft survival weight and a critic beyond the horizon. ``--bptt_decay lam`` scales the gradient through the
imagined state by lam at every step."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..policies.critic import Critic
from ..policies.quadrotor_policy import QuadPolicy
from ..robots.quadrotor import NUM_U
from ..tasks.hdmi import relative_rot, rot_angle
from ..tasks.quadrotor_tracking import GOAL_FRAME_DIM, OBS_DIM, STATE_DIM, QuadTask, unflatten_state
from ..worldmodels.quadrotor import QuadWorldModel
from .common import set_requires_grad
from .hand_fast import Collector, finite_step, optimizer, survive
from .supertrack import LossBalancer

WINDOW_KEYS = ("s", "s1", "u", "done")
ACC_KEYS = ("r", "steps", "done", "term", "ep_len", "pos_err")  # the log's sums over a row's collection steps


def replay_fields() -> dict:
    """A transition: the state, the command ``u``, the next state, the action before this step's, the frame."""
    return {
        "s": (STATE_DIM, torch.float32),
        "s1": (STATE_DIM, torch.float32),
        "u": (NUM_U, torch.float32),
        "a_prev": (NUM_U, torch.float32),
        "f": ((), torch.long),
        "done": ((), torch.bool),
    }


class QuadSuperMimic:
    kind = "increment"  # the world model's parameterization

    def __init__(self, args, task: QuadTask, dev: str, substeps: int = 4):
        self.args, self.task, self.dev = args, task, dev
        self.goal_frames, self.horizon = args.goal_frames, args.horizon
        self.bptt_decay = float(getattr(args, "bptt_decay", 1.0))
        if not 0 < self.bptt_decay <= 1:
            raise ValueError(f"--bptt_decay must be in (0, 1], got {self.bptt_decay}")
        goal_dim = GOAL_FRAME_DIM * self.goal_frames
        self.wm = QuadWorldModel(hidden=args.wm_hidden, act=args.wm_act, layer_norm=not args.wm_no_ln, kind=self.kind, substeps=substeps).to(dev)
        self.pi = QuadPolicy(OBS_DIM, goal_dim).to(dev)
        self.critic = Critic(OBS_DIM + goal_dim).to(dev)
        self.opt_wm = optimizer(args, dev, self.wm, "wm")
        self.opt_pi = optimizer(args, dev, self.pi, "pi")
        self.opt_v = optimizer(args, dev, self.critic, "critic")
        self.no_sync = bool(getattr(args, "no_sync", False))
        self.wm_rollout = torch.compile(self._wm_rollout)
        self.imag_step = torch.compile(self._imag_step)

    @property
    def optimizers(self):
        return self.opt_wm, self.opt_pi, self.opt_v

    def state_dict(self) -> dict:
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "critic": self.critic.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict()}  # fmt: skip

    def load_state_dict(self, ck: dict):
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"]), self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"]), self.opt_v.load_state_dict(ck["opt_v"])

    # ------------------------------------------------------------------ collection
    @torch.no_grad()
    def act(self, S, a_prev: torch.Tensor, f: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        """The policy's action in the simulator, with Gaussian exploration noise; fits the policy's input normalizer."""
        x = self.task.obs(S, a_prev), self.task.goal(S, f, self.goal_frames)
        if noise > 0:
            self.pi.norm.update(torch.cat(x, -1))
        a = self.pi(*x)
        return (a + noise * torch.randn_like(a)).clamp(-1, 1) if noise > 0 else a

    @torch.no_grad()
    def update_wm_norms(self, S, S1, u: torch.Tensor):
        self.wm.in_norm.update(self.wm.inputs(S, u))
        self.wm.out_norm.update(self.wm.targets(S, S1))
        if self.wm.err_norm is not None:  # the acceleration model: the increments' spread, the error's units
            self.wm.err_norm.update(self.wm.increments(S, S1))

    # ------------------------------------------------------------------ world model
    def _wm_rollout(self, s0: torch.Tensor, u: torch.Tensor, s1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The k-step rollout from ``s0`` (B, D) under ``u`` (B, k, 4) against ``s1`` (B, k, D): the normalized squared
        error summed over the valid steps, mean over windows."""
        S, loss = unflatten_state(s0), 0.0
        for j in range(u.shape[1]):
            S = self.wm.step(S, u[:, j])
            e = self.wm.error(S, unflatten_state(s1[:, j])).pow(2).mean(-1)
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
    def _imag_step(self, S, f: torch.Tensor, a_prev: torch.Tensor, x: torch.Tensor):
        """One imagined step from the state ``S`` with the features ``x`` (the observation and the goal there, the policy's
        input and the critic's): the policy's action, the world model's next state, the reward, the termination probability
        and the features there."""
        task = self.task
        a = self.pi.from_features(x)
        S = self.wm.step(S, task.setpoint(a, f))
        if self.bptt_decay != 1.0:  # --bptt_decay: the gradient through the imagined state, scaled at every step
            S = tuple(self.bptt_decay * x + (1 - self.bptt_decay) * x.detach() for x in S)
        f = task.ref.next_frame(f)
        return S, f, a, task.reward(S, f, a, a_prev), task.soft_termination(S, f), task.critic_features(S, f, a, self.goal_frames)

    def _starts(self, buffer: ReplayBuffer):
        age, env = buffer.sample_start_points(self.args.pi_batch)
        return unflatten_state(buffer.at("s", age, env)), buffer.at("f", age, env), buffer.at("a_prev", age, env)

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states, then the critic's on their lambda-returns. Returns
        (policy loss, value loss, imagined reward as a share of the maximum, gradient norm, imagined survival)."""
        args, horizon, B = self.args, self.horizon, self.args.pi_batch
        S, f, a_prev = self._starts(buffer)
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        x = self.task.critic_features(S, f, a_prev, self.goal_frames)
        X, rews, dones = [x], [], []
        for _ in range(horizon):
            S, f, a_prev, r, d, x = self.imag_step(S, f, a_prev, x)
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


class QuadSuperTrack(QuadSuperMimic):
    kind = "accel"
    TERMS = ("pos", "rot", "vel", "ang")

    def __init__(self, args, task: QuadTask, dev: str, substeps: int = 4):
        super().__init__(args, task, dev, substeps)
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
            weights, widths = (c.w_pos, c.w_rot, c.w_vel, c.w_ang), (c.pos_std, c.rot_std, c.vel_std, c.ang_std)
            self.bal.w.copy_(torch.tensor([w / s for w, s in zip(weights, widths, strict=True)], device=dev) / self.horizon)
            self.bal.scale.copy_(torch.tensor(widths))
            self.bal.ready.fill_(True)
            self.step_worth = sum(weights)  # every error at its kernel's width
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

    def _errors(self, S, f: torch.Tensor) -> torch.Tensor:
        """(B, terms) L1 errors of the state ``S`` against the reference's frame ``f``."""
        p, R, v, w = S
        ref = self.task.ref
        return torch.stack(((p - ref.p[f]).abs().sum(-1), rot_angle(relative_rot(ref.R[f], R)), (v - ref.v[f]).abs().sum(-1),
                            (w - ref.w[f]).abs().sum(-1)), -1)  # fmt: skip

    def _st_step(self, S, f: torch.Tensor, a_prev: torch.Tensor, noise: torch.Tensor):
        """One imagined step: the policy's action (with ``noise`` on it in the rollout), the world model's next state, the
        errors against the frame it ends at, the termination probability, the reward."""
        task = self.task
        a = self.pi(task.obs(S, a_prev), task.goal(S, f, self.goal_frames))
        S = self.wm.step(S, task.setpoint(a + noise, f))
        f = task.ref.next_frame(f)
        return S, f, a, self._errors(S, f), task.soft_termination(S, f), task.reward(S, f, a, a_prev)

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states. Returns (policy loss, NaN, imagined reward as a share
        of the maximum, gradient norm, imagined survival), the layout of SuperMimic's."""
        horizon, B = self.horizon, self.args.pi_batch
        S, f, a_prev = self._starts(buffer)
        set_requires_grad(self.wm, False)
        errs, dones, acts, rews = [], [], [], []
        for _ in range(horizon):
            noise = self.imag_noise * torch.randn(B, NUM_U, device=self.dev) if self.imag_noise else torch.zeros(B, NUM_U, device=self.dev)
            S, f, a_prev, e, d, r = self.st_step(S, f, a_prev, noise)
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


QUAD_ALGOS = {"supertrack": QuadSuperTrack, "supermimic": QuadSuperMimic}


def collect_step(sim, task, algo, buffer, args, a_prev: torch.Tensor, ep_len: torch.Tensor, acc: dict, step, add):
    """One collection step on the loop's state kept in fixed tensors (the previous action, the episode lengths, the log's
    sums ``acc``), without a host read: ``step(u)`` the simulator's control step with its resets, ``add(transition)`` the
    replay's write."""
    s, f = sim.state(), sim.f.clone()
    S = unflatten_state(s)
    a = algo.act(S, a_prev, f, noise=args.action_noise)
    u = task.setpoint(a, f)
    s1, term, done = step(u)
    S1, f1 = unflatten_state(s1), task.ref.next_frame(f)
    algo.update_wm_norms(S, S1, u)
    add({"s": s, "s1": s1, "u": u, "a_prev": a_prev, "f": f, "done": done})
    ep_len.add_(1)
    acc["r"] += task.reward(S1, f1, a, a_prev).sum() / task.max_step_reward
    acc["pos_err"] += task.errors(S1, f1)["pos"].sum()
    acc["steps"] += sim.N
    acc["done"] += done.sum()
    acc["term"] += term.sum()
    acc["ep_len"] += torch.where(done, ep_len, 0.0).sum()
    ep_len.masked_fill_(done, 0.0)
    a_prev.copy_(torch.where(done.unsqueeze(-1), 0.0, a))


class QuadCollector(Collector):
    """The collection step for --no_sync and --collect_graph: hand_fast.Collector's streams and capture around this task's
    step (``collect_step`` with QuadSim.step_nosync)."""

    def __init__(self, sim, task, algo, buffer, args):
        self.sim, self.task, self.algo, self.buffer, self.args = sim, task, algo, buffer, args
        self.graphed = bool(args.collect_graph)
        if self.graphed and buffer.size_t is None:
            raise ValueError("--collect_graph writes at the replay's device-side head (ReplayBuffer.use_device_ptr(device_size=True))")
        self.a_prev = torch.zeros(sim.N, NUM_U, device=sim.device)
        self.ep_len = torch.zeros(sim.N, device=sim.device)
        self.acc = {k: torch.zeros((), device=sim.device) for k in ACC_KEYS}
        self.calls, self.stream, self.graph, self.record = 0, None, None, None

    @torch.no_grad()
    def core(self, physics=None):
        add = self.buffer.add_device if self.graphed else self.buffer.add
        collect_step(self.sim, self.task, self.algo, self.buffer, self.args, self.a_prev, self.ep_len, self.acc,
                     lambda u: self.sim.step_nosync(u, physics), add)  # fmt: skip
