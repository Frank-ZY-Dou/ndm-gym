"""SuperTrack and SuperMimic for quadruped motion tracking (tasks.quadruped_tracking), after the hand-object task's
(algos.hand_algos). Both fit the world model on k-step windows of simulator transitions and train the policy through
imagined rollouts of the frozen world model from replayed states.

The world model is worldmodels.mjcf's model of the Go2's generalized coordinates (``world_model``): the inputs do not
depend on the trunk's heading and horizontal position, the free joint integrates as MuJoCo's, and the position servos'
forces (robots.go2.load_scene) are among its inputs; its step is one control step. SuperTrack's predicts accelerations,
SuperMimic's increments.

SuperTrack (``QuadrupedSuperTrack``): the policy minimizes L1 errors of the quantities the task tracks over the imagined
window, each term weighted once to 1 at calibration (or by the task's weights, ``--st_task_weights``); no reward, no critic.

SuperMimic (``QuadrupedSuperMimic``): the policy maximizes the imagined tracking return with a soft survival weight and a
critic beyond the horizon. ``--bptt_decay lam`` scales the gradient through the imagined state by lam at every step.

The systems options are the hand-object task's (algos.hand_fast, train_quadruped): the agents build their optimizers with
its ``optimizer`` and take the policy's step with its ``finite_step`` on the survival of ``survive``; ``QuadrupedCollector``
is its collection step without host synchronization with this task's log sums."""

from __future__ import annotations

import dataclasses

import mujoco
import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..policies.critic import Critic
from ..policies.quadruped_policy import QuadrupedPolicy
from ..robots.go2 import Go2Kinematics
from ..state import CONTROL_DT
from ..tasks.hdmi import relative_rot, rot_angle
from ..tasks.quadruped_tracking import QuadrupedTask, feet_local
from ..worldmodels.mjcf import MjcfLayout, MjcfWorldModel, mean_velocity_lead
from .common import set_requires_grad
from .hand_fast import Collector, finite_step, optimizer, survive
from .supertrack import LossBalancer

WINDOW_KEYS = ("s", "s1", "u", "done")


def replay_fields(task=QuadrupedTask) -> dict:
    """A transition: the state, the position targets ``u``, the next state, the action before this step's, the frame;
    the sizes ``task``'s (the Go2's by default; a humanoid's, tasks.humanoid_tracking)."""
    return {
        "s": (task.state_dim, torch.float32),
        "s1": (task.state_dim, torch.float32),
        "u": (task.num_joints, torch.float32),
        "a_prev": (task.num_joints, torch.float32),
        "f": ((), torch.long),
        "done": ((), torch.bool),
    }


def world_model(model: mujoco.MjModel, kind: str, hidden=(1024, 1024, 1024), act: str = "silu", layer_norm: bool = True,
                mean_velocity: bool = False) -> MjcfWorldModel:
    """The world model of the scene ``model`` (the Go2's, or a humanoid's) over one control step. ``mean_velocity``
    (SuperTrack's accelerations only; train_humanoid --accel_mean_velocity): the position advances with the mean velocity
    of the control step's physics steps under a constant acceleration (worldmodels.mjcf.mean_velocity_lead), not the
    step's last velocity."""
    lay = dataclasses.replace(MjcfLayout.from_model(model), dt=CONTROL_DT)
    lead = {"pos_lead": mean_velocity_lead(model, round(CONTROL_DT / model.opt.timestep))} if mean_velocity and kind == "accel" else {}
    return MjcfWorldModel(lay, hidden=hidden, act=act, layer_norm=layer_norm, kind=kind, **lead)


class QuadrupedSuperMimic:
    kind = "increment"  # the world model's parameterization

    def __init__(self, args, model: mujoco.MjModel, kin: Go2Kinematics, task: QuadrupedTask, dev: str):
        self.args, self.kin, self.task, self.dev = args, kin, task, dev
        self.goal_frames, self.horizon = args.goal_frames, args.horizon
        self.bptt_decay = float(getattr(args, "bptt_decay", 1.0))
        if not 0 < self.bptt_decay <= 1:
            raise ValueError(f"--bptt_decay must be in (0, 1], got {self.bptt_decay}")
        self.no_sync = bool(getattr(args, "no_sync", False))  # train_quadruped --no_sync: the policy update without host reads
        self.fk = kin.forward  # the collection's forward kinematics, run eagerly; train_quadruped --compile_eager compiles them
        self.unflatten = task.unflatten  # the state's layout: the Go2's, or a humanoid's (tasks.humanoid_tracking)
        goal_dim = task.goal_frame_dim * self.goal_frames
        self.wm = world_model(model, self.kind, hidden=args.wm_hidden, act=args.wm_act, layer_norm=not args.wm_no_ln,
                              mean_velocity=bool(getattr(args, "accel_mean_velocity", False))).to(dev)
        self.bound = float(getattr(args, "action_bound", 1.0))  # train_humanoid --action_bound: the actions' range (the Go2's: 1)
        self.pi = QuadrupedPolicy(task.obs_dim, goal_dim, num_actions=task.num_joints, bound=self.bound).to(dev)
        self.critic = Critic(task.obs_dim + goal_dim).to(dev)
        self.opt_wm = optimizer(args, dev, self.wm, "wm")
        self.opt_pi = optimizer(args, dev, self.pi, "pi")
        self.opt_v = optimizer(args, dev, self.critic, "critic")
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

    # ------------------------------------------------------------------ collection
    @torch.no_grad()
    def act(self, S, a_prev: torch.Tensor, f: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        """The policy's action in the simulator, with Gaussian exploration noise; fits the policy's input normalizer."""
        fk = self.fk(S[0])
        x = self.task.obs(S, fk, a_prev), self.task.goal(S, fk, f, self.goal_frames)
        if noise > 0:
            self.pi.norm.update(torch.cat(x, -1))
        a = self.pi(*x)
        return (a + noise * torch.randn_like(a)).clamp(-self.bound, self.bound) if noise > 0 else a

    @torch.no_grad()
    def update_wm_norms(self, S, S1, u: torch.Tensor):
        self.wm.update_norms(*S, u, *S1)

    # ------------------------------------------------------------------ world model
    def _wm_rollout(self, s0: torch.Tensor, u: torch.Tensor, s1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The k-step rollout from ``s0`` (B, D) under ``u`` (B, k, J) against ``s1`` (B, k, D): the normalized squared
        error summed over the valid steps, mean over windows."""
        (qpos, qvel), loss = self.unflatten(s0), 0.0
        for j in range(u.shape[1]):
            qpos, qvel = self.wm.step(qpos, qvel, u[:, j])
            e = self.wm.error(qpos, qvel, *self.unflatten(s1[:, j])).pow(2).mean(-1)
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
        fk = self.kin.forward(S[0])
        return fk, self.task.critic_features(S, fk, f, a_prev, self.goal_frames)

    def _imag_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor):
        """One imagined step: the policy's action, the world model's next state and its forward kinematics, the reward,
        the termination probability and the critic's input there."""
        task = self.task
        a = self.pi(task.obs(S, fk, a_prev), task.goal(S, fk, f, self.goal_frames))
        S = self.wm.step(*S, task.setpoint(a, f))
        if self.bptt_decay != 1.0:  # --bptt_decay: the gradient through the imagined state, scaled at every step
            S = tuple(self.bptt_decay * x + (1 - self.bptt_decay) * x.detach() for x in S)
        fk, f = self.kin.forward(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, task.reward(S, fk, f, a, a_prev), task.soft_termination(S, fk, f), task.critic_features(S, fk, f, a, self.goal_frames)

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states, then the critic's on their lambda-returns. Returns
        (policy loss, value loss, imagined reward as a share of the maximum, gradient norm, imagined survival)."""
        args, horizon, B = self.args, self.horizon, self.args.pi_batch
        age, env = buffer.sample_start_points(B)
        S, f, a_prev = self.unflatten(buffer.at("s", age, env)), buffer.at("f", age, env), buffer.at("a_prev", age, env)
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        fk, x = self.start_features(S, f, a_prev)
        X, rews, dones = [x], [], []
        for _ in range(horizon):
            S, fk, f, a_prev, r, d, x = self.imag_step(S, fk, f, a_prev)
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


class QuadrupedSuperTrack(QuadrupedSuperMimic):
    """SuperTrack on the quadruped task. The policy minimizes, over the imagined window, the L1 errors of ``TERMS``: the
    feet about the trunk in its heading frame and in the world, the trunk's position, height and rotation, the joints, the
    trunk's linear and angular velocity and the joints' velocities, each term weighted once to 1 at calibration
    (LossBalancer). ``--st_task_weights``: no calibration; the task's six tracked quantities each weigh, per unit of error
    and imagined step, the task's reward weight over its kernel's width (tasks.quadruped_tracking.RewardCfg), the other terms
    nothing. Plus the failure penalty on the task's soft termination (``--st_fail_w``: every imagined step after a failure
    costs that many steps' worth of loss) and the regularizer on the actions (``--st_reg``), which are offsets about the
    reference pose. ``--st_imag_noise`` adds Gaussian noise to the imagined actions. No reward, no critic: ``policy_update``
    returns NaN for the value loss and the imagined reward of the same rollouts for the log."""

    kind = "accel"
    TERMS = ("feet", "feet_world", "trunk_pos", "trunk_height", "trunk_ori", "joint_pos", "lin_vel", "ang_vel", "joint_vel")

    def __init__(self, args, model: mujoco.MjModel, kin: Go2Kinematics, task: QuadrupedTask, dev: str):
        super().__init__(args, model, kin, task, dev)
        if self.bptt_decay != 1.0:
            raise ValueError("--bptt_decay is SuperMimic's option here")
        self.critic = self.opt_v = None
        self.bal = LossBalancer(len(self.TERMS)).to(dev)
        self.term_mult = torch.ones(len(self.TERMS), device=dev)  # each term's weight on top of the balancing (a subclass's)
        self.fail_w, self.reg, self.imag_noise = float(args.st_fail_w), float(args.st_reg), float(args.st_imag_noise)
        if self.fail_w < 0 or self.reg < 0 or self.imag_noise < 0:
            raise ValueError(f"--st_fail_w >= 0, --st_reg >= 0, --st_imag_noise >= 0, got {self.fail_w}, {self.reg}, {self.imag_noise}")
        self.step_worth = float(len(self.TERMS))  # a step's worth of loss, the failure penalty's unit: every term at its calibrated 1
        if getattr(args, "st_task_weights", False):  # the task's weights and widths in the calibration's place
            per_unit, self.step_worth = self.task_weights(task.cfg)
            self.bal.w.copy_(torch.tensor([per_unit.get(t, 0.0) for t in self.TERMS], device=dev) / self.horizon)
            self.bal.ready.fill_(True)
        self.st_step = torch.compile(self._st_step)

    @staticmethod
    def task_weights(c) -> tuple[dict, float]:
        """--st_task_weights: per term of ``TERMS`` the task's reward weight over its kernel's width (the others nothing),
        and a step's worth of loss, the reward's tracking weights' sum."""
        per_unit = {"feet": c.w_feet / c.feet_std, "trunk_pos": c.w_trunk_pos / c.trunk_pos_std, "trunk_ori": c.w_trunk_ori / c.trunk_ori_std,
                    "joint_pos": c.w_joint / c.joint_std, "lin_vel": c.w_lin_vel / c.lin_vel_std, "ang_vel": c.w_ang_vel / c.ang_vel_std}  # fmt: skip
        return per_unit, c.w_feet + c.w_trunk_pos + c.w_trunk_ori + c.w_joint + c.w_lin_vel + c.w_ang_vel

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
        """(B, terms) L1 errors of the state ``S`` against the reference's frame ``f``: per foot or joint the mean, per
        vector the sum of its components' absolute values."""
        qpos, qvel = S
        keypoints, R = fk
        ref, task = self.task.ref, self.task
        errs = [(feet_local(fk) - task.ref_feet[f]).abs().sum(-1).mean(-1), (keypoints[..., 1:, :] - ref.keypoints[f][..., 1:, :]).abs().sum(-1).mean(-1),
                (qpos[..., :3] - ref.qpos[f, :3]).abs().sum(-1), (qpos[..., 2] - ref.qpos[f, 2]).abs(), rot_angle(relative_rot(ref.R[f], R)),
                (qpos[..., 7:] - ref.qpos[f, 7:]).abs().mean(-1), (qvel[..., :3] - ref.qvel[f, :3]).abs().sum(-1),
                (qvel[..., 3:6] - ref.qvel[f, 3:6]).abs().sum(-1), (qvel[..., 6:] - ref.qvel[f, 6:]).abs().mean(-1)]  # fmt: skip
        return torch.stack(errs, -1)

    def _st_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, noise: torch.Tensor):
        """One imagined step: the policy's action (with ``noise`` on it in the rollout), the world model's next state and
        its forward kinematics, the errors against the frame it ends at, the termination probability, the reward."""
        task = self.task
        a = self.pi(task.obs(S, fk, a_prev), task.goal(S, fk, f, self.goal_frames))
        S = self.wm.step(*S, task.setpoint(a + noise, f))
        fk, f = self.kin.forward(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, self._errors(S, fk, f), task.soft_termination(S, fk, f), task.reward(S, fk, f, a, a_prev)

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states. Returns (policy loss, NaN, imagined reward as a share
        of the maximum, gradient norm, imagined survival), the layout of SuperMimic's."""
        horizon, B = self.horizon, self.args.pi_batch
        age, env = buffer.sample_start_points(B)
        S, f, a_prev = self.unflatten(buffer.at("s", age, env)), buffer.at("f", age, env), buffer.at("a_prev", age, env)
        set_requires_grad(self.wm, False)
        fk = self.kin.forward(S[0])
        errs, dones, acts, rews = [], [], [], []
        for _ in range(horizon):
            noise = self.imag_noise * torch.randn(B, self.task.num_joints, device=self.dev)
            S, fk, f, a_prev, e, d, r = self.st_step(S, fk, f, a_prev, noise)
            errs.append(e), dones.append(d), acts.append(a_prev), rews.append(r)
        E, d = torch.stack(errs), torch.stack(dones)  # (H, B, terms), (H, B)
        terms = E.sum(0).mean(0)
        if not self.bal.calibrated:
            self.bal.calibrate(terms.detach().clamp(min=1e-6), torch.ones_like(terms))
        loss = (self.bal.w * self.term_mult * terms).sum()
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


class QuadrupedCollector(Collector):
    """train_quadruped's collection step for --no_sync (hand_fast.Collector) with the task's state layout and log sums (the
    Go2's trunk and feet errors, a humanoid's own: the task's ``log_errors``)."""

    acc_keys = ("r", "steps", "done", "term", "ep_len", "trunk_err", "feet_err")

    def __init__(self, sim, task, algo, buffer, args):
        super().__init__(sim, task, algo, buffer, args)
        self.unflatten = task.unflatten

    def log_errors(self, errors: dict):
        (_, first), (_, second) = self.task.log_errors
        self.acc["trunk_err"] += errors[first].sum()
        self.acc["feet_err"] += errors[second].mean(-1).sum()


QUADRUPED_ALGOS = {"supertrack": QuadrupedSuperTrack, "supermimic": QuadrupedSuperMimic}
