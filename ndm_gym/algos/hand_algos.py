"""SuperTrack and SuperMimic for hand-object tracking (tasks.hand_tracking). Both fit the world model on k-step windows
of simulator transitions and train the policy through imagined rollouts of the frozen world model from replayed states.

SuperTrack (algos/supertrack.py, hoi_algos.HOISuperTrack; ``HandSuperTrack``): the acceleration world model; the policy
minimizes L1 errors of the quantities the task tracks over the imagined window, each term weighted once to 1 at
calibration or by the task's weights; no reward, no critic.

SuperMimic (algos/supermimic.py; Liao et al., CoRL 2026; ``HandSuperMimic``): the increment world model; the policy
maximizes the imagined tracking return with a soft survival weight and a critic beyond the horizon (paper §3, Eq. 1-3).
``--bptt_decay lam`` scales the gradient through the imagined state by lam at every step."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..policies.critic import Critic
from ..policies.hand_policy import HandPolicy
from ..robots.xhand import NUM_J, NUM_PALM, HandKinematics
from ..tasks.hand_tracking import GOAL_FRAME_DIM, OBS_DIM, STATE_DIM, HandTask, unflatten_state
from ..tasks.hdmi import relative_rot, rot_angle
from ..worldmodels.hand import HandWorldModel
from .common import set_requires_grad
from .hand_fast import finite_step, optimizer, survive
from .supertrack import LossBalancer

WINDOW_KEYS = ("s", "s1", "u", "done")


def replay_fields() -> dict:
    """A transition: the state, the position targets ``u``, the next state, the action before this step's, the frame."""
    return {
        "s": (STATE_DIM, torch.float32),
        "s1": (STATE_DIM, torch.float32),
        "u": (NUM_J, torch.float32),
        "a_prev": (NUM_J, torch.float32),
        "f": ((), torch.long),
        "done": ((), torch.bool),
    }


class HandSuperMimic:
    kind = "increment"  # the world model's parameterization

    def __init__(self, args, kin: HandKinematics, task: HandTask, dev: str):
        self.args, self.kin, self.task, self.dev = args, kin, task, dev
        self.goal_frames, self.horizon = args.goal_frames, args.horizon
        self.bptt_decay = float(getattr(args, "bptt_decay", 1.0))
        if not 0 < self.bptt_decay <= 1:
            raise ValueError(f"--bptt_decay must be in (0, 1], got {self.bptt_decay}")
        goal_dim = GOAL_FRAME_DIM * self.goal_frames
        self.wm = HandWorldModel(hidden=args.wm_hidden, act=args.wm_act, layer_norm=not args.wm_no_ln, kind=self.kind).to(dev)
        self.pi = HandPolicy(OBS_DIM, goal_dim).to(dev)
        self.critic = Critic(OBS_DIM + goal_dim).to(dev)
        # Adam at --lr, the policy's with betas (0.9, 0.95), unless a systems option says otherwise (algos.hand_fast)
        self.opt_wm = optimizer(args, dev, self.wm, "wm")
        self.opt_pi = optimizer(args, dev, self.pi, "pi")
        self.opt_v = optimizer(args, dev, self.critic, "critic")
        self.no_sync = bool(getattr(args, "no_sync", False))  # train_hand --no_sync: the policy update without host reads
        self.fk = kin.forward  # the collection's forward kinematics, run eagerly; train_hand --compile_eager compiles them
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
        x = self.task.obs(S, self.fk(S[0]), a_prev), self.task.goal(S, f, self.goal_frames)
        if noise > 0:
            self.pi.norm.update(torch.cat(x, -1))
        a = self.pi(*x)
        return (a + noise * torch.randn_like(a)).clamp(-1, 1) if noise > 0 else a

    @torch.no_grad()
    def update_wm_norms(self, S, S1, u: torch.Tensor):
        self.wm.in_norm.update(self.wm.inputs(S, self.fk(S[0]), u))
        self.wm.out_norm.update(self.wm.targets(S, S1))
        if self.wm.err_norm is not None:  # the acceleration model: the increments' spread, the error's units
            self.wm.err_norm.update(self.wm.increments(S, S1))

    # ------------------------------------------------------------------ world model (Eq. 1)
    def _wm_rollout(self, s0: torch.Tensor, u: torch.Tensor, s1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The k-step rollout from ``s0`` (B, D) under ``u`` (B, k, J) against ``s1`` (B, k, D): the normalized squared
        error summed over the valid steps, mean over windows."""
        S, loss = unflatten_state(s0), 0.0
        for j in range(u.shape[1]):
            S = self.wm.step(S, self.kin.forward(S[0]), u[:, j])
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

    # ------------------------------------------------------------------ policy (Eq. 2) and critic (Eq. 3)
    def _start_features(self, S, f: torch.Tensor, a_prev: torch.Tensor):
        fk = self.kin.forward(S[0])
        return fk, self.task.critic_features(S, fk, f, a_prev, self.goal_frames)

    def _imag_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor):
        """One imagined step: the policy's action, the world model's next state and its forward kinematics, the reward,
        the termination probability and the critic's input there."""
        task = self.task
        a = self.pi(task.obs(S, fk, a_prev), task.goal(S, f, self.goal_frames))
        S = self.wm.step(S, fk, task.setpoint(a, f))
        if self.bptt_decay != 1.0:  # --bptt_decay: the gradient through the imagined state, scaled at every step
            S = tuple(self.bptt_decay * x + (1 - self.bptt_decay) * x.detach() for x in S)
        fk, f = self.kin.forward(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, task.reward(S, fk, f, a, a_prev), task.soft_termination(S, fk, f), task.critic_features(S, fk, f, a, self.goal_frames)

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states, then the critic's on their lambda-returns. Returns
        (policy loss, value loss, imagined reward as a share of the maximum, gradient norm, imagined survival)."""
        args, horizon, B = self.args, self.horizon, self.args.pi_batch
        age, env = buffer.sample_start_points(B)
        S, f, a_prev = unflatten_state(buffer.at("s", age, env)), buffer.at("f", age, env), buffer.at("a_prev", age, env)
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


class HandSuperTrack(HandSuperMimic):
    """SuperTrack on the hand-object task. The policy minimizes, over the imagined window, the L1 errors of what the task
    tracks (``TERMS``: the keypoints' positions in the world and, for the fingertips, about the palm in the palm's frame;
    the palm's rotation; the finger joints; the object's position and rotation), each term weighted once to 1 at
    calibration (LossBalancer); ``--st_vel_terms`` adds the velocities' (``VEL_TERMS``: the joints', the object's linear
    and angular), calibrated as the others; ``--st_obj_w`` then weighs the object's position and rotation terms that many
    times. ``--st_unit_scales``: terms of one unit share a calibrated weight (``UNIT_OF``): the fingertips' positions about
    the palm and the object's position weigh per metre as the keypoints' positions in the world, the object's rotation per
    radian as the palm's; a clip whose reference pose alone does not move the object starts with a large error of the
    object and so, calibrated by itself, with a small weight on it. ``--st_task_weights``: no calibration; the terms are
    the task's five tracked quantities and each weighs, per unit of error and imagined step, the task's reward weight over
    its kernel's width (tasks.hand_tracking.RewardCfg: an error as large as the width costs the reward's weight), so that
    the loss trades the hand's pose against the object as the task does; the fingertips about the palm, which the task
    does not track, weigh nothing. Plus the failure penalty on the task's soft termination (``--st_fail_w``, as LAFAN1's
    and HOI's: every imagined step after a failure costs that many steps' worth of loss) and the regularizer on the
    actions (``--st_reg``), which are offsets about the reference pose. ``--st_imag_noise`` adds Gaussian noise to the imagined actions. No reward, no
    critic: ``policy_update`` returns NaN for the value loss and the imagined reward of the same rollouts for the log."""

    kind = "accel"
    TERMS = ("keypoint_pos", "keypoint_pos_local", "palm_ori", "finger_pos", "object_pos", "object_ori")
    VEL_TERMS = ("joint_vel", "object_lin_vel", "object_ang_vel")
    OBJECT_TERMS = ("object_pos", "object_ori")
    UNIT_OF = {"keypoint_pos_local": "keypoint_pos", "object_pos": "keypoint_pos", "object_ori": "palm_ori"}  # --st_unit_scales

    def __init__(self, args, kin: HandKinematics, task: HandTask, dev: str):
        super().__init__(args, kin, task, dev)
        if self.bptt_decay != 1.0:
            raise ValueError("--bptt_decay is SuperMimic's option here")
        self.critic = self.opt_v = None
        self.vel_terms = bool(getattr(args, "st_vel_terms", False))
        self.terms = self.TERMS + (self.VEL_TERMS if self.vel_terms else ())
        self.bal = LossBalancer(len(self.terms)).to(dev)
        self.obj_w, self.fail_w, self.reg = float(getattr(args, "st_obj_w", 1.0)), float(args.st_fail_w), float(args.st_reg)
        self.imag_noise = float(args.st_imag_noise)
        if self.obj_w <= 0 or self.fail_w < 0 or self.reg < 0 or self.imag_noise < 0:
            raise ValueError(f"--st_obj_w > 0, --st_fail_w >= 0, --st_reg >= 0, --st_imag_noise >= 0, got {self.obj_w}, {self.fail_w}, "
                             f"{self.reg}, {self.imag_noise}")  # fmt: skip
        self.term_scale = torch.tensor([self.obj_w if t in self.OBJECT_TERMS else 1.0 for t in self.terms], device=dev)
        # the term whose error at calibration sets each term's weight: itself, or with --st_unit_scales the term of its unit
        units = self.UNIT_OF if getattr(args, "st_unit_scales", False) else {}
        self.cal_idx = torch.tensor([self.terms.index(units.get(t, t)) for t in self.terms], device=dev)
        self.step_worth = float(len(self.terms))  # a step's worth of loss, the failure penalty's unit: every term at its calibrated 1
        if getattr(args, "st_task_weights", False):  # the task's weights and widths in the calibration's place
            if self.vel_terms or units:
                raise ValueError("--st_task_weights weighs the task's five tracked quantities: not with --st_vel_terms or --st_unit_scales")
            c = task.cfg
            per_unit = {"keypoint_pos": c.w_keypoint / c.keypoint_std, "keypoint_pos_local": 0.0, "palm_ori": c.w_palm_ori / c.palm_ori_std,
                        "finger_pos": c.w_finger / c.finger_std, "object_pos": c.w_object_pos / c.object_pos_std,
                        "object_ori": c.w_object_ori / c.object_ori_std}  # fmt: skip
            self.bal.w.copy_(torch.tensor([per_unit[t] for t in self.terms], device=dev) / self.horizon)
            self.bal.scale.copy_(torch.tensor([c.keypoint_std, c.keypoint_std, c.palm_ori_std, c.finger_std, c.object_pos_std, c.object_ori_std]))
            self.bal.ready.fill_(True)
            self.step_worth = c.w_keypoint + c.w_palm_ori + c.w_finger + c.w_object_pos + c.w_object_ori  # every error at its kernel's width
        ref = task.ref
        self.ref_local = (ref.keypoints[:, 1:] - ref.keypoints[:, :1]) @ ref.palm_R  # the fingertips about the palm, the palm's frame
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
        q, qd, p, R, v, w = S
        keypoints, palm_R = fk
        ref = self.task.ref
        local = (keypoints[..., 1:, :] - keypoints[..., :1, :]) @ palm_R
        errs = [(keypoints - ref.keypoints[f]).abs().sum(-1).mean(-1), (local - self.ref_local[f]).abs().sum(-1).mean(-1),
                rot_angle(relative_rot(ref.palm_R[f], palm_R)), (q[..., NUM_PALM:] - ref.q[f][..., NUM_PALM:]).abs().mean(-1),
                (p - ref.p[f]).abs().sum(-1), rot_angle(relative_rot(ref.R[f], R))]  # fmt: skip
        if self.vel_terms:
            errs += [(qd - ref.qd[f]).abs().mean(-1), (v - ref.v[f]).abs().sum(-1), (w - ref.w[f]).abs().sum(-1)]
        return torch.stack(errs, -1)

    def _st_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, noise: torch.Tensor):
        """One imagined step: the policy's action (with ``noise`` on it in the rollout), the world model's next state and
        its forward kinematics, the errors against the frame it ends at, the termination probability, the reward."""
        task = self.task
        a = self.pi(task.obs(S, fk, a_prev), task.goal(S, f, self.goal_frames))
        S = self.wm.step(S, fk, task.setpoint(a + noise, f))
        fk, f = self.kin.forward(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, self._errors(S, fk, f), task.soft_termination(S, fk, f), task.reward(S, fk, f, a, a_prev)

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states. Returns (policy loss, NaN, imagined reward as a share
        of the maximum, gradient norm, imagined survival), the layout of SuperMimic's."""
        horizon, B = self.horizon, self.args.pi_batch
        age, env = buffer.sample_start_points(B)
        S, f, a_prev = unflatten_state(buffer.at("s", age, env)), buffer.at("f", age, env), buffer.at("a_prev", age, env)
        set_requires_grad(self.wm, False)
        fk = self.kin.forward(S[0])
        errs, dones, acts, rews = [], [], [], []
        for _ in range(horizon):
            noise = self.imag_noise * torch.randn(B, NUM_J, device=self.dev) if self.imag_noise else torch.zeros(B, NUM_J, device=self.dev)
            S, fk, f, a_prev, e, d, r = self.st_step(S, fk, f, a_prev, noise)
            errs.append(e), dones.append(d), acts.append(a_prev), rews.append(r)
        E, d = torch.stack(errs), torch.stack(dones)  # (H, B, terms), (H, B)
        terms = E.sum(0).mean(0)
        if not self.bal.calibrated:
            self.bal.calibrate(terms.detach().clamp(min=1e-6)[self.cal_idx], torch.ones_like(terms))
        loss = (self.bal.w * self.term_scale * terms).sum()
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


HAND_ALGOS = {"supertrack": HandSuperTrack, "supermimic": HandSuperMimic}
