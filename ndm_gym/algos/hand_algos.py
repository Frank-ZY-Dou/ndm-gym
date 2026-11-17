"""SuperTrack and SuperMimic for hand-object tracking (tasks.hand_tracking). Both fit the world model on k-step windows
of simulator transitions and train the policy through imagined rollouts of the frozen world model from replayed states.

SuperTrack (algos/supertrack.py, hoi_algos.HOISuperTrack; ``HandSuperTrack``): the acceleration world model; the policy
minimizes L1 errors of the quantities the task tracks over the imagined window, each term weighted once to 1 at
calibration or by the task's weights; no reward, no critic.

SuperMimic (algos/supermimic.py; Liao et al., CoRL 2026; ``HandSuperMimic``): the increment world model; the policy
maximizes the imagined tracking return with a soft survival weight and a critic beyond the horizon (paper §3, Eq. 1-3).
``--bptt_decay lam`` scales the gradient through the imagined state by lam at every step."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..policies.critic import Critic
from ..policies.hand_policy import HandPolicy
from ..robots.hands import HandKinematics
from ..tasks.hand_tracking import HandTask, unflatten_state
from ..tasks.hdmi import relative_rot, rot_angle
from ..worldmodels.hand import HandWorldModel
from .common import set_requires_grad
from .hand_fast import finite_step, optimizer, survive
from .supertrack import LossBalancer

if TYPE_CHECKING:
    import argparse

    from ..core.muon import MuonAdam
    from ..worldmodels.hand import HandState

WINDOW_KEYS = ("s", "s1", "u", "done")
START_KEYS = ("s", "a_prev", "f")  # an imagination start's fields (sample_valid_starts_nosync; train_hand --layout: algos.hand_layout)


def replay_fields(task: HandTask) -> dict:
    """A transition: the state, the position targets ``u``, the next state, the action before this step's, the frame."""
    return {
        "s": (task.state_dim, torch.float32),
        "s1": (task.state_dim, torch.float32),
        "u": (task.kin.num_joints, torch.float32),
        "a_prev": (task.kin.num_joints, torch.float32),
        "f": ((), torch.long),
        "done": ((), torch.bool),
    }


class HandSuperMimic:
    kind = "increment"  # the world model's parameterization
    decode_replay_state = staticmethod(unflatten_state)  # the task state consumed by the imagined rollout

    def __init__(self, args: argparse.Namespace, kin: HandKinematics, task: HandTask, dev: str) -> None:
        self.args, self.kin, self.task, self.dev = args, kin, task, dev
        self.goal_frames, self.horizon = args.goal_frames, args.horizon
        self.bptt_decay = float(getattr(args, "bptt_decay", 1.0))
        if not 0 < self.bptt_decay <= 1:
            raise ValueError(f"--bptt_decay must be in (0, 1], got {self.bptt_decay}")
        goal_dim = task.goal_frame_dim * self.goal_frames
        self.wm = HandWorldModel(kin.num_joints, kin.palm_translations, kin.palm_keypoints.tolist(), hidden=args.wm_hidden, act=args.wm_act,
                                 layer_norm=not args.wm_no_ln, kind=self.kind).to(dev)  # fmt: skip
        self.pi = HandPolicy(task.obs_dim, goal_dim, kin.num_joints).to(dev)
        self.critic = Critic(task.obs_dim + goal_dim).to(dev)
        # Adam at --lr, the policy's with betas (0.9, 0.95), unless a systems option says otherwise (algos.hand_fast)
        self.opt_wm = optimizer(args, dev, self.wm, "wm")
        self.opt_pi = optimizer(args, dev, self.pi, "pi")
        self.opt_v = optimizer(args, dev, self.critic, "critic")
        self.no_sync = bool(getattr(args, "no_sync", False))  # train_hand --no_sync: the policy update without host reads
        self.lap = None  # train_hand --phase_times without --update_graphs: hand_fast.PhaseTimes.lap, for the critic's update timed apart
        self.fk = kin.forward  # the collection's forward kinematics, run eagerly; train_hand --compile_eager compiles them
        self.wm_fk = kin.palm if getattr(args, "wm_palm_fk", False) else kin.forward  # the world model's fit: --wm_palm_fk, the palm's alone
        self.wm_rollout = torch.compile(self._wm_rollout)
        self.imag_step = torch.compile(self._imag_step)
        self.start_features = torch.compile(self._start_features)
        # train_hand --compile_updates: the updates' parts that run eagerly otherwise, compiled (algos.hand_fast)
        self.compile_updates = bool(getattr(args, "compile_updates", False))
        self.reuse_input = bool(getattr(args, "sm_reuse_input", False))  # train_hand --sm_reuse_input (SuperMimic's imagined steps)
        self.sm_objective, self.critic_targets, self.critic_loss = self._sm_objective, self._critic_targets, self._critic_loss
        if self.compile_updates:
            self.sm_objective, self.critic_targets, self.critic_loss = (torch.compile(f) for f in (self._sm_objective, self._critic_targets,
                                                                                                    self._critic_loss))  # fmt: skip
        # train_hand --compile_rollout: the policy update's imagined rollout (the start's kinematics and the horizon's steps) as one
        # compiled function, whose backward is one graph (the agent's ``_rollout``; algos.hand_fast)
        self.rollout = torch.compile(self._rollout) if getattr(args, "compile_rollout", False) else None

    @property
    def optimizers(self) -> tuple[torch.optim.Adam | MuonAdam, ...]:
        return self.opt_wm, self.opt_pi, self.opt_v

    def state_dict(self) -> dict:
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "critic": self.critic.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict()}  # fmt: skip

    def load_state_dict(self, ck: dict) -> None:
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"]), self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"]), self.opt_v.load_state_dict(ck["opt_v"])

    # ------------------------------------------------------------------ collection
    @torch.no_grad()
    def act(self, S: HandState, a_prev: torch.Tensor, f: torch.Tensor, noise: float = 0.0, z: torch.Tensor | None = None) -> torch.Tensor:
        """The policy's action in the simulator, with Gaussian exploration noise; fits the policy's input normalizer.
        ``z``: the noise's standard normal draw, made before (train_hand --compile_collect); None: drawn here."""
        x = self.task.obs(S, self.fk(S[0]), a_prev), self.task.goal(S, f, self.goal_frames)
        if noise > 0:
            self.pi.norm.update(torch.cat(x, -1))
        a = self.pi(*x)
        if noise > 0 and z is None:
            z = torch.randn_like(a)
        return (a + noise * z).clamp(-1, 1) if noise > 0 else a

    @torch.no_grad()
    def update_wm_norms(self, S: HandState, S1: HandState, u: torch.Tensor) -> None:
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
            S = self.wm.step(S, self.wm_fk(S[0]), u[:, j])
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

    # ------------------------------------------------------------------ train_hand --layout (algos.layouts.Split, algos.hand_layout)
    average = None  # with --pi_ema: the policy average of this object's policy (hand_fast.PolicyAverage), on the policy learner

    @property
    def pi_avg(self) -> torch.nn.Module | None:
        """The policy average's model (layouts.Split reads its count of steps), or None."""
        return None if self.average is None else self.average.avg

    def sample_valid_starts_nosync(self, buffer: ReplayBuffer, B: int) -> tuple[dict, dict]:
        """The policy update's draw of ``B`` imagination starts from the replay, without a host read: each the state, the
        frame and the action before it (``START_KEYS``; every replayed state is a start, the reference holding its last
        frame), without a history. Under --layout the policy learner's draw is the batch the actor drew (layouts.GivenStarts)."""
        age, env = buffer.sample_start_points(B)
        return {"s": buffer.at("s", age, env), "f": buffer.at("f", age, env), "a_prev": buffer.at("a_prev", age, env)}, {}

    def wm_update_graphable(self, buffer: ReplayBuffer) -> tuple:
        """--layout: the iteration's world-model updates (``wm_update``), the last loss in a tuple (layouts.Split's graph)."""
        return (self.wm_update(buffer),)

    def policy_update_graphable(self, buffer: ReplayBuffer) -> tuple:
        """--layout: the policy update and the policy average's step, as hand_fast.update_functions calls them, in one
        function for the policy learner's CUDA graph; the loop counts the average's steps."""
        out = self.policy_update(buffer)
        if self.average is not None:
            self.average.step()
        return out

    # ------------------------------------------------------------------ policy (Eq. 2) and critic (Eq. 3)
    def _start_features(self, S, f: torch.Tensor, a_prev: torch.Tensor):
        fk = self.kin.forward(S[0])
        return fk, self.task.critic_features(S, fk, f, a_prev, self.goal_frames)

    def _imag_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, x: torch.Tensor | None = None):
        """One imagined step: the policy's action, the world model's next state and its forward kinematics, the reward,
        the termination probability and the critic's input there. ``x`` (--sm_reuse_input): the policy's input at ``S``,
        the critic's input the step before returned (the same observation and goal of the same state; None: computed)."""
        task = self.task
        if x is None:
            a = self.pi(task.obs(S, fk, a_prev), task.goal(S, f, self.goal_frames))
        else:
            a = self.pi.from_input(x)
        S = self.wm.step(S, fk, task.setpoint(a, f))
        if self.bptt_decay != 1.0:  # --bptt_decay: the gradient through the imagined state, scaled at every step
            S = tuple(self.bptt_decay * x + (1 - self.bptt_decay) * x.detach() for x in S)
        fk, f = self.kin.forward(S[0]), task.ref.next_frame(f)
        return S, fk, f, a, task.reward(S, fk, f, a, a_prev), task.soft_termination(S, fk, f), task.critic_features(S, fk, f, a, self.goal_frames)

    def _rollout(self, S, f: torch.Tensor, a_prev: torch.Tensor):
        """--compile_rollout: ``policy_update``'s imagined rollout from the start ``S`` at frame ``f`` after ``a_prev``, the
        start's features and ``horizon`` imagined steps, as one function. Returns the critic's inputs (horizon + 1 of them),
        the rewards and the termination probabilities (horizon each)."""
        fk, x = self._start_features(S, f, a_prev)
        X, rews, dones = [x], [], []
        for _ in range(self.horizon):
            if self.reuse_input:
                S, fk, f, a_prev, r, d, x = self._imag_step(S, fk, f, a_prev, x)
            else:
                S, fk, f, a_prev, r, d, x = self._imag_step(S, fk, f, a_prev)
            rews.append(r), dones.append(d), X.append(x)
        return X, rews, dones

    def _sm_objective(self, rews: list, dones: list, x_last: torch.Tensor):
        """The policy's loss from the imagined rewards and termination probabilities (``horizon`` of (B,) each) and the
        critic's input at the window's end: minus the mean discounted return with the soft survival and the critic's value
        beyond the horizon (Eq. 2). Also returns the rewards, the termination probabilities, the survival and its weights."""
        args, horizon = self.args, self.horizon
        r, d = torch.stack(rews), torch.stack(dones)
        alive = survive(d, self.no_sync)  # c_{t+1}: cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, r.shape[1], device=self.dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(horizon, device=self.dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0) + args.gamma**horizon * alive[-1] * self.critic(x_last)
        return -J.mean(), r, d, alive, c_t

    def _critic_targets(self, X: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """The critic's normalizer fit to the imagined window's inputs ``X`` (horizon + 1, B, D), then the lambda-returns of
        the rewards ``r`` and termination probabilities ``d`` (horizon, B) on its values, flattened (Eq. 3)."""
        args = self.args
        self.critic.norm.update(X)
        with torch.no_grad():
            V = self.critic(X)
            R, targets = V[-1], []
            for t in range(self.horizon - 1, -1, -1):
                R = r[t] + args.gamma * (1 - d[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            return torch.stack(targets[::-1]).reshape(-1)

    def _critic_loss(self, inputs: torch.Tensor, targets: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """The critic's squared error on the minibatch ``idx`` of its inputs and targets."""
        return (self.critic(inputs[idx]) - targets[idx]).pow(2).mean()

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states, then the critic's on their lambda-returns. Returns
        (policy loss, value loss, imagined reward as a share of the maximum, gradient norm, imagined survival)."""
        args, horizon, B = self.args, self.horizon, self.args.pi_batch
        s, _ = self.sample_valid_starts_nosync(buffer, B)  # under --layout: the batch the actor drew
        S, f, a_prev = self.decode_replay_state(s["s"]), s["f"], s["a_prev"]
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        if self.rollout is not None:  # --compile_rollout: the rollout below as one compiled function
            X, rews, dones = self.rollout(S, f, a_prev)
        else:
            fk, x = self.start_features(S, f, a_prev)
            X, rews, dones = [x], [], []
            for _ in range(horizon):
                if self.reuse_input:  # the policy's input is the critic's input of the step before (start_features' at the start)
                    S, fk, f, a_prev, r, d, x = self.imag_step(S, fk, f, a_prev, x)
                else:
                    S, fk, f, a_prev, r, d, x = self.imag_step(S, fk, f, a_prev)
                rews.append(r), dones.append(d), X.append(x)
        loss, r, d, alive, c_t = self.sm_objective(rews, dones, X[-1])  # --compile_updates: compiled
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        finite_step(self.opt_pi, self.pi.parameters(), grad_norm, self.no_sync)  # the step, unless the gradient is non-finite
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
        if self.lap is not None:  # --phase_times: the policy's step up to here, the critic's update from here
            self.lap("policy")

        X = torch.stack(X).detach()  # (horizon + 1, B, D)
        r, d, c_t = r.detach(), d.detach(), c_t.detach()
        targets = self.critic_targets(X, r, d)  # --compile_updates: compiled
        inputs = X[:-1].reshape(-1, X.shape[-1])
        minibatch = inputs.shape[0] // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(inputs.shape[0], device=self.dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * minibatch : (i + 1) * minibatch]
                v_loss = self.critic_loss(inputs, targets, idx)  # --compile_updates: compiled
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
        if self.lap is not None:
            self.lap("critic")
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

    def __init__(self, args: argparse.Namespace, kin: HandKinematics, task: HandTask, dev: str) -> None:
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
        self.ref_local = kin.fingertips_about_palms(ref.keypoints, ref.palm_R)
        self.st_step = torch.compile(self._st_step)
        # --compile_updates: the start states' kinematics (SuperMimic's start_features compile them) and the objective compiled
        self.start_fk, self.st_objective = kin.forward, self._st_objective
        if self.compile_updates:
            self.start_fk, self.st_objective = torch.compile(kin.forward), torch.compile(self._st_objective)

    @property
    def optimizers(self) -> tuple[torch.optim.Adam | MuonAdam, ...]:
        return self.opt_wm, self.opt_pi

    def state_dict(self) -> dict:
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "bal": self.bal.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict()}  # fmt: skip

    def load_state_dict(self, ck: dict) -> None:
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"]), self.bal.load_state_dict(ck["bal"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"])

    def _errors(self, S, fk, f: torch.Tensor) -> torch.Tensor:
        """(B, terms) L1 errors of the state ``S`` against the reference's frame ``f``."""
        q, qd, p, R, v, w = S
        keypoints, palm_R = fk
        ref = self.task.ref
        local = self.kin.fingertips_about_palms(keypoints, palm_R)
        errs = [(keypoints - ref.keypoints[f]).abs().sum(-1).mean(-1), (local - self.ref_local[f]).abs().sum(-1).mean(-1),
                rot_angle(relative_rot(ref.palm_R[f], palm_R)).mean(-1), (q - ref.q[f])[..., self.kin.finger_joints].abs().mean(-1),
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

    def _rollout(self, S, f: torch.Tensor, a_prev: torch.Tensor, noises: list):
        """--compile_rollout: ``policy_update``'s imagined rollout from the start ``S`` at frame ``f`` after ``a_prev`` as one
        function, the start's kinematics and a step per noise of ``noises`` (drawn before, in the steps' order). Returns the
        errors (horizon, B, terms), the termination probabilities (horizon, B), the actions and the rewards."""
        fk = self.kin.forward(S[0])
        errs, dones, acts, rews = [], [], [], []
        for noise in noises:
            S, fk, f, a_prev, e, d, r = self._st_step(S, fk, f, a_prev, noise)
            errs.append(e), dones.append(d), acts.append(a_prev), rews.append(r)
        return torch.stack(errs), torch.stack(dones), acts, rews

    def _st_objective(self, terms: torch.Tensor, d: torch.Tensor, acts: list):
        """The policy's loss from the window's summed errors per term ``terms``, the termination probabilities ``d``
        (horizon, B) and the actions (``horizon`` of (B, J)): the weighted terms, the failure penalty on the survival and
        the actions' regularizer. Also returns the survival."""
        horizon = self.horizon
        loss = (self.bal.w * self.term_scale * terms).sum()
        alive = survive(d, self.no_sync)  # cumprod(1 - d, 0)
        if self.fail_w:  # every imagined step after a failure costs st_fail_w steps' worth of loss
            loss = loss + self.fail_w * (self.step_worth / horizon) * (1 - alive).sum(0).mean()
        A = torch.stack(acts)
        return loss + self.reg * ((A**2).mean() + A.abs().mean()), alive

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """One policy step on imagined rollouts from replayed states. Returns (policy loss, NaN, imagined reward as a share
        of the maximum, gradient norm, imagined survival), the layout of SuperMimic's."""
        horizon, B = self.horizon, self.args.pi_batch
        s, _ = self.sample_valid_starts_nosync(buffer, B)  # under --layout: the batch the actor drew
        S, f, a_prev = self.decode_replay_state(s["s"]), s["f"], s["a_prev"]
        set_requires_grad(self.wm, False)
        if self.rollout is not None:  # --compile_rollout: the rollout below as one compiled function, the noises drawn first
            noises = [self.imag_noise * torch.randn_like(a_prev) if self.imag_noise else torch.zeros_like(a_prev) for _ in range(horizon)]
            E, d, acts, rews = self.rollout(S, f, a_prev, noises)
        else:
            fk = self.start_fk(S[0]) if self.start_fk is not None else None  # --compile_updates: compiled for hand-object tracking
            errs, dones, acts, rews = [], [], [], []
            for _ in range(horizon):
                noise = self.imag_noise * torch.randn_like(a_prev) if self.imag_noise else torch.zeros_like(a_prev)
                S, fk, f, a_prev, e, d, r = self.st_step(S, fk, f, a_prev, noise)
                errs.append(e), dones.append(d), acts.append(a_prev), rews.append(r)
            E, d = torch.stack(errs), torch.stack(dones)  # (H, B, terms), (H, B)
        terms = E.sum(0).mean(0)
        if not self.bal.calibrated:
            self.bal.calibrate(terms.detach().clamp(min=1e-6)[self.cal_idx], torch.ones_like(terms))
        loss, alive = self.st_objective(terms, d, acts)  # --compile_updates: compiled
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
