"""SuperTrack and SuperMimic on MuscleVAE's muscle-actuated character (docs/musclevae_design.md §4), each with the settings
selected on LAFAN1: a world model of the bodies and the muscle state (worldmodels.muscle) fit on replay windows, and a
policy (policies.muscle_policy) trained by backpropagation through imagined rollouts on the task's terms
(tasks.muscle_tracking). The world model is frozen in the policy's update; gradients reach the policy through every
imagined step.

Imagination starts are replay states that the guard did not flag and whose clip holds the horizon, with the reference
frame their recorded step tracked, the episode step and MuscleVAE's off count before it; the reference then advances one
frame per step (the simulator's random switches are not imagined; a state after one is a replay state like any other).
--imag_start_perturb kicks every body of a start by one horizontal velocity N(0, s^2) m/s: the G1's root kick with the
joints unchanged. Terminations are MuscleVAE's, straight-through (tasks.muscle_tracking.soft_termination).

Settings in action units (the G1's normalized setpoints there, the muscle targets' a here) are mapped by
tasks.muscle_tracking.ACTION_UNIT, MuscleVAE's own collection noise (0.05 on a, MV:72) over SuperMimic's default (0.1):
--action_noise and --st_imag_noise times 0.5, and the task's action rate in the G1's units. Everything else keeps its
LAFAN1 value.

* SuperTrack: muscle targets about the reference action, the PD offset about the reference pose (the action
  l_ref / l_Tpose - 1 of the tracked frame); the world model fit by the L1 window loss on world-space body positions,
  velocities, rotations (log) and rotation rates, plus the muscle state; the policy by SuperTrack's L1 terms in the
  root's frame (algos.supertrack.LOCAL_TERMS: positions, velocities, rotations, rotation rates, heights, up direction, on
  all 23 bodies; no world-space root terms, tasks.muscle_tracking), each weighted once to 1 at calibration
  (LossBalancer) and the heights and up direction by --st_balance_w on top, the regularizer --st_reg on the offsets, the
  task's action-rate penalty at the reward's ratio to its full tracking score (as on LAFAN1), the failure penalty
  --st_fail_w on the soft terminations, and noise --st_imag_noise on the imagined offsets. The action rate is that of the
  executed targets without the noise, the first step's against the replay's previous action. Options: LAFAN1's
  --st_late_w, and --st_ref_replay, DiffMimic's demonstration replay in the imagined window (MuscleAgent._replay).
* SuperMimic: muscle targets about the T-pose (a = 0), the setpoint about the default pose; the world model fit by the
  normalized squared increment error; the policy maximizes the imagined return of the tracking reward (widths times
  --reward_std_scale) with the soft survival weight and a critic bootstrap at the horizon; the critic regresses
  GAE(lambda) targets.

Both algorithms take LAFAN1's --wm_input_noise and --wm_smooth_w for the world model's fit and --pi_rollout_clip for the
policy's gradient (algos.common), and BlinkMimic's options: the policy average (--pi_ema), Muon (--pi_opt, --wm_opt; --wm_opt also with the RigidFormer, on its blocks' matrices as in LAFAN1 training),
the updates written without host synchronization (--no_sync: ``wm_update_graphable``, ``policy_update_graphable``), which
--update_graphs captures as CUDA graphs.

MV = MuscleVAECore/Model/musclevae.py of wangchek/MuscleVAE_ @ 54c001c (no license: nothing is copied from it).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.kinematics import mat_mat, mat_vec
from ..core.muon import MuonAdam
from ..core.rotations import mat_to_6d, so3_log, yaw_mat
from ..policies.critic import Critic
from ..policies.muscle_policy import GOAL_DIM, OBS_DIM, MusclePolicy, goal_table, policy_observation
from ..robots.musclevae_char import NUM_MUSCLES
from ..tasks import muscle_tracking as mt
from ..worldmodels.muscle import MuscleWorldModel, increments, rigidformer_kwargs
from .blinkmimic import policy_average
from .common import (adam_kwargs, check_smoothness_args, clip_rollout_policy_grads, ema_step, gated_step, set_requires_grad, smoothness_penalty,
                     survival, wm_optimizer, zero_nonfinite_grads)
from .hoi_guard import DeviceSkippedUpdates
from .muscle_guard import SkippedUpdates, masked_update
from .supertrack import LOCAL_TERMS, WM_TERMS, LossBalancer

ACTION_UNIT = mt.ACTION_UNIT
WINDOW_KEYS = ("s", "s1", "fat", "fat1", "a", "done", "bad")
ST_WM_TERMS = WM_TERMS + ("fatigue",)
CRITIC_DIM = OBS_DIM + GOAL_DIM + 3 * mt.NB + 1
# SuperMimic's floors on the error normalization, per body (position, rotation, velocity, rotation rate: the G1 root's),
# and of the muscle state
ERR_FLOOR = ((3e-3,) * 3 + (5e-3,) * 3 + (2e-2,) * 3 + (5e-2,) * 3) * mt.NB + (1e-3,) * mt.FATIGUE_DIM


def replay_fields() -> dict:
    """A transition in the replay (ndm_gym.buffer.ReplayBuffer): the states before and after the step
    (tasks.muscle_tracking), the muscle states, the action, the reference frame the step tracked, the episode step and
    MuscleVAE's off count before it, the flags, and whether the guard flagged it."""
    f32, NB = torch.float32, mt.NB
    return {"s": ((NB, mt.BODY_DIM), f32), "s1": ((NB, mt.BODY_DIM), f32), "fat": (mt.FATIGUE_DIM, f32), "fat1": (mt.FATIGUE_DIM, f32),
            "a": (NUM_MUSCLES, f32), "t": ((), torch.long), "ep_step": ((), torch.int32), "off": ((), torch.int32), "done": ((), torch.bool),
            "term": ((), torch.bool), "bad": ((), torch.bool)}  # fmt: skip


def local_terms(S) -> tuple:
    """SuperTrack's Local(X) (algos.supertrack.local_features, Eq. 1-7) with the 3 x 3 products written elementwise:
    positions, velocities, 6-D rotations and rotation rates in the root's frame, the heights and the up direction."""
    p, R, v, w = S
    Rt = R[..., :1, :, :].transpose(-1, -2)
    return mat_vec(Rt, p - p[..., :1, :]), mat_vec(Rt, v), mat_to_6d(mat_mat(Rt, R)), mat_vec(Rt, w), p[..., 2], R[..., 0, 2, :]


def wm_terms(S_hat, F_hat, S, F) -> torch.Tensor:
    """L1 errors (..., 5) of world-space body positions, velocities, rotations (log), rotation rates and the muscle state."""
    return torch.stack(((S_hat[0] - S[0]).abs().sum((-1, -2)), (S_hat[2] - S[2]).abs().sum((-1, -2)),
                        so3_log(mat_mat(S[1].transpose(-1, -2), S_hat[1])).abs().sum((-1, -2)), (S_hat[3] - S[3]).abs().sum((-1, -2)),
                        (F_hat - F).abs().sum(-1)), -1)  # fmt: skip


class MuscleAgent:
    """One learner on the muscle character; ``kind`` is the world model's parameterization (SuperTrack "accel",
    SuperMimic "increment"); ``residual``: targets about the reference action."""

    kind: str = ""
    residual: bool = False
    imag_noise: float = 0.0  # std of Gaussian noise on imagined actions
    replay: float = 0.0  # SuperTrack's --st_ref_replay threshold (m; 0: off)

    def __init__(self, args, ref: mt.MuscleReference, dev):
        self.args, self.ref, self.dev = args, ref, dev
        self.wm = MuscleWorldModel(self.kind, hidden=args.wm_hidden, act=args.wm_act, layer_norm=not args.wm_no_ln,
                                   rigidformer=rigidformer_kwargs(args)).to(dev)  # fmt: skip
        self.pi = MusclePolicy().to(dev)
        self.goal = goal_table(ref)  # (F, 371)
        # optional: Adam's fused kernel (--fused_adam); with --update_graphs the
        # capturable one and device learning rates, one per optimizer (adam_kwargs at each); Muon on the MLPs' hidden matrices
        # (--wm_opt, --pi_opt), and with --wm_opt on the RigidFormer's block matrices, as LAFAN1 training's
        # (algos.common.wm_optimizer)
        self.adam_kw = {"fused": True} if getattr(args, "fused_adam", False) else {}
        self.update_graphs, self.no_sync = getattr(args, "update_graphs", False), getattr(args, "no_sync", False)
        opt_kw = lambda: adam_kwargs(args.lr, dev, self.update_graphs, self.adam_kw)  # noqa: E731
        self.opt_wm = wm_optimizer(self.wm, args, opt_kw(), self.update_graphs, self.adam_kw)
        if getattr(args, "pi_opt", "adam") == "muon":
            self.opt_pi = MuonAdam(self.pi, [self.pi.enc, self.pi.dec], lr=args.lr, betas=(0.9, 0.95), capturable=self.update_graphs, **self.adam_kw)
        else:
            self.opt_pi = torch.optim.Adam(self.pi.parameters(), betas=(0.9, 0.95), **opt_kw())
        # optional (--pi_ema d): an exponential moving average of the policy; checkpoints store it as "pi" and the trained one
        # as "pi_live"
        self.pi_avg = policy_average(self.pi, args.pi_ema) if getattr(args, "pi_ema", 0.0) > 0 else None
        self.B, self.H = args.pi_batch, args.horizon
        self.action_noise = ACTION_UNIT * args.action_noise  # the collection's, on a
        # updates skipped for a non-finite gradient; --no_sync: counted on the device, as are the policy updates whose 2B
        # candidate starts held fewer than B valid ones (``starts_nosync``)
        self.wm_skips, self.pi_skips = (DeviceSkippedUpdates(dev), DeviceSkippedUpdates(dev)) if self.no_sync else (SkippedUpdates(), SkippedUpdates())
        self.short_starts = torch.zeros((), dtype=torch.long, device=dev)
        # optional, as on LAFAN1 (algos.common): the model's fit with noise on its normalized inputs (--wm_input_noise) and a
        # finite-difference Jacobian penalty (--wm_smooth_w); each rollout's part of the policy gradient at most k times the
        # median (--pi_rollout_clip)
        self.input_noise = getattr(args, "wm_input_noise", 0.0)
        if self.input_noise < 0:
            raise ValueError(f"--wm_input_noise must be >= 0, got {self.input_noise}")
        self.smooth_w = check_smoothness_args(args)
        self.pi_rollout_clip = getattr(args, "pi_rollout_clip", 0.0)
        if self.pi_rollout_clip < 0:
            raise ValueError(f"--pi_rollout_clip must be >= 0 (0: off), got {self.pi_rollout_clip}")
        # compiled as the LAFAN1 algorithms' world-model rollout and imagination step; also the normalizers' updates of
        # each collection step (a few fused kernels instead of many eager ones)
        self.wm_window = torch.compile(self._wm_window)
        self.wm_smoothness = torch.compile(self._wm_smoothness)
        self.imag_step = torch.compile(self._imag_step)
        self.pi_norm_update = torch.compile(self._pi_norm_update)
        self.wm_norm_update = torch.compile(self._wm_norm_update)

    @property
    def optimizers(self) -> tuple:
        return (self.opt_wm, self.opt_pi)

    def state_dict(self) -> dict:
        """Everything a resumed run continues from: the networks with their normalizers and the optimizers' states."""
        sd = {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict()}
        if self.pi_avg is not None:  # --pi_ema: "pi" is the average
            sd.update(pi=self.pi_avg.module.state_dict(), pi_live=self.pi.state_dict(), pi_avg=self.pi_avg.state_dict())
        return sd

    def load_state_dict(self, ck: dict):
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"] if self.pi_avg is None else ck["pi_live"])
        if self.pi_avg is not None:
            self.pi_avg.load_state_dict(ck["pi_avg"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"])

    @torch.no_grad()
    def update_pi_avg(self):
        """--pi_ema: one step of the policy's moving average, after a policy update (a no-op without it)."""
        if self.pi_avg is not None:
            self.pi_avg.update_parameters(self.pi)

    def average_graphable(self):
        """--pi_ema in the graphable policy updates: the average's step without a host read (algos.common.ema_step); its first
        step, which copies the policy, runs eagerly before any capture."""
        if self.pi_avg is not None:
            ema_step(self.pi_avg, self.pi, self.args.pi_ema) if int(self.pi_avg.n_averaged) > 0 else self.pi_avg.update_parameters(self.pi)

    def act(self, obs: torch.Tensor, f: torch.Tensor, tap: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """The action (muscle targets a) of a step that tracks reference frame ``f``, and the policy's output; ``tap``
        (--pi_rollout_clip): a zero tensor added to the output, whose gradient is the output's adjoint."""
        out = self.pi(obs, self.goal[f])
        if tap is not None:
            out = out + tap
        return (out + self.ref.action[f] if self.residual else out), out

    # ------------------------------------------------------------------ normalizers
    def _pi_norm_update(self, obs: torch.Tensor, f: torch.Tensor):
        keep = torch.isfinite(obs).all(-1)
        masked_update(self.pi.obs_norm, obs, keep)
        masked_update(self.pi.goal_norm, self.goal[f], keep)

    @torch.no_grad()
    def update_pi_norms(self, obs: torch.Tensor, f: torch.Tensor):
        """The policy's normalizers from the observations it acts on and the goals of the frames ``f``."""
        self.pi_norm_update(obs, f)

    def _wm_norm_update(self, tr: dict):
        S, S1, keep = mt.unflatten_bodies(tr["s"]), mt.unflatten_bodies(tr["s1"]), ~tr["bad"]
        masked_update(self.wm.in_norm, self.wm.inputs(S, tr["fat"], tr["a"]), keep)
        masked_update(self.wm.out_norm, self.wm.targets(S, tr["fat"], S1, tr["fat1"]), keep)

    @torch.no_grad()
    def update_wm_norms(self, tr: dict):
        """The world model's normalizers from a step's transitions, without those the guard flagged."""
        self.wm_norm_update(tr)

    # ------------------------------------------------------------------ world model
    def _wm_window(self, w: dict):
        """The window loss of the open-loop rollout from each window's first state; a flagged step and the window's steps
        after it are left out."""
        S, F = mt.unflatten_bodies(w["s"][:, 0]), w["fat"][:, 0]
        mask = w["mask"] * torch.cumprod((~w["bad"]).to(w["mask"].dtype), 1)
        preds = []
        for j in range(w["a"].shape[1]):
            S, F = self.wm.step(S, F, w["a"][:, j], self.input_noise)
            preds.append((S, F))
        return self._window_loss(preds, w, mask)

    def _wm_smoothness(self, w: dict) -> torch.Tensor:
        """--wm_smooth_w's penalty (algos.common.smoothness_penalty, LAFAN1's) at the model's normalized inputs of the
        windows' first steps, those the window loss keeps."""
        x = self.wm.in_norm(self.wm.inputs(mt.unflatten_bodies(w["s"][:, 0]), w["fat"][:, 0], w["a"][:, 0]))
        a = self.args
        return smoothness_penalty(self.wm.net, x, a.wm_smooth_eps, a.wm_smooth_dirs, (w["mask"][:, 0] > 0) & ~w["bad"][:, 0])

    def _window_loss(self, preds: list, w: dict, mask: torch.Tensor):
        raise NotImplementedError

    def _wm_objective(self, out) -> torch.Tensor:
        return out

    def wm_update(self, buffer) -> torch.Tensor | None:
        """One world-model update on a batch of replay windows; skipped (and counted) if its gradient is not finite.
        Returns the loss per window step (the fit loss, without --wm_smooth_w's penalty), None when skipped."""
        a = self.args
        w = buffer.sample_windows(a.wm_batch, a.wm_window)
        loss = self._wm_objective(self.wm_window(w))
        objective = loss if self.smooth_w == 0 else loss + self.smooth_w * self.wm_smoothness(w)
        self.opt_wm.zero_grad(set_to_none=True)
        objective.backward()
        gn = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        ok = bool(torch.isfinite(gn))
        self.wm_skips(not ok)
        if not ok:
            return None
        self.opt_wm.step()
        return loss.detach() / a.wm_window

    def wm_update_graphable(self, buffer) -> tuple:
        """``wm_update`` for --no_sync and --update_graphs: no host synchronization, the optimizer step skipped on the device
        for a non-finite gradient (counted in ``wm_skips``). Returns the loss per window step and whether the update
        stepped (0-dim bool). SuperTrack's model loss balancer is calibrated by the first, eager update."""
        a = self.args
        w = buffer.sample_windows(a.wm_batch, a.wm_window)
        loss = self._wm_objective(self.wm_window(w))
        objective = loss if self.smooth_w == 0 else loss + self.smooth_w * self.wm_smoothness(w)
        self.opt_wm.zero_grad(set_to_none=True)
        objective.backward()
        gn = nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        ok = torch.isfinite(gn)
        self.wm_skips(~ok)
        zero_nonfinite_grads(self.wm.parameters(), ok)
        gated_step(self.opt_wm, ok)
        return loss.detach() / a.wm_window, ok

    # ------------------------------------------------------------------ imagination
    @torch.no_grad()
    def starts(self, buffer) -> dict:
        """B replay states (module docstring): the bodies (kicked with --imag_start_perturb), the muscle state, the frame
        the recorded step tracked, the episode step and the off count before it, the previous step's action if the
        episode and the replay have one."""
        D, B, H, dev = buffer.data, self.B, self.H, self.dev
        ages, envs, n = [], [], 0
        while n < B:
            age = torch.randint(0, buffer.size, (2 * B,), device=dev)
            env = torch.randint(0, buffer.N, (2 * B,), device=dev)
            row = buffer._row(age)
            t = D["t"][row, env]
            ok = ((t + H <= self.ref.clip_last[t]) & ~D["bad"][row, env]).nonzero(as_tuple=False).squeeze(-1)[: B - n]
            ages.append(age[ok]), envs.append(env[ok])
            n += len(ok)
        return self._starts_at(buffer, torch.cat(ages), torch.cat(envs))

    @torch.no_grad()
    def starts_nosync(self, buffer) -> dict:
        """``starts`` without reading the valid count on the host (--no_sync): 2B candidates (the
        default's first draw), stable-sorted valid first, the first B kept: the default's starts whenever that draw holds B
        valid ones; a draw with fewer is counted in ``short_starts``, which train_muscles checks at every log row."""
        D, B, H = buffer.data, self.B, self.H
        age, env = buffer.sample_start_points(2 * B)
        row = buffer._row(age)
        t = D["t"][row, env]
        ok = (t + H <= self.ref.clip_last[t]) & ~D["bad"][row, env]
        idx = torch.sort((~ok).to(torch.uint8), stable=True).indices[:B]
        self.short_starts.add_((ok.sum() < B).long())
        return self._starts_at(buffer, age[idx], env[idx])

    def _starts_at(self, buffer, age: torch.Tensor, env: torch.Tensor) -> dict:
        """The starts at replay ages ``age`` (0 = oldest) of ``env`` (``starts``)."""
        D, B, dev = buffer.data, self.B, self.dev
        row = buffer._row(age)
        p, R, v, w = mt.unflatten_bodies(D["s"][row, env])
        if self.args.imag_start_perturb != 0:
            kick = self.args.imag_start_perturb * torch.randn(B, 3, device=dev)
            kick[:, 2] = 0.0
            v = v + kick.unsqueeze(1)
        ep = D["ep_step"][row, env].long()
        return {"S": (p, R, v, w), "F": D["fat"][row, env], "t": D["t"][row, env], "ep": ep, "count": D["off"][row, env].float(),
                "a_prev": D["a"][buffer._row(age - 1), env], "has_prev": (ep > 0) & (age > 0)}  # fmt: skip

    def _imag_step(self, S, F, f, count, ep, a_prev, has_prev, tap=None, obs=None):
        """One imagined step that tracks frame ``f``: the policy's action (with noise for SuperTrack), the world model,
        MuscleVAE's soft termination; returns the next state, muscle state and off counts, the step's termination
        probability, its record for the loss (``_record``) and the action without the noise. The action rate is against
        ``a_prev`` where ``has_prev``, else 0. ``tap`` and ``obs`` (--pi_rollout_clip): a zero tensor added to the policy's
        output and the policy's observation of S and F, made by the caller. With SuperTrack's --st_ref_replay, the rollouts
        the replay replaces continue from the reference (``_replay``), and the record ends with which ones they are."""
        a, out = self.act(policy_observation(S, F) if obs is None else obs, f, tap)
        rate = mt.action_rate(a, torch.where(has_prev.unsqueeze(-1), a_prev, a))
        a_exec = a + self.imag_noise * torch.randn_like(a) if self.imag_noise else a
        S1, F1 = self.wm.step(S, F, a_exec)
        d, count = mt.soft_termination(S1, self.ref.head_z[f], count, ep)
        if not self.replay:
            return S1, F1, count, d, self._record(S1, f, out, rate), a
        rec = self._record(S1, f, out, rate)
        S1, count, hit = self._replay(S1, f, count)
        return S1, F1, count, d, rec + (hit,), a

    def _replay(self, S1, f: torch.Tensor, count: torch.Tensor):
        """SuperTrack's --st_ref_replay (DiffMimic's demonstration replay, arXiv 2304.03274) after an imagined step whose
        losses are recorded: every rollout whose bodies' mean distance to the reference frame ``f`` re-anchored to its root
        (tasks.muscle_tracking.anchored_reference, the reward's frame, since the task scores no drift) exceeds the
        threshold continues from that anchored reference state (its bodies' positions, rotations and velocities, with no
        gradient through it), its off count restarted. The muscle state stays the rollout's, and so does the failure
        penalty: the step's termination probability is the rollout's, and a failure before a replacement still costs every
        later step of the window. Returns the next step's state and off counts and which rollouts were replaced (B,)."""
        ref = mt.anchored_reference(tuple(x.detach() for x in S1), self.ref, f)
        hit = (S1[0].detach() - ref[0]).norm(dim=-1).mean(-1) > self.replay
        sel = lambda r, y: torch.where(hit.view(-1, *([1] * (y.dim() - 1))), r, y)  # noqa: E731
        return tuple(sel(r, y) for r, y in zip(ref, S1)), torch.where(hit, 0.0, count), hit

    def _record(self, S1, f: torch.Tensor, out: torch.Tensor, rate: torch.Tensor):
        raise NotImplementedError

    def imagine(self, st: dict, taps: list | None = None):
        """H imagined steps from the starts; per step the state, muscle state, off count, termination probability and
        record, and the frame the next step would track. ``taps`` (a list, for --pi_rollout_clip): per step the policy's
        observation and goal and a zero tensor added to its output are appended."""
        S, F, f, count, ep = st["S"], st["F"], st["t"], st["count"], st["ep"]
        a_prev, has_prev = st["a_prev"], st["has_prev"]
        steps = []
        for _ in range(self.H):
            if taps is None:
                S, F, count, d, rec, a_prev = self.imag_step(S, F, f, count, ep, a_prev, has_prev)
            else:
                taps.append((policy_observation(S, F), self.goal[f], torch.zeros(self.B, NUM_MUSCLES, device=self.dev, requires_grad=True)))
                S, F, count, d, rec, a_prev = self.imag_step(S, F, f, count, ep, a_prev, has_prev, taps[-1][2], taps[-1][0])
            steps.append((S, F, count, d, rec))
            f, ep, has_prev = f + 1, ep + 1, torch.ones_like(has_prev)
        return steps, f

    def _finish(self, loss: torch.Tensor, extra: tuple = (), taps: list | None = None) -> tuple[torch.Tensor, bool]:
        """The policy's backward pass (with ``taps``, --pi_rollout_clip's caps after it) and step, skipped (and counted) if
        its gradient or any of ``extra`` is not finite."""
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if taps:
            clip_rollout_policy_grads(self.pi, taps, self.pi_rollout_clip)
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        ok = bool(torch.stack([torch.isfinite(gn)] + [torch.isfinite(x).all() for x in extra]).all())
        self.pi_skips(not ok)
        if ok:
            self.opt_pi.step()
        return gn, ok

    def _finish_graphable(self, loss: torch.Tensor, extra: tuple = ()) -> tuple[torch.Tensor, torch.Tensor]:
        """``_finish`` without host synchronization (the graphable updates): the step skipped on the device where the
        gradient or any of ``extra`` is not finite (counted), then the policy average's step. Returns the gradient norm
        and whether the update stepped (0-dim bool)."""
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        ok = torch.stack([torch.isfinite(gn)] + [torch.isfinite(x).all() for x in extra]).all()
        self.pi_skips(~ok)
        zero_nonfinite_grads(self.pi.parameters(), ok)
        gated_step(self.opt_pi, ok)
        self.average_graphable()
        return gn, ok


class MuscleSuperTrack(MuscleAgent):
    """SuperTrack on the muscle character (module docstring)."""

    kind, residual = "accel", True

    def __init__(self, args, ref, dev):
        super().__init__(args, ref, dev)
        self.imag_noise = ACTION_UNIT * args.st_imag_noise
        with torch.no_grad():
            self.ref_local = local_terms(ref.state)  # Local(K) of every reference frame
        self.n_track = len(LOCAL_TERMS)
        self.n_el = torch.tensor([float(x[0].numel()) for x in self.ref_local], device=dev)
        self.bal_wm, self.bal_pi = LossBalancer(len(ST_WM_TERMS)).to(dev), LossBalancer(self.n_track).to(dev)
        self.term_mult = torch.ones(self.n_track, device=dev)
        self.term_mult[[LOCAL_TERMS.index("hei"), LOCAL_TERMS.index("up")]] = args.st_balance_w
        # the task's action-rate penalty at the reward's ratio to its full tracking score, on the n_track unit-scaled terms
        self.w_rate = -mt.MuscleRewardCfg().w_action_rate * self.n_track / mt.FULL_TRACKING_SCORE
        # --st_late_w (LAFAN1's): later imagined steps weigh more, 1 at the first step to 1 + w at the last
        late_w = getattr(args, "st_late_w", 0.0)
        self.late = 1 + late_w * torch.arange(self.H, device=dev) / max(self.H - 1, 1) if late_w != 0 else None
        # --st_ref_replay: DiffMimic's demonstration replay in the imagined window (MuscleAgent._replay)
        self.replay = getattr(args, "st_ref_replay", 0.0)
        if self.replay < 0:
            raise ValueError(f"--st_ref_replay must be >= 0 (0: off), got {self.replay}")

    def state_dict(self) -> dict:
        return {**super().state_dict(), "bal_wm": self.bal_wm.state_dict(), "bal_pi": self.bal_pi.state_dict()}

    def load_state_dict(self, ck: dict):
        super().load_state_dict(ck)
        self.bal_wm.load_state_dict(ck["bal_wm"]), self.bal_pi.load_state_dict(ck["bal_pi"])

    def _window_loss(self, preds, w, mask):
        """Per-term L1 errors summed over each window's valid steps, mean over windows: (5,)."""
        E = torch.stack([wm_terms(S, F, mt.unflatten_bodies(w["s1"][:, j]), w["fat1"][:, j]) for j, (S, F) in enumerate(preds)], 1)
        return (E * mask.unsqueeze(-1)).sum(1).mean(0)

    def _wm_objective(self, terms):
        if not self.bal_wm.calibrated:
            self.bal_wm.calibrate(terms.detach(), torch.ones_like(terms))
        return (self.bal_wm.w * terms).sum()

    def _record(self, S1, f, out, rate):
        """The step's L1 tracking errors per term (B, 6), the policy's offsets and the action rate."""
        E = torch.stack([(x - r[f]).abs().flatten(1).sum(-1) for x, r in zip(local_terms(S1), self.ref_local)], -1)
        return E, out, rate

    def _loss(self, steps: list, survive) -> tuple:
        """The policy loss of the imagined steps and their survival products ``survive(d)`` (cumprod(1 - d, 0)) of the
        termination probabilities d (H, B). The first call calibrates the loss balancer (eagerly, before any capture)."""
        a = self.args
        E, A = torch.stack([s[4][0] for s in steps]), torch.stack([s[4][1] for s in steps])  # (H, B, 6), (H, B, 284)
        if self.late is not None:  # --st_late_w, before the calibration (LAFAN1's)
            E = E * self.late[:, None, None]
        terms = E.sum(0).mean(0)
        if not self.bal_pi.calibrated:
            self.bal_pi.calibrate(terms.detach(), self.n_el * self.H)
        rate = torch.stack([s[4][2] for s in steps])  # (H, B)
        loss = (self.bal_pi.w * self.term_mult * terms).sum() + a.st_reg * ((A**2).mean() + A.abs().mean()) + self.w_rate * rate.mean()
        alive = survive(torch.stack([s[3] for s in steps]))
        if a.st_fail_w != 0:  # the failure penalty: each imagined step after a termination costs st_fail_w steps' worth of loss
            loss = loss + a.st_fail_w * (self.n_track / self.H) * (1 - alive).sum(0).mean()
        return loss, alive

    @property
    def graph_keys(self) -> tuple:
        """policy_update_graphable's outputs before ok."""
        return ("pi_loss", "pi_gradnorm", "imag_survival") + (("replay_frac",) if self.replay else ())

    def policy_update(self, buffer) -> dict:
        st = self.starts(buffer)
        set_requires_grad(self.wm, False)
        taps = [] if self.pi_rollout_clip > 0 else None
        steps, _ = self.imagine(st, taps)
        loss, alive = self._loss(steps, lambda d: torch.cumprod(1 - d, 0))
        gn, ok = self._finish(loss, taps=taps)
        set_requires_grad(self.wm, True)
        if not ok:
            return {"skipped": True}
        out = {"pi_loss": loss.detach(), "pi_gradnorm": gn.detach(), "imag_survival": alive[-1].detach().mean(), "skipped": False}
        if self.replay:  # --st_ref_replay: the fraction of imagined steps after which a rollout was replaced
            out["replay_frac"] = torch.stack([s[4][-1] for s in steps]).float().mean()
        return out

    def policy_update_graphable(self, buffer) -> tuple:
        """``policy_update`` for --no_sync and --update_graphs: the starts selected
        on the device, the survival products a loop of products, the step gated on the device, the policy average's step.
        Returns the default update's values (``graph_keys``) and whether it stepped (0-dim bool)."""
        st = self.starts_nosync(buffer)
        set_requires_grad(self.wm, False)
        steps, _ = self.imagine(st)
        loss, alive = self._loss(steps, survival)
        gn, ok = self._finish_graphable(loss)
        set_requires_grad(self.wm, True)
        out = (loss.detach(), gn.detach(), alive[-1].detach().mean())
        if self.replay:
            out += (torch.stack([s[4][-1] for s in steps]).float().mean(),)
        return out + (ok,)


class MuscleSuperMimic(MuscleAgent):
    """SuperMimic on the muscle character (module docstring)."""

    kind = "increment"

    def __init__(self, args, ref, dev):
        super().__init__(args, ref, dev)
        self.reward_cfg = mt.MuscleRewardCfg().scaled(args.reward_std_scale)
        self.err_floor = torch.tensor(ERR_FLOOR, device=dev)
        self.critic = Critic(CRITIC_DIM).to(dev)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), **adam_kwargs(args.lr, dev, self.update_graphs, self.adam_kw))
        self.critic_input_c = torch.compile(self.critic_input)

    @property
    def optimizers(self) -> tuple:
        return (self.opt_wm, self.opt_pi, self.opt_v)

    def state_dict(self) -> dict:
        return {**super().state_dict(), "critic": self.critic.state_dict(), "opt_v": self.opt_v.state_dict()}

    def load_state_dict(self, ck: dict):
        super().load_state_dict(ck)
        self.critic.load_state_dict(ck["critic"]), self.opt_v.load_state_dict(ck["opt_v"])

    def _window_loss(self, preds, w, mask):
        """SuperMimic's normalized squared error of the increment layout, summed over the valid steps, mean over windows."""
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        err = [increments(mt.unflatten_bodies(w["s1"][:, j]), w["fat1"][:, j], S, F) for j, (S, F) in enumerate(preds)]
        e = torch.stack([((x / std) ** 2).mean(-1) for x in err], 1)
        return (e * mask).sum() / e.shape[0]

    def _record(self, S1, f, out, rate):
        return mt.reward(S1, self.ref, f, self.reward_cfg, rate)

    def critic_input(self, S, F, f: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """The critic's input for a state that should match reference frame ``f`` (the next step tracks f + 1): the
        policy's observation and goal, the re-anchored body position errors in the root's heading frame, the off count."""
        ref_p = mt.anchored_reference(S, self.ref, f)[0]
        H = yaw_mat(S[1][..., 0, :, :]).unsqueeze(-3)
        e_body = mat_vec(H.transpose(-1, -2), ref_p - S[0]).flatten(-2)
        return torch.cat((policy_observation(S, F), self.goal[f + 1], e_body, (count / mt.ERR_LENGTH).unsqueeze(-1)), -1)

    def _actor_loss(self, steps: list, f_end: torch.Tensor, r: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        """Minus the mean imagined return: the rewards ``r`` (H, B) discounted and weighted by the survival products
        ``alive`` (H, B), the critic's value at the horizon."""
        a, dev, B, H = self.args, self.dev, self.B, self.H
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = a.gamma ** torch.arange(H, device=dev, dtype=torch.float32).unsqueeze(-1)
        S_H, F_H, count_H = steps[-1][0], steps[-1][1], steps[-1][2]
        J = (disc * c_t * r).sum(0) + (a.gamma**H) * alive[-1] * self.critic(self.critic_input(S_H, F_H, f_end - 1, count_H))
        return -J.mean()

    graph_keys = ("pi_loss", "v_loss", "imag_r", "pi_gradnorm", "imag_survival")  # policy_update_graphable's outputs before ok

    def policy_update(self, buffer) -> dict:
        st = self.starts(buffer)
        set_requires_grad(self.wm, False), set_requires_grad(self.critic, False)
        taps = [] if self.pi_rollout_clip > 0 else None
        steps, f_end = self.imagine(st, taps)
        r, d = torch.stack([s[4] for s in steps]), torch.stack([s[3] for s in steps])  # (H, B)
        alive = torch.cumprod(1 - d, 0)
        loss = self._actor_loss(steps, f_end, r, alive)
        gn, ok = self._finish(loss, (r, d), taps)
        set_requires_grad(self.wm, True), set_requires_grad(self.critic, True)
        if not ok:
            return {"skipped": True}
        v_loss = self._critic_update(st, steps, r, d)
        return {"pi_loss": loss.detach(), "v_loss": v_loss.detach(), "imag_r": (r.detach().mean() / mt.MAX_STEP_REWARD), "pi_gradnorm": gn.detach(),
                "imag_survival": alive[-1].detach().mean(), "skipped": False}  # fmt: skip

    def policy_update_graphable(self, buffer) -> tuple:
        """``policy_update`` for --no_sync and --update_graphs: the starts selected
        on the device, the survival products a loop of products, the actor's and the critic's steps (and the critic
        normalizer's update) skipped on the device where the gradient or the imagined rollout is not finite, the policy
        average's step. Returns the default update's values (``graph_keys``) and whether it stepped (0-dim bool)."""
        st = self.starts_nosync(buffer)
        set_requires_grad(self.wm, False), set_requires_grad(self.critic, False)
        steps, f_end = self.imagine(st)
        r, d = torch.stack([s[4] for s in steps]), torch.stack([s[3] for s in steps])  # (H, B)
        alive = survival(d)
        loss = self._actor_loss(steps, f_end, r, alive)
        gn, ok = self._finish_graphable(loss, (r, d))
        set_requires_grad(self.wm, True), set_requires_grad(self.critic, True)
        v_loss = self._critic_update(st, steps, r, d, ok)
        return loss.detach(), v_loss.detach(), r.detach().mean() / mt.MAX_STEP_REWARD, gn.detach(), alive[-1].detach().mean(), ok

    def _critic_update(self, st: dict, steps: list, r: torch.Tensor, d: torch.Tensor, ok: torch.Tensor | None = None) -> torch.Tensor:
        """The critic's GAE(lambda) targets over the imagined transitions and its minibatch steps, as SuperMimic's on
        LAFAN1; ``ok`` (the graphable update): a 0-dim bool, the normalizer's update and the steps skipped on the device
        where it is false. Returns the last minibatch's loss."""
        a, dev, H = self.args, self.dev, self.H
        with torch.no_grad():
            f0 = st["t"] - 1
            Xs = [self.critic_input_c(st["S"], st["F"], f0, st["count"])]
            Xs += [self.critic_input_c(s[0], s[1], f0 + k + 1, s[2]) for k, s in enumerate(steps)]
            X = torch.stack([x.detach() for x in Xs])  # (H + 1, B, D)
            if ok is None:
                self.critic.norm.update(X)
            else:
                self.critic.norm.update(X, gate=ok)
            V = self.critic(X)
            R, targets = V[-1], []
            rews, dones = r.detach(), d.detach()
            for t in range(H - 1, -1, -1):
                R = rews[t] + a.gamma * (1 - dones[t]) * ((1 - a.lam) * V[t + 1] + a.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])  # (H, B)
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        mb = Xf.shape[0] // a.critic_minibatches
        v_loss = torch.zeros((), device=dev)
        for _ in range(a.critic_epochs):
            perm = torch.randperm(Xf.shape[0], device=dev)
            for i in range(a.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                if ok is None:
                    self.opt_v.step()
                else:
                    zero_nonfinite_grads(self.critic.parameters(), ok)
                    gated_step(self.opt_v, ok)
        return v_loss


MUSCLE_ALGOS = {"supertrack": MuscleSuperTrack, "supermimic": MuscleSuperMimic}

__all__ = ["ACTION_UNIT", "MUSCLE_ALGOS", "MuscleSuperMimic", "MuscleSuperTrack", "WINDOW_KEYS", "local_terms", "replay_fields", "wm_terms"]
