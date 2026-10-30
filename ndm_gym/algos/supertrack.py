"""SuperTrack (Fussell, Bergamin, Holden, TOG 2021) on the shared pipeline.

Only the algorithm differs from SuperMimic. The world model predicts accelerations that are integrated with
semi-implicit Euler (paper §3.4, Eq. 8-12; :mod:`ndm_gym.worldmodels.accel`) and is fit with an L1 window loss on
world-space body states (Alg. 1). The policy outputs PD offsets about the reference pose and is trained by BPTT
through the frozen world model on an L1 tracking loss in root-local space (§3.3, §3.5, Alg. 2), plus world-space root
position and rotation terms as in the paper's §4.2 extension for tasks that score global drift. There is no reward,
no survival weight and no critic. Added on top of the paper (each with an ablation flag): the task's own action-rate
and joint-limit penalties, at the reward's ratio to its full tracking score. Optional tricks behind flags (off by
default) include a failure penalty on imagined steps past the task's termination thresholds (``--st_fail_w``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..core.kinematics import ArticulationKinematics
from ..core.rotations import mat_to_6d, so3_log, yaw_mat
from ..motion.library import MotionLib
from ..robots.g1 import G1_JOINT_NAMES, action_scale
from ..state import CONTROL_DT, normalize_xi, unflatten_state
from ..tasks.tracking import MAX_STEP_REWARD
from ..worldmodels.accel import AccelWorldModel
from ..worldmodels.rigidformer import rigidformer_kwargs
from .common import Algo, cap_rollout_grads, reference_setpoints, set_requires_grad

WM_TERMS = ("pos", "vel", "rot", "ang")
LOCAL_TERMS = ("lpos", "lvel", "lrot", "lang", "hei", "up")
TRACK_TERMS = LOCAL_TERMS + ("rpos", "rrot")  # + world-space root position / rotation (paper §4.2)


def local_features(bp, bR, bv, bw):
    """Local(X) of SuperTrack Eq. 1-7 for world-space body states (..., B, ...) whose body 0 is the root:
    root-frame positions, velocities, 6D rotations, angular velocities, body heights, and the up vector."""
    Rt = bR[..., :1, :, :].transpose(-1, -2)
    lp = (Rt @ (bp - bp[..., :1, :]).unsqueeze(-1)).squeeze(-1)
    lv = (Rt @ bv.unsqueeze(-1)).squeeze(-1)
    lr = mat_to_6d(Rt @ bR)
    lw = (Rt @ bw.unsqueeze(-1)).squeeze(-1)
    return lp, lv, lr, lw, bp[..., 2], bR[..., 0, 2, :]


def com_features(kin: ArticulationKinematics, body, p, R):
    """Whole-body center of mass relative to the root and its velocity, both in the root's heading frame: (N, 6)."""
    c, cv = kin.com(*body)
    Ht = yaw_mat(R).transpose(-1, -2)
    return torch.cat(((Ht @ (c - p).unsqueeze(-1)).squeeze(-1), (Ht @ cv.unsqueeze(-1)).squeeze(-1)), -1)


class ReferenceLocal:
    """Local(K) of every reference frame over all bodies, precomputed once, plus the reference root pose and the
    reference center-of-mass features."""

    def __init__(self, motion: MotionLib, kin: ArticulationKinematics, chunk: int = 65536):
        assert kin.body_names[0] == "pelvis", kin.body_names[0]
        parts, com = [[] for _ in LOCAL_TERMS], []
        with torch.no_grad():
            for s in range(0, motion.num_frames, chunk):
                f = torch.arange(s, min(s + chunk, motion.num_frames), device=motion.device)
                S = motion.ref_state(f)
                body = kin.forward(*S, all_bodies=True)
                for i, x in enumerate(local_features(*body)):
                    parts[i].append(x)
                com.append(com_features(kin, body, S[0], S[1]))
        self.feats = tuple(torch.cat(x) for x in parts) + (motion.root_p, motion.root_R)
        self.com = torch.cat(com)


def body_errors(body, S, ref: tuple, f: torch.Tensor) -> list[torch.Tensor]:
    """Per-element absolute tracking errors of every term in TRACK_TERMS, (N, n_term) each, from all-body FK of S."""
    P = local_features(*body)
    errs = [(x - r[f]).abs().flatten(1) for x, r in zip(P, ref[: len(LOCAL_TERMS)])]
    root_p, root_R = ref[len(LOCAL_TERMS) :]
    errs.append((S[0] - root_p[f]).abs())
    errs.append(so3_log(root_R[f].transpose(-1, -2) @ S[1]).abs())
    return errs


def track_terms(kin: ArticulationKinematics, S, ref: tuple, f: torch.Tensor) -> torch.Tensor:
    """L1 tracking errors of states S against reference frames f, summed within each term: (N, len(TRACK_TERMS))."""
    return torch.stack([e.sum(-1) for e in body_errors(kin.forward(*S, all_bodies=True), S, ref, f)], -1)


def wm_window_terms(kin: ArticulationKinematics, S_hat, S, squared: bool = False) -> torch.Tensor:
    """L1 (or, with ``squared``, L2) errors of world-space body positions, velocities, rotations (log) and angular
    velocities: (N, 4)."""
    ph, Rh, vh, wh = kin.forward(*S_hat, all_bodies=True)
    with torch.no_grad():
        pt, Rt, vt, wt = kin.forward(*S, all_bodies=True)
    norm = (lambda x: x.square()) if squared else (lambda x: x.abs())  # noqa: E731
    return torch.stack(
        (
            norm(ph - pt).sum((-1, -2)),
            norm(vh - vt).sum((-1, -2)),
            norm(so3_log(Rt.transpose(-1, -2) @ Rh)).sum((-1, -2)),
            norm(wh - wt).sum((-1, -2)),
        ),
        -1,
    )


class LossBalancer(nn.Module):
    """Fixed term weights, set once so that every term contributes 1 at the start of training (paper §3.4-3.5)."""

    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("w", torch.zeros(n))
        self.register_buffer("scale", torch.zeros(n))  # typical per-element error of each term at calibration
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def calibrate(self, terms: torch.Tensor, n_elements: torch.Tensor):
        assert bool(torch.isfinite(terms).all()) and bool((terms > 0).all()), terms
        self.w.copy_(1.0 / terms)
        self.scale.copy_(terms / n_elements)
        self.ready.fill_(True)


class SuperTrack(Algo):
    residual = True

    def __init__(self, args, kin, motion, mdp, dev):
        super().__init__(args, kin, motion, mdp, dev)
        ref = ReferenceLocal(motion, kin)
        self.ref_local, self.ref_com = ref.feats, ref.com
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
        self.ref_u = reference_setpoints(motion, mdp.default_q, self.alpha)
        self.terms = list(LOCAL_TERMS if args.st_no_root_terms else TRACK_TERMS) + (["com"] if args.st_com else [])
        self.n_track = len(self.terms)
        # task penalties at the reward's own ratio to its full tracking score (5), on the n_track unit-scaled terms
        full_score = MAX_STEP_REWARD / CONTROL_DT
        self.w_rate = 0.0 if args.st_no_task_regs else -mdp.cfg.w_action_rate * self.n_track / full_score
        self.w_limit = 0.0 if args.st_no_task_regs else -mdp.cfg.w_joint_limit * self.n_track / full_score
        self.bal_wm, self.bal_pi = LossBalancer(len(WM_TERMS)).to(dev), LossBalancer(self.n_track).to(dev)
        # optional trick: extra weight on the balance terms (body heights, up vector) on top of the balancing
        self.term_mult = torch.ones(self.n_track, device=dev)
        self.term_mult[[self.terms.index("hei"), self.terms.index("up")]] = args.st_balance_w
        # optional trick: later steps of the imagined window weigh more (1 at the first step, 1 + st_late_w at the last)
        self.late = 1 + args.st_late_w * torch.arange(self.HZ, device=dev) / (self.HZ - 1) if args.st_late_w != 0 else None
        self.st_wm_terms = torch.compile(self._st_wm_terms)
        self.st_imag_step = torch.compile(self._st_imag_step)
        self.st_track = torch.compile(self._st_track)

    @staticmethod
    def world_model_cls(a):
        return AccelWorldModel(hidden=a["wm_hidden"], substeps=a["st_substeps"], act=a["wm_act"], layer_norm=not a["wm_no_ln"],
                               rigidformer=rigidformer_kwargs(a))

    def setpoint(self, a, f):
        return a + self.ref_u[f + 1]  # PD offset about the reference pose of the next frame

    def state_dict(self):
        return {**super().state_dict(), "bal_wm": self.bal_wm.state_dict(), "bal_pi": self.bal_pi.state_dict()}

    def load_state_dict(self, ck):
        super().load_state_dict(ck)
        self.bal_wm.load_state_dict(ck["bal_wm"]), self.bal_pi.load_state_dict(ck["bal_pi"])

    # ------------------------------------------------------------------ world model (Alg. 1)
    def _st_wm_terms(self, S_hat, s1, mask):
        """Per-window terms (sum over the valid steps) and their mean over windows."""
        B, k = s1.shape[:2]
        E = wm_window_terms(self.kin, tuple(x.flatten(0, 1) for x in S_hat), unflatten_state(s1.flatten(0, 1)),
                            squared=self.args.st_wm_loss == "l2").view(B, k, -1)  # fmt: skip
        Ew = (E * mask.unsqueeze(-1)).sum(1)
        return Ew.mean(0), Ew

    def wm_loss(self, S_hat, w):
        terms, per_window = self.st_wm_terms(S_hat, w["s1"], w["mask"])
        if not self.bal_wm.ready:
            self.bal_wm.calibrate(terms.detach(), torch.ones_like(terms))
        return (self.bal_wm.w * terms).sum(), torch.full((), float("nan"), device=self.dev), (per_window.detach() * self.bal_wm.w).sum(-1)

    # ------------------------------------------------------------------ policy (Alg. 2)
    def _st_imag_step(self, S, o_hist, f, xi, xi_n, g):
        a = self.pi(o_hist, g)
        f1 = f + 1
        u = a + self.args.st_imag_noise * torch.randn_like(a) + self.ref_u[f1]  # PD offset about the reference pose
        S1 = self.wm.step(S, u, xi_n)
        if self.args.bptt_clip_k > 0:
            S1 = cap_rollout_grads(S1, self.args.bptt_clip_k)
        o1 = self.mdp.add_noise(self.mdp.obs(S1, u, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, self.motion.goal(f1, self.K)

    def _st_track(self, S_all, f_all):
        """Per-term L1 errors (and squared errors with --st_quad_w): sum over the imagined window, mean over rollouts,
        plus the number of elements of every term."""
        body = self.kin.forward(*S_all, all_bodies=True)
        errs = body_errors(body, S_all, self.ref_local, f_all)[: len(LOCAL_TERMS) if self.args.st_no_root_terms else len(TRACK_TERMS)]
        if self.args.st_com:
            errs.append((com_features(self.kin, body, S_all[0], S_all[1]) - self.ref_com[f_all]).abs())
        n_el = torch.tensor([float(e.shape[-1]) for e in errs], device=self.dev)
        E = torch.stack([e.sum(-1) for e in errs], -1).view(self.HZ, -1, self.n_track)
        if self.late is not None:
            E = E * self.late[:, None, None]
        if self.args.st_quad_w == 0:
            return E.sum(0).mean(0), None, n_el
        Q = torch.stack([(e**2).sum(-1) for e in errs], -1).view(self.HZ, -1, self.n_track)
        if self.late is not None:
            Q = Q * self.late[:, None, None]
        return E.sum(0).mean(0), Q.sum(0).mean(0), n_el

    def policy_update(self, buffer: ReplayBuffer):
        args, mdp = self.args, self.mdp
        s, h = self.sample_valid_starts(buffer, args.pi_batch)
        S = unflatten_state(s["s"])
        xi, f = s["xi"], s["f"]
        xi_n = normalize_xi(xi)
        o_hist = self.start_obs_history(h)
        S = self.perturb_start(S)
        set_requires_grad(self.wm, False)
        if args.wm_bptt_fp32:
            self.wm.amp = False
        g = self.motion.goal(f, self.K)
        Ss, As, Fs = [], [], []
        for t in range(self.HZ):
            S, a, o_hist, f, g = self.st_imag_step(S, o_hist, f, xi, xi_n, g)
            Ss.append(S), As.append(a), Fs.append(f)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all, f_all = tuple(flat([St[i] for St in Ss]) for i in range(6)), flat(Fs)
        terms, sq_terms, n_el = self.st_track(S_all, f_all)
        if not self.bal_pi.ready:
            self.bal_pi.calibrate(terms.detach(), n_el * self.HZ)
        A = torch.stack(As)
        loss = (self.bal_pi.w * self.term_mult * terms).sum() + args.st_reg * ((A**2).mean() + A.abs().mean())
        nan = torch.full((), float("nan"), device=self.dev)
        surv = nan
        if args.st_fail_w != 0:
            # optional trick: failure penalty. Every imagined step after the state crosses the task's termination
            # thresholds (the soft termination SuperMimic uses; --st_fail_margin < 1 moves them closer) costs st_fail_w
            # times the per-step tracking loss at calibration, so failing at step t costs about st_fail_w * (H - t)
            # steps' worth. The tracking loss itself stays unweighted: no survival weighting, no reward, no critic.
            d = mdp.soft_termination(S_all, f_all, margin=args.st_fail_margin)
            alive = torch.cumprod(1 - d.view(self.HZ, -1), 0)
            loss = loss + args.st_fail_w * (self.n_track / self.HZ) * (1 - alive).sum(0).mean()
            surv = alive[-1].detach().mean()
        if sq_terms is not None:
            # optional trick: e^2 / s next to |e|, equal at the typical error s and growing for larger deviations
            loss = loss + args.st_quad_w * (self.bal_pi.w * self.term_mult * sq_terms / self.bal_pi.scale).sum()
        if not args.st_no_task_regs:
            U = A + self.ref_u[torch.stack(Fs)]  # executed setpoints without the exploration noise
            rate = ((U - torch.cat((s["a_prev"].unsqueeze(0), U[:-1]), 0)) ** 2).sum(-1).mean()
            q = torch.stack([St[4] for St in Ss])
            limit = (torch.relu(q - mdp.soft_hi) + torch.relu(mdp.soft_lo - q)).sum(-1).mean()
            loss = loss + self.w_rate * rate + self.w_limit * limit
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        if torch.isfinite(gn):
            self.opt_pi.step()
        set_requires_grad(self.wm, True)
        self.wm.amp = True
        return loss.detach(), nan, nan, gn.detach(), surv
