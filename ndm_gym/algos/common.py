"""Machinery shared by the algorithms: world-model fitting on replay windows and imagination start states."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..core.kinematics import ArticulationKinematics
from ..motion.library import MotionLib
from ..policies.tracking_policy import TrackingPolicy
from ..state import normalize_xi, unflatten_state
from ..tasks.tracking import MDP, OBS_DIM


def lr_at(t: float, lr: float, decay: float) -> float:
    """Cosine decay to 0.03x over ``decay``, then constant (``t`` and ``decay`` in iterations or in training hours)."""
    if t >= decay:
        return lr * 0.03
    return lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * t / decay)))


def reference_setpoints(motion: MotionLib, default_q: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Reference pose of every frame as a normalized joint setpoint (residual policies add their offsets to it)."""
    return (motion.q - default_q) / alpha


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


class _CapRowGrad(torch.autograd.Function):
    """Identity; in the backward pass, every row whose gradient norm exceeds k times the median over the rows with a
    nonzero gradient is scaled down to that norm."""

    @staticmethod
    def forward(ctx, x, k):
        ctx.k = k
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        n = g.flatten(1).norm(dim=1)
        cap = ctx.k * torch.where(n > 0, n, torch.nan).nanmedian()  # NaN when every row is zero: nothing is scaled
        s = torch.where(n > cap, cap / n, 1.0)
        return g * s.view(-1, *([1] * (g.dim() - 1))), None


def cap_rollout_grads(S: tuple, k: float) -> tuple:
    """Optional imagination trick (``--bptt_clip_k k``), applied to the state after every imagined step: identity in
    the forward pass; in the policy's BPTT, each rollout's gradient on each state component is scaled down to at most
    k times the median over the batch, so that a few rollouts whose gradients grow through the world model cannot
    dominate the policy gradient. The other rollouts' gradients are unchanged."""
    return tuple(_CapRowGrad.apply(x, k) for x in S)


class Algo:
    """One learning algorithm on the shared pipeline.

    The base class owns the world model, the policy and their optimizers, fits the world model on k-step replay
    windows, and samples imagination start states. Subclasses define ``make_world_model``, ``setpoint`` (policy
    output -> normalized joint setpoint sent to the simulator), ``wm_loss``, ``policy_update`` and checkpoint state.
    """

    residual = False  # True: the policy outputs offsets about the reference pose

    @classmethod
    def is_residual(cls, a: dict) -> bool:
        """Whether a run with training arguments ``a`` (``vars(args)``; also a checkpoint's ``args``) outputs offsets
        about the reference pose."""
        return cls.residual

    def __init__(self, args, kin: ArticulationKinematics, motion: MotionLib, mdp: MDP, dev: str):
        self.args, self.kin, self.motion, self.mdp, self.dev = args, kin, motion, mdp, dev
        self.K, self.Hh, self.HZ = args.goal_frames, args.hist, args.horizon
        self.wm = self.make_world_model().to(dev)
        self.pi = TrackingPolicy(OBS_DIM, 73 * self.K, hist=self.Hh).to(dev)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=args.lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
        # a Transformer's k-step rollout is one large graph (~10 min to compile): compile its step instead (~2 min)
        rf = args.wm_backbone == "rigidformer"
        self.wm_step = torch.compile(self.wm.step) if rf else self.wm.step
        self.wm_rollout = self._wm_rollout if rf else torch.compile(self._wm_rollout)
        self.fail_starts = 0

    # ------------------------------------------------------------------ interface
    @staticmethod
    def world_model_cls(a: dict) -> nn.Module:
        """The world model for training arguments ``a`` (``vars(args)``; also a checkpoint's ``args``)."""
        raise NotImplementedError

    def make_world_model(self) -> nn.Module:
        return self.world_model_cls(vars(self.args))

    def setpoint(self, a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Normalized joint setpoint for policy output ``a`` at reference frame ``f`` (target frame f + 1)."""
        raise NotImplementedError

    def wm_loss(self, S_hat, w: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(loss, k-step error for logging, per-window error for prioritized sampling) of the predicted windows
        ``S_hat`` against the replay windows ``w``."""
        raise NotImplementedError

    def policy_update(self, buffer: ReplayBuffer) -> tuple:
        """(policy loss, value loss, imagined reward, grad norm, imagined survival); nan where not defined."""
        raise NotImplementedError

    @property
    def optimizers(self) -> tuple:
        return (self.opt_wm, self.opt_pi)

    def state_dict(self) -> dict:
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "opt_wm": self.opt_wm.state_dict(),
                "opt_pi": self.opt_pi.state_dict()}  # fmt: skip

    def load_state_dict(self, ck: dict):
        self.wm.load_state_dict(ck["wm"]), self.pi.load_state_dict(ck["pi"])
        self.opt_wm.load_state_dict(ck["opt_wm"]), self.opt_pi.load_state_dict(ck["opt_pi"])

    # ------------------------------------------------------------------ world model
    def _wm_rollout(self, s, a, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(self.args.wm_window):
            S = self.wm_step(S, a[:, j], normalize_xi(xi[:, j]), self.args.wm_input_noise)
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)

    @torch.no_grad()
    def wm_probe(self, S_hat, s1, mask):
        """Model-independent k-step prediction error for logging: tracked-body position (m) and joint angle (rad)."""
        m = mask[:, -1] > 0
        Sh = tuple(x[:, -1][m] for x in S_hat)
        St = unflatten_state(s1[:, -1][m])
        return (self.kin.forward(*Sh)[0] - self.kin.forward(*St)[0]).norm(dim=-1).mean(), (Sh[4] - St[4]).abs().mean()

    def wm_update(self, buffer: ReplayBuffer, probe: bool = False):
        """One world-model step on a batch of replay windows. Optional (``--wm_fail_windows p``): a fraction p of the
        windows ends in the last ``--wm_fail_window`` steps before the replay's terminations, while it holds any."""
        args = self.args
        prioritized = args.wm_sample == "prioritized"
        n_fail = round(args.wm_fail_windows * args.wm_batch)
        if prioritized and n_fail:
            raise ValueError("--wm_sample prioritized and --wm_fail_windows cannot be combined")
        wf = None
        if n_fail:
            failed = lambda s1, f: self.mdp.termination(unflatten_state(s1), f + 1)  # noqa: E731  (as TrackingSim.step)
            wf = buffer.pre_failure_windows(n_fail, args.wm_window, args.wm_fail_window, failed)
        n_rest = args.wm_batch - (0 if wf is None else n_fail)
        if prioritized:
            w = buffer.sample_windows_prioritized(n_rest, args.wm_window, args.wm_prio_alpha)
        else:
            w = buffer.sample_windows(n_rest, args.wm_window)
        if wf is not None:
            w = {key: torch.cat((w[key], wf[key])) for key in w}
        S_hat = self.wm_rollout(w["s"], w["a"], w["xi"])
        loss, e8, per_window = self.wm_loss(S_hat, w)
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        if prioritized:
            buffer.update_priorities(w["start"], per_window)
        if probe:
            return (loss.detach() / self.args.wm_window, e8) + self.wm_probe(S_hat, w["s1"], w["mask"])
        return loss.detach() / self.args.wm_window, e8

    # ------------------------------------------------------------------ imagination
    def sample_valid_starts(self, buffer: ReplayBuffer, B: int):
        """Replay start states (and their observation histories) whose clip has H more frames. Optional trick
        (``--imag_fail_starts p``): a fraction p of them comes from the last ``--imag_fail_window`` steps before the
        replay's terminations, while it holds any, so that the policy also rehearses the states it failed from; how
        many did is kept in ``self.fail_starts`` for the training log."""
        starts, hists, n = [], [], 0

        def keep(s, h, want):
            f = s["f"]
            ok = f + self.HZ <= self.motion.clip_end[self.motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[:want]
            starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
            return len(idx)

        n_fail = round(self.args.imag_fail_starts * B)
        failed = lambda s1, f: self.mdp.termination(unflatten_state(s1), f + 1)  # noqa: E731  (as TrackingSim.step)
        while n < n_fail:
            sh = buffer.pre_failure_starts(2 * n_fail, self.Hh, self.args.imag_fail_window, failed)
            got = 0 if sh is None else keep(*sh, n_fail - n)
            if got == 0:  # no termination in the replay yet, or all of them too close to their clip's end
                break
            n += got
        self.fail_starts = n
        while n < B:
            s, h = buffer.sample_starts(2 * B, self.Hh)
            n += keep(s, h, B - n)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)

    def perturb_start(self, S):
        """Optional imagination trick: Gaussian horizontal root-velocity kicks on the start states (m/s), so the
        policy also learns to recover. Off (exact no-op, no RNG draw) at ``--imag_start_perturb 0``."""
        if self.args.imag_start_perturb == 0:
            return S
        p, R, v, w, q, qd = S
        kick = self.args.imag_start_perturb * torch.randn_like(v)
        kick[:, 2] = 0.0
        return (p, R, v + kick, w, q, qd)

    def start_obs_history(self, h: dict) -> torch.Tensor:
        """Noisy observation history of the start states, as the deployed policy would see it."""
        with torch.no_grad():
            o_hist = self.mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        return self.mdp.add_noise(o_hist)
