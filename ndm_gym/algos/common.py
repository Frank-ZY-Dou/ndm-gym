"""Machinery shared by the algorithms: world-model fitting on replay windows and imagination start states."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..buffer import ReplayBuffer
from ..core.kinematics import ArticulationKinematics
from ..core.muon import MuonAdam
from ..core.nets import run_mlp
from ..motion.library import MotionLib
from ..policies.tracking_policy import TrackingPolicy
from ..state import normalize_xi, unflatten_state
from ..tasks.tracking import MDP, OBS_DIM
from .blinkmimic import Mirror, policy_average


def lr_at(t: float, lr: float, decay: float) -> float:
    """Cosine decay to 0.03x over ``decay``, then constant (``t`` and ``decay`` in iterations or in training hours)."""
    if t >= decay:
        return lr * 0.03
    return lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * t / decay)))


def set_lr(opt: torch.optim.Optimizer, lr: float):
    """Every parameter group's learning rate; a tensor one (capturable Adam, --critic_cudagraphs) is written in place, for
    the CUDA graph that reads it."""
    for group in opt.param_groups:
        if torch.is_tensor(group["lr"]):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr


def reference_setpoints(motion: MotionLib, default_q: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Reference pose of every frame as a normalized joint setpoint (residual policies add their offsets to it)."""
    return (motion.q - default_q) / alpha


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


def adam_kwargs(lr: float, dev, update_graphs: bool, adam_kw: dict) -> dict:
    """Adam's keyword arguments: with --update_graphs the capturable fused kernel and a device learning rate, a new tensor
    at each call. Each optimizer takes its own call: optimizers built with one tensor would share their learning rate, which
    set_lr writes in place (--wm_lr would set the policy's too)."""
    if update_graphs:
        return {"lr": torch.tensor(float(lr), device=dev), "fused": True, "capturable": True}
    return {"lr": lr, **adam_kw}


def wm_optimizer(wm: nn.Module, args, opt_kw: dict, capturable: bool, adam_kw: dict):
    """The world model's optimizer (LAFAN1 and muscle training): Adam with ``opt_kw`` (``adam_kwargs``, a call of its own) on
    all its parameters, or with --wm_opt muon MuonAdam (Adam's betas 0.9, 0.999; ``capturable``: --update_graphs;
    ``adam_kw``: --fused_adam): Muon on the hidden matrices of the MLP, or on those of the RigidFormer (its network's
    ``hidden_matrices``, the blocks' weight matrices: RigidFormerNet, the muscle character's MuscleRigidFormerNet), Adam on
    the rest."""
    if getattr(args, "wm_opt", "adam") != "muon":
        return torch.optim.Adam(wm.parameters(), **opt_kw)
    if isinstance(wm.net, nn.Sequential):
        return MuonAdam(wm, [wm.net], lr=args.lr, betas=(0.9, 0.999), capturable=capturable, **adam_kw)
    if not hasattr(wm.net, "hidden_matrices"):
        raise ValueError("--wm_opt muon: the MLP and RigidFormer world models only")
    return MuonAdam(wm, None, lr=args.lr, betas=(0.9, 0.999), capturable=capturable, matrices=wm.net.hidden_matrices(), **adam_kw)


def gated_step(opt, ok: torch.Tensor):
    """--no_sync: ``opt``'s step where the 0-dim bool ``ok`` holds, skipped on the device elsewhere (parameters and all
    optimizer state unchanged): the fused Adam's found_inf, MuonAdam's gate. The gradients must be finite."""
    if isinstance(opt, MuonAdam):
        return opt.step(ok=ok)
    if not opt.defaults.get("fused"):
        raise ValueError("--no_sync skips non-finite steps with the fused Adam (--fused_adam)")
    opt.found_inf = (~ok).float()  # the fused kernel skips the step and undoes its step count
    try:
        opt.step()
    finally:
        opt.found_inf = None


@torch.no_grad()
def zero_nonfinite_grads(params, finite: torch.Tensor):
    """--no_sync: every gradient of ``params`` set to zero where the 0-dim bool ``finite`` is false (a gated step needs
    finite gradients), without a host read."""
    for p in params:
        if p.grad is not None:
            p.grad.copy_(torch.where(finite, p.grad, 0.0))


@torch.no_grad()
def ema_step(pi_avg, pi: nn.Module, decay: float):
    """The policy average's step (``AveragedModel.update_parameters``, the same lerp) without its host-side test of the
    first call; the training loop advances the host-side counter n_averaged (--update_graphs)."""
    avg = list(pi_avg.module.parameters()) + list(pi_avg.module.buffers())
    live = [t.detach() for t in list(pi.parameters()) + list(pi.buffers())]
    torch._foreach_lerp_(avg, live, 1 - decay)


def survival(d: torch.Tensor) -> torch.Tensor:
    """cumprod(1 - d, 0) as a loop of products (the graphable updates': cumprod's backward tests for zeros on the host)."""
    acc, cs = torch.ones_like(d[0]), []
    for t in range(d.shape[0]):
        acc = acc * (1 - d[t])
        cs.append(acc)
    return torch.stack(cs)


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


class _ScaleGrad(torch.autograd.Function):
    """Identity; in the backward pass, the gradient times a constant."""

    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


@torch.no_grad()
def update_in_place(norm, x):
    """``norm.update(x)`` for CUDA graphs, which read the normalizer's statistics where they were captured:
    RunningNorm.update replaces its variance and count tensors, so the new values are written into the old ones."""
    var, count = norm.var, norm.count
    norm.update(x)
    var.copy_(norm.var), count.copy_(norm.count)
    norm.var, norm.count = var, count


def cap_rollout_grads(S: tuple, k: float) -> tuple:
    """Optional imagination trick (``--bptt_clip_k k``), applied to the state after every imagined step: identity in
    the forward pass; in the policy's BPTT, each rollout's gradient on each state component is scaled down to at most
    k times the median over the batch, so that a few rollouts whose gradients grow through the world model cannot
    dominate the policy gradient. The other rollouts' gradients are unchanged."""
    return tuple(_CapRowGrad.apply(x, k) for x in S)


def clip_rollout_policy_grads(pi: TrackingPolicy, taps: list, k: float) -> int:
    """Optional trick (``--pi_rollout_clip k``), after the BPTT's backward pass: each rollout's part of the policy
    gradient, sum_t J_tb^T abar_tb over the horizon (``rollout_grads``), is scaled down to at most k times the median
    over the rollouts, so that a few rollouts whose gradients grow through the world model (up to 100 times the median
    on the study's checkpoints) cannot dominate the update. A rollout's size is measured in the policy's output: the norm
    over the horizon of its adjoints abar_tb, the gradients that reached the output (``taps``: per imagined step, the
    policy's inputs (obs history, goals) and a zero tensor added to its output, whose .grad is abar), which ranks the
    rollouts as the parameter-space norm does (log correlation 0.84-0.85; 3 of 4 of the largest 1% the same). The
    backward pass has accumulated all rollouts' parts in the parameters' .grad; the policy runs again on the clipped
    rollouts' inputs and adds (w_b - 1) times their parts, w_b = k median / size_b < 1. Rollouts without a gradient do
    not count in the median. Returns the number of clipped rollouts."""
    def rerun(idx):
        obs, goal = (torch.stack([x[i][idx].detach() for x in taps]).flatten(0, 1) for i in (0, 1))
        return pi(obs, goal)

    return clip_rollout_output_grads(rerun, [t for _, _, t in taps], k)


def clip_rollout_output_grads(rerun, outs: list, k: float) -> int:
    """``clip_rollout_policy_grads`` for a policy of any inputs: ``outs`` per imagined step the zero tensor (B, J) added to
    the policy's output, whose .grad is the output's adjoint, and ``rerun(idx)`` the policy's output again on the steps'
    inputs of the rollouts ``idx``, steps outer ((H * len(idx), J), detached inputs). Returns the number of clipped rollouts."""
    if any(t.grad is None for t in outs):  # every step's output reaches a later loss: a missing gradient is a broken tap
        raise RuntimeError("--pi_rollout_clip: no gradient reached the policy's output at some imagined step")
    A = torch.stack([t.grad for t in outs])  # (H, B, J)
    size = A.square().sum((0, 2)).sqrt()
    live = size > 0
    if not bool(live.any()):
        return 0
    w = k * size[live].median() / size.clamp_min(1e-30)
    idx = (live & (w < 1)).nonzero().squeeze(-1)
    if len(idx):
        cot = ((w[idx] - 1)[None, :, None] * A[:, idx]).flatten(0, 1)
        with torch.enable_grad():
            rerun(idx).backward(cot)
    return len(idx)


def smoothness_penalty(net: nn.Module, x: torch.Tensor, eps: float, n: int, keep: torch.Tensor | None = None) -> torch.Tensor:
    """``--wm_smooth_w``'s penalty of a world model's network ``net`` (normalized inputs to normalized outputs) at its
    normalized inputs ``x`` (N, D): the mean over the rows (those where ``keep`` (N,) holds, if given) and ``n`` random
    unit directions u of ||net(x + eps u) - net(x)||^2 / eps^2, both passes in fp32 (``Algo._wm_smoothness``)."""
    u = torch.randn((n,) + x.shape, device=x.device)
    u = u / u.norm(dim=-1, keepdim=True)
    y = run_mlp(net, torch.cat((x.unsqueeze(0), x + eps * u)).flatten(0, 1), amp=False).unflatten(0, (n + 1, -1))
    d = ((y[1:] - y[:1]) ** 2).sum(-1)  # (n, N)
    if keep is None:
        return d.mean() / eps**2
    return torch.where(keep, d, 0.0).sum() / (n * keep.sum().clamp(min=1)) / eps**2  # a left-out row's value may be non-finite


def check_smoothness_args(args):
    """``--wm_smooth_w``'s arguments as ``Algo`` checks them; the defaults where an entry point's arguments lack them."""
    w = getattr(args, "wm_smooth_w", 0.0)
    if w < 0 or (w > 0 and args.wm_smooth_dirs < 1):
        raise ValueError("--wm_smooth_w must be >= 0, and with it --wm_smooth_dirs >= 1")
    if w > 0 and args.wm_smooth_eps < 0.01:
        raise ValueError(f"--wm_smooth_eps {args.wm_smooth_eps}: must be >= 0.01 with --wm_smooth_w. The finite difference runs in fp32, "
                         "whose matmuls round their inputs to TF32 under allow_tf32 (trained world models: x1.07 at 0.01, x4 at 0.001)")
    return w


def decay_rollout_grads(S: tuple, lam: float) -> tuple:
    """Optional imagination trick (``--bptt_decay lam``), applied to the state after every imagined step (after
    ``--bptt_clip_k``'s cap): identity in the forward pass; in the policy's BPTT the state's gradient is multiplied by
    lam, so that a loss d imagined steps after an action reaches it with weight lam^d (every path back through the
    states, the world model's and the observations' alike). Gradients that grow through many steps of the world model
    count less; the losses next to an action keep their relative weights."""
    return tuple(_ScaleGrad.apply(x, lam) for x in S)


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
        # optional (train.py --fused_adam): Adam's fused kernel, not bitwise equal to the default implementation
        self.adam_kw = {"fused": True} if getattr(args, "fused_adam", False) else {}
        # optional (train.py --update_graphs, BlinkMimic): whole updates as CUDA graphs, so the optimizers are capturable,
        # their learning rates 0-dim device tensors (set_lr fills them), one per optimizer (optimizer_kw at each)
        self.update_graphs = getattr(args, "update_graphs", False)
        self.no_sync = getattr(args, "no_sync", False)
        # optional (--wm_opt muon, --pi_opt muon; InstantTrack, BlinkMimic): Muon on the MLPs' hidden weight matrices (the
        # RigidFormer world model's: its blocks' matrices, RigidFormerNet.hidden_matrices; wm_optimizer)
        self.opt_wm = wm_optimizer(self.wm, args, self.optimizer_kw(args.lr), self.update_graphs, self.adam_kw)
        if getattr(args, "pi_opt", "adam") == "muon":
            self.opt_pi = MuonAdam(self.pi, [self.pi.enc, self.pi.dec], lr=args.lr, betas=(0.9, 0.95), capturable=self.update_graphs, **self.adam_kw)
        else:
            self.opt_pi = torch.optim.Adam(self.pi.parameters(), betas=(0.9, 0.95), **self.optimizer_kw(args.lr))
        # --no_sync: policy updates skipped for a non-finite gradient, counted on the device (the log's nonfinite column)
        self.nonfinite = torch.zeros((), dtype=torch.long, device=dev)
        # optional (--pi_ema d; InstantTrack, BlinkMimic): an exponential moving average of the policy, updated after every
        # policy update (update_pi_avg); checkpoints store it as "pi", the policy evaluate.py deploys, and the trained one as
        # "pi_live"
        self.pi_avg = policy_average(self.pi, args.pi_ema) if getattr(args, "pi_ema", 0.0) > 0 else None
        # optional (--wm_mirror p; BlinkMimic): a share p of every world-model batch replaced by its left-right mirror image
        self.mirror = Mirror(dev) if getattr(args, "wm_mirror", 0.0) > 0 else None
        if getattr(args, "wm_structured", False) and not getattr(self.wm, "structured", False):
            raise ValueError("--wm_structured: SuperMimic's MLP world model only")
        # optional (train.py --wm_cudagraphs, --pi_cudagraphs): the world-model update's and the policy update's compiled
        # functions replayed as CUDA graphs
        self.wm_graphs = getattr(args, "wm_cudagraphs", False)
        self.wm_compile = {"mode": "reduce-overhead"} if self.wm_graphs else {}
        self.pi_graphs = getattr(args, "pi_cudagraphs", False)
        self.pi_compile = {"mode": "reduce-overhead"} if self.pi_graphs else {}
        # a Transformer's k-step rollout is one large graph (~10 min to compile): compile its step instead (~2 min)
        rf = args.wm_backbone == "rigidformer"
        self.wm_step = torch.compile(self.wm.step, **self.wm_compile) if rf else self.wm.step
        self.wm_rollout = self._wm_rollout if rf else torch.compile(self._wm_rollout, **self.wm_compile)
        # optional trick (--wm_smooth_w): the world model's network penalized for steep changes about the replay's inputs
        self.smooth_w = check_smoothness_args(args)
        self.wm_smoothness = torch.compile(self._wm_smoothness, **self.wm_compile)
        # optional trick (--bptt_decay): the policy's BPTT gradient through the imagined states decays by lam per step
        self.bptt_decay = getattr(args, "bptt_decay", 1.0)
        if not 0 < self.bptt_decay <= 1:
            raise ValueError(f"--bptt_decay must be in (0, 1], got {self.bptt_decay}")
        # optional trick (--pi_rollout_clip): each rollout's part of the policy gradient at most k times the median
        self.pi_rollout_clip = getattr(args, "pi_rollout_clip", 0.0)
        if self.pi_rollout_clip < 0:
            raise ValueError(f"--pi_rollout_clip must be >= 0 (0: off), got {self.pi_rollout_clip}")
        # optional (train.py --compile_eager): the input features the world model's normalizer is fit on, of every collected
        # transition, as a compiled function (eagerly, the RigidFormer's forward kinematics are hundreds of small launches)
        self.compile_eager = getattr(args, "compile_eager", False)
        self.norm_inputs = torch.compile(self.wm.inputs) if self.compile_eager else self.wm.inputs
        self.fail_starts = 0

    def optimizer_kw(self, lr: float) -> dict:
        """Adam's keyword arguments (``adam_kwargs``; one call per optimizer: with --update_graphs its own learning rate)."""
        return adam_kwargs(lr, self.dev, self.update_graphs, self.adam_kw)

    def optimizer_step(self, opt, ok: torch.Tensor):
        """--no_sync: ``opt``'s step gated on the device (``gated_step``)."""
        return gated_step(opt, ok)

    def ema_update(self):
        """The policy average's step without a host read (``ema_step``)."""
        ema_step(self.pi_avg, self.pi, self.args.pi_ema)

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

    def log_lr(self, lr: float) -> float:
        """The learning rate train.py's log reports: its schedule's ``lr``."""
        return lr

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
        sd = {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(), "opt_wm": self.opt_wm.state_dict(),
              "opt_pi": self.opt_pi.state_dict()}  # fmt: skip
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

    # ------------------------------------------------------------------ world model
    def _wm_rollout(self, s, a, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(self.args.wm_window):
            S = self.wm_step(S, a[:, j], normalize_xi(xi[:, j]), self.args.wm_input_noise)
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)

    def _wm_smoothness(self, s0, a0, xi0):
        """Finite-difference Jacobian penalty of the world model's network (``--wm_smooth_w``) at the windows' first
        inputs x (normalized; replay states): the mean over windows and ``--wm_smooth_dirs`` random unit directions u
        of ||f(x + eps u) - f(x)||^2 / eps^2, f the network from normalized inputs to normalized outputs (the mean
        squared directional derivative, ||J||_F^2 / dim(x) in expectation), eps ``--wm_smooth_eps``. The starts of the
        uniformly drawn windows cover the replay, so this regularizes where the rollout's steps go at 1/k of its rows.
        Both forward passes run in fp32, without the world model's bf16 autocast, whose rounding of the network's inputs and
        activations (8-bit mantissa) is of the order of the step: on the study's trained world models and replay states, the
        bf16 penalty was 1.05 times the exact one at eps 0.1 and 3.5 times at 0.01. Under train.py's allow_tf32 the fp32
        matmuls round their inputs to TF32 (10-bit mantissa): 1.0004 times the exact one at 0.1, 1.007 at 0.03, 1.07 at
        0.01, hence the floor of 0.01 on eps."""
        x = self.wm.in_norm(self.wm.inputs(unflatten_state(s0), a0, normalize_xi(xi0)))
        return smoothness_penalty(self.wm.net, x, self.args.wm_smooth_eps, self.args.wm_smooth_dirs)

    @torch.no_grad()
    def update_wm_norms(self, S0, S1, a, xi):
        """Fit the world model's input and output normalizers on simulator transitions (states S0 -> S1, setpoints a,
        context xi)."""
        self._update_norm(self.wm.in_norm, self.norm_inputs(S0, a, normalize_xi(xi)))
        self._update_norm(self.wm.out_norm, self.wm.targets(S0, S1))
        if getattr(self.wm, "structured", False):  # --wm_structured: the plain increments' scale normalizes the fit's loss
            self._update_norm(self.wm.inc_norm, self.wm.increments(S0, S1))

    @torch.no_grad()
    def update_pi_norms(self, o, g):
        """Fit the policy's observation and goal normalizers on the observations and goals of collection."""
        self._update_norm(self.pi.obs_norm, o)
        self._update_norm(self.pi.goal_norm, g)

    def _update_norm(self, norm, x):
        if not (self.wm_graphs or self.pi_graphs):
            return norm.update(x)
        update_in_place(norm, x)

    @torch.no_grad()
    def wm_probe(self, S_hat, s1, mask):
        """Model-independent k-step prediction error for logging: tracked-body position (m) and joint angle (rad)."""
        m = mask[:, -1] > 0
        Sh = tuple(x[:, -1][m] for x in S_hat)
        St = unflatten_state(s1[:, -1][m])
        return (self.kin.forward(*Sh)[0] - self.kin.forward(*St)[0]).norm(dim=-1).mean(), (Sh[4] - St[4]).abs().mean()

    def wm_update_graphable(self, buffer: ReplayBuffer):
        """``wm_update`` for --update_graphs: no host synchronization, so the whole update (sampling, mirror, rollout, loss,
        backward, clipping, optimizer step) can be captured in a CUDA graph. Returns (loss per step, k-step error)."""
        args = self.args
        w = buffer.sample_windows(args.wm_batch, args.wm_window)
        if self.mirror is not None:
            w = self.mirror.windows(w, int(args.wm_batch * args.wm_mirror))
        S_hat = self.wm_rollout(w["s"], w["a"], w["xi"])
        loss, e8, _ = self.wm_loss(S_hat, w)
        objective = loss
        if self.smooth_w > 0:
            objective = loss + self.smooth_w * self.wm_smoothness(w["s"][:, 0], w["a"][:, 0], w["xi"][:, 0])
        self.opt_wm.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        return loss.detach() / args.wm_window, e8

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
        if self.mirror is not None:
            w = self.mirror.windows(w, int(args.wm_batch * args.wm_mirror))
        if self.wm_graphs:
            torch.compiler.cudagraph_mark_step_begin()  # the previous update's graph outputs are no longer used
        S_hat = self.wm_rollout(w["s"], w["a"], w["xi"])
        loss, e8, per_window = self.wm_loss(S_hat, w)
        objective = loss  # the fit loss is what the log reports
        if self.smooth_w > 0:
            objective = loss + self.smooth_w * self.wm_smoothness(w["s"][:, 0], w["a"][:, 0], w["xi"][:, 0])
        self.opt_wm.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        if prioritized:
            buffer.update_priorities(w["start"], per_window)
        if probe:
            return (loss.detach() / self.args.wm_window, e8) + self.wm_probe(S_hat, w["s1"], w["mask"])
        return loss.detach() / self.args.wm_window, e8

    # ------------------------------------------------------------------ imagination
    def mark_static_tensors(self):
        """For CUDA graphs of the imagination (``--pi_cudagraphs``): the tables it reads from plain objects (motion
        library, task, kinematics, the algorithm's own) are static inputs, which the graphs would otherwise copy at every
        replay (module parameters and buffers are static already)."""
        for obj in (self.motion, self.mdp, self.kin, self):
            for v in vars(obj).values():
                for t in v if isinstance(v, (tuple, list)) else (v,):
                    if isinstance(t, torch.Tensor) and t.is_cuda:
                        torch._dynamo.mark_static_address(t)

    def sample_valid_starts(self, buffer: ReplayBuffer, B: int):
        """Replay start states (and their observation histories) whose clip has H more frames. Optional trick
        (``--imag_fail_starts p``): a fraction p of them comes from the last ``--imag_fail_window`` steps before the
        replay's terminations, while it holds any, so that the policy also rehearses the states it failed from; how
        many did is kept in ``self.fail_starts`` for the training log."""
        starts, hists, n = [], [], 0

        def keep(points, want):
            # the first ``want`` candidate points whose clip has H more frames; only their states and histories are read
            age, env = points
            f = buffer.at("f", age, env)
            ok = f + self.HZ <= self.motion.clip_end[self.motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[:want]
            s, h = buffer.starts_at(age[idx], env[idx], self.Hh)
            starts.append(s), hists.append(h)
            return len(idx)

        n_fail = round(self.args.imag_fail_starts * B)
        failed = lambda s1, f: self.mdp.termination(unflatten_state(s1), f + 1)  # noqa: E731  (as TrackingSim.step)
        while n < n_fail:
            points = buffer.pre_failure_points(2 * n_fail, self.args.imag_fail_window, failed)
            got = 0 if points is None else keep(points, n_fail - n)
            if got == 0:  # no termination in the replay yet, or all of them too close to their clip's end
                break
            n += got
        self.fail_starts = n
        while n < B:
            n += keep(buffer.sample_start_points(2 * B), B - n)
        if len(starts) == 1:  # one draw held enough valid starts (the usual case)
            return starts[0], hists[0]
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)

    def sample_valid_starts_nosync(self, buffer: ReplayBuffer, B: int):
        """``sample_valid_starts`` without reading the valid count on the host (--no_sync; BlinkMimic): 2B candidates,
        stable-sorted valid first (only starts within H frames of a clip's end are invalid, under 1%), the first B kept."""
        age, env = buffer.sample_start_points(2 * B)
        f = buffer.at("f", age, env)
        ok = f + self.HZ <= self.motion.clip_end[self.motion.clip_id[f]]
        idx = torch.sort((~ok).to(torch.uint8), stable=True).indices[:B]
        return buffer.starts_at(age[idx], env[idx], self.Hh)

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
