"""Inference copies of the G1 world models (``AccelWorldModel``, ``IncrementWorldModel``) for simulation throughput, as
``HOIWorldModel.inference_copy`` for HDMI's scenes: the LAFAN1 tracking and the two-G1 boxing scene's NDMs.

``inference_copy(wm, dtype, attention, fused)`` returns a ``G1InferenceModel`` (no parameter requires gradients) whose
``step(S, a, xi_n)`` is the world model's:
  dtype      bf16: the network's parameters in bf16 and the step without autocast, so the network runs in bf16 end to end
             (the RigidFormer's token features, residual stream, norms, attention and heads: ``InferenceRigidFormerNet``;
             the MLP's layers); the input features, kinematics, normalizers and integration stay fp32. None keeps the fp32
             parameters under the step's bf16 autocast, as the model as trained.
  attention  the RigidFormer's attention: "token" (``ndm_gym.core.token_attention``) or "sdpa"; its relation bias is
             gathered once. The MLP takes None.
  fused      the features (the RigidFormer's forward kinematics and token features, the MLP's invariant features) and the
             integration as two Triton kernels (``g1_fused``), the normalizers fused with the network's casts; CUDA only.
The RigidFormer's blocks or its whole network can then run as fused Triton kernels (``kernels.rf_block.fused_copy`` of an
fp32 copy, dtype None; a bf16 copy is refused there).

Every copy is compared with the model it was made from on the device before use (``check``) and refused on a mismatch:
the fused features and integration against the model's own ``inputs`` and ``step`` in fp32 (float rounding), the network
against the model's in fp32: its distance may not exceed that of the model under bf16 autocast (the as-trained path) by
more than 2x on average and 4x at the largest. Forward only: the copies refuse to run with gradients enabled.
"""

from __future__ import annotations

import copy
import types

import torch
import torch.nn as nn

from ..core.nets import run_mlp
from ..core.rotations import so3_exp
from ..robots.g1 import NUM_J
from ..state import XI_DIM
from .accel import AccelWorldModel
from .increment import IncrementWorldModel
from .rigidformer import GLOBAL_DIM, RigidFormerNet


def _no_grad(what: str):
    if torch.is_grad_enabled():
        raise RuntimeError(f"{what} is inference-only (no backward): run it under torch.no_grad() or torch.inference_mode()")


class InferenceRigidFormerNet(RigidFormerNet):
    """``RigidFormerNet`` of an inference copy: its forward in the parameters' precision (bf16 end to end; fp32 parameters
    run under the caller's autocast as the model as trained, the residual stream fp32), the relation bias gathered once
    (``bias_table``), the blocks' attention by ``attention``, fp32 outputs. Made by ``from_net`` (a copy). The blocks run
    as Blocks or as the fused kernels' ``FusedBlock`` (``fused_copy``), never as the training kernels of ``fused_train``."""

    @classmethod
    def from_net(cls, net: RigidFormerNet, dtype: torch.dtype | None, attention: str) -> InferenceRigidFormerNet:
        if type(net) is not RigidFormerNet:
            raise TypeError(f"expected a RigidFormerNet, got {type(net).__name__}")
        if attention not in ("token", "sdpa"):
            raise ValueError(f"attention must be 'token' or 'sdpa', got {attention!r}")
        m = copy.deepcopy(net).requires_grad_(False)
        m.__class__ = cls
        if dtype is not None:
            for mod in m.modules():
                for name, prm in list(mod.named_parameters(recurse=False)):
                    setattr(mod, name, nn.Parameter(prm.detach().to(dtype), requires_grad=False))
        m.fused_train = m.keep_ops = False
        with torch.no_grad():
            table = m.rel_bias[:, m.rel].unsqueeze(0).detach().clone() if m.rel_bias is not None else None
        m.register_buffer("bias_table", table, persistent=False)
        for blk in m.blocks:
            blk.attention = attention
        return m

    def forward(self, x):
        _no_grad("InferenceRigidFormerNet")
        return super().forward(x).float()

    def _blocks(self, x, pos):
        """``RigidFormerNet._blocks`` in the parameters' precision, with the gathered bias."""
        N, B = x.shape[0], self.n_body
        dt = self.embed.weight.dtype
        tok = x[:, : B * self.tok_dim].unflatten(-1, (B, self.tok_dim))
        ctx = x[:, B * self.tok_dim : B * self.tok_dim + GLOBAL_DIM] if self.n_anchor > 1 else x[:, B * self.tok_dim :]
        c = None
        if self.ctx_net is None:
            tok = torch.cat((tok, ctx.unsqueeze(1).expand(-1, B, -1)), -1)
        else:
            c = self.ctx_net(ctx.to(dt))
        if self.merge != "none":  # RigidFormerNet._blocks's merged tokens: the members' features side by side, the last's position
            tok = torch.cat((tok, tok.new_zeros(N, 1, self.tok_dim)), 1)[:, self.slots].flatten(-2)
            pos = pos[:, self.rep]
        h = self.embed(tok.to(dt)).to(dt) + self.body_emb  # fp32 parameters: the model's .float() (the embedding under autocast is bf16)
        if self.n_reg:
            h = torch.cat((h, self.registers.expand(N, -1, -1).to(h.dtype)), 1)
            pos = torch.cat((pos, pos.new_zeros(N, self.n_reg, *pos.shape[2:])), 1)
        cos, sin = self.rope(pos) if self.rope is not None else (None, None)
        hs = [h]
        for blk in self.blocks:
            h = blk(h, cos, sin, self.bias_table, c)
            hs.append(h)
        return [hs[i] for i in self.head_layers]


class CastNet(nn.Module):
    """An MLP of an inference copy with ``dtype`` parameters: its input cast to them, its output to fp32."""

    def __init__(self, net: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.net, self.dtype = net, dtype

    def forward(self, x):
        _no_grad("CastNet")
        return self.net(x.to(self.dtype)).float()


class G1InferenceModel(nn.Module):
    """An inference copy of a G1 world model (``inference_copy``): ``wm``, a copy of it with its network made for
    inference, and ``step(S, a, xi_n)``: ``wm.step``, or with ``fused`` the step's features and integration as Triton
    kernels around the same network."""

    def __init__(self, wm: nn.Module, fused: bool):
        super().__init__()
        self.wm, self.fused = wm, fused
        self.accel = isinstance(wm, AccelWorldModel)
        self.rf = isinstance(wm.net, RigidFormerNet)
        with torch.no_grad():
            self.register_buffer("in_scale", 1 / torch.sqrt(wm.in_norm.var + wm.in_norm.eps**2), persistent=False)
            self.register_buffer("out_std", wm.out_norm.std(), persistent=False)
        if fused:
            if wm.in_norm.mean.device.type != "cuda":
                raise ValueError("fused=True runs Triton kernels: move the model to a CUDA device first")
            if self.rf:
                net = wm.net
                if net.n_anchor != 1 or net.ext_wrench:
                    raise ValueError("the fused features write one rotary point per body (its link origin) and no external wrench: "
                                     f"anchors {net.n_anchor}, ext_wrench {net.ext_wrench} are not supported")  # fmt: skip
                from . import hoi_fused

                self.fk = (hoi_fused.kinematic_tree(net.kin), net.q0.tolist(), net.alpha.tolist(), net.kp.tolist(), net.kd.tolist(),
                           net.tau_max.tolist())  # fmt: skip

    @property
    def net(self):
        return self.wm.net

    @property
    def in_norm(self):
        return self.wm.in_norm

    @property
    def out_norm(self):
        return self.wm.out_norm

    def features(self, S, a, xi_n) -> torch.Tensor:
        """The network's normalized input as ``wm.in_norm(wm.inputs(S, a, xi_n))``, from the fused kernel (``fused``)."""
        from . import g1_fused

        p, R, v, w, q, qd = S
        if self.rf:
            d = self.wm.in_dim
            raw = g1_fused.rf_features(p, R, v, w, q, qd, a, xi_n, *self.fk, d).t()
            return torch.cat(((raw[:, :d] - self.wm.in_norm.mean) * self.in_scale, raw[:, d:]), -1)
        f = g1_fused.mlp_features(p, R, v, w).t()
        return (torch.cat((f, q, qd, a, xi_n), -1) - self.wm.in_norm.mean) * self.in_scale

    def integrate(self, y, S):
        """The next state from the network's normalized outputs y, by the fused kernel (``fused``)."""
        from . import g1_fused

        return tuple(g1_fused.integrate(y * self.out_std + self.wm.out_norm.mean, *S, self.accel, getattr(self.wm, "pos_coef", 0.0)))

    def step(self, S, a, xi_n, input_noise: float = 0.0):
        _no_grad("G1InferenceModel")
        if input_noise != 0:
            raise ValueError("an inference copy takes no input noise (a fitting option)")
        if not self.fused:
            return self.wm.step(S, a, xi_n)
        return self.integrate(run_mlp(self.wm.net, self.features(S, a, xi_n), amp=self.wm.amp), S)


def inference_copy(wm: nn.Module, dtype: torch.dtype | None = torch.bfloat16, attention: str | None = "token", fused: bool = False,
                   check: bool = True) -> G1InferenceModel:  # fmt: skip
    """An inference copy of the G1 world model ``wm`` (module docstring); with ``check``, compared with ``wm`` on the
    device first (``check_copy``)."""
    if not isinstance(wm, (AccelWorldModel, IncrementWorldModel)):
        raise TypeError(f"expected an AccelWorldModel or IncrementWorldModel, got {type(wm).__name__}")
    if fused and getattr(wm, "structured", False):
        raise ValueError("the fused integration takes plain increments, not the semi-implicit world model's outputs (--wm_structured)")
    rf = isinstance(wm.net, RigidFormerNet)
    if type(wm.net) not in (RigidFormerNet, nn.Sequential):
        raise TypeError(f"the world model's network must be as trained (a RigidFormerNet or the MLP), got {type(wm.net).__name__}")
    if rf != (attention is not None):
        raise ValueError(f"attention={attention!r}: the RigidFormer takes 'token' or 'sdpa', the MLP None")
    c = copy.deepcopy(wm).requires_grad_(False)
    if rf:
        c.net = InferenceRigidFormerNet.from_net(wm.net, dtype, attention)
    elif dtype is not None:
        for mod in c.net.modules():
            for name, prm in list(mod.named_parameters(recurse=False)):
                setattr(mod, name, nn.Parameter(prm.detach().to(dtype), requires_grad=False))
        c.net = CastNet(c.net, dtype)
    if dtype is not None:
        c.amp = False  # the network runs in its parameters' precision, without autocast
    m = G1InferenceModel(c, fused)
    if check:
        check_copy(wm, m)
    return m


def random_inputs(n: int, device, seed: int = 0) -> tuple:
    """States S, setpoints a and normalized contexts xi_n of n robots for the checks: the root near standing height,
    random rotations and velocities, joints within about +-0.5 rad."""
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=device, generator=g)  # noqa: E731
    p = 0.5 * r(n, 3) + torch.tensor([0.0, 0.0, 0.8], device=device)
    S = (p, so3_exp(0.7 * r(n, 3)), r(n, 3), 2.0 * r(n, 3), 0.5 * r(n, NUM_J), 2.0 * r(n, NUM_J))
    return tuple(t.contiguous() for t in S), r(n, NUM_J), r(n, XI_DIM)


def _fp32(fn):
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast("cuda", enabled=False):
            return fn()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def network_errors(ref_net: nn.Module, net: nn.Module, x: torch.Tensor, amp: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    """(max, mean) |net(x) - ref_net(x) in fp32| for ``net`` as a copy runs it (under bf16 autocast with ``amp``) and for
    ``ref_net`` under bf16 autocast (the model as trained)."""
    ref32 = _fp32(lambda: ref_net(x).float())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref16 = ref_net(x).float()
    y = run_mlp(net, x, amp=amp) if amp else net(x).float()
    d, r = (y - ref32).abs(), (ref16 - ref32).abs()
    return (float(d.max()), float(d.mean())), (float(r.max()), float(r.mean()))


def check_copy(wm: nn.Module, m: G1InferenceModel, n: int = 64):
    """Raise unless the inference copy ``m`` matches the world model ``wm`` it was made from on its device: the fused
    features and integration within float rounding of ``wm.inputs`` / ``wm.step`` in fp32, the network within the
    bf16-autocast reference's error (module docstring)."""
    dev = wm.in_norm.mean.device
    if dev.type != "cuda":
        return  # the copies' kernels (token attention, the fused step) are CUDA-only; a CPU copy runs the model's own code
    S, a, xi_n = random_inputs(n, dev)
    with torch.no_grad():
        if m.fused:
            raw = _fp32(lambda: wm.inputs(S, a, xi_n))
            ref = _fp32(lambda: wm.in_norm(raw))
            got = _fp32(lambda: m.features(S, a, xi_n))
            if not torch.allclose(got, ref, rtol=1e-4, atol=1e-4):
                raise RuntimeError(f"the fused features differ from the model's by {float((got - ref).abs().max()):.3g} on "
                                   f"{torch.cuda.get_device_name()}")  # fmt: skip
            y = 0.5 * torch.randn(n, wm.out_norm.mean.numel(), device=dev, generator=torch.Generator(device=dev).manual_seed(1))
            proxy = types.SimpleNamespace(in_norm=lambda x: x, inputs=lambda *s: torch.zeros(n, 1, device=dev), net=lambda x: y, amp=False,
                                          out_norm=wm.out_norm, pos_coef=getattr(wm, "pos_coef", None))  # fmt: skip
            ref_next = _fp32(lambda: type(wm).step(proxy, S, a, xi_n))
            got_next = _fp32(lambda: m.integrate(y, S))
            for r_, g_ in zip(ref_next, got_next):
                if not torch.allclose(g_, r_, rtol=1e-4, atol=1e-5):
                    raise RuntimeError(f"the fused integration differs from the model's step by {float((g_ - r_).abs().max()):.3g} on "
                                       f"{torch.cuda.get_device_name()}")  # fmt: skip
        x = _fp32(lambda: wm.in_norm(wm.inputs(S, a, xi_n)))
        (fmax, fmean), (rmax, rmean) = network_errors(wm.net, m.wm.net, x, m.wm.amp)
        if not (fmean <= 2 * rmean + 1e-7 and fmax <= 4 * rmax + 1e-6):
            raise RuntimeError(f"the inference copy's network differs from the model's in fp32 by {fmax:.3g} at most, {fmean:.3g} on average, "
                               f"the model under bf16 autocast by {rmax:.3g}, {rmean:.3g}, on {torch.cuda.get_device_name()}: refused")  # fmt: skip
