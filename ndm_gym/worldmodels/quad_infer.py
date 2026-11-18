"""Inference copies of the quadrotor world model (``QuadWorldModel``, both parameterizations) for simulation throughput, as
``g1_infer.inference_copy`` makes the G1 world models' and ``mjcf.inference_copy`` the MuJoCo models'.

``inference_copy(wm, dtype, fused)`` returns a ``QuadInferenceModel`` (no parameter requires gradients) whose ``step(S, u)`` is the
world model's:
  dtype  bf16: the network's parameters in bf16 and the step without autocast, so the network runs in bf16 end to end (the features,
         normalizers and integration stay fp32). None keeps the network in fp32 (TF32 products where the caller allows them), the
         precision quadrotor training runs it in (``--amp none``).
  fused  the features with the input's normalization (written in the network's dtype), and the output's de-normalization with the
         integration, as two Triton kernels (``quad_fused``) around the network; the state stays four tensors (p, R, v, w), and
         ``step_into`` writes the next one into given tensors (a simulation alternating between two buffers copies nothing). CUDA only.

Every copy is compared with the model it was made from on the device before use (``check_copy``) and refused on a mismatch: the fused
features and integration against the model's own ``inputs`` and ``step`` in fp32 (float rounding), the network against the model's in
fp32: its distance may not exceed that of the model under bf16 autocast by more than 2x on average and 4x at the largest (the rule of
``g1_infer``). Forward only: the copies refuse to run with gradients enabled.
"""

from __future__ import annotations

import copy
import math
import types

import torch
import torch.nn as nn

from ..core.rotations import so3_exp
from ..robots.quadrotor import NUM_U
from .g1_infer import CastNet, _fp32, _no_grad, network_errors
from .quadrotor import WM_IN, QuadWorldModel


class QuadInferenceModel(nn.Module):
    """An inference copy of a quadrotor world model (``inference_copy``): ``wm``, a copy of it with its network made for inference, and
    ``step(S, u)``: ``wm.step``, or with ``fused`` the step's features and integration as Triton kernels around the same network."""

    def __init__(self, wm: QuadWorldModel, fused: bool) -> None:
        super().__init__()
        self.wm, self.fused = wm, fused
        self.accel = wm.kind == "accel"
        self.bf16 = isinstance(wm.net, CastNet)
        with torch.no_grad():
            self.register_buffer("in_mean", wm.in_norm.mean.detach().float().contiguous(), persistent=False)
            self.register_buffer("in_scale", (1 / wm.in_norm.std()).detach().float().contiguous(), persistent=False)
            self.register_buffer("out_mean", wm.out_norm.mean.detach().float().contiguous(), persistent=False)
            self.register_buffer("out_std", wm.out_norm.std().detach().float().contiguous(), persistent=False)
        if fused and self.in_mean.device.type != "cuda":
            raise ValueError("fused=True runs Triton kernels: move the model to a CUDA device first")

    def features(self, S, u: torch.Tensor) -> torch.Tensor:
        """The network's normalized input as ``wm.in_norm(wm.inputs(S, u))``, in the network's dtype, from the fused kernel."""
        from . import quad_fused

        _, R, v, w = S
        return quad_fused.features(R, v, w, u, self.in_mean, self.in_scale, self.bf16).t()

    def integrate(self, y: torch.Tensor, S):
        """The next state from the network's normalized outputs y (fp32 or bf16), by the fused kernel."""
        from . import quad_fused

        return tuple(quad_fused.integrate(y, *S, self.out_mean, self.out_std, self.accel, float(self.wm.pos_coef)))

    def integrate_into(self, y: torch.Tensor, S, out) -> None:
        """``integrate`` into the tensors ``out`` (p, R, v, w)."""
        from . import quad_fused

        quad_fused.integrate_into(y, S, out, self.out_mean, self.out_std, self.accel, float(self.wm.pos_coef))

    def step(self, S, u: torch.Tensor):
        _no_grad("QuadInferenceModel")
        if not self.fused:
            return self.wm.step(S, u)
        return self.integrate(self.wm.net(self.features(S, u)), S)

    def step_into(self, S, u: torch.Tensor, out, net: nn.Module | None = None) -> None:
        """The fused step with the next state written into ``out``; ``net``: the network to run (default the copy's; e.g. a compiled one)."""
        _no_grad("QuadInferenceModel")
        if not self.fused:
            raise ValueError("step_into is the fused path's: make the copy with fused=True")
        self.integrate_into((self.wm.net if net is None else net)(self.features(S, u)), S, out)


def inference_copy(wm: QuadWorldModel, dtype: torch.dtype | None = torch.bfloat16, fused: bool = False, check: bool = True) -> QuadInferenceModel:
    """An inference copy of the quadrotor world model ``wm`` (module docstring); with ``check``, compared with ``wm`` on its device first."""
    if not isinstance(wm, QuadWorldModel):
        raise TypeError(f"expected a QuadWorldModel, got {type(wm).__name__}")
    if type(wm.net) is not nn.Sequential:
        raise TypeError(f"the world model's network must be the MLP as trained, got {type(wm.net).__name__}")
    if dtype not in (None, torch.bfloat16):
        raise ValueError(f"dtype must be torch.bfloat16 or None, got {dtype}")
    c = copy.deepcopy(wm).requires_grad_(False).eval()
    if dtype is not None:
        for mod in c.net.modules():
            for name, prm in list(mod.named_parameters(recurse=False)):
                setattr(mod, name, nn.Parameter(prm.detach().to(dtype), requires_grad=False))
        c.net = CastNet(c.net, dtype)
    m = QuadInferenceModel(c, fused)
    if check:
        check_copy(wm, m)
    return m


def random_states(n: int, device, seed: int = 0):
    """States (p, R, v, w) and commands u of n drones for the checks: positions within metres, random rotations, velocities of a few
    m/s, body rates of a few rad/s, commands about the hover."""
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, device=device)  # noqa: E731
    S = (3 * r(n, 3), so3_exp(math.pi * torch.tanh(r(n, 3))), 5 * r(n, 3), 2 * r(n, 3))
    u = torch.cat((8.5 + 2 * r(n, 1), r(n, NUM_U - 1)), -1)
    return tuple(t.contiguous() for t in S), u


def check_copy(wm: QuadWorldModel, m: QuadInferenceModel, n: int = 256) -> None:
    """Raise unless the inference copy ``m`` matches the world model ``wm`` it was made from on its device: the fused features and
    integration within float rounding of ``wm.inputs`` / ``wm.step`` in fp32, the network within the bf16-autocast reference's error
    (module docstring)."""
    dev = wm.in_norm.mean.device
    if dev.type != "cuda":
        raise ValueError("the inference copy is checked on its CUDA device: move the model there first")
    S, u = random_states(n, dev)
    with torch.no_grad():
        if m.fused:
            from . import quad_fused

            gpu = torch.cuda.get_device_name()
            ref = _fp32(lambda: wm.in_norm(wm.inputs(S, u)))
            got = _fp32(lambda: quad_fused.features(S[1], S[2], S[3], u, m.in_mean, m.in_scale, False).t())
            if not torch.allclose(got, ref, rtol=1e-4, atol=1e-4):
                raise RuntimeError(f"the fused features differ from the model's by {float((got - ref).abs().max()):.3g} on {gpu}")
            if m.bf16 and not torch.allclose(m.features(S, u).float(), got, rtol=4e-3, atol=1e-30):  # bf16's rounding: half of 2^-7
                raise RuntimeError(f"the fused features in bf16 differ from their fp32 values by more than bf16's rounding on {gpu}")
            y = 0.5 * torch.randn(n, wm.out_norm.mean.numel(), device=dev, generator=torch.Generator(device=dev).manual_seed(1))
            for y_ in (y, y.bfloat16()) if m.bf16 else (y,):  # the network's outputs as the copy hands them on
                proxy = types.SimpleNamespace(in_norm=lambda x: x, inputs=lambda S_, u_: torch.zeros(n, WM_IN, device=dev), net=lambda x: y_.float(),
                                              out_norm=wm.out_norm, kind=wm.kind, pos_coef=wm.pos_coef)  # fmt: skip
                ref_next = _fp32(lambda: QuadWorldModel.step(proxy, S, u))  # noqa: B023 (called at once)
                got_next = _fp32(lambda: m.integrate(y_, S))  # noqa: B023
                for name, r_, g_ in zip(("p", "R", "v", "w"), ref_next, got_next, strict=True):
                    if not torch.allclose(g_, r_, rtol=1e-4, atol=1e-5):
                        raise RuntimeError(f"the fused integration ({y_.dtype}) differs from the model's step in {name} by "
                                           f"{float((g_ - r_).abs().max()):.3g} on {gpu}")  # fmt: skip
        x = _fp32(lambda: wm.in_norm(wm.inputs(S, u)))
        (fmax, fmean), (rmax, rmean) = network_errors(wm.net, m.wm.net, x, amp=False)
        if not (fmean <= 2 * rmean + 1e-7 and fmax <= 4 * rmax + 1e-6):
            raise RuntimeError(f"the inference copy's network differs from the model's in fp32 by {fmax:.3g} at most, {fmean:.3g} on average, "
                               f"the model under bf16 autocast by {rmax:.3g}, {rmean:.3g}, on {torch.cuda.get_device_name()}: refused")  # fmt: skip
