"""The quadrotor world model's step outside its network as two Triton kernels, for inference (``quad_infer.inference_copy(fused=True)``),
as ``g1_fused`` does for the G1's.

``features``: the network's normalized input from the state and the command in one kernel: the world's up axis in the body's frame
(the rotation's last row), the linear velocity in the body's frame (R^T v), the body rates and the command (``QuadWorldModel.inputs``),
normalized as its ``in_norm``, written feature-major ((13, N), coalesced) in the network's dtype and read transposed. ``integrate``: the
next state from the network's normalized outputs (fp32 or bf16), de-normalized in the kernel as ``QuadWorldModel.step`` does: the acceleration model's
(the linear acceleration in the body's frame and the body rates' derivative, integrated in closed form over the simulator's substeps:
positions gain dt (v + pc a), velocities dt a, the rotation R exp(dt (w + pc aw))) or the increment model's (p + R dp, R exp(dth),
v + R dv, w + dw), into new tensors or (``integrate_into``) into given ones, e.g. the other of two state buffers that a simulation
alternates between. The state stays four tensors (p, R, v, w): no quaternion is packed or unpacked. fp32 throughout, equal to
``QuadWorldModel.inputs`` and ``.step`` up to float rounding (the rotation's exponential written out, as ``so3_exp(elementwise=True)``).
Each program handles a block of samples, one per lane. Forward pass only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..robots.quadrotor import CONTROL_DT
from .hoi_fused import _exp, _ld, _mm3, _mv3, _rows

BLOCK, WARPS = 64, 2  # samples per program (one per lane) and warps, as hoi_fused
N_IN = 13


@triton.jit
def _quad_features_kernel(R, V, W, U, MEAN, SCALE, OUT, N, sr, sv, sw, su, OUT_BF16: tl.constexpr, BLOCK: tl.constexpr):
    """(R[2, :], R^T v, w, u) of every sample, normalized ((x - mean) * scale), stored feature-major."""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    r = (_ld(R, n * sr, m, 0), _ld(R, n * sr, m, 1), _ld(R, n * sr, m, 2), _ld(R, n * sr, m, 3), _ld(R, n * sr, m, 4), _ld(R, n * sr, m, 5),
         _ld(R, n * sr, m, 6), _ld(R, n * sr, m, 7), _ld(R, n * sr, m, 8))  # fmt: skip
    v0, v1, v2 = _ld(V, n * sv, m, 0), _ld(V, n * sv, m, 1), _ld(V, n * sv, m, 2)
    f = (r[6], r[7], r[8], r[0] * v0 + r[3] * v1 + r[6] * v2, r[1] * v0 + r[4] * v1 + r[7] * v2, r[2] * v0 + r[5] * v1 + r[8] * v2,
         _ld(W, n * sw, m, 0), _ld(W, n * sw, m, 1), _ld(W, n * sw, m, 2),
         _ld(U, n * su, m, 0), _ld(U, n * su, m, 1), _ld(U, n * su, m, 2), _ld(U, n * su, m, 3))  # fmt: skip
    for k in tl.static_range(13):
        x = (f[k] - tl.load(MEAN + k)) * tl.load(SCALE + k)
        if OUT_BF16:
            x = x.to(tl.bfloat16)
        tl.store(OUT + k * N + n, x, mask=m)


@triton.jit
def _quad_integrate_kernel(Y, OMEAN, OSTD, P, R, V, W, NP, NR, NV, NW, N, sy, sp, sr, sv, sw, snp, snr, snv, snw, dt, pc, ACCEL: tl.constexpr,
                           BLOCK: tl.constexpr):  # fmt: skip
    """The next (p, R, v, w) from the network's normalized outputs Y (de-normalized here: y * std + mean, in fp32)."""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    row = n * sy
    D: tl.constexpr = 6 if ACCEL else 12
    y = ()
    for k in tl.static_range(D):
        y = y + (_ld(Y, row, m, k).to(tl.float32) * tl.load(OSTD + k) + tl.load(OMEAN + k),)
    p = (_ld(P, n * sp, m, 0), _ld(P, n * sp, m, 1), _ld(P, n * sp, m, 2))
    r = (_ld(R, n * sr, m, 0), _ld(R, n * sr, m, 1), _ld(R, n * sr, m, 2), _ld(R, n * sr, m, 3), _ld(R, n * sr, m, 4), _ld(R, n * sr, m, 5),
         _ld(R, n * sr, m, 6), _ld(R, n * sr, m, 7), _ld(R, n * sr, m, 8))  # fmt: skip
    v = (_ld(V, n * sv, m, 0), _ld(V, n * sv, m, 1), _ld(V, n * sv, m, 2))
    w = (_ld(W, n * sw, m, 0), _ld(W, n * sw, m, 1), _ld(W, n * sw, m, 2))
    if ACCEL:
        av = _mv3(r, y[0], y[1], y[2])  # the linear acceleration in the world's frame; the body rates' derivative stays in the body's
        p1 = (p[0] + dt * (v[0] + pc * av[0]), p[1] + dt * (v[1] + pc * av[1]), p[2] + dt * (v[2] + pc * av[2]))
        r1 = _mm3(r, _exp(dt * (w[0] + pc * y[3]), dt * (w[1] + pc * y[4]), dt * (w[2] + pc * y[5])))
        v1 = (v[0] + dt * av[0], v[1] + dt * av[1], v[2] + dt * av[2])
        w1 = (w[0] + dt * y[3], w[1] + dt * y[4], w[2] + dt * y[5])
    else:
        dp, dv = _mv3(r, y[0], y[1], y[2]), _mv3(r, y[6], y[7], y[8])
        p1 = (p[0] + dp[0], p[1] + dp[1], p[2] + dp[2])
        r1 = _mm3(r, _exp(y[3], y[4], y[5]))
        v1 = (v[0] + dv[0], v[1] + dv[1], v[2] + dv[2])
        w1 = (w[0] + y[9], w[1] + y[10], w[2] + y[11])
    for k in tl.static_range(3):
        tl.store(NP + n * snp + k, p1[k], mask=m)
        tl.store(NV + n * snv + k, v1[k], mask=m)
        tl.store(NW + n * snw + k, w1[k], mask=m)
    for k in tl.static_range(9):
        tl.store(NR + n * snr + k, r1[k], mask=m)


@torch.library.custom_op("ndm_gym::quad_features", mutates_args=())
def features(R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, u: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, bf16: bool) -> torch.Tensor:
    """The network's normalized input (QuadWorldModel.in_norm(QuadWorldModel.inputs(S, u))), feature-major: (13, N), fp32 or bf16."""
    N = R.shape[0]
    out = torch.empty(N_IN, N, device=R.device, dtype=torch.bfloat16 if bf16 else torch.float32)
    _quad_features_kernel[(triton.cdiv(N, BLOCK),)](R, v, w, u, mean, scale, out, N, _rows(R, 9), _rows(v, 3), _rows(w, 3), _rows(u, 4),
                                                    OUT_BF16=int(bf16), BLOCK=BLOCK, num_warps=WARPS)  # fmt: skip
    return out


@features.register_fake
def _(R, v, w, u, mean, scale, bf16):
    return R.new_empty(N_IN, R.shape[0], dtype=torch.bfloat16 if bf16 else torch.float32)


def integrate_into(y: torch.Tensor, S, out, out_mean: torch.Tensor, out_std: torch.Tensor, accel: bool, pos_coef: float) -> None:
    """Writes the next state into ``out`` (p, R, v, w; rows of contiguous values, any row stride, not overlapping S) from y (N, 6
    accelerations or 12 increments), the network's normalized outputs in fp32 or bf16; pos_coef: the acceleration model's position
    coefficient (s), unused for increments."""
    p, R, v, w = S
    N = p.shape[0]
    width = 6 if accel else 12
    if y.stride(-1) != 1 or y.dtype not in (torch.float32, torch.bfloat16) or y.shape[-1] != width:
        raise ValueError(f"quad_fused.integrate: y needs fp32 or bf16, a unit last stride and {width} columns, got {y.dtype} {tuple(y.shape)}")
    st = [y.stride(0), _rows(p, 3), _rows(R, 9), _rows(v, 3), _rows(w, 3), _rows(out[0], 3), _rows(out[1], 9), _rows(out[2], 3), _rows(out[3], 3)]
    _quad_integrate_kernel[(triton.cdiv(N, BLOCK),)](y, out_mean, out_std, p, R, v, w, *out, N, *st, CONTROL_DT, pos_coef, ACCEL=int(accel),
                                                     BLOCK=BLOCK, num_warps=WARPS)  # fmt: skip


@torch.library.custom_op("ndm_gym::quad_integrate", mutates_args=())
def integrate(y: torch.Tensor, p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, out_mean: torch.Tensor, out_std: torch.Tensor,
              accel: bool, pos_coef: float) -> list[torch.Tensor]:  # fmt: skip
    """``integrate_into`` new tensors: the next [p, R, v, w]."""
    nxt = [torch.empty(t.shape, device=t.device, dtype=t.dtype) for t in (p, R, v, w)]
    integrate_into(y, (p, R, v, w), nxt, out_mean, out_std, accel, pos_coef)
    return nxt


@integrate.register_fake
def _(y, p, R, v, w, out_mean, out_std, accel, pos_coef):
    return [t.new_empty(t.shape) for t in (p, R, v, w)]
