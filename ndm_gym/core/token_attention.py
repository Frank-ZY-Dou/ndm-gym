"""Attention over the few tokens of each sample, forward pass only (inference of the token world models,
``worldmodels/rigidformer.Block`` with ``attention = "token"``).

One Triton program per (sample, head) holds the T x T scores in registers (T <= 64 tokens, padded to a power of two):
softmax(q k^T / sqrt(hd) + bias) v with fp32 scores and accumulation (fp32 inputs: full fp32 products), an optional
additive bias. At 33 tokens, 4 heads of width 32, PyTorch's memory-efficient kernel spends most of its time in the
setup of its 64 x 64 tiles: on an A100 this kernel takes 0.14 instead of 0.40 ms for 4096 samples and 0.04 instead of
0.10 ms for 1024 (FlexAttention with the bias as a score_mod: 0.38-0.41 ms at 4096 for block sizes 32-128; flash
attention without any bias: 0.49 ms). Results equal ``F.scaled_dot_product_attention``'s up to rounding (the
probabilities enter the second product in the inputs' precision, as in the fused kernels).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _token_attention_fwd(Q, K, V, Bias, Out, sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, so_b, so_h, so_t, sb_b, sb_h, sb_t,
                         scale, H: tl.constexpr, T: tl.constexpr, TP: tl.constexpr, HD: tl.constexpr, HAS_BIAS: tl.constexpr,
                         PRECISION: tl.constexpr):  # fmt: skip
    pid = tl.program_id(0)
    b, h = pid // H, pid % H
    t, d = tl.arange(0, TP), tl.arange(0, HD)
    row = t < T
    q = tl.load(Q + b * sq_b + h * sq_h + t[:, None] * sq_t + d[None, :], mask=row[:, None], other=0.0)
    k = tl.load(K + b * sk_b + h * sk_h + t[:, None] * sk_t + d[None, :], mask=row[:, None], other=0.0)
    v = tl.load(V + b * sv_b + h * sv_h + t[:, None] * sv_t + d[None, :], mask=row[:, None], other=0.0)
    s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale
    if HAS_BIAS:
        s += tl.load(Bias + b * sb_b + h * sb_h + t[:, None] * sb_t + t[None, :], mask=row[:, None] & row[None, :], other=0.0).to(tl.float32)
    s = tl.where(row[None, :], s, float("-inf"))
    p = tl.exp(s - tl.max(s, 1)[:, None])
    o = tl.dot(p.to(v.dtype), v, input_precision=PRECISION) / tl.sum(p, 1)[:, None]
    tl.store(Out + b * so_b + h * so_h + t[:, None] * so_t + d[None, :], o.to(Out.dtype.element_ty), mask=row[:, None])


@torch.library.custom_op("ndm_gym::token_attention", mutates_args=())
def token_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """q, k, v (N, H, T, hd) CUDA tensors of one dtype with unit stride in the last dimension, bias (1 or N, H, T, T) of
    q's dtype or None -> (N, H, T, hd), stored as (N, T, H, hd) like the fused SDPA kernels' outputs. An opaque custom
    op, so that torch.compile launches the kernel as compiled by Triton itself (with its argument specializations;
    traced as a user kernel instead, Inductor's launch of it took 4-5x longer here)."""
    N, H, T, hd = q.shape
    if not (q.is_cuda and k.shape == q.shape and v.shape == q.shape and q.dtype == k.dtype == v.dtype):
        raise ValueError(f"token_attention: q, k, v must be CUDA tensors of one shape and dtype, got {q.shape} {k.shape} {v.shape}")
    if T > 64 or hd < 16 or hd & (hd - 1) or min(q.stride(-1), k.stride(-1), v.stride(-1)) != 1:
        raise ValueError(f"token_attention: needs T <= 64 tokens, a power-of-two head width >= 16 and unit last strides (T={T}, hd={hd})")
    if bias is not None and (bias.shape[1:] != (H, T, T) or bias.shape[0] not in (1, N) or bias.dtype != q.dtype or bias.stride(-1) != 1):
        raise ValueError(f"token_attention: bias must be (1 or N, H, T, T) of q's dtype with unit last stride, got {tuple(bias.shape)}")
    o = torch.empty_strided((N, H, T, hd), (T * H * hd, hd, H * hd, 1), device=q.device, dtype=q.dtype)
    sb = (0, 0, 0) if bias is None else ((bias.stride(0) if bias.shape[0] > 1 else 0), bias.stride(1), bias.stride(2))
    _token_attention_fwd[(N * H,)](q, k, v, q if bias is None else bias, o, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *o.stride()[:3],
                                   *sb, hd**-0.5, H=H, T=T, TP=max(16, triton.next_power_of_2(T)), HD=hd, HAS_BIAS=bias is not None,
                                   PRECISION="ieee" if q.dtype == torch.float32 else "tf32", num_warps=2)  # fmt: skip
    return o


@token_attention.register_fake
def _(q, k, v, bias=None):
    N, H, T, hd = q.shape
    return q.new_empty_strided((N, H, T, hd), (T * H * hd, hd, H * hd, 1))
