"""Fused training passes of RigidFormer's Transformer block (``ndm_gym.worldmodels.rigidformer.Block``) in Triton, for
``RigidFormerNet(fused_train=True)`` (``--rf_fused_train``): the forward pass, and the backward pass for the world-model
fit (gradients to the block's input, rotary tables, attention bias, FiLM vectors and parameters) and for the policy's
BPTT through the frozen world model (gradients to the input and the rotary tables only).

The forward pass is one kernel, as ``rf_block``'s inference kernel: one program holds the tokens of E = BLOCK_M // T
environments (4 of the G1's 32 tokens in 128 rows) on chip through the block: RMSNorm, the q/k/v/gate products head by
head (the even and odd channels of q and k as separate products, so that the rotation needs no shuffles), QK-RMSNorm,
rotation, attention over the tile's rows with a precomputed (H, BLOCK_M, BLOCK_M) tile that holds the bias within an
environment and -inf between environments (``attention_tile``), softmax, the gate, the output projection,
FiLM, RMSNorm, SwiGLU. It also stores the residual stream after the attention (x1), from which the backward pass
starts its second half.

The backward pass: (1) one kernel for the SwiGLU half, from x1 (FiLM, RMSNorm and the SwiGLU recomputed, their
gradients, FiLM's gamma and beta summed over each environment's tokens), holding only the tile's normalized input, dy
(bf16) and the gradient accumulator through the loop over hidden chunks; (2) the attention half between cuBLAS
products, which do its products with qkvg's and the output projection's weights faster than a fused kernel did (the
products' many narrow tiles and the attention's gradients left too few registers): a row kernel for the RMSNorm, z =
xn Wqkvg^T and dog = dx1 Wp (bf16), one kernel for the attention's backward pass per environment (QK-RMSNorm, rotation,
softmax and the gate recomputed from z; the gradients of the q/k/v/gate columns written as dz), dxn = dz Wqkvg (fp32),
and a row kernel for the RMSNorm's backward pass. The parameters' gradients are products over all tokens: the kernels
write the products' operands in bf16 (the normalized inputs, the gate-weighted attention output, the SwiGLU's hidden
values, and the gradients of the products' outputs) and cuBLAS forms the products with fp32 outputs (``torch.mm(...,
out_dtype=float32)``); each input operand carries a column of ones, so that the same product also sums the output
gradients over the tokens, the bias's gradient. The norm-weight and attention-bias gradients are summed per program,
then over programs. Every sum has a fixed order (no atomics): the same inputs give the same bits.

The matrices reach the kernels as bf16 copies in tile layouts, each tile contiguous (``pack``, made from the current
parameters at every call of the network, so that an optimizer step is always seen), streamed from L2 and software-
pipelined over the loops (``stages``).

Numerics as ``rf_block``'s: every matrix product takes bf16 operands (the normalized tokens, the weights, q and k after
the rotation, the attention probabilities, v, the gated attention output, the SwiGLU's hidden values, and in the
backward pass the output gradients) and accumulates in fp32; everything else is fp32 (RMSNorms, softmax, gates, FiLM,
which the default path runs partly in bf16 under autocast). The q/k/v/gate products are rounded to bf16 before their
bias is added, in the forward pass as in the backward pass, which takes them from cuBLAS (the default path's autocast
rounds them after the bias). The attention bias is rounded to bf16 as the default path's ``bias.to(bf16)``. The softmax
is normalized before the product with v. tests/test_rf_train.py compares the forward pass and every gradient with the
fp32 block, against the bf16-autocast block's own errors.

Supported: CUDA GPUs of compute capability 8.0 and later, fp32 parameters and inputs, token counts T, widths D and head
widths that are powers of two (T >= 16, head width >= 32), FiLM; the rotation and the attention bias optional; outside
autocast or under bf16 autocast (the kernels compute as under it either way). Anything else is refused (``check_block``,
``rf_block_train``). ``RigidFormerNet(fused_train=True)`` calls them for its passes under bf16 autocast only.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

LOG2E = math.log2(math.e)
# the block's parameters in the order the kernels and the backward pass take them
PARAMS = ("norm1.weight", "qkvg.weight", "qkvg.bias", "q_norm.weight", "k_norm.weight", "proj.weight", "proj.bias", "norm2.weight",
          "w12.weight", "w12.bias", "w3.weight", "w3.bias")  # fmt: skip
CONFIG = ("T", "D", "H", "HD", "R", "HID", "BH", "HAS_BIAS", "BM_FWD", "BM_FFN", "BM_ATTN", "WARPS_FWD", "WARPS_FFN", "WARPS_ATTN", "STAGES",
          "KEEP_OPS")  # fmt: skip
ROWS = 64  # rows per program of the RMSNorm row kernels
BM_HID, WARPS_HID = 64, 4  # the hidden-chunk kernel's rows per program and warps (keep_ops; A100: 0.27 ms at 2048 x 32 tokens)


def attention_tile(bias: torch.Tensor | None, heads: int, T: int, block_m: int, device) -> torch.Tensor:
    """(H, block_m, block_m) bf16: bias[0, h, t_i, t_j] * log2(e) for rows i, j of one environment (row i = e T + t,
    idle rows past E T count as one more environment), -inf between environments."""
    i = torch.arange(block_m, device=device)
    e, t = i // T, i % T
    tile = torch.zeros(heads, block_m, block_m, device=device) if bias is None else bias[0][:, t[:, None], t[None, :]].float() * LOG2E
    return tile.masked_fill(e[:, None] != e[None, :], float("-inf")).to(torch.bfloat16).contiguous()


@triton.jit
def _rms(x, eps, D: tl.constexpr):
    """x / RMS(x) over the rows (BM, D) and the reciprocal RMS (BM,)."""
    r = tl.rsqrt(tl.sum(x * x, 1) / D + eps)
    return x * r[:, None], r


@triton.jit
def _tile_rows(E: tl.constexpr, T: tl.constexpr, n_env):
    """Rows of the program's tile: the token row in the (N T, .) inputs (int64), its environment, the row mask (BM, 1)."""
    i = tl.arange(0, E * T)
    env = tl.program_id(0) * E + i // T
    row = env.to(tl.int64) * T + i % T
    return row, env, (env < n_env)[:, None]


@triton.jit
def _rope_tables(COS, SIN, row, rmask, BM: tl.constexpr, R: tl.constexpr, HD2: tl.constexpr):
    """cos, sin (BM, HD/2) of the rotation of every channel pair of a row; pairs past the R rotated ones: 1, 0."""
    m = tl.arange(0, HD2)
    if R > 0:
        msk = rmask & (m[None, :] < R)
        cs = tl.load(COS + row[:, None] * R + m[None, :], mask=msk, other=1.0)
        sn = tl.load(SIN + row[:, None] * R + m[None, :], mask=msk, other=0.0)
    else:
        cs = tl.zeros([BM, HD2], dtype=tl.float32) + 1.0
        sn = tl.zeros([BM, HD2], dtype=tl.float32)
    return cs, sn


@triton.jit
def _qk_head(xn, h, WQK, BQK, QW, cs, sn, eps, Z, zrow, rmask, D: tl.constexpr, HD: tl.constexpr, J: tl.constexpr, SAVE: tl.constexpr):
    """q (J = 0) or k (J = 2) of head h: the products of the even and odd channels (rounded to bf16 before the bias, as
    the backward pass's cuBLAS product), QK-RMSNorm (weight QW), rotation. Returns the rotated pairs (even, odd), the
    normalized channels before the weight (he, ho) and the reciprocal RMS. With SAVE the products, channels in order,
    also go to their columns of z (Z + zrow, (N T, 4 D) in qkvg's order)."""
    HD2: tl.constexpr = HD // 2
    m = tl.arange(0, HD2)
    w = tl.arange(0, D)[:, None] * HD2 + m[None, :]
    base = WQK + (h * 4 + J) * D * HD2
    eb = tl.dot(xn, tl.load(base + w, cache_modifier=".cg")).to(tl.bfloat16)
    ob = tl.dot(xn, tl.load(base + D * HD2 + w, cache_modifier=".cg")).to(tl.bfloat16)
    if SAVE:
        tl.store(Z + zrow + (J // 2) * D + h * HD + tl.arange(0, HD)[None, :], tl.reshape(tl.join(eb, ob), [xn.shape[0], HD]), mask=rmask)
    e, o = eb.to(tl.float32), ob.to(tl.float32)
    e += tl.load(BQK + (h * 4 + J) * HD2 + m)[None, :]
    o += tl.load(BQK + (h * 4 + J + 1) * HD2 + m)[None, :]
    r = tl.rsqrt((tl.sum(e * e, 1) + tl.sum(o * o, 1)) / HD + eps)
    he, ho = e * r[:, None], o * r[:, None]
    ne, no = he * tl.load(QW + 2 * m)[None, :], ho * tl.load(QW + 2 * m + 1)[None, :]
    return ne * cs - no * sn, ne * sn + no * cs, he, ho, r


@triton.jit
def _vg_head(xn, h, WVG, BVG, Z, zrow, rmask, D: tl.constexpr, HD: tl.constexpr, J: tl.constexpr, SAVE: tl.constexpr):
    """v (J = 0) or the gate's input (J = 1) of head h (BM, HD) fp32, the product rounded to bf16 before the bias; with
    SAVE the product also goes to its columns of z (as ``_qk_head``)."""
    c = tl.arange(0, HD)
    w = tl.arange(0, D)[:, None] * HD + c[None, :]
    zb = tl.dot(xn, tl.load(WVG + (h * 2 + J) * D * HD + w, cache_modifier=".cg")).to(tl.bfloat16)
    if SAVE:
        tl.store(Z + zrow + (2 + J) * D + h * HD + c[None, :], zb, mask=rmask)
    return zb.to(tl.float32) + tl.load(BVG + (h * 2 + J) * HD + c)[None, :]


@triton.jit
def _fwd_kernel(X, Y, X1, XN1, Z, XN2, COS, SIN, BT, GB, N1, QN, KN, BP, N2, B3, WQK, BQK, WVG, BVG, WP, W12, B12, W3, n_env, eps, scale2,
                T: tl.constexpr, E: tl.constexpr, D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, R: tl.constexpr,
                NCH: tl.constexpr, BH: tl.constexpr, SAVE_X1: tl.constexpr, SAVE_OPS: tl.constexpr, STAGES: tl.constexpr):  # fmt: skip
    """The block on the tokens of environments [E pid, E pid + E): x (N T, D) -> y; with SAVE_X1 also x1, the residual
    after the attention (before FiLM). BT (H, BM, BM) bf16: the attention bias times log2(e) within an environment,
    -inf between environments. GB (N, 2 D): FiLM's gamma, beta of every environment. Packed weights as ``pack``'s. With
    SAVE_OPS (the fit, ``keep_ops``) also the bf16 operands the backward pass would recompute: the normalized input and
    the normalized FiLM output (XN1, XN2 (N T, D + 8), each with a column of ones) and the q/k/v/gate products without
    their bias (Z (N T, 4 D), qkvg's order)."""
    BM: tl.constexpr = E * T
    row, env, rmask = _tile_rows(E, T, n_env)
    i = tl.arange(0, BM)
    d = tl.arange(0, D)
    c = tl.arange(0, HD)
    off = row[:, None] * D + d[None, :]
    x = tl.load(X + off, mask=rmask, other=0.0)
    xh, _ = _rms(x, eps, D)
    xn = (xh * tl.load(N1 + d)[None, :]).to(tl.bfloat16)
    if SAVE_OPS:
        tl.store(XN1 + row[:, None] * (D + 8) + d[None, :], xn, mask=rmask)
        _ones_column(XN1, row, rmask, D + 8, D)
    zrow = row[:, None] * 4 * D
    cs, sn = _rope_tables(COS, SIN, row, rmask, BM, R, HD // 2)
    acc = x + tl.load(BP + d)[None, :]
    for h in tl.range(H, num_stages=STAGES):
        qre, qro, _, _, _ = _qk_head(xn, h, WQK, BQK, QN, cs, sn, eps, Z, zrow, rmask, D, HD, 0, SAVE_OPS)
        kre, kro, _, _, _ = _qk_head(xn, h, WQK, BQK, KN, cs, sn, eps, Z, zrow, rmask, D, HD, 2, SAVE_OPS)
        s = tl.dot(qre.to(tl.bfloat16), tl.trans(kre.to(tl.bfloat16)))
        s = tl.dot(qro.to(tl.bfloat16), tl.trans(kro.to(tl.bfloat16)), s)
        s = s * scale2 + tl.load(BT + h * BM * BM + i[:, None] * BM + i[None, :]).to(tl.float32)
        p = tl.exp2(s - tl.max(s, 1)[:, None])
        p = p / tl.sum(p, 1)[:, None]
        o = tl.dot(p.to(tl.bfloat16), _vg_head(xn, h, WVG, BVG, Z, zrow, rmask, D, HD, 0, SAVE_OPS).to(tl.bfloat16))
        og = (o * tl.sigmoid(_vg_head(xn, h, WVG, BVG, Z, zrow, rmask, D, HD, 1, SAVE_OPS))).to(tl.bfloat16)
        acc = tl.dot(og, tl.load(WP + h * HD * D + c[:, None] * D + d[None, :], cache_modifier=".cg"), acc)
    if SAVE_X1:
        tl.store(X1 + off, acc, mask=rmask)
    gb = GB + env.to(tl.int64)[:, None] * 2 * D + d[None, :]
    x2 = acc * (1.0 + tl.load(gb, mask=rmask, other=0.0)) + tl.load(gb + D, mask=rmask, other=0.0)
    xh2, _ = _rms(x2, eps, D)
    xn2 = (xh2 * tl.load(N2 + d)[None, :]).to(tl.bfloat16)
    if SAVE_OPS:
        tl.store(XN2 + row[:, None] * (D + 8) + d[None, :], xn2, mask=rmask)
        _ones_column(XN2, row, rmask, D + 8, D)
    acc = x2 + tl.load(B3 + d)[None, :]
    cb = tl.arange(0, BH)
    w12 = d[:, None] * BH + cb[None, :]
    for j in tl.range(NCH, num_stages=STAGES):
        u = tl.dot(xn2, tl.load(W12 + j * 2 * D * BH + w12, cache_modifier=".cg")) + tl.load(B12 + j * 2 * BH + cb)[None, :]
        z = tl.dot(xn2, tl.load(W12 + j * 2 * D * BH + D * BH + w12, cache_modifier=".cg")) + tl.load(B12 + j * 2 * BH + BH + cb)[None, :]
        acc = tl.dot((u * z * tl.sigmoid(z)).to(tl.bfloat16), tl.load(W3 + j * BH * D + cb[:, None] * D + d[None, :], cache_modifier=".cg"), acc)
    tl.store(Y + off, acc, mask=rmask)


@triton.jit
def _ones_column(P, row, rmask, LD: tl.constexpr, COL: tl.constexpr):
    """Columns [COL, COL + 8) of the given rows of a row-major bf16 operand with leading dimension LD: 1, 0, ..., 0. In a
    weight's gradient A^T P, the column of ones sums A over the rows: the bias's gradient, from the same product."""
    k = tl.arange(0, 8)
    v = tl.zeros([row.shape[0], 8], dtype=tl.float32) + (k == 0).to(tl.float32)[None, :]
    tl.store(P + row[:, None] * LD + COL + k[None, :], v.to(tl.bfloat16), mask=rmask)


@triton.jit
def _bwd_ffn_kernel(DY, X1, GB, N2, W12, B12, W3T, W12T, DX1, DX1B, DGB, XN2, DUZ, HH, DYB, PART, n_env, eps, T: tl.constexpr, E: tl.constexpr,
                    D: tl.constexpr, NCH: tl.constexpr, BH: tl.constexpr, FILM_GRAD: tl.constexpr, WGRAD: tl.constexpr,
                    STAGES: tl.constexpr):  # fmt: skip
    """Backward pass of the block's second half (FiLM, RMSNorm, SwiGLU, residual) from x1 and dy: dx1 (fp32, and in bf16
    DX1B for its product with the output projection's weight), and with FILM_GRAD FiLM's (dgamma, dbeta) per environment
    (DGB (N, 2 D)). With WGRAD the products' operands for the parameters' gradients (bf16: XN2 (N T, D + 8) the
    normalized input and a column of ones, whose product gives the bias's gradient, DUZ the gradients of the value and
    gate columns, HH (N T, HID + 8) the hidden values and a column of ones, DYB dy) and the program's sums PART[pid] =
    dnorm2 (D). Only the normalized input, dy (bf16) and the gradient accumulator are held through the loop over hidden
    chunks; the rest is reloaded."""
    BM: tl.constexpr = E * T
    HID: tl.constexpr = NCH * BH
    row, env, rmask = _tile_rows(E, T, n_env)
    d = tl.arange(0, D)
    off = row[:, None] * D + d[None, :]
    gbp = GB + env.to(tl.int64)[:, None] * 2 * D + d[None, :]
    pid = tl.program_id(0)
    x2 = tl.load(X1 + off, mask=rmask, other=0.0) * (1.0 + tl.load(gbp, mask=rmask, other=0.0)) + tl.load(gbp + D, mask=rmask, other=0.0)
    xh2, _ = _rms(x2, eps, D)
    n2 = tl.load(N2 + d)
    xn2 = (xh2 * n2[None, :]).to(tl.bfloat16)
    dyb = tl.load(DY + off, mask=rmask, other=0.0).to(tl.bfloat16)
    if WGRAD:
        tl.store(XN2 + row[:, None] * (D + 8) + d[None, :], xn2, mask=rmask)
        _ones_column(XN2, row, rmask, D + 8, D)
        _ones_column(HH, row, rmask, HID + 8, HID)
        tl.store(DYB + off, dyb, mask=rmask)
    dxn2 = tl.zeros([BM, D], dtype=tl.float32)
    cb = tl.arange(0, BH)
    w12 = d[:, None] * BH + cb[None, :]
    w12t = cb[:, None] * D + d[None, :]
    for j in tl.range(NCH, num_stages=STAGES):
        u = tl.dot(xn2, tl.load(W12 + j * 2 * D * BH + w12, cache_modifier=".cg")) + tl.load(B12 + j * 2 * BH + cb)[None, :]
        z = tl.dot(xn2, tl.load(W12 + j * 2 * D * BH + D * BH + w12, cache_modifier=".cg")) + tl.load(B12 + j * 2 * BH + BH + cb)[None, :]
        dh = tl.dot(dyb, tl.load(W3T + j * D * BH + w12, cache_modifier=".cg"))
        sg = tl.sigmoid(z)
        sl = z * sg
        du = dh * sl
        dz = dh * u * sg * (1.0 + z * (1.0 - sg))
        dub, dzb = du.to(tl.bfloat16), dz.to(tl.bfloat16)
        if WGRAD:
            ru = j * BH + cb
            tl.store(HH + row[:, None] * (HID + 8) + ru[None, :], (u * sl).to(tl.bfloat16), mask=rmask)
            tl.store(DUZ + row[:, None] * 2 * HID + ru[None, :], dub, mask=rmask)
            tl.store(DUZ + row[:, None] * 2 * HID + HID + ru[None, :], dzb, mask=rmask)
        dxn2 = tl.dot(dub, tl.load(W12T + j * 2 * BH * D + w12t, cache_modifier=".cg"), dxn2)
        dxn2 = tl.dot(dzb, tl.load(W12T + j * 2 * BH * D + BH * D + w12t, cache_modifier=".cg"), dxn2)
    x1 = tl.load(X1 + off, mask=rmask, other=0.0)
    gm = tl.load(gbp, mask=rmask, other=0.0)
    xh2, r2 = _rms(x1 * (1.0 + gm) + tl.load(gbp + D, mask=rmask, other=0.0), eps, D)
    dy = tl.load(DY + off, mask=rmask, other=0.0)
    dxh2 = dxn2 * n2[None, :]
    dx2 = dy + r2[:, None] * (dxh2 - xh2 * (tl.sum(dxh2 * xh2, 1) / D)[:, None])
    tl.store(DX1 + off, dx2 * (1.0 + gm), mask=rmask)
    tl.store(DX1B + off, (dx2 * (1.0 + gm)).to(tl.bfloat16), mask=rmask)
    if FILM_GRAD:
        e = tl.arange(0, E)
        emask = (pid * E + e < n_env)[:, None]
        dgb = DGB + (pid * E + e).to(tl.int64)[:, None] * 2 * D + d[None, :]
        tl.store(dgb, tl.sum(tl.reshape(dx2 * x1, [E, T, D]), 1), mask=emask)
        tl.store(dgb + D, tl.sum(tl.reshape(dx2, [E, T, D]), 1), mask=emask)
    if WGRAD:
        tl.store(PART + pid * D + d, tl.sum(dxn2 * xh2, 0))


@triton.jit
def _ffn_hidden_bwd_kernel(XN2, DYB, W12, B12, W3T, DUZ, HH, n_rows, D: tl.constexpr, NCH: tl.constexpr, BH: tl.constexpr,
                           BM: tl.constexpr):  # fmt: skip
    """The SwiGLU's backward pass in hidden chunk pid0 for rows [BM pid1, BM pid1 + BM) (keep_ops: xn2 from the forward
    pass, dy in bf16 (DYB); the chunk index varies fastest, so that a row tile's xn2 and dy stay in L2 across its chunks):
    u, z = xn2 W12's chunk (+ bias), dh = dy W3's chunk; du = dh silu(z), dz = dh u silu'(z) (DUZ, bf16) and h = u silu(z)
    (HH, bf16), the operands of the weights' gradients and of dxn2 = duz W12. Chunk 0 also stores HH's column of ones."""
    HID: tl.constexpr = NCH * BH
    j = tl.program_id(0)
    row = (tl.program_id(1) * BM + tl.arange(0, BM)).to(tl.int64)
    msk = (row < n_rows)[:, None]
    d = tl.arange(0, D)
    cb = tl.arange(0, BH)
    xn2 = tl.load(XN2 + row[:, None] * (D + 8) + d[None, :], mask=msk, other=0.0)
    dyb = tl.load(DYB + row[:, None] * D + d[None, :], mask=msk, other=0.0)  # bf16 (converted before: fewer registers than here)
    w12 = d[:, None] * BH + cb[None, :]
    u = tl.dot(xn2, tl.load(W12 + j * 2 * D * BH + w12, cache_modifier=".cg")) + tl.load(B12 + j * 2 * BH + cb)[None, :]
    z = tl.dot(xn2, tl.load(W12 + j * 2 * D * BH + D * BH + w12, cache_modifier=".cg")) + tl.load(B12 + j * 2 * BH + BH + cb)[None, :]
    dh = tl.dot(dyb, tl.load(W3T + j * D * BH + w12, cache_modifier=".cg"))
    sg = tl.sigmoid(z)
    sl = z * sg
    ru = j * BH + cb
    tl.store(HH + row[:, None] * (HID + 8) + ru[None, :], (u * sl).to(tl.bfloat16), mask=msk)
    tl.store(DUZ + row[:, None] * 2 * HID + ru[None, :], (dh * sl).to(tl.bfloat16), mask=msk)
    tl.store(DUZ + row[:, None] * 2 * HID + HID + ru[None, :], (dh * u * sg * (1.0 + z * (1.0 - sg))).to(tl.bfloat16), mask=msk)
    if j == 0:
        _ones_column(HH, row, msk, HID + 8, HID)


@triton.jit
def _ffn_tail_bwd_kernel(DXN2, X1, GB, N2, DY, DX1, DX1B, DGB, PART, n_env, eps, T: tl.constexpr, D: tl.constexpr, FILM_GRAD: tl.constexpr):
    """The backward pass of RMSNorm2 and FiLM for the tokens of environment pid (keep_ops), from dxn2 = duz W12 (fp32):
    dx1 (fp32, and in bf16 DX1B), with FILM_GRAD FiLM's (dgamma, dbeta) of the environment, and the program's sums
    PART[pid] = dxn2 x / RMS(x) (the norm weight's gradient), as ``_bwd_ffn_kernel``'s last part."""
    row, env, rmask = _tile_rows(1, T, n_env)
    d = tl.arange(0, D)
    off = row[:, None] * D + d[None, :]
    gbp = GB + env.to(tl.int64)[:, None] * 2 * D + d[None, :]
    x1 = tl.load(X1 + off, mask=rmask, other=0.0)
    gm = tl.load(gbp, mask=rmask, other=0.0)
    xh2, r2 = _rms(x1 * (1.0 + gm) + tl.load(gbp + D, mask=rmask, other=0.0), eps, D)
    dxn2 = tl.load(DXN2 + off, mask=rmask, other=0.0)
    pid = tl.program_id(0)
    tl.store(PART + pid * D + d, tl.sum(dxn2 * xh2, 0))
    dxh2 = dxn2 * tl.load(N2 + d)[None, :]
    dx2 = tl.load(DY + off, mask=rmask, other=0.0) + r2[:, None] * (dxh2 - xh2 * (tl.sum(dxh2 * xh2, 1) / D)[:, None])
    tl.store(DX1 + off, dx2 * (1.0 + gm), mask=rmask)
    tl.store(DX1B + off, (dx2 * (1.0 + gm)).to(tl.bfloat16), mask=rmask)
    if FILM_GRAD:
        dgb = DGB + pid.to(tl.int64) * 2 * D + d
        tl.store(dgb, tl.sum(dx2 * x1, 0))
        tl.store(dgb + D, tl.sum(dx2, 0))


@triton.jit
def _qk_grad(dre, dro, he, ho, r, QW, cs, sn, dcs, dsn, ROPE_GRAD: tl.constexpr, WGRAD: tl.constexpr, HD: tl.constexpr):
    """Backward pass of one head's rotation and QK-RMSNorm, from the gradients of the rotated pairs (dre, dro): the
    gradients of the product's even and odd channels, with WGRAD the norm weight's (even, odd; summed over rows, else
    zeros), and the rotary tables' accumulated (dcs, dsn)."""
    HD2: tl.constexpr = HD // 2
    m = tl.arange(0, HD2)
    we, wo = tl.load(QW + 2 * m)[None, :], tl.load(QW + 2 * m + 1)[None, :]
    if ROPE_GRAD:  # the rotated pair (n_e c - n_o s, n_e s + n_o c), n = h w: its derivatives in c and s
        dcs += dre * he * we + dro * ho * wo
        dsn += dro * he * we - dre * ho * wo
    dne, dno = dre * cs + dro * sn, dro * cs - dre * sn
    if WGRAD:
        dwe, dwo = tl.sum(dne * he, 0), tl.sum(dno * ho, 0)
    else:
        dwe, dwo = tl.zeros([HD2], dtype=tl.float32), tl.zeros([HD2], dtype=tl.float32)
    dhe, dho = dne * we, dno * wo
    mr = (tl.sum(dhe * he, 1) + tl.sum(dho * ho, 1)) / HD
    return r[:, None] * (dhe - he * mr[:, None]), r[:, None] * (dho - ho * mr[:, None]), dwe, dwo, dcs, dsn


@triton.jit
def _rms_rows_kernel(X, NW, XN, n_rows, eps, D: tl.constexpr, BM: tl.constexpr, WGRAD: tl.constexpr):
    """xn = x / RMS(x) * NW (bf16, XN (n_rows, D + 8)) for rows [BM pid, BM pid + BM) of x (n_rows, D) fp32; with WGRAD
    XN's column of ones for qkvg's bias."""
    row = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    d = tl.arange(0, D)
    msk = (row < n_rows)[:, None]
    xh, _ = _rms(tl.load(X + row[:, None] * D + d[None, :], mask=msk, other=0.0), eps, D)
    tl.store(XN + row[:, None] * (D + 8) + d[None, :], (xh * tl.load(NW + d)[None, :]).to(tl.bfloat16), mask=msk)
    if WGRAD:
        _ones_column(XN, row, msk, D + 8, D)


@triton.jit
def _rms_bwd_rows_kernel(DXN, X, DX1, NW, DX, PART, n_rows, eps, D: tl.constexpr, BM: tl.constexpr, WGRAD: tl.constexpr):
    """dx = dx1 + the RMSNorm's backward pass of dxn (the gradient of xn = x / RMS(x) * NW) for rows [BM pid, BM pid +
    BM); with WGRAD the program's sums PART[pid] = dxn x / RMS(x) (D), the norm weight's gradient."""
    row = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    d = tl.arange(0, D)
    msk = (row < n_rows)[:, None]
    off = row[:, None] * D + d[None, :]
    xh, r = _rms(tl.load(X + off, mask=msk, other=0.0), eps, D)
    dxn = tl.load(DXN + off, mask=msk, other=0.0)
    dxh = dxn * tl.load(NW + d)[None, :]
    tl.store(DX + off, tl.load(DX1 + off, mask=msk, other=0.0) + r[:, None] * (dxh - xh * (tl.sum(dxh * xh, 1) / D)[:, None]), mask=msk)
    if WGRAD:
        tl.store(PART + tl.program_id(0) * D + d, tl.sum(dxn * xh, 0))


@triton.jit
def _qk_from_z(zr, rmask, BQ, QW, cs, sn, eps, col, BM: tl.constexpr, HD: tl.constexpr):
    """q or k of a head from its columns [col, col + HD) of z = xn Wqkvg^T (bf16 rows zr) plus the bias: QK-RMSNorm and
    rotation, as ``_qk_head``."""
    HD2: tl.constexpr = HD // 2
    m = tl.arange(0, HD2)
    c = tl.arange(0, HD)
    z = tl.load(zr + col + c[None, :], mask=rmask, other=0.0).to(tl.float32) + tl.load(BQ + col + c)[None, :]
    e, o = tl.split(tl.reshape(z, [BM, HD2, 2]))  # the head's channels loaded in order, split in registers
    r = tl.rsqrt((tl.sum(e * e, 1) + tl.sum(o * o, 1)) / HD + eps)
    he, ho = e * r[:, None], o * r[:, None]
    ne, no = he * tl.load(QW + 2 * m)[None, :], ho * tl.load(QW + 2 * m + 1)[None, :]
    return ne * cs - no * sn, ne * sn + no * cs, he, ho, r


@triton.jit
def _attn_core_bwd_kernel(Z, BQ, DOG, COS, SIN, BT, QN, KN, DZ, OG, DCOS, DSIN, PART, n_env, eps, scale2, scale, T: tl.constexpr, E: tl.constexpr,
                          D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, R: tl.constexpr, HAS_BIAS: tl.constexpr, ROPE_GRAD: tl.constexpr,
                          WGRAD: tl.constexpr, STAGES: tl.constexpr):  # fmt: skip
    """The attention's backward pass on the tokens of environments [E pid, E pid + E), all heads, between two cuBLAS
    products: from z = xn Wqkvg^T (bf16 (N T, 4 D), without the bias BQ, added here in fp32) and dog = dx1 Wp (bf16 (N
    T, D), the gated attention output's gradient) to the gradients of the q/k/v/gate columns (DZ, bf16 (N T, 4 D),
    qkvg's order), and with ROPE_GRAD the rotary tables' (DCOS, DSIN (N T, R)). With WGRAD the gated attention output
    (OG, bf16 (N T, D + 8), and a column of ones for the output projection's bias) and the program's sums PART[pid] =
    [dq_norm (HD), dk_norm (HD), dbias (H, T, T)]."""
    BM: tl.constexpr = E * T
    HD2: tl.constexpr = HD // 2
    PW: tl.constexpr = 2 * HD + H * T * T
    row, env, rmask = _tile_rows(E, T, n_env)
    i = tl.arange(0, BM)
    m = tl.arange(0, HD2)
    c = tl.arange(0, HD)
    t = tl.arange(0, T)
    pw = PART + tl.program_id(0) * PW
    cs, sn = _rope_tables(COS, SIN, row, rmask, BM, R, HD2)
    dcs = tl.zeros([BM, HD2], dtype=tl.float32)
    dsn = tl.zeros([BM, HD2], dtype=tl.float32)
    dqwe = tl.zeros([HD2], dtype=tl.float32)
    dqwo = tl.zeros([HD2], dtype=tl.float32)
    dkwe = tl.zeros([HD2], dtype=tl.float32)
    dkwo = tl.zeros([HD2], dtype=tl.float32)
    zr = Z + row[:, None] * 4 * D
    dz = DZ + row[:, None] * 4 * D
    if WGRAD:
        _ones_column(OG, row, rmask, D + 8, D)
    for h in tl.range(H, num_stages=STAGES):
        rv = 2 * D + h * HD + c
        qre, qro, qhe, qho, rq = _qk_from_z(zr, rmask, BQ, QN, cs, sn, eps, h * HD, BM, HD)
        kre, kro, khe, kho, rk = _qk_from_z(zr, rmask, BQ, KN, cs, sn, eps, D + h * HD, BM, HD)
        qeb, qob, keb, kob = qre.to(tl.bfloat16), qro.to(tl.bfloat16), kre.to(tl.bfloat16), kro.to(tl.bfloat16)
        s = tl.dot(qob, tl.trans(kob), tl.dot(qeb, tl.trans(keb)))
        s = s * scale2 + tl.load(BT + h * BM * BM + i[:, None] * BM + i[None, :]).to(tl.float32)
        p = tl.exp2(s - tl.max(s, 1)[:, None])
        p = p / tl.sum(p, 1)[:, None]
        vb = (tl.load(zr + rv[None, :], mask=rmask, other=0.0).to(tl.float32) + tl.load(BQ + rv)[None, :]).to(tl.bfloat16)
        o = tl.dot(p.to(tl.bfloat16), vb)
        sg = tl.sigmoid(tl.load(zr + D + rv[None, :], mask=rmask, other=0.0).to(tl.float32) + tl.load(BQ + D + rv)[None, :])
        dog = tl.load(DOG + row[:, None] * D + h * HD + c[None, :], mask=rmask, other=0.0).to(tl.float32)
        do = dog * sg
        tl.store(dz + D + rv[None, :], (dog * o * sg * (1.0 - sg)).to(tl.bfloat16), mask=rmask)
        if WGRAD:
            tl.store(OG + row[:, None] * (D + 8) + h * HD + c[None, :], (o * sg).to(tl.bfloat16), mask=rmask)
        delta = tl.sum(do * o, 1)
        dob = do.to(tl.bfloat16)
        ds = p * (tl.dot(dob, tl.trans(vb)) - delta[:, None])
        tl.store(dz + rv[None, :], tl.dot(tl.trans(p.to(tl.bfloat16)), dob).to(tl.bfloat16), mask=rmask)
        if WGRAD and HAS_BIAS:  # within-environment entries only (ds is 0 between environments): the sum of the diagonal blocks
            tl.store(pw + 2 * HD + h * T * T + t[:, None] * T + t[None, :], tl.sum(tl.sum(tl.reshape(ds, [E, T, E, T]), 2), 0))
        dsb = ds.to(tl.bfloat16)
        dqe, dqo, dwe, dwo, dcs, dsn = _qk_grad(tl.dot(dsb, keb) * scale, tl.dot(dsb, kob) * scale, qhe, qho, rq, QN, cs, sn, dcs, dsn,
                                                ROPE_GRAD, WGRAD, HD)  # fmt: skip
        dqwe += dwe
        dqwo += dwo
        tl.store(dz + h * HD + c[None, :], tl.reshape(tl.join(dqe, dqo), [BM, HD]).to(tl.bfloat16), mask=rmask)  # channels back in order
        dst = tl.trans(dsb)
        dke, dko, dwe, dwo, dcs, dsn = _qk_grad(tl.dot(dst, qeb) * scale, tl.dot(dst, qob) * scale, khe, kho, rk, KN, cs, sn, dcs, dsn,
                                                ROPE_GRAD, WGRAD, HD)  # fmt: skip
        dkwe += dwe
        dkwo += dwo
        tl.store(dz + D + h * HD + c[None, :], tl.reshape(tl.join(dke, dko), [BM, HD]).to(tl.bfloat16), mask=rmask)
    if ROPE_GRAD:
        msk = rmask & (m[None, :] < R)
        tl.store(DCOS + row[:, None] * R + m[None, :], dcs, mask=msk)
        tl.store(DSIN + row[:, None] * R + m[None, :], dsn, mask=msk)
    if WGRAD:
        tl.store(pw + 2 * m, dqwe)
        tl.store(pw + 2 * m + 1, dqwo)
        tl.store(pw + HD + 2 * m, dkwe)
        tl.store(pw + HD + 2 * m + 1, dkwo)


def _cfg(config):
    return dict(zip(CONFIG, config))


def pack(params: list[torch.Tensor], heads: int, chunk: int, keep_ops: bool = False) -> list[torch.Tensor]:
    """bf16 copies of a block's matrices (``params`` in PARAMS' order) in the kernels' layouts, each tile contiguous;
    biases fp32 in the same orders. The forward kernel's: WQK (H, 4, D, HD/2) the columns of q (even, odd channels) and
    k (even, odd) of each head; BQK (H, 4, HD/2); WVG (H, 2, D, HD) v and the gate; BVG (H, 2, HD); WP (H, HD, D) the
    output projection's rows of each head; W12 (NCH, 2, D, chunk) the SwiGLU's value and gate columns in chunks; B12
    (NCH, 2, chunk); W3 (NCH, chunk, D). The backward pass's: WQ (4 D, D) and WPB (D, D), qkvg's and the output
    projection's weights as they are (for cuBLAS); W12T (NCH, 2, chunk, D) and W3T (NCH, D, chunk), the SwiGLU's tiles
    transposed; with keep_ops also W12F (2 HID, D), the SwiGLU's input weight as it is (for cuBLAS)."""
    _, wq, bq, _, _, wp, _, _, w12, b12, w3, _ = params
    D = wq.shape[1]
    hd, hid = D // heads, w3.shape[1]
    nch = hid // chunk
    with torch.no_grad():
        w4, b4 = wq.view(4, heads, hd, D), bq.view(4, heads, hd)
        wqkt = torch.stack((w4[0, :, 0::2], w4[0, :, 1::2], w4[1, :, 0::2], w4[1, :, 1::2]), 1)
        bqk = torch.stack((b4[0, :, 0::2], b4[0, :, 1::2], b4[1, :, 0::2], b4[1, :, 1::2]), 1)
        w12t = w12.view(2, nch, chunk, D).transpose(0, 1)
        w3t = w3.t().reshape(nch, chunk, D)
        bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
        b12p = b12.view(2, nch, chunk).transpose(0, 1).float().contiguous()
        out = [bf(wqkt.transpose(-1, -2)), bqk.float().contiguous(), bf(w4[2:].transpose(0, 1).transpose(-1, -2)),
               b4[2:].transpose(0, 1).float().contiguous(), bf(wp.t().reshape(heads, hd, D)), bf(w12t.transpose(-1, -2)), b12p, bf(w3t),
               bf(wq), bf(wp), bf(w12t), bf(w3t.transpose(-1, -2))]  # fmt: skip
        return out + [bf(w12)] if keep_ops else out


def _ops_shapes(x: torch.Tensor, config: list[int], save_x1: bool) -> list[tuple[int, ...]]:
    """Shapes of the operands the forward pass keeps for the fit (keep_ops): xn1 (N T, D + 8), z (N T, 4 D), xn2 (N T,
    D + 8); (0,) each without."""
    k = _cfg(config)
    if not (save_x1 and k["KEEP_OPS"]):
        return [(0,)] * 3
    M, D = x.shape[0] * x.shape[1], k["D"]
    return [(M, D + 8), (M, 4 * D), (M, D + 8)]


@torch.library.custom_op("ndm_gym::rf_train_fwd", mutates_args=())
def rf_train_fwd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, bias: torch.Tensor, tile_f: torch.Tensor, tile_b: torch.Tensor,
                 gb: torch.Tensor, params: list[torch.Tensor], packed: list[torch.Tensor], eps: float, save_x1: bool,
                 config: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """The block's forward pass (``_fwd_kernel``) on x (N, T, D) -> (y, x1, xn1, z, xn2); x1 is empty unless
    ``save_x1``, and the products' operands xn1, z, xn2 (bf16, the fit's backward pass reads them instead of recomputing
    them) are empty unless also config's KEEP_OPS. ``bias``, ``tile_b`` and the backward pass's matrices of ``packed`` are
    inputs only for the backward pass."""
    k = _cfg(config)
    E = k["BM_FWD"] // k["T"]
    y = torch.empty_like(x)
    x1 = torch.empty_like(x) if save_x1 else x.new_empty(0)
    xn1, z, xn2 = (torch.empty(s, device=x.device, dtype=torch.bfloat16) for s in _ops_shapes(x, config, save_x1))
    n1, _, _, qn, kn, _, bp, n2, _, _, _, b3 = params
    _fwd_kernel[(triton.cdiv(x.shape[0], E),)](x, y, x1, xn1, z, xn2, cos, sin, tile_f, gb, n1, qn, kn, bp, n2, b3, *packed[:8], x.shape[0],
                                               eps, k["HD"] ** -0.5 * LOG2E, T=k["T"], E=E, D=k["D"], H=k["H"], HD=k["HD"], R=k["R"],
                                               NCH=k["HID"] // k["BH"], BH=k["BH"], SAVE_X1=save_x1, SAVE_OPS=z.numel() > 0,
                                               STAGES=k["STAGES"], num_warps=k["WARPS_FWD"])  # fmt: skip
    return y, x1, xn1, z, xn2


@rf_train_fwd.register_fake
def _(x, cos, sin, bias, tile_f, tile_b, gb, params, packed, eps, save_x1, config):
    ops = [x.new_empty(s, dtype=torch.bfloat16) for s in _ops_shapes(x, config, save_x1)]
    return torch.empty_like(x), torch.empty_like(x) if save_x1 else x.new_empty(0), *ops


@torch.library.custom_op("ndm_gym::rf_train_bwd", mutates_args=())
def rf_train_bwd(dy: torch.Tensor, x: torch.Tensor, x1: torch.Tensor, ops: list[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor,
                 tile_b: torch.Tensor, gb: torch.Tensor, params: list[torch.Tensor], packed: list[torch.Tensor], eps: float, config: list[int],
                 need: list[int]) -> list[torch.Tensor]:  # fmt: skip
    """The block's backward pass: [dx, dcos, dsin, dbias, dgb] + the gradients of ``params`` (PARAMS' order). ``need``
    = [rotary tables, attention bias, FiLM vectors, parameters] (0/1); the attention bias's gradient comes with the
    parameters'; a gradient not needed is an empty tensor. A weight's and its bias's gradients are one product: the
    weight's input operand carries a column of ones (bf16 operands with 8 extra columns, the ones in the first). ``ops``:
    the forward pass's xn1, z, xn2 (keep_ops, with the parameters' gradients) or empty tensors; with them the SwiGLU half
    runs as ``_ffn_hidden_bwd_kernel``, cuBLAS dxn2 = duz W12 and ``_ffn_tail_bwd_kernel``, and the attention half
    reads xn1 and z instead of recomputing them."""
    k = _cfg(config)
    N, T, D = x.shape
    HID, HD, H, R, BH = k["HID"], k["HD"], k["H"], k["R"], k["BH"]
    E2, E3 = k["BM_FFN"] // T, k["BM_ATTN"] // T
    rope_g, bias_g, film_g, wgrad = (bool(v) for v in need)
    nt2, nt3, ntr = triton.cdiv(N, E2), triton.cdiv(N, E3), triton.cdiv(N * T, ROWS)
    e0 = lambda dtype=x.dtype: x.new_empty(0, dtype=dtype)  # noqa: E731  (a distinct empty tensor for every output not needed)
    bf = lambda n, need=wgrad: torch.empty(N * T, n, device=x.device, dtype=torch.bfloat16) if need else e0(torch.bfloat16)  # noqa: E731
    n1, _, bq, qn, kn, _, _, n2, _, _, _, _ = params
    _, _, _, _, _, w12, b12, _, wq, wpb, w12t, w3t = packed[:12]
    kept = wgrad and ops[1].numel() > 0  # the fit with the forward pass's operands (keep_ops)
    # the SwiGLU half: dx1 (fp32, and bf16 for the product with the output projection's weight)
    dx1, dx1b = torch.empty_like(x), bf(D, True)
    dgb = torch.empty_like(gb) if film_g else e0()
    xn2, duz, hh, dyb = ops[2] if kept else bf(D + 8), bf(2 * HID), bf(HID + 8), bf(D)
    if kept:
        dyb.copy_(dy.view(N * T, D))  # bf16, also the operand of the output weight's gradient
        _ffn_hidden_bwd_kernel[(HID // BH, triton.cdiv(N * T, BM_HID))](xn2, dyb, w12, b12, w3t, duz, hh, N * T, D=D, NCH=HID // BH, BH=BH,
                                                                        BM=BM_HID, num_warps=WARPS_HID)  # fmt: skip
        part2 = x.new_empty(N, D)
        _ffn_tail_bwd_kernel[(N,)](torch.mm(duz, packed[12], out_dtype=torch.float32), x1, gb, n2, dy, dx1, dx1b, dgb, part2, N, eps, T=T, D=D,
                                   FILM_GRAD=film_g, num_warps=4)  # fmt: skip
    else:
        part2 = x.new_empty(nt2, D) if wgrad else e0()
        _bwd_ffn_kernel[(nt2,)](dy, x1, gb, n2, w12, b12, w3t, w12t, dx1, dx1b, dgb, xn2, duz, hh, dyb, part2, N, eps, T=T, E=E2, D=D,
                                NCH=HID // BH, BH=BH, FILM_GRAD=film_g, WGRAD=wgrad, STAGES=k["STAGES"], num_warps=k["WARPS_FFN"])  # fmt: skip
    # the attention half, between cuBLAS products with qkvg's and the output projection's weights
    og = bf(D + 8)
    if kept:
        xn1, z = ops[0], ops[1]
    else:
        xn1 = bf(D + 8, True)
        _rms_rows_kernel[(ntr,)](x, n1, xn1, N * T, eps, D=D, BM=ROWS, WGRAD=wgrad, num_warps=4)
        z = torch.mm(xn1[:, :D], wq.t())
    dz = torch.empty(N * T, 4 * D, device=x.device, dtype=torch.bfloat16)
    dcos = x.new_empty(N, T, R) if rope_g else e0()
    dsin = x.new_empty(N, T, R) if rope_g else e0()
    # every program writes all of its row with a bias; without one the bias entries stay zero
    part3 = (x.new_empty if k["HAS_BIAS"] else x.new_zeros)(nt3, 2 * HD + H * T * T) if wgrad else e0()
    _attn_core_bwd_kernel[(nt3,)](z, bq, torch.mm(dx1b, wpb), cos, sin, tile_b, qn, kn, dz, og, dcos, dsin, part3, N,
                                  eps, HD**-0.5 * LOG2E, HD**-0.5, T=T, E=E3, D=D, H=H, HD=HD, R=R, HAS_BIAS=bool(k["HAS_BIAS"]),
                                  ROPE_GRAD=rope_g, WGRAD=wgrad, STAGES=k["STAGES"], num_warps=k["WARPS_ATTN"])  # fmt: skip
    dx = torch.empty_like(x)
    parte = x.new_empty(ntr, D) if wgrad else e0()
    _rms_bwd_rows_kernel[(ntr,)](torch.mm(dz, wq, out_dtype=torch.float32), x, dx1, n1, dx, parte, N * T, eps, D=D, BM=ROWS, WGRAD=wgrad,
                                 num_warps=4)  # fmt: skip
    if not wgrad:
        return [dx, dcos, dsin, e0(), dgb] + [e0() for _ in PARAMS]
    s3 = part3.sum(0)
    dbias = s3[2 * HD :].view(H, T, T).clone() if bias_g else e0()
    mm = lambda a, b: torch.mm(a.t(), b, out_dtype=torch.float32)  # noqa: E731  (a^T b over all tokens, fp32 output)
    gq, gp, g12, g3 = mm(dz, xn1), mm(dx1b, og), mm(duz, xn2), mm(dyb, hh)  # the weights' gradients, the biases' in column D (HID)
    return [dx, dcos, dsin, dbias, dgb, parte.sum(0), gq[:, :D].contiguous(), gq[:, D].contiguous(), s3[:HD].clone(), s3[HD : 2 * HD].clone(),
            gp[:, :D].contiguous(), gp[:, D].contiguous(), part2.sum(0), g12[:, :D].contiguous(), g12[:, D].contiguous(),
            g3[:, :HID].contiguous(), g3[:, HID].contiguous()]  # fmt: skip


@rf_train_bwd.register_fake
def _(dy, x, x1, ops, cos, sin, tile_b, gb, params, packed, eps, config, need):
    k = _cfg(config)
    N, T, D = x.shape
    e0 = lambda: x.new_empty(0)  # noqa: E731
    out = [torch.empty_like(x), x.new_empty(N, T, k["R"]) if need[0] else e0(), x.new_empty(N, T, k["R"]) if need[0] else e0(),
           x.new_empty(k["H"], T, T) if need[1] and need[3] else e0(), torch.empty_like(gb) if need[2] else e0()]  # fmt: skip
    return out + ([torch.empty_like(p) for p in params] if need[3] else [e0() for _ in PARAMS])


def _setup_context(ctx, inputs, output):
    x, cos, sin, bias, tile_f, tile_b, gb, params, packed, eps, save_x1, config = inputs
    ctx.unusable = not save_x1
    if save_x1:  # rf_block_train saves x1 whenever a gradient is needed
        ctx.save_for_backward(x, *output[1:], cos, sin, tile_b, gb, *params, *packed)
        ctx.eps, ctx.config, ctx.n_params = eps, config, len(params)
        ctx.mark_non_differentiable(*output[1:])
    ctx.set_materialize_grads(False)  # no zero-filled gradients for the outputs the backward pass does not read


def _backward(ctx, dy, *_):
    if ctx.unusable:
        raise RuntimeError("rf_train_fwd ran without saving x1 (no input required gradients), so it has no backward pass")
    x, x1, xn1, z, xn2, cos, sin, tile_b, gb, *rest = ctx.saved_tensors
    params, packed = rest[: ctx.n_params], rest[ctx.n_params :]
    if dy is None:  # the block's output reached no loss
        return None, None, None, None, None, None, None, [None] * len(params), [None] * len(packed), None, None, None
    ng = ctx.needs_input_grad
    wgrad = any(ng[7]) or ng[3]  # the attention bias's gradient is summed with the parameters' (the world-model fit)
    need = [int(ng[1] or ng[2]), int(ng[3]), int(ng[6]), int(wgrad)]
    out = rf_train_bwd(dy.contiguous(), x, x1, [xn1, z, xn2], cos, sin, tile_b, gb, params, packed, ctx.eps, ctx.config, need)
    dx, dcos, dsin, dbias, dgb = out[:5]
    dparams = [g if w else None for g, w in zip(out[5:], ng[7])]
    return (dx, dcos if need[0] else None, dsin if need[0] else None, dbias if need[1] else None, None, None, dgb if need[2] else None,
            dparams, [None] * len(packed), None, None, None)  # fmt: skip


rf_train_fwd.register_autograd(_backward, setup_context=_setup_context)


def check_block(blk, T: int, chunk: int):
    """Raise unless the fused training kernels support this Block at T tokens per environment (see the module doc)."""
    D, hd, hid = blk.norm1.weight.numel(), blk.hd, blk.w3.in_features
    p2 = lambda n: n > 0 and n & (n - 1) == 0  # noqa: E731
    dev = blk.norm1.weight.device
    if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] < 8:
        raise ValueError(f"--rf_fused_train needs a CUDA GPU of compute capability 8.0 or later (bf16 tensor cores), got {dev}")
    if not (p2(T) and T >= 16 and p2(D) and p2(hd) and hd >= 32 and p2(chunk) and chunk >= 16 and hid % chunk == 0):
        raise ValueError(f"--rf_fused_train supports T (tokens), width and head width powers of two (T >= 16, head width >= 32) and a "
                         f"SwiGLU width that is a multiple of the hidden chunk (a power of two >= 16); got T {T}, width {D}, head width "
                         f"{hd}, SwiGLU width {hid}, chunk {chunk}")  # fmt: skip
    if blk.film is None:
        raise ValueError("--rf_fused_train needs FiLM conditioning (film=True)")
    eps = {m.eps for m in (blk.norm1, blk.q_norm, blk.k_norm, blk.norm2)}
    if len(eps) != 1 or None in eps:
        raise ValueError(f"--rf_fused_train needs one explicit RMSNorm eps in the block, got {eps}")
    dtypes = {p.dtype for p in blk.parameters()}
    if dtypes != {torch.float32}:
        raise ValueError(f"--rf_fused_train takes fp32 parameters, got {dtypes}")


def rf_block_train(blk, x: torch.Tensor, cos, sin, bias, c: torch.Tensor, packed: list[torch.Tensor] | None = None,
                   block_m: tuple[int, int, int] = (128, 64, 32), chunk: int = 64, warps: tuple[int, int, int] = (8, 8, 2),
                   stages: int = 2, keep_ops: bool = False) -> torch.Tensor:  # fmt: skip
    """``blk(x, cos, sin, bias, c)`` (a ``Block``) by the fused kernels as under bf16 autocast (whether autocast is on or
    off; refused under autocast to another dtype), differentiable: x (N, T, D) fp32; cos, sin (N, 1, T, r) fp32 or None;
    bias (1, H, T, T) or None; c (N, C), FiLM's context. ``packed``: ``pack(params, heads, chunk)``
    of the block's current parameters (made here if None). FiLM's (gamma, beta) are computed here in fp32 (autocast
    off), as everything outside the products in the kernels. ``block_m`` and ``warps``: the rows per program and the
    warps of the forward, the SwiGLU-backward and the attention-backward kernels (the defaults the fastest measured on an
    A100: 4 environments of 32 tokens per forward program, 2 per SwiGLU-backward and 1 per attention-backward program);
    ``chunk``: the SwiGLU's hidden columns per loop step; ``stages``: software pipelining of the loops. ``keep_ops``
    (``--rf_fit_keep_ops``; effective when the parameters require gradients): the forward pass also keeps the bf16
    operands the backward pass would recompute (the normalized input and FiLM output, the q/k/v/gate products: 3
    tensors of (N T, D + 8), (N T, 4 D), (N T, D + 8)), and the backward pass takes them, with the SwiGLU half as a
    hidden-chunk kernel, cuBLAS dxn2 and a row kernel: equal up to float rounding."""
    N, T, D = x.shape
    check_block(blk, T, chunk)
    if x.dtype != torch.float32:
        raise ValueError(f"x must be fp32, got {x.dtype}")
    if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != torch.bfloat16:
        raise ValueError(f"the fused kernels compute the block as under bf16 autocast, not {torch.get_autocast_dtype('cuda')}")
    if len(block_m) != 3 or len(warps) != 3 or any(bm % T or (bm // T) & (bm // T - 1) for bm in block_m):
        raise ValueError(f"block_m {block_m} must be three power-of-two multiples of the {T} tokens, with three warps {warps}")
    with torch.autocast("cuda", enabled=False):
        gb = F.linear(c.float(), blk.film.weight, blk.film.bias)
    R = 0 if cos is None else cos.shape[-1]
    e0 = x.new_empty(0)
    cos = e0 if cos is None else cos.reshape(N, T, R).contiguous()
    sin = e0 if sin is None else sin.reshape(N, T, R).contiguous()
    bias = e0 if bias is None else bias.reshape(blk.heads, T, T)
    with torch.no_grad():
        b4 = None if bias.numel() == 0 else bias.unsqueeze(0)
        tile_f = attention_tile(b4, blk.heads, T, block_m[0], x.device)
        tile_b = attention_tile(b4, blk.heads, T, block_m[2], x.device)
    params = [blk.get_parameter(n) for n in PARAMS]
    save = torch.is_grad_enabled() and any(t.requires_grad for t in [x, cos, sin, bias, gb, *params])
    keep_ops = keep_ops and save and any(p.requires_grad for p in params)  # only the fit's backward pass reads them
    if packed is None:
        packed = pack(params, blk.heads, chunk, keep_ops)
    elif keep_ops and len(packed) < 13:
        raise ValueError("keep_ops needs pack(..., keep_ops=True)'s matrices")
    config = [T, D, blk.heads, blk.hd, R, blk.w3.in_features, chunk, int(bias.numel() > 0), *block_m, *warps, stages, int(keep_ops)]
    y = rf_train_fwd(x.contiguous(), cos, sin, bias, tile_f, tile_b, gb.contiguous(), params, packed, blk.norm1.eps, save, config)[0]
    return y
