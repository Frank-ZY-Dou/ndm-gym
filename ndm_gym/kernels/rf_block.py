"""Fused forward (inference) of RigidFormer's Transformer block, ``ndm_gym.worldmodels.rigidformer.Block``, and of the
whole ``HOIRigidFormerNet`` or G1 ``RigidFormerNet``, in Triton.

One program holds the tokens of E = BLOCK_M // T environments (3 environments of 33 tokens in 128 rows for HDMI's
scenes) on chip through the whole block: RMSNorm, the q/k/v/gate projection head by head, QK-RMSNorm, rotary rotation,
attention with the per-head additive bias (exact softmax over the tile's rows; a precomputed (H, BLOCK_M, BLOCK_M) tile
holds the bias of every pair of rows of one environment and -inf between environments), the sigmoid gate, the output
projection, FiLM, RMSNorm, SwiGLU. The residual stream stays fp32; it is the initial value of the output projection's
and of the SwiGLU's accumulators, so no second copy of it is held. The weights (bf16 copies made once, each tile
contiguous) are streamed from L2, prefetched one loop iteration ahead (the head loop and the SwiGLU's hidden chunks are
software-pipelined; ``stages`` >= 2: unpipelined, Triton 3.6 miscompiles the kernel on an H100).

``FusedBlock`` runs one block per launch (the residual read and written once per block). ``FusedHOIRigidFormerNet``
(opt-in, and refused on an H100 until it passes there: with the rotation computed in the kernel from the token
positions, Triton 3.6 compiled it to a kernel that reads past its shared memory; tools/bench_rf_kernel.py --diagnose
bisects it on a GPU) runs the whole network in one launch: the tile's environments enter as the network's inputs and
leave as its outputs; the token embedding (one product over the features of every token type), all blocks with the
residual on chip, the read-out SwiGLU on the embedding and the last block's tokens (the embedding recomputed there),
and the read-out heads as one product whose outputs are scattered to their places. Before it, in PyTorch: the
per-environment FiLM vectors (the context network and every block's FiLM layer) and the rotation's cos, sin tables.
``FusedG1RigidFormerNet`` runs the G1's ``RigidFormerNet`` (LAFAN1, the two-G1 scene) through the same kernel with its
own tables (one token type, the root and joint heads).
Both kernels are launched as opaque operators (``torch.library.custom_op``): torch.compile and CUDA graphs run them as
Triton compiled them. They take models with fp32 parameters, also ``HOIWorldModel.inference_copy(None, ...)`` (its
step outside the network as Triton kernels); a bf16 inference copy is refused.

Numerics. Every matrix product takes bf16 operands (the token features, the normalized tokens, the weights, q and k
after the rotation, the attention probabilities, v, the gated attention output, the SwiGLU's hidden product, the
read-out tokens) and accumulates in fp32; all else is fp32. Against the network under bf16 autocast: the products'
outputs are not rounded to bf16 (the embedding, q, k, v, the gate, the projection, the read-out), the attention bias is
rounded to bf16 as the reference's ``bias.to(bf16)``, the softmax runs in fp32 with exp2 and is normalized after the
product with v, FiLM's gamma and beta are computed (by PyTorch's fp32 matmuls, TF32 where enabled) and applied in fp32
(the reference rounds 1 + gamma to bf16), the gates and the SwiGLUs run in fp32 (the reference in bf16). So the result
is closer to the fp32 network than the autocast reference is (tests/test_rf_kernel.py). Deterministic: no atomics,
every sum in a fixed order, the same inputs give the same bits; within a tile, the other environments' entries are
exact zeros after the softmax. ``self_check`` compares a fused block or network with its reference on the device
before use (``fused_copy`` runs it), so a setting miscompiled on some GPU is refused instead of returning wrong numbers.

Forward only: there is no backward, and the fused modules refuse to run while gradients are enabled.
"""

from __future__ import annotations

import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from ..robots.g1 import NUM_J
from .rf_train import attention_tile
from ..tasks.hdmi import DECIMATION
from ..worldmodels.hoi import HOIRigidFormerNet
from ..worldmodels.rigidformer import POS_SCALE, Block, RigidFormerNet

LOG2E = math.log2(math.e)
BLOCK_BUFFERS = ("n1", "wqk", "bqk", "qkn", "wvg", "bvg", "wp", "bp", "n2", "w12", "b12", "w3", "b3")


@triton.jit
def _rope(t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, R: tl.constexpr, HD2: tl.constexpr, ROPE: tl.constexpr):
    """cos, sin (BM, HD/2) of the rotation of each channel pair of a tile's rows (pairs past the rotated ones: 1, 0). ROPE
    1: from the tables COS, SIN (N T, R) (row: a row's index); 2: from the token positions in x (PIDX (T, HD/2) the column
    of the coordinate of each pair of a token, -1: none; FRQ (HD/2,) the pair's frequency; t: a row's token)."""
    c2 = tl.arange(0, HD2)
    if ROPE == 1:
        rmask = xmask & (c2[None, :] < R)
        cs = tl.load(COS + row[:, None] * R + c2[None, :], mask=rmask, other=1.0)
        sn = tl.load(SIN + row[:, None] * R + c2[None, :], mask=rmask, other=0.0)
    else:
        pi = tl.load(PIDX + t[:, None] * HD2 + c2[None, :])
        ang = tl.load(X + env[:, None] * xd + pi, mask=xmask & (pi >= 0), other=0.0) * pos_scale * tl.load(FRQ + c2)[None, :]
        cs, sn = tl.cos(ang), tl.sin(ang)
    return cs, sn


@triton.jit
def _ex2_f16(x):
    """2^x of an fp16 tensor, two per instruction (PTX ex2.approx.f16x2, relative error about 2^-10); -inf -> 0."""
    return tl.inline_asm_elementwise("ex2.approx.f16x2 $0, $1;", "=r,r", [x], dtype=tl.float16, is_pure=True, pack=2)


@triton.jit
def _halves(x, M: tl.constexpr, K: tl.constexpr):
    """(M, 2K) -> its first and its last K columns (in registers where the layout keeps a row's columns in a thread)."""
    return tl.split(tl.permute(tl.reshape(x, [M, 2, K]), [0, 2, 1]))


@triton.jit
def _qkvg(xn, h, WQK, BQK, BM: tl.constexpr, D: tl.constexpr, HD: tl.constexpr):
    """Head h's merged product (BM, 4 HD) fp32, the bias its initial value: q even, q odd, k even, k odd, v, gate."""
    d = tl.arange(0, D)
    c4 = tl.arange(0, 4 * HD)
    b = tl.zeros([BM, 4 * HD], dtype=tl.float32) + tl.load(BQK + h * 4 * HD + c4)[None, :]
    return tl.dot(xn, tl.load(WQK + h * D * 4 * HD + d[:, None] * 4 * HD + c4[None, :], cache_modifier=".cg"), b)


@triton.jit
def _scores(qe, qo, ke, ko, h, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, QKN, BT, eps, qk_scale, BM: tl.constexpr,
            HD: tl.constexpr, R: tl.constexpr, ROPE: tl.constexpr, FAST_EXP: tl.constexpr, SOFTMAX: tl.constexpr, QKNORM: tl.constexpr):  # fmt: skip
    """Head h's attention probabilities, not normalized (fp16 with FAST_EXP), and their row sums, from its q and k
    channels (even, odd): QK-RMSNorm, rotation, scores with the bias tile, softmax in log2 units."""
    HD2: tl.constexpr = HD // 2
    i = tl.arange(0, BM)
    c2 = tl.arange(0, HD2)
    if QKNORM:
        rq = tl.rsqrt((tl.sum(qe * qe, 1) + tl.sum(qo * qo, 1)) / HD + eps)[:, None] * qk_scale  # QK-RMSNorm; q takes the scale
        rk = tl.rsqrt((tl.sum(ke * ke, 1) + tl.sum(ko * ko, 1)) / HD + eps)[:, None]
        qe, qo = qe * rq * tl.load(QKN + c2)[None, :], qo * rq * tl.load(QKN + HD2 + c2)[None, :]
        ke, ko = ke * rk * tl.load(QKN + 2 * HD2 + c2)[None, :], ko * rk * tl.load(QKN + 3 * HD2 + c2)[None, :]
        if ROPE != 0:  # pairs (2m, 2m + 1) of the leading channels turn; the others pass (cos 1, sin 0)
            cs, sn = _rope(t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, R, HD2, ROPE)
            qe, qo = qe * cs - qo * sn, qe * sn + qo * cs
            ke, ko = ke * cs - ko * sn, ke * sn + ko * cs
    q = tl.reshape(tl.join(qe, qo), [BM, HD]).to(tl.bfloat16)  # channels back in order: the product is one K = HD dot
    k = tl.reshape(tl.join(ke, ko), [BM, HD]).to(tl.bfloat16)
    s = tl.dot(q, tl.trans(k)) + tl.load(BT + h * BM * BM + i[:, None] * BM + i[None, :]).to(tl.float32)  # log2 units
    if not SOFTMAX:
        p = s.to(tl.float16) if FAST_EXP else s
        den = tl.zeros([BM], dtype=tl.float32) + 1.0
    elif FAST_EXP:  # two exponentials per instruction; the probabilities and v in fp16 for their product
        p = _ex2_f16((s - tl.max(s, 1)[:, None]).to(tl.float16))
        den = tl.sum(p.to(tl.float32), 1)
    else:
        p = tl.exp2(s - tl.max(s, 1)[:, None])
        den = tl.sum(p, 1)
    return p, den


@triton.jit
def _pv(p, v, den, FAST_EXP: tl.constexpr):
    """The attention's output: the probabilities' product with v, normalized."""
    if FAST_EXP:
        return tl.dot(p, v.to(tl.float16)) / den[:, None]
    return tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16)) / den[:, None]


@triton.jit
def _merged_head(z4, h, acc, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, QKN, BT, WP, eps, qk_scale, BM: tl.constexpr,
                 D: tl.constexpr, HD: tl.constexpr, R: tl.constexpr, ROPE: tl.constexpr, FAST_EXP: tl.constexpr, SOFTMAX: tl.constexpr,
                 QKNORM: tl.constexpr):  # fmt: skip
    """Head h from its merged product z4 (``_qkvg``): attention, gate and output projection, added to acc."""
    HD2: tl.constexpr = HD // 2
    d = tl.arange(0, D)
    c = tl.arange(0, HD)
    qk, vg = _halves(z4, BM, 2 * HD)
    qeo, keo = _halves(qk, BM, HD)
    qe, qo = _halves(qeo, BM, HD2)
    ke, ko = _halves(keo, BM, HD2)
    v, g = _halves(vg, BM, HD)
    p, den = _scores(qe, qo, ke, ko, h, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, QKN, BT, eps, qk_scale, BM, HD, R, ROPE,
                     FAST_EXP, SOFTMAX, QKNORM)  # fmt: skip
    o = _pv(p, v, den, FAST_EXP)
    return tl.dot((o * tl.sigmoid(g)).to(tl.bfloat16), tl.load(WP + h * HD * D + c[:, None] * D + d[None, :], cache_modifier=".cg"), acc)


@triton.jit
def _block(x, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, BT, GB, gb_stride, N1, WQK, BQK, QKN, WVG, BVG, WP, BP, N2, W12, B12,
           W3, B3, eps, qk_scale, BM: tl.constexpr, D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, R: tl.constexpr, NCH: tl.constexpr,
           BH: tl.constexpr, ROPE: tl.constexpr, HAS_FILM: tl.constexpr, STAGES: tl.constexpr, MERGE: tl.constexpr,
           FAST_EXP: tl.constexpr, ABL: tl.constexpr, PREFETCH: tl.constexpr):  # fmt: skip
    """One block on a tile's residual x (BM, D) fp32 -> the new residual. t, row (BM,): a row's token and index; env (BM,)
    int64, xmask (BM, 1) the rows of environments; the rotation (``_rope``: none with ROPE 0) recomputed for every head,
    so that it is not held through the block. ABL (0) leaves parts out to time the rest, wrong outputs (bits: 1 the
    attention, 2 the SwiGLU, 4 the softmax, 8 the QK-RMSNorm and rotation; tools/bench_rf_kernel.py --ablate).

    Weights (bf16, each tile contiguous): WQK (H, 4, D, HD/2) the even and odd channels of q and k per head; WVG (H, 2,
    D, HD) v and the gate; WP (H, HD, D) the output projection's rows of each head; W12 (NCH, 2, D, BH) the SwiGLU's
    value and gate columns in chunks of BH; W3 (NCH, BH, D). Biases and norm weights fp32 in the same orders. Products
    merged and split in registers: MERGE 1, one per head, WQK (H, D, 4 HD) the columns of q (even, odd), k (even, odd),
    v and the gate, BQK (H, 4 HD), WVG and BVG unused; MERGE 2, two per head, WQK (H, D, 2 HD) q and k with their
    channels in order, WVG (H, D, 2 HD) v and the gate, BQK and BVG (H, 2 HD); both, one per chunk, W12 (NCH, D, 2 BH),
    B12 (NCH, 2 BH). BT (H, BM, BM) bf16: the attention bias times log2(e) within an environment, -inf between
    environments. GB: FiLM's gamma, beta (D each) of every environment, rows gb_stride apart.
    """
    HD2: tl.constexpr = HD // 2
    ATTN: tl.constexpr = ABL % 2 == 0
    MLP: tl.constexpr = ABL // 2 % 2 == 0
    SOFTMAX: tl.constexpr = ABL // 4 % 2 == 0
    QKNORM: tl.constexpr = ABL // 8 % 2 == 0
    d = tl.arange(0, D)
    c2 = tl.arange(0, HD2)
    c = tl.arange(0, HD)
    xn = (x * tl.rsqrt(tl.sum(x * x, 1) / D + eps)[:, None] * tl.load(N1 + d)[None, :]).to(tl.bfloat16)
    wqk = d[:, None] * HD2 + c2[None, :]
    wvg = d[:, None] * HD + c[None, :]
    acc = x + tl.load(BP + d)[None, :]  # the residual, the output projection's initial value
    if MERGE == 1 and PREFETCH and ATTN:  # the next head's product in flight while this head's attention and gating run
        z4 = _qkvg(xn, 0, WQK, BQK, BM, D, HD)
        for h in tl.range(H - 1, num_stages=STAGES):
            z4n = _qkvg(xn, h + 1, WQK, BQK, BM, D, HD)
            acc = _merged_head(z4, h, acc, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, QKN, BT, WP, eps, qk_scale, BM, D, HD, R,
                               ROPE, FAST_EXP, SOFTMAX, QKNORM)  # fmt: skip
            z4 = z4n
        acc = _merged_head(z4, H - 1, acc, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, QKN, BT, WP, eps, qk_scale, BM, D, HD, R,
                           ROPE, FAST_EXP, SOFTMAX, QKNORM)  # fmt: skip
    else:
        for h in tl.range(H if ATTN else 0, num_stages=STAGES):
            if MERGE == 1:
                acc = _merged_head(_qkvg(xn, h, WQK, BQK, BM, D, HD), h, acc, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, QKN,
                                   BT, WP, eps, qk_scale, BM, D, HD, R, ROPE, FAST_EXP, SOFTMAX, QKNORM)  # fmt: skip
            elif MERGE == 2:  # [q | k] and [v | gate], channels in order; the rotation's pairs (2m, 2m + 1) split in registers
                c2h = tl.arange(0, 2 * HD)
                w2 = d[:, None] * 2 * HD + c2h[None, :]
                q, k = _halves(tl.dot(xn, tl.load(WQK + h * D * 2 * HD + w2, cache_modifier=".cg")) + tl.load(BQK + h * 2 * HD + c2h)[None, :],
                               BM, HD)  # fmt: skip
                qe, qo = tl.split(tl.reshape(q, [BM, HD2, 2]))
                ke, ko = tl.split(tl.reshape(k, [BM, HD2, 2]))
                p, den = _scores(qe, qo, ke, ko, h, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, QKN, BT, eps, qk_scale, BM, HD,
                                 R, ROPE, FAST_EXP, SOFTMAX, QKNORM)  # fmt: skip
                v, g = _halves(tl.dot(xn, tl.load(WVG + h * D * 2 * HD + w2, cache_modifier=".cg")) + tl.load(BVG + h * 2 * HD + c2h)[None, :],
                               BM, HD)  # fmt: skip
                o = _pv(p, v, den, FAST_EXP)
                wp = tl.load(WP + h * HD * D + c[:, None] * D + d[None, :], cache_modifier=".cg")
                acc = tl.dot((o * tl.sigmoid(g)).to(tl.bfloat16), wp, acc)
            else:
                wq, bq = WQK + h * 4 * D * HD2, BQK + h * 4 * HD2
                qe = tl.dot(xn, tl.load(wq + wqk, cache_modifier=".cg")) + tl.load(bq + c2)[None, :]
                qo = tl.dot(xn, tl.load(wq + D * HD2 + wqk, cache_modifier=".cg")) + tl.load(bq + HD2 + c2)[None, :]
                ke = tl.dot(xn, tl.load(wq + 2 * D * HD2 + wqk, cache_modifier=".cg")) + tl.load(bq + 2 * HD2 + c2)[None, :]
                ko = tl.dot(xn, tl.load(wq + 3 * D * HD2 + wqk, cache_modifier=".cg")) + tl.load(bq + 3 * HD2 + c2)[None, :]
                p, den = _scores(qe, qo, ke, ko, h, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, QKN, BT, eps, qk_scale, BM, HD,
                                 R, ROPE, FAST_EXP, SOFTMAX, QKNORM)  # fmt: skip
                wv, bv = WVG + h * 2 * D * HD, BVG + h * 2 * HD
                o = _pv(p, tl.dot(xn, tl.load(wv + wvg, cache_modifier=".cg")) + tl.load(bv + c)[None, :], den, FAST_EXP)
                g = tl.dot(xn, tl.load(wv + D * HD + wvg, cache_modifier=".cg")) + tl.load(bv + HD + c)[None, :]
                wp = tl.load(WP + h * HD * D + c[:, None] * D + d[None, :], cache_modifier=".cg")
                acc = tl.dot((o * tl.sigmoid(g)).to(tl.bfloat16), wp, acc)
    x = acc
    if HAS_FILM:
        gb = GB + env[:, None] * gb_stride + d[None, :]
        x = x * (1.0 + tl.load(gb, mask=xmask, other=0.0)) + tl.load(gb + D, mask=xmask, other=0.0)
    xn = (x * tl.rsqrt(tl.sum(x * x, 1) / D + eps)[:, None] * tl.load(N2 + d)[None, :]).to(tl.bfloat16)
    cb = tl.arange(0, BH)
    cb2 = tl.arange(0, 2 * BH)
    w12 = d[:, None] * BH + cb[None, :]
    acc = x + tl.load(B3 + d)[None, :]  # the residual, the SwiGLU's initial value
    for j in tl.range(NCH if MLP else 0, num_stages=STAGES):
        w, b = W12 + j * 2 * D * BH, B12 + j * 2 * BH
        if MERGE:
            u, z = _halves(tl.dot(xn, tl.load(w + d[:, None] * 2 * BH + cb2[None, :], cache_modifier=".cg")) + tl.load(b + cb2)[None, :], BM, BH)
        else:
            u = tl.dot(xn, tl.load(w + w12, cache_modifier=".cg")) + tl.load(b + cb)[None, :]
            z = tl.dot(xn, tl.load(w + D * BH + w12, cache_modifier=".cg")) + tl.load(b + BH + cb)[None, :]
        acc = tl.dot((u * z * tl.sigmoid(z)).to(tl.bfloat16), tl.load(W3 + j * BH * D + cb[:, None] * D + d[None, :], cache_modifier=".cg"), acc)
    return acc


@triton.jit
def _block_kernel(X, Y, COS, SIN, BT, GB, gb_stride, N1, WQK, BQK, QKN, WVG, BVG, WP, BP, N2, W12, B12, W3, B3, n_env, eps, qk_scale,
                  T: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, R: tl.constexpr,
                  NCH: tl.constexpr, BH: tl.constexpr, ROPE: tl.constexpr, HAS_FILM: tl.constexpr, STAGES: tl.constexpr,
                  MERGE: tl.constexpr, FAST_EXP: tl.constexpr, ABL: tl.constexpr, PREFETCH: tl.constexpr):  # fmt: skip
    """One block on the tokens of environments [E pid, E pid + E): rows i = e T + t of the tile; rows past E T idle.
    COS, SIN (N T, R): the rotation of the leading R channel pairs (ROPE 1; 0: none)."""
    i = tl.arange(0, BM)
    e_loc = i // T
    t = i % T
    env = (tl.program_id(0) * E + e_loc).to(tl.int64)
    xmask = ((e_loc < E) & (env < n_env))[:, None]
    row = env * T + t
    d = tl.arange(0, D)
    x = tl.load(X + row[:, None] * D + d[None, :], mask=xmask, other=0.0)
    y = _block(x, t, row, env, xmask, COS, SIN, X, 0, X, X, 0.0, BT, GB, gb_stride, N1, WQK, BQK, QKN, WVG, BVG, WP, BP, N2, W12, B12, W3, B3, eps,
               qk_scale, BM, D, H, HD, R, NCH, BH, ROPE, HAS_FILM, STAGES, MERGE, FAST_EXP, ABL, PREFETCH)  # fmt: skip
    tl.store(Y + row[:, None] * D + d[None, :], y, mask=xmask)


@triton.jit
def _embed(X, xd, GIDX, WE, CT, env, t, xmask, D: tl.constexpr, KE: tl.constexpr):
    """Token embeddings (BM, D) fp32 of a tile: GIDX (T, KE) the column of x of every channel of a token's features (-1:
    none; each token type has its own channels), WE (KE, D) bf16 the embeddings' weights stacked, CT (T, D) fp32 every
    token's constant (its embedding's bias and token embedding; a register's value)."""
    k = tl.arange(0, KE)
    d = tl.arange(0, D)
    gi = tl.load(GIDX + t[:, None] * KE + k[None, :])
    f = tl.load(X + env[:, None] * xd + gi, mask=xmask & (gi >= 0), other=0.0)
    return tl.dot(f.to(tl.bfloat16), tl.load(WE + k[:, None] * D + d[None, :])) + tl.load(CT + t[:, None] * D + d[None, :])


@triton.jit
def _net_block(x, blk, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, BT, GB, gb_stride, N1, WQK, BQK, QKN, WVG, BVG, WP, BP, N2, W12,
               B12, W3, B3, eps, qk_scale, BM: tl.constexpr, D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, R: tl.constexpr,
               NCH: tl.constexpr, BH: tl.constexpr, ROPE: tl.constexpr, STAGES: tl.constexpr, MERGE: tl.constexpr,
               FAST_EXP: tl.constexpr, PREFETCH: tl.constexpr):  # fmt: skip
    """Block ``blk`` of the network's stacked blocks on a tile's residual x."""
    return _block(x, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, BT, GB + blk * 2 * D, gb_stride, N1 + blk * D,
                  WQK + blk * H * (2 + 2 * (MERGE == 1)) * D * HD, BQK + blk * H * (2 + 2 * (MERGE == 1)) * HD, QKN + blk * 2 * HD,
                  WVG + blk * H * 2 * D * HD, BVG + blk * H * 2 * HD, WP + blk * H * HD * D, BP + blk * D, N2 + blk * D,
                  W12 + blk * NCH * 2 * D * BH, B12 + blk * NCH * 2 * BH, W3 + blk * NCH * BH * D, B3 + blk * D, eps, qk_scale, BM, D, H, HD, R, NCH,
                  BH, ROPE, True, STAGES, MERGE, FAST_EXP, 0, PREFETCH)  # fmt: skip


@triton.jit
def _net_kernel(X, Y, xd, yd, GIDX, WE, CT, PIDX, FRQ, COS, SIN, BT, GB, gb_stride, N1, WQK, BQK, QKN, WVG, BVG, WP, BP, N2, W12, B12, W3, B3, HNW,
                HW12, HB12, HW3, HB3, WOUT, BOUT, DEST, n_env, eps, qk_scale, pos_scale, T: tl.constexpr, E: tl.constexpr, BM: tl.constexpr,
                D: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, R: tl.constexpr, NCH: tl.constexpr, BH: tl.constexpr, KE: tl.constexpr,
                HNCH: tl.constexpr, HBH: tl.constexpr, NO: tl.constexpr, NB: tl.constexpr, STAGES: tl.constexpr, OUTER: tl.constexpr,
                ROPE: tl.constexpr, SKIP_EMBED: tl.constexpr, SKIP_BLOCKS: tl.constexpr, SKIP_HEAD: tl.constexpr, SKIP_STORE: tl.constexpr,
                MERGE: tl.constexpr, FAST_EXP: tl.constexpr, PREFETCH: tl.constexpr):  # fmt: skip
    """HOIRigidFormerNet.forward on the environments [E pid, E pid + E): x rows (xd apart) in, y rows (yd apart) out.

    PIDX (T, HD/2): the column of x of the coordinate of each rotary pair of a token (-1: none), FRQ (HD/2,) its
    frequency; the NB blocks' weights stacked (NB, ...) as _block's; GB rows (NB, 2 D); HNW (2 D,) the read-out norm's
    weight; HW12 (HNCH, 4, D, HBH): the read-out SwiGLU's value and gate columns on the embedding's and the last block's
    halves; HB12 (HNCH, 2, HBH); HW3 (HNCH, HBH, D); WOUT (D, NO) bf16 all read-out heads' weights; DEST (T, NO) the
    column of y of each head output of a token (-1: none). The loop over blocks: OUTER 0 a loop, 1 a loop that is not
    software-pipelined, 2 unrolled. The rotation: ROPE 1 from the tables COS, SIN (N T, R) computed before, 2 from the
    token positions in the kernel. Diagnosis only (FusedHOIRigidFormerNet.debug; wrong outputs): ROPE 0 no rotation, SKIP_EMBED the
    embedding's product, SKIP_BLOCKS, SKIP_HEAD the read-out SwiGLU, SKIP_STORE.
    """
    i = tl.arange(0, BM)
    e_loc = i // T
    t = i % T
    env = (tl.program_id(0) * E + e_loc).to(tl.int64)
    row = env * T + t
    xmask = ((e_loc < E) & (env < n_env))[:, None]
    d = tl.arange(0, D)
    if SKIP_EMBED:
        x = tl.zeros([BM, D], dtype=tl.float32) + tl.load(CT + t[:, None] * D + d[None, :])
    else:
        x = _embed(X, xd, GIDX, WE, CT, env, t, xmask, D, KE)
    if not SKIP_BLOCKS:
        if OUTER == 2:
            for blk in tl.static_range(NB):
                x = _net_block(x, blk, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, BT, GB, gb_stride, N1, WQK, BQK, QKN, WVG, BVG,
                               WP, BP, N2, W12, B12, W3, B3, eps, qk_scale, BM, D, H, HD, R, NCH, BH, ROPE, STAGES, MERGE,
                               FAST_EXP, PREFETCH)  # fmt: skip
        elif OUTER == 1:
            for blk in tl.range(NB, num_stages=1):
                x = _net_block(x, blk, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, BT, GB, gb_stride, N1, WQK, BQK, QKN, WVG, BVG,
                               WP, BP, N2, W12, B12, W3, B3, eps, qk_scale, BM, D, H, HD, R, NCH, BH, ROPE, STAGES, MERGE,
                               FAST_EXP, PREFETCH)  # fmt: skip
        else:
            for blk in range(NB):
                x = _net_block(x, blk, t, row, env, xmask, COS, SIN, X, xd, PIDX, FRQ, pos_scale, BT, GB, gb_stride, N1, WQK, BQK, QKN, WVG, BVG,
                               WP, BP, N2, W12, B12, W3, B3, eps, qk_scale, BM, D, H, HD, R, NCH, BH, ROPE, STAGES, MERGE,
                               FAST_EXP, PREFETCH)  # fmt: skip
    if SKIP_HEAD:
        acc = x
    else:
        if SKIP_EMBED:
            h0 = tl.zeros([BM, D], dtype=tl.float32) + tl.load(CT + t[:, None] * D + d[None, :])
        else:
            h0 = _embed(X, xd, GIDX, WE, CT, env, t, xmask, D, KE)  # recomputed: cheaper than holding it through the blocks
        r = tl.rsqrt((tl.sum(h0 * h0, 1) + tl.sum(x * x, 1)) / (2 * D) + eps)[:, None]
        xa = (h0 * r * tl.load(HNW + d)[None, :]).to(tl.bfloat16)
        xb = (x * r * tl.load(HNW + D + d)[None, :]).to(tl.bfloat16)
        cb = tl.arange(0, HBH)
        cb2 = tl.arange(0, 2 * HBH)
        w12 = d[:, None] * HBH + cb[None, :]
        w12m = d[:, None] * 2 * HBH + cb2[None, :]
        acc = tl.zeros([BM, D], dtype=tl.float32) + tl.load(HB3 + d)[None, :]
        for j in tl.range(HNCH, num_stages=STAGES):
            w, b = HW12 + j * 4 * D * HBH, HB12 + j * 2 * HBH
            if MERGE:  # value and gate as one product per half of the input (HW12 (HNCH, 2, D, 2 HBH), HB12 (HNCH, 2 HBH))
                uz = tl.dot(xa, tl.load(w + w12m, cache_modifier=".cg"))
                u, z = _halves(tl.dot(xb, tl.load(w + 2 * D * HBH + w12m, cache_modifier=".cg"), uz) + tl.load(b + cb2)[None, :], BM, HBH)
            else:
                u = tl.dot(xa, tl.load(w + w12, cache_modifier=".cg"))
                u = tl.dot(xb, tl.load(w + D * HBH + w12, cache_modifier=".cg"), u) + tl.load(b + cb)[None, :]
                z = tl.dot(xa, tl.load(w + 2 * D * HBH + w12, cache_modifier=".cg"))
                z = tl.dot(xb, tl.load(w + 3 * D * HBH + w12, cache_modifier=".cg"), z) + tl.load(b + HBH + cb)[None, :]
            w3 = tl.load(HW3 + j * HBH * D + cb[:, None] * D + d[None, :], cache_modifier=".cg")
            acc = tl.dot((u * z * tl.sigmoid(z)).to(tl.bfloat16), w3, acc)
    if not SKIP_STORE:
        o = tl.arange(0, NO)
        out = tl.dot(acc.to(tl.bfloat16), tl.load(WOUT + d[:, None] * NO + o[None, :])) + tl.load(BOUT + o)[None, :]
        dest = tl.load(DEST + t[:, None] * NO + o[None, :])
        tl.store(Y + env[:, None] * yd + dest, out, mask=xmask & (dest >= 0))


# The kernels are launched as opaque operators: torch.compile calls them as they are (and CUDA graphs capture their
# launches) instead of compiling the Triton kernels a second time, which ran 25-40 % slower than the launch's own.
BLOCK_CONFIG = ("T", "E", "BM", "D", "H", "HD", "R", "NCH", "BH", "ROPE", "HAS_FILM", "STAGES", "MERGE", "FAST_EXP", "ABL", "PREFETCH", "num_warps")
NET_CONFIG = ("T", "E", "BM", "R", "NB", "STAGES", "MERGE", "FAST_EXP", "PREFETCH", "D", "H", "HD", "NCH", "BH", "KE", "HNCH", "HBH", "NO",
              "OUTER", "ROPE", "SKIP_EMBED", "SKIP_BLOCKS", "SKIP_HEAD", "SKIP_STORE", "num_warps", "num_stages")  # fmt: skip


@torch.library.custom_op("ndm_gym::rf_block", mutates_args=())
def rf_block_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, tile: torch.Tensor, gb: torch.Tensor, gb_stride: int,
                weights: list[torch.Tensor], eps: float, qk_scale: float, config: list[int]) -> torch.Tensor:  # fmt: skip
    """One block (``_block_kernel``) on x (N, T, D) -> y; weights in _block_kernel's order, config as BLOCK_CONFIG."""
    y = torch.empty_like(x)
    kw = dict(zip(BLOCK_CONFIG, config))
    _block_kernel[(triton.cdiv(x.shape[0], kw["E"]),)](x, y, cos, sin, tile, gb, gb_stride, *weights, x.shape[0], eps, qk_scale, **kw)
    return y


@rf_block_op.register_fake
def _(x, cos, sin, tile, gb, gb_stride, weights, eps, qk_scale, config):
    return torch.empty_like(x)


@torch.library.custom_op("ndm_gym::rf_hoi_net", mutates_args=())
def rf_hoi_net_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, tile: torch.Tensor, gb: torch.Tensor, tensors: list[torch.Tensor],
                  out_size: int, eps: float, qk_scale: float, pos_scale: float, config: list[int]) -> torch.Tensor:  # fmt: skip
    """The whole network (``_net_kernel``) on x (N, in_dim + 3 tokens) -> y (N, out_size); tensors: the embedding's
    five (GIDX, WE, CT, PIDX, FRQ), the blocks' thirteen, the read-out's eight, in _net_kernel's order; config as
    NET_CONFIG."""
    y = x.new_empty(x.shape[0], out_size)
    kw = dict(zip(NET_CONFIG, config))
    _net_kernel[(triton.cdiv(x.shape[0], kw["E"]),)](x, y, x.stride(0), y.stride(0), *tensors[:5], cos, sin, tile, gb, gb.stride(0), *tensors[5:],
                                                     x.shape[0], eps, qk_scale, pos_scale, **kw)  # fmt: skip
    return y


@rf_hoi_net_op.register_fake
def _(x, cos, sin, tile, gb, tensors, out_size, eps, qk_scale, pos_scale, config):
    return x.new_empty(x.shape[0], out_size)


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _fp32_parameters(module: nn.Module):
    """Refuse a module whose parameters are not fp32 (e.g. HOIWorldModel.inference_copy with bf16): the kernels run its
    fp32 model with an fp32 residual stream, while a bf16 copy's reference keeps everything in bf16."""
    dtypes = sorted({str(p.dtype) for p in module.parameters()} - {str(torch.float32)})
    if dtypes:
        raise ValueError(f"the fused kernels take a model with fp32 parameters (an inference copy with dtype=None), got {', '.join(dtypes)}")


def _next_pow2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


def _no_grad_check(what: str):
    if torch.is_grad_enabled():
        raise RuntimeError(f"{what} is inference-only (no backward): run it under torch.no_grad() or torch.inference_mode()")


class FusedBlock(nn.Module):
    """Inference-only stand-in for a ``Block``: the same ``forward(x, cos, sin, bias, c)`` as one Triton kernel.

    The block's weights are copied at construction (matrices bf16, vectors fp32), so later changes of the block are not
    seen. x (N, T, D) fp32; cos, sin (N, 1, T, r) or None; bias (1, H, T, T) or None (one bias for all environments: a
    per-environment bias is not supported); c (N, C) or None; T at most ``block_m``. FiLM's gamma and beta are computed
    per environment in fp32 before the kernel. ``block_m``, ``hidden_chunk``, ``num_warps``, ``stages`` (software
    pipelining depth, at least 2), ``merge`` (1: q, k, v and the gate of a head as one product; 2: [q, k] and [v, gate]
    as two; both: the SwiGLU's value and gate of a chunk as one; split in registers; 0: a product each) and
    ``fast_exp`` (1: the attention probabilities by
    fp16 exponentials, two per instruction, and their product with v in fp16; 0: fp32 exponentials, the product in
    bf16) are the kernel's tuning knobs. ``stages`` and ``merge`` None take the fastest measured on the device's
    generation (tools/bench_rf_kernel.py --sweep): on compute capability 9 and later (an H100) stages 3 and merge 1
    (one wide product beats several narrow ones there: 5.9 instead of 6.7 ms per block at 65536 environments), before
    (an A100) stages 2 (3 does not fit its shared memory) and merge 0 (5 % faster there). Only ``fast_exp`` changes
    the numbers' rounding: fp16 (10-bit mantissa) instead of bf16 for the probabilities and v (v must stay below fp16's
    65504 in magnitude), within the same bf16-level error. ``prefetch`` 1 (with merge 1): the next head's product is issued
    before this head's attention and gating, so that on a GPU with asynchronous tensor-core products (an H100) it runs
    meanwhile, at the cost of its accumulator's registers.
    """

    def __init__(self, block: Block, block_m: int = 128, hidden_chunk: int = 64, num_warps: int = 8, stages: int | None = None,
                 merge: int | None = None, fast_exp: int = 0, prefetch: int = 0):  # fmt: skip
        super().__init__()
        D, H, hd, hid = block.norm1.weight.numel(), block.heads, block.hd, block.w3.in_features
        if not (_pow2(D) and _pow2(hd) and hd >= 32 and _pow2(block_m) and _pow2(hidden_chunk) and hidden_chunk >= 16 and hid % hidden_chunk == 0):
            raise ValueError(f"unsupported block: width {D}, head width {hd}, hidden {hid}, block_m {block_m}, hidden_chunk {hidden_chunk} "
                             "(width, head width >= 32, block_m and hidden_chunk powers of two, hidden a multiple of hidden_chunk)")  # fmt: skip
        if block.norm1.weight.device.type != "cuda":
            raise ValueError(f"the block must be on a CUDA device, not {block.norm1.weight.device}")
        _fp32_parameters(block)
        hopper = torch.cuda.get_device_capability(block.norm1.weight.device)[0] >= 9
        stages = (3 if hopper else 2) if stages is None else stages
        merge = int(hopper) if merge is None else merge
        if stages < 2:  # the loops unpipelined: on an H100 (Triton 3.6) wrong results or illegal memory accesses; an A100 is unaffected
            raise ValueError(f"stages must be at least 2, got {stages}: without software pipelining Triton 3.6 miscompiles this kernel on an "
                             "H100 (wrong results, illegal memory accesses)")  # fmt: skip
        eps = {m.eps for m in (block.norm1, block.q_norm, block.k_norm, block.norm2)}
        if len(eps) != 1 or None in eps:
            raise ValueError(f"the block's RMSNorms need one explicit eps, got {eps}")
        if merge not in (0, 1, 2) or fast_exp not in (0, 1) or prefetch not in (0, 1) or (prefetch and merge != 1):
            raise ValueError(f"merge must be 0, 1 or 2, fast_exp and prefetch 0 or 1, prefetch needs merge 1; got {merge}, {fast_exp}, {prefetch}")
        self.dim, self.heads, self.hd, self.hidden, self.eps = D, H, hd, hid, float(eps.pop())
        self.block_m, self.hidden_chunk, self.num_warps, self.stages = block_m, hidden_chunk, num_warps, stages
        self.merge, self.fast_exp, self.prefetch = merge, fast_exp, prefetch
        self.debug = dict(ABL=0)  # diagnosis only: parts of the block left out (_block's ABL), wrong outputs
        self.has_film = block.film is not None
        nch, bh = hid // hidden_chunk, hidden_chunk
        with torch.no_grad():
            f32 = lambda t: t.detach().float().contiguous()  # noqa: E731
            bf = lambda t: t.detach().to(torch.bfloat16).contiguous()  # noqa: E731
            w, b = block.qkvg.weight.detach().float().t(), block.qkvg.bias.detach().float()  # (D, 4D): columns part, head, channel
            cols = lambda part, h: part * D + h * hd + torch.arange(hd, device=w.device)  # noqa: E731
            qk = [[cols(p, h)[par::2] for p in (0, 1) for par in (0, 1)] for h in range(H)]  # q even, q odd, k even, k odd
            vg = [[cols(p, h) for p in (2, 3)] for h in range(H)]
            self.register_buffer("wqk", bf(torch.stack([torch.stack([w[:, i] for i in r]) for r in qk])), persistent=False)
            self.register_buffer("bqk", f32(torch.stack([torch.stack([b[i] for i in r]) for r in qk])), persistent=False)
            qn, kn = block.q_norm.weight, block.k_norm.weight
            self.register_buffer("qkn", f32(torch.stack((qn[0::2], qn[1::2], kn[0::2], kn[1::2]))), persistent=False)
            self.register_buffer("wvg", bf(torch.stack([torch.stack([w[:, i] for i in r]) for r in vg])), persistent=False)
            self.register_buffer("bvg", f32(torch.stack([torch.stack([b[i] for i in r]) for r in vg])), persistent=False)
            self.register_buffer("wp", bf(block.proj.weight.t().reshape(H, hd, D)), persistent=False)
            self.register_buffer("bp", f32(block.proj.bias), persistent=False)
            self.register_buffer("n1", f32(block.norm1.weight), persistent=False)
            self.register_buffer("n2", f32(block.norm2.weight), persistent=False)
            w12, b12 = block.w12.weight.detach().t(), block.w12.bias.detach()  # (D, 2 hid): value columns, then gate columns
            chunk = lambda t, j: torch.stack((t[..., j * bh : (j + 1) * bh], t[..., hid + j * bh : hid + (j + 1) * bh]), 0)  # noqa: E731
            self.register_buffer("w12", bf(torch.stack([chunk(w12, j) for j in range(nch)])), persistent=False)
            self.register_buffer("b12", f32(torch.stack([chunk(b12, j) for j in range(nch)])), persistent=False)
            self.register_buffer("w3", bf(block.w3.weight.t().reshape(nch, bh, D)), persistent=False)
            self.register_buffer("b3", f32(block.w3.bias), persistent=False)
            cols_of = lambda t: t.permute(0, 2, 1, 3).flatten(2)  # noqa: E731  (n, parts, D, c) -> (n, D, parts c)
            if merge == 1:  # one product per head (q even, q odd, k even, k odd, v, gate)
                self.wqk = torch.cat((cols_of(self.wqk), cols_of(self.wvg)), -1).contiguous()
                self.bqk = torch.cat((self.bqk.flatten(1), self.bvg.flatten(1)), -1).contiguous()
            elif merge == 2:  # two per head: q and k with their channels in order, v and the gate
                nat = [[cols(p, h) for p in (0, 1)] for h in range(H)]
                self.wqk = bf(torch.stack([torch.cat([w[:, i] for i in r], -1) for r in nat]))
                self.bqk = f32(torch.stack([torch.cat([b[i] for i in r]) for r in nat]))
                self.wvg, self.bvg = cols_of(self.wvg).contiguous(), self.bvg.flatten(1).contiguous()
            if merge:  # one product per chunk (value, gate)
                self.w12, self.b12 = cols_of(self.w12).contiguous(), self.b12.flatten(1).contiguous()
            if self.has_film:
                self.register_buffer("film_w", f32(block.film.weight), persistent=False)
                self.register_buffer("film_b", f32(block.film.bias), persistent=False)

    def film(self, c: torch.Tensor) -> torch.Tensor:
        """FiLM's (gamma, beta) of every environment in fp32: (N, 2D)."""
        with torch.autocast("cuda", enabled=False):
            return F.linear(c.float(), self.film_w, self.film_b)

    def forward(self, x, cos=None, sin=None, bias=None, c=None):
        _no_grad_check("FusedBlock")
        N, T, D = x.shape
        if D != self.dim or x.dtype != torch.float32 or x.device != self.n1.device:
            raise ValueError(f"x must be an fp32 (N, T, {self.dim}) tensor on {self.n1.device}, got {x.dtype} {tuple(x.shape)} on {x.device}")
        E = self.block_m // T
        if E < 1:
            raise ValueError(f"{T} tokens do not fit in block_m {self.block_m}")
        R = 0
        if cos is not None:
            R = cos.shape[-1]
            if cos.shape != (N, 1, T, R) or sin is None or sin.shape != cos.shape or 2 * R > self.hd:
                got = (tuple(cos.shape), None if sin is None else tuple(sin.shape))
                raise ValueError(f"cos, sin must be (N, 1, T, r) with 2 r <= {self.hd}, got {got}")
            cos, sin = cos.float().contiguous(), sin.float().contiguous()
        if bias is not None and bias.shape != (1, self.heads, T, T):
            raise ValueError(f"bias must be (1, {self.heads}, T, T) (one bias for all environments), got {tuple(bias.shape)}")
        gb = None
        if self.has_film:
            if c is None:
                raise ValueError("the block has FiLM: c is required")
            gb = self.film(c)
        elif c is not None:
            raise ValueError("the block has no FiLM: c must be None")
        x = x.contiguous()
        tile = attention_tile(bias, self.heads, T, self.block_m, x.device)
        config = [T, E, self.block_m, self.dim, self.heads, self.hd, R, self.hidden // self.hidden_chunk, self.hidden_chunk, 0 if cos is None else 1,
                  int(self.has_film), self.stages, self.merge, self.fast_exp, self.debug["ABL"], self.prefetch, self.num_warps]  # fmt: skip
        weights = [getattr(self, n) for n in BLOCK_BUFFERS]
        return rf_block_op(x, x if cos is None else cos, x if sin is None else sin, tile, x if gb is None else gb, 0 if gb is None else gb.stride(0),
                           weights, self.eps, self.hd**-0.5 * LOG2E, config)  # fmt: skip


class FusedHOIRigidFormerNet(HOIRigidFormerNet):
    """``HOIRigidFormerNet`` whose forward is one Triton kernel (and the per-environment FiLM vectors in PyTorch):
    inference only. Made from a network by ``from_net`` (a copy, so ``inputs``, ``make_input_norm`` and the parameters
    are the network's; the kernel uses copies of the parameters made then). Returns fp32 outputs in the network's layout.
    Refused on GPUs of compute capability 9.x (H100): there Triton 3.6 compiled the kernel with the rotation from the
    token positions to one that reads past its shared memory (an illegal access that ends the process) or returns NaN;
    with the rotation from tables (now the default) it ran clean in the diagnosis, and the refusal stays until the full
    tests pass there. ``force`` (tools/bench_rf_kernel.py --diagnose) or NDM_RF_WHOLE_ON_SM90=1 (the tests) build it
    anyway, to verify it.
    """

    @classmethod
    def from_net(cls, net: HOIRigidFormerNet, block_m: int = 128, hidden_chunk: int = 64, head_chunk: int = 32, num_warps: int = 8,
                 stages: int | None = None, merge: int | None = None, fast_exp: int = 0, prefetch: int = 0,
                 force: bool = False) -> FusedHOIRigidFormerNet:  # fmt: skip
        if type(net) is not HOIRigidFormerNet:
            raise TypeError(f"expected an HOIRigidFormerNet, got {type(net).__name__}")
        _fp32_parameters(net)
        cap = torch.cuda.get_device_capability(net.tok_emb.device)
        if cap[0] == 9 and not (force or os.environ.get("NDM_RF_WHOLE_ON_SM90") == "1"):  # the variable: to verify it there
            raise RuntimeError(f"the whole-network kernel is refused on compute capability {cap[0]}.{cap[1]}: Triton 3.6 miscompiles it there "
                               "(shared-memory reads past the allocation, NaN); fuse the blocks only (fused_copy's default)")  # fmt: skip
        m = copy.deepcopy(net)
        m.__class__ = cls  # the same module with this forward
        m._pack(block_m, hidden_chunk, head_chunk, num_warps, stages, merge, fast_exp, prefetch)
        return m

    def _pack(self, block_m, hidden_chunk, head_chunk, num_warps, stages, merge, fast_exp, prefetch):
        B, P, NR, D = self.n_body, self.n_props, self.n_reg, self.tok_emb.shape[1]
        NT = B + 1 + P  # tokens with features: bodies, the object, props; then the registers
        self.n_tok = T = NT + NR
        fbs = [FusedBlock(b, block_m, hidden_chunk, num_warps, stages, merge, fast_exp, prefetch) for b in self.blocks]
        fb = fbs[0]
        merge, stages = fb.merge, fb.stages  # resolved for the device
        hid = self.head_w3.in_features
        if not (block_m // T >= 1 and _pow2(head_chunk) and head_chunk >= 16 and hid % head_chunk == 0 and fb.has_film):
            raise ValueError(f"unsupported: {T} tokens in block_m {block_m}, read-out hidden {hid} in chunks of {head_chunk}")
        hd, dev = fb.hd, self.tok_emb.device
        ar = lambda n, s=0: torch.arange(s, s + n, dtype=torch.int32)  # noqa: E731
        f32 = lambda t: t.detach().float().contiguous().to(dev)  # noqa: E731
        bf = lambda t: t.detach().to(torch.bfloat16).contiguous().to(dev)  # noqa: E731
        with torch.no_grad():
            # token embedding: one product over KE channels, a body's features in [0, kb), the object's in [kb, kb + ko), a prop's after
            kb, ko, kp = self.body_dim, self.obj_dim, self.prop_dim if P else 0
            KE = max(16, _next_pow2(kb + ko + kp))
            we = torch.zeros(KE, D, device=dev)
            we[:kb], we[kb : kb + ko] = self.embed_body.weight.t(), self.embed_obj.weight.t()
            gidx = torch.full((T, KE), -1, dtype=torch.int32)
            for b in range(B):
                gidx[b, :kb] = ar(kb, b * kb)
            gidx[B, kb : kb + ko] = ar(ko, B * kb)
            ct = torch.zeros(T, D, device=dev)
            ct[:NT] = self.tok_emb
            ct[:B] += self.embed_body.bias
            ct[B] += self.embed_obj.bias
            if P:
                we[kb + ko : kb + ko + kp] = self.embed_prop.weight.t()
                for k in range(P):
                    gidx[B + 1 + k, kb + ko : kb + ko + kp] = ar(kp, B * kb + ko + k * kp)
                ct[B + 1 : NT] += self.embed_prop.bias
            if NR:
                ct[NT:] = self.registers[0]
            # rotary angles from the token positions, x[:, in_dim + 3 t + coordinate] * POS_SCALE; pair m: coordinate m // F, frequency m % F
            freqs, HD2 = self.rope.freqs.detach().float().cpu(), hd // 2
            F_, R = freqs.numel(), 3 * freqs.numel()
            if R > HD2:
                raise ValueError(f"{R} rotary pairs in a head of {hd} channels")
            pidx, frq = torch.full((T, HD2), -1, dtype=torch.int32), torch.zeros(HD2)
            for m in range(R):
                frq[m] = freqs[m % F_]
                pidx[:NT, m] = self.in_dim + 3 * ar(NT) + m // F_
            # FiLM of all blocks from the context network's output: (L 2D, C); the read-out SwiGLU in chunks of head_chunk
            film_w = torch.cat([b.film.weight for b in self.blocks])
            film_b = torch.cat([b.film.bias for b in self.blocks])
            w12, b12 = self.head_w12.weight.detach(), self.head_w12.bias.detach()  # (2 hid, 2D): value rows, then gate rows
            hc = head_chunk
            rows = lambda j, gate: slice(gate * hid + j * hc, gate * hid + (j + 1) * hc)  # noqa: E731
            hw12 = torch.stack([torch.stack([w12[rows(j, g), half * D : (half + 1) * D].t() for g in (0, 1) for half in (0, 1)])
                                for j in range(hid // hc)])  # fmt: skip
            hb12 = torch.stack([torch.stack([b12[rows(j, g)] for g in (0, 1)]) for j in range(hid // hc)])
            if merge:  # value and gate as one product per half of the input: (HNCH, 2, D, 2 hc), biases (HNCH, 2 hc)
                hw12 = torch.stack((torch.cat((hw12[:, 0], hw12[:, 2]), -1), torch.cat((hw12[:, 1], hw12[:, 3]), -1)), 1)
                hb12 = hb12.flatten(1)
            # read-out heads as one product (D, NO); DEST: the column of y of a token's head output (HOIRigidFormerNet.forward's layout)
            heads = [self.root_head, self.joint_head, self.obj_head, self.eef_head, self.feet_head]
            widths = [h.out_features for h in heads]
            NO = max(16, _next_pow2(sum(widths)))
            start = [sum(widths[:k]) for k in range(len(heads))]
            wout, bout = torch.zeros(D, NO, device=dev), torch.zeros(NO, device=dev)
            for h, s in zip(heads, start):
                wout[:, s : s + h.out_features], bout[s : s + h.out_features] = h.weight.t(), h.bias
            ro, jo, oo, fo = widths[0], widths[1], widths[2], widths[4]
            base_o = ro + NUM_J * jo
            base_e = base_o + oo
            base_f = base_e + 3 * len(self.eef_tok)
            dest = torch.full((T, NO), -1, dtype=torch.int32)
            dest[0, start[0] : start[0] + ro] = ar(ro)
            for tok in range(1, B):  # joint j's outputs at ro + o J + j
                dest[tok, start[1] : start[1] + jo] = ro + ar(jo) * NUM_J + int(self.joint_of_body[tok - 1])
            dest[B, start[2] : start[2] + oo] = ar(oo, base_o)
            for e, tok in enumerate(self.eef_tok):
                dest[tok, start[3] : start[3] + 3] = ar(3, base_e + 3 * e)
            for f, tok in enumerate(self.feet_tok):  # (substep, foot, axis)
                k = ar(fo)
                dest[tok, start[4] : start[4] + fo] = base_f + (k // 3) * 6 + f * 3 + k % 3
            self.out_size = base_f + DECIMATION * 2 * 3
            written = dest[dest >= 0].sort().values
            if not torch.equal(written, torch.arange(self.out_size, dtype=torch.int32)):
                raise RuntimeError("the read-out heads do not write every output exactly once")
            bufs = dict(gidx=gidx, we=bf(we), ct=f32(ct), pidx=pidx, frq=frq, film_w=f32(film_w), film_b=f32(film_b), hnw=f32(self.head_norm.weight),
                        hw12=bf(hw12), hb12=f32(hb12), hw3=bf(self.head_w3.weight.t().reshape(hid // hc, hc, D)), hb3=f32(self.head_w3.bias),
                        wout=bf(wout), bout=f32(bout), dest=dest)  # fmt: skip
            bufs |= {f"blk_{n}": torch.stack([getattr(f, n) for f in fbs]) for n in BLOCK_BUFFERS}
            for name, buf in bufs.items():
                self.register_buffer("k_" + name, buf.to(dev), persistent=False)
        eps = {fb.eps, self.head_norm.eps}
        if len(eps) != 1:
            raise ValueError(f"the RMSNorms need one eps, got {eps}")
        self.knobs = dict(block_m=block_m, hidden_chunk=hidden_chunk, head_chunk=head_chunk, num_warps=num_warps, stages=stages, merge=merge,
                          fast_exp=fast_exp, prefetch=prefetch)
        # diagnosis of a GPU on which the kernel fails: _net_kernel's constexprs (anything but these defaults gives wrong outputs, except
        # OUTER) and the launch's num_stages (the default for loops without their own: the loop over blocks)
        self.debug = dict(OUTER=0, ROPE=1, SKIP_EMBED=False, SKIP_BLOCKS=False, SKIP_HEAD=False, SKIP_STORE=False, LAUNCH_STAGES=stages)
        self.k_eps, self.k_dims = fb.eps, dict(D=D, H=fb.heads, HD=hd, NCH=fb.hidden // hidden_chunk, BH=hidden_chunk, KE=KE,
                                                  HNCH=hid // hc, HBH=hc, NO=NO)  # fmt: skip
        self.k_scal = B * kb + ko + P * kp  # the context's scalars: x[:, k_scal : in_dim]

    def forward(self, x):
        _no_grad_check("FusedHOIRigidFormerNet")
        N = x.shape[0]
        if x.dtype != torch.float32 or x.device != self.k_we.device or x.shape[1] != self.in_dim + 3 * (self.n_tok - self.n_reg):
            raise ValueError(f"x must be an fp32 (N, {self.in_dim + 3 * (self.n_tok - self.n_reg)}) tensor on {self.k_we.device}, got {x.dtype} "
                             f"{tuple(x.shape)} on {x.device}")  # fmt: skip
        x = x.contiguous()
        with torch.autocast("cuda", enabled=False):  # the FiLM vectors of every block and environment, fp32
            c = F.silu(F.linear(x[:, self.k_scal : self.in_dim], self.ctx_net[0].weight, self.ctx_net[0].bias))
            gb = F.linear(c, self.k_film_w, self.k_film_b)
        bm = self.knobs["block_m"]
        T, E = self.n_tok, bm // self.n_tok
        tile = attention_tile(self.rel_bias[:, self.rel].unsqueeze(0), self.k_dims["H"], T, bm, x.device)
        cos = sin = x
        R = 3 * self.rope.freqs.numel()
        if self.debug["ROPE"] == 1:  # the rotation from cos, sin tables computed here (as FusedBlock's; in the kernel: ROPE 2)
            pos = POS_SCALE * x[:, self.in_dim :].unflatten(-1, (self.n_tok - self.n_reg, 3))
            cos, sin = (t.reshape(N * T, R).contiguous() for t in self.rope(torch.cat((pos, pos.new_zeros(N, self.n_reg, 3)), 1)))
        tensors = [self.k_gidx, self.k_we, self.k_ct, self.k_pidx, self.k_frq] + [getattr(self, "k_blk_" + n) for n in BLOCK_BUFFERS]
        tensors += [self.k_hnw, self.k_hw12, self.k_hb12, self.k_hw3, self.k_hb3, self.k_wout, self.k_bout, self.k_dest]
        kw = dict(T=T, E=E, BM=bm, R=R, NB=len(self.blocks), STAGES=self.knobs["stages"], MERGE=self.knobs["merge"],
                  FAST_EXP=self.knobs["fast_exp"], PREFETCH=self.knobs["prefetch"], num_warps=self.knobs["num_warps"],
                  num_stages=self.debug["LAUNCH_STAGES"]) | self.k_dims | {k: v for k, v in self.debug.items() if k != "LAUNCH_STAGES"}  # fmt: skip
        return rf_hoi_net_op(x, cos, sin, tile, gb, tensors, self.out_size, self.k_eps, self.k_dims["HD"] ** -0.5 * LOG2E, POS_SCALE,
                             [int(kw[k]) for k in NET_CONFIG])  # fmt: skip


class FusedG1RigidFormerNet(RigidFormerNet):
    """``RigidFormerNet`` (the G1's tracking world models, LAFAN1 and the two-G1 scene) whose forward is the whole-network
    kernel of ``FusedHOIRigidFormerNet`` (``_net_kernel``: the token embedding, all blocks with the residual on chip, the
    read-out), with the G1's tables: one token type (the bodies' features, each body's embedding added to the embedding's
    bias), the registers, the rotary positions of the bodies' link origins, the root head on the pelvis token and the
    joint head on the other bodies' tokens (``RigidFormerNet._readout``'s layout). Before it, in PyTorch: the per-
    environment FiLM vectors and the rotation's cos, sin tables. Inference only; made from a network by ``from_net`` (a
    copy, so ``inputs`` and ``make_input_norm`` are the network's; the kernel uses copies of the parameters made then).
    Takes a network with fp32 parameters, FiLM, the rotary positions and the tree bias, one rotary point per body, no
    ``rope_center``, one token per body (no ``merge``; the kernel embeds one body per token and reads every body out of
    its own token) and the read-out on the embedding and the last block (the defaults); refused on compute capability 9.x
    (an H100), as ``FusedHOIRigidFormerNet``, unless ``force`` or NDM_RF_WHOLE_ON_SM90=1."""

    @classmethod
    def from_net(cls, net: RigidFormerNet, block_m: int = 128, hidden_chunk: int = 64, head_chunk: int = 32, num_warps: int = 8,
                 stages: int | None = None, merge: int | None = None, fast_exp: int = 0, prefetch: int = 0,
                 force: bool = False) -> FusedG1RigidFormerNet:  # fmt: skip
        if not isinstance(net, RigidFormerNet) or isinstance(net, FusedG1RigidFormerNet):
            raise TypeError(f"expected a RigidFormerNet, got {type(net).__name__}")
        _fp32_parameters(net)
        cap = torch.cuda.get_device_capability(net.embed.weight.device)
        if cap[0] == 9 and not (force or os.environ.get("NDM_RF_WHOLE_ON_SM90") == "1"):  # the variable: to verify it there
            raise RuntimeError(f"the whole-network kernel is refused on compute capability {cap[0]}.{cap[1]}: Triton 3.6 miscompiles it there "
                               "(shared-memory reads past the allocation, NaN); fuse the blocks only (fused_copy's default)")  # fmt: skip
        unsupported = {"no FiLM": net.ctx_net is None, "no rotary positions": net.rope is None, "no tree bias": net.rel_bias is None,
                       "anchors": net.n_anchor != 1, "rope_center": net.rope_center,
                       "merged tokens (--rf_merge: one token per body only)": net.merge != "none",
                       "a read-out on other layers than the embedding and the last block": tuple(net.head_layers) != (0, len(net.blocks))}  # fmt: skip
        if any(unsupported.values()):
            raise ValueError(f"the whole-network kernel does not support {', '.join(k for k, v in unsupported.items() if v)}")
        m = copy.deepcopy(net)
        m.__class__ = cls  # the same module with this forward
        m._pack(block_m, hidden_chunk, head_chunk, num_warps, stages, merge, fast_exp, prefetch)
        return m

    def _pack(self, block_m, hidden_chunk, head_chunk, num_warps, stages, merge, fast_exp, prefetch):
        B, NR, D, td = self.n_body, self.n_reg, self.embed.weight.shape[0], self.tok_dim
        self.n_tok = T = B + NR
        fbs = [FusedBlock(b, block_m, hidden_chunk, num_warps, stages, merge, fast_exp, prefetch) for b in self.blocks]
        fb = fbs[0]
        merge, stages = fb.merge, fb.stages  # resolved for the device
        hid = self.head_w3.in_features
        if not (block_m // T >= 1 and _pow2(head_chunk) and head_chunk >= 16 and hid % head_chunk == 0 and fb.has_film):
            raise ValueError(f"unsupported: {T} tokens in block_m {block_m}, read-out hidden {hid} in chunks of {head_chunk}")
        hd, dev = fb.hd, self.embed.weight.device
        ar = lambda n, s=0: torch.arange(s, s + n, dtype=torch.int32)  # noqa: E731
        f32 = lambda t: t.detach().float().contiguous().to(dev)  # noqa: E731
        bf = lambda t: t.detach().to(torch.bfloat16).contiguous().to(dev)  # noqa: E731
        with torch.no_grad():
            # token embedding: the bodies' td features, body b's at columns [b td, (b + 1) td) of x; each body's constant
            # its embedding's bias plus its body embedding; the registers' their values
            KE = max(16, _next_pow2(td))
            we = torch.zeros(KE, D, device=dev)
            we[:td] = self.embed.weight.t()
            gidx = torch.full((T, KE), -1, dtype=torch.int32)
            for b in range(B):
                gidx[b, :td] = ar(td, b * td)
            ct = torch.zeros(T, D, device=dev)
            ct[:B] = self.embed.bias + self.body_emb
            if NR:
                ct[B:] = self.registers[0]
            # rotary angles from the bodies' positions, x[:, in_dim + 3 b + coordinate] * POS_SCALE; pair m: coordinate m // F, frequency m % F
            freqs, HD2 = self.rope.freqs.detach().float().cpu(), hd // 2
            F_, R = freqs.numel(), 3 * freqs.numel()
            if R > HD2:
                raise ValueError(f"{R} rotary pairs in a head of {hd} channels")
            pidx, frq = torch.full((T, HD2), -1, dtype=torch.int32), torch.zeros(HD2)
            for m in range(R):
                frq[m] = freqs[m % F_]
                pidx[:B, m] = self.in_dim + 3 * ar(B) + m // F_
            film_w = torch.cat([b.film.weight for b in self.blocks])
            film_b = torch.cat([b.film.bias for b in self.blocks])
            w12, b12 = self.head_w12.weight.detach(), self.head_w12.bias.detach()  # (2 hid, 2D): value rows, then gate rows
            hc = head_chunk
            rows = lambda j, gate: slice(gate * hid + j * hc, gate * hid + (j + 1) * hc)  # noqa: E731
            hw12 = torch.stack([torch.stack([w12[rows(j, g), half * D : (half + 1) * D].t() for g in (0, 1) for half in (0, 1)])
                                for j in range(hid // hc)])  # fmt: skip
            hb12 = torch.stack([torch.stack([b12[rows(j, g)] for g in (0, 1)]) for j in range(hid // hc)])
            if merge:  # value and gate as one product per half of the input: (HNCH, 2, D, 2 hc), biases (HNCH, 2 hc)
                hw12 = torch.stack((torch.cat((hw12[:, 0], hw12[:, 2]), -1), torch.cat((hw12[:, 1], hw12[:, 3]), -1)), 1)
                hb12 = hb12.flatten(1)
            # the root and joint heads as one product (D, NO); DEST: the column of y of a token's head output (_readout's layout:
            # the root's outputs, then each joint output o of the J joints at root_out + o J + j)
            ro, jo = self.root_head.out_features, self.joint_head.out_features
            NO = max(16, _next_pow2(ro + jo))
            wout, bout = torch.zeros(D, NO, device=dev), torch.zeros(NO, device=dev)
            wout[:, :ro], bout[:ro] = self.root_head.weight.t(), self.root_head.bias
            wout[:, ro : ro + jo], bout[ro : ro + jo] = self.joint_head.weight.t(), self.joint_head.bias
            dest = torch.full((T, NO), -1, dtype=torch.int32)
            dest[0, :ro] = ar(ro)
            for tok in range(1, B):
                dest[tok, ro : ro + jo] = ro + ar(jo) * NUM_J + int(self.joint_of_body[tok - 1])
            self.out_size = ro + NUM_J * jo
            written = dest[dest >= 0].sort().values
            if not torch.equal(written, torch.arange(self.out_size, dtype=torch.int32)):
                raise RuntimeError("the read-out heads do not write every output exactly once")
            bufs = dict(gidx=gidx, we=bf(we), ct=f32(ct), pidx=pidx, frq=frq, film_w=f32(film_w), film_b=f32(film_b), hnw=f32(self.head_norm.weight),
                        hw12=bf(hw12), hb12=f32(hb12), hw3=bf(self.head_w3.weight.t().reshape(hid // hc, hc, D)), hb3=f32(self.head_w3.bias),
                        wout=bf(wout), bout=f32(bout), dest=dest)  # fmt: skip
            bufs |= {f"blk_{n}": torch.stack([getattr(f, n) for f in fbs]) for n in BLOCK_BUFFERS}
            for name, buf in bufs.items():
                self.register_buffer("k_" + name, buf.to(dev), persistent=False)
        eps = {fb.eps, self.head_norm.eps}
        if len(eps) != 1:
            raise ValueError(f"the RMSNorms need one eps, got {eps}")
        self.knobs = dict(block_m=block_m, hidden_chunk=hidden_chunk, head_chunk=head_chunk, num_warps=num_warps, stages=stages, merge=merge,
                          fast_exp=fast_exp, prefetch=prefetch)
        self.debug = dict(OUTER=0, ROPE=1, SKIP_EMBED=False, SKIP_BLOCKS=False, SKIP_HEAD=False, SKIP_STORE=False, LAUNCH_STAGES=stages)
        self.k_eps, self.k_dims = fb.eps, dict(D=D, H=fb.heads, HD=hd, NCH=fb.hidden // hidden_chunk, BH=hidden_chunk, KE=KE,
                                                  HNCH=hid // hc, HBH=hc, NO=NO)  # fmt: skip
        self.k_scal = B * td  # the global context: x[:, k_scal : in_dim]

    def forward(self, x):
        _no_grad_check("FusedG1RigidFormerNet")
        N, B = x.shape[0], self.n_body
        if x.dtype != torch.float32 or x.device != self.k_we.device or x.shape[1] != self.in_dim + 3 * B:
            raise ValueError(f"x must be an fp32 (N, {self.in_dim + 3 * B}) tensor on {self.k_we.device}, got {x.dtype} {tuple(x.shape)} on "
                             f"{x.device}")  # fmt: skip
        x = x.contiguous()
        with torch.autocast("cuda", enabled=False):  # the FiLM vectors of every block and environment, fp32
            c = F.silu(F.linear(x[:, self.k_scal : self.in_dim], self.ctx_net[0].weight, self.ctx_net[0].bias))
            gb = F.linear(c, self.k_film_w, self.k_film_b)
        bm = self.knobs["block_m"]
        T, E = self.n_tok, bm // self.n_tok
        tile = attention_tile(self.rel_bias[:, self.rel].unsqueeze(0), self.k_dims["H"], T, bm, x.device)
        cos = sin = x
        R = 3 * self.rope.freqs.numel()
        if self.debug["ROPE"] == 1:  # the rotation from cos, sin tables computed here (as FusedBlock's; in the kernel: ROPE 2)
            pos = POS_SCALE * x[:, self.in_dim :].unflatten(-1, (B, 3))
            cos, sin = (t.reshape(N * T, R).contiguous() for t in self.rope(torch.cat((pos, pos.new_zeros(N, self.n_reg, 3)), 1)))
        tensors = [self.k_gidx, self.k_we, self.k_ct, self.k_pidx, self.k_frq] + [getattr(self, "k_blk_" + n) for n in BLOCK_BUFFERS]
        tensors += [self.k_hnw, self.k_hw12, self.k_hb12, self.k_hw3, self.k_hb3, self.k_wout, self.k_bout, self.k_dest]
        kw = dict(T=T, E=E, BM=bm, R=R, NB=len(self.blocks), STAGES=self.knobs["stages"], MERGE=self.knobs["merge"],
                  FAST_EXP=self.knobs["fast_exp"], PREFETCH=self.knobs["prefetch"], num_warps=self.knobs["num_warps"],
                  num_stages=self.debug["LAUNCH_STAGES"]) | self.k_dims | {k: v for k, v in self.debug.items() if k != "LAUNCH_STAGES"}  # fmt: skip
        return rf_hoi_net_op(x, cos, sin, tile, gb, tensors, self.out_size, self.k_eps, self.k_dims["HD"] ** -0.5 * LOG2E, POS_SCALE,
                             [int(kw[k]) for k in NET_CONFIG])  # fmt: skip


def self_check(ref: nn.Module, fused: nn.Module, T: int, rope_pairs: int, bias: bool, n: int = 7):
    """Raise if a fused block (``ref`` its Block) or fused network (``ref`` its HOIRigidFormerNet; T, rope_pairs and bias
    unused), as compiled for this GPU and setting, is not within bf16 rounding of the reference in fp32 on random inputs
    of n environments: its maximum error must not exceed twice the bf16-autocast reference's (tests/test_rf_kernel.py
    asserts once). Catches a miscompiled setting (as stages 1 on an H100) before it returns wrong numbers."""
    g = torch.Generator(device="cuda").manual_seed(0)
    if isinstance(fused, FusedBlock):
        x = torch.randn(n, T, fused.dim, device="cuda", generator=g)
        cos = sin = b = c = None
        if rope_pairs:
            ang = 10.0 * torch.rand(n, 1, T, rope_pairs, device="cuda", generator=g)
            cos, sin = ang.cos(), ang.sin()
        if bias:
            b = 0.5 * torch.randn(1, fused.heads, T, T, device="cuda", generator=g)
        if fused.has_film:
            c = torch.randn(n, fused.film_w.shape[1], device="cuda", generator=g)
        args = (x, cos, sin, b, c)
    else:  # normalized features ~ N(0, 1), positions within a metre (in units of POS_SCALE)
        nt = fused.n_tok - fused.n_reg
        pos = (2 * torch.rand(n, 3 * nt, device="cuda", generator=g) - 1) / POS_SCALE
        args = (torch.cat((torch.randn(n, fused.in_dim, device="cuda", generator=g), pos), 1),)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.autocast("cuda", enabled=False):  # fp32 even inside the caller's autocast
                ref32 = ref(*args).float()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref16 = ref(*args).float()
        y = fused(*args)
    err, tol = float((y - ref32).abs().max()), 2.0 * float((ref16 - ref32).abs().max())
    if not err <= tol:
        knobs = getattr(fused, "knobs", None) or {k: getattr(fused, k) for k in ("block_m", "num_warps", "stages", "hidden_chunk")}
        raise RuntimeError(f"{type(fused).__name__} ({knobs}) differs from {type(ref).__name__} in fp32 by {err:.3g} on "
                           f"{torch.cuda.get_device_name()}, more than twice the bf16-autocast reference's {tol / 2:.3g}: this setting is "
                           "miscompiled here")  # fmt: skip


def fused_copy(module: nn.Module, whole_network: bool = False, check: bool = True, **kw) -> nn.Module:
    """A copy of ``module`` (an ``HOIRigidFormerNet``, a ``RigidFormerNet``, or a world model holding one, also an
    inference copy of either) for inference with the fused kernels: every ``Block`` becomes a ``FusedBlock`` (a
    RigidFormerNet's ``fused_train``, the training kernels, which need Blocks, is turned off in the copy); with
    ``whole_network``, every HOIRigidFormerNet instead becomes a ``FusedHOIRigidFormerNet`` and every RigidFormerNet a
    ``FusedG1RigidFormerNet`` (one launch for the whole network; opt-in: it faults on an H100 with Triton 3.6, under
    investigation, and a fault kills the process before the self-check can refuse it). ``kw`` are the kernels' knobs
    (FusedBlock's; the whole networks' also ``head_chunk``). The original is left as it is; parameters changed later are
    not seen by the copy. With ``check``, every fused piece is first compared with its reference on the device
    (``self_check``)."""
    m = copy.deepcopy(module)
    head_kw = {k: kw.pop(k) for k in ("head_chunk",) if k in kw}
    n = 0
    whole = (FusedHOIRigidFormerNet, FusedG1RigidFormerNet)
    if whole_network:
        for parent in [None] + list(m.modules()):
            children = [("", m)] if parent is None else list(parent.named_children())
            for name, net in children:
                if type(net) is HOIRigidFormerNet:
                    fused = FusedHOIRigidFormerNet.from_net(net, **kw, **head_kw)
                elif isinstance(net, RigidFormerNet) and not isinstance(net, FusedG1RigidFormerNet):
                    fused = FusedG1RigidFormerNet.from_net(net, **kw, **head_kw)
                else:
                    continue
                if check:
                    self_check(net, fused, 0, 0, True)
                if parent is None:
                    return fused
                setattr(parent, name, fused)
                n += 1
    for sub in list(m.modules()):
        blocks = getattr(sub, "blocks", None)
        if isinstance(sub, whole) or not (isinstance(blocks, nn.ModuleList) and all(isinstance(b, Block) for b in blocks)):
            continue
        # tokens: RigidFormerNet's (n_tok, fewer than its bodies with merge), HOIRigidFormerNet's bodies, object and props
        T = getattr(sub, "n_tok", sub.n_body) + sub.n_reg + (1 + sub.n_props if hasattr(sub, "n_props") else 0)
        rope_pairs = 3 * sub.rope.freqs.numel() if sub.rope is not None else 0
        for i, blk in enumerate(blocks):
            # the SwiGLU's columns per step: 64 (the default) unless its width needs 32 (a 64-wide block's 160)
            fused = FusedBlock(blk, **({"hidden_chunk": 64 if blk.w3.in_features % 64 == 0 else 32} | kw))
            if check:
                self_check(blk, fused, T, rope_pairs, sub.rel_bias is not None)
            blocks[i] = fused
            n += 1
        if isinstance(sub, RigidFormerNet):  # its blocks are FusedBlocks now, which rf_block_train (fused_train) cannot take
            sub.fused_train = sub.keep_ops = False
    if n == 0:
        raise ValueError(f"{type(module).__name__} holds no RigidFormerNet or HOIRigidFormerNet")
    return m
