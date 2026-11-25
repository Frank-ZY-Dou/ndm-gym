"""Fused forward and backward passes of ``RigidFormerNet``'s read-out (``RigidFormerNet._readout``) with frozen
parameters, in Triton: the policy's BPTT through the world model, where the read-out's products, its SwiGLU and its norm
with their backward passes are a third of a lightweight RigidFormer's time.

One program holds the bodies of EB environments, a row per body slot (BP slots per environment, a power of two; BMB = EB BP
rows). A body reads the features of its token on the read-out's two layers (the embedding and the last block; with merged
tokens, ``--rf_merge``, several bodies share a token and add their read-out embedding): RMSNorm over the two halves, the
SwiGLU chunk by chunk, the output projection and the root and joint heads as one product whose outputs are scattered to
their places (the forward kernel is also the read-out of the merged-token networks' inference in two launches,
``kernels.rf_block.rf_merged_net_op``). The backward pass recomputes the forward pass's products chunk by chunk (nothing
is kept but the read-out's inputs), goes back through the heads, the SwiGLU and the norm, and sums the bodies of a token
with a one-hot product, so that no two programs write one row and the result is deterministic.

A warp holds 16 of the tile's rows and the whole of each weight tile it multiplies by in its registers, 255 a thread at
most (ptxas -v reports a kernel's count and its spills; ``tools/kernel_stats.py``). The SwiGLU's output projection and
the heads are two linear maps in a row: both passes multiply by their product, made once from the parameters
(``pack_head``), so the forward pass holds a (rows, head outputs) accumulator instead of a (rows, D) one and the
backward pass takes the outputs' gradient to a chunk's SwiGLU columns in one small product. The backward pass sums the
norm's scalar in its loop, so that after it each half's gradient is formed from its own accumulator alone. A kernel's
time is mostly the fetch of its weight tiles from L2 for every tile of rows (the same loop without fetches runs 2.3
times faster), so fewer and larger tiles per loop step are faster as long as a thread's registers hold them.

Numerics as the blocks' kernels: every product takes bf16 operands and accumulates in fp32, all else is fp32; the backward
pass also rounds the bodies' gradients to bf16 before the product that sums them per token. No parameter gradients: the
read-out of a network whose parameters train runs in PyTorch.

Merged tokens are read out from the tokens' rows instead (``_token_readout``): the read-out's first products per token,
16 rows an environment instead of the 32 body slots, and a one-hot product per group of 16 body slots that copies a
token's chunk to its bodies' rows. Inference runs it inside the whole-network kernel (``kernels.rf_block._net_kernel``),
the training passes as their own pair of kernels (``_head_tok_fwd_kernel``, ``_head_tok_bwd_kernel``; ``head_tables``'
token_rows), whose backward pass sums a token's bodies with the transposed one-hot product and multiplies by the
transposed weights on the token rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..robots.g1 import NUM_J
from .rf_train import _env_cols, _sigmoid

# one kernel's dimensions and launch options (maxnreg: a thread's registers, 0 the assembler's choice); a read-out's configuration: the
# forward pass's, then the backward pass's
HEAD_CFG = ("T", "EB", "BP", "BMB", "D", "HNCH", "HBH", "NO", "STAGES", "num_warps", "maxnreg")
TOK_CFG = ("T", "E", "BP", "D", "HNCH", "HBH", "NO", "STAGES", "num_warps", "maxnreg")  # the token-row kernels': forward, then backward


@triton.jit
def _inputs(H0, H1, TOB, UNM, HNW, n_env, eps, T: tl.constexpr, EB: tl.constexpr, BP: tl.constexpr, BMB: tl.constexpr, D: tl.constexpr):
    """The tile's bodies: their token's features on the two layers plus their read-out embedding (h0, x), the norm's
    scale r, the slots' body index, environment, token and mask."""
    i = tl.arange(0, BMB)
    b = i % BP
    env = (tl.program_id(0) * EB + i // BP).to(tl.int64)
    tok = tl.load(TOB + b)
    live = (env < n_env) & (tok >= 0)
    row = env * T + tl.where(tok >= 0, tok, 0)
    d = tl.arange(0, D)
    h0 = tl.load(H0 + row[:, None] * D + d[None, :], mask=live[:, None], other=0.0) + tl.load(UNM + b[:, None] * 2 * D + d[None, :])
    x = tl.load(H1 + row[:, None] * D + d[None, :], mask=live[:, None], other=0.0) + tl.load(UNM + b[:, None] * 2 * D + D + d[None, :])
    r = tl.rsqrt((tl.sum(h0 * h0, 1) + tl.sum(x * x, 1)) / (2 * D) + eps)[:, None]
    return h0, x, r, b, env, tok, live


@triton.jit
def _head_fwd_kernel(H0, H1, Y, yd, TOB, UNM, HNW, HW12, HB12, HM3, BOUT, DEST, n_env, eps, T: tl.constexpr, EB: tl.constexpr,
                     BP: tl.constexpr, BMB: tl.constexpr, D: tl.constexpr, HNCH: tl.constexpr, HBH: tl.constexpr, NO: tl.constexpr,
                     STAGES: tl.constexpr):  # fmt: skip
    """The read-out of the environments [EB pid, EB pid + EB): H0, H1 (N T, D) fp32 the tokens' features on the embedding
    and the last block; TOB (BP,) each slot's token (-1: no body); UNM (BP, 2 D) its read-out embedding (zeros where the
    token has one body); HNW (2 D,) the norm's weight; HW12 (HNCH, 4, D, HBH) bf16 the SwiGLU's value and gate columns on
    the two halves, HB12 (HNCH, 2, HBH); HM3 (HNCH, HBH, NO) bf16 the SwiGLU's output projection times the heads' weights
    and BOUT (NO,) their bias (``pack_head``); DEST (BP, NO) the column of y of each head output of a slot (-1: none)."""
    h0, x, r, b, env, tok, live = _inputs(H0, H1, TOB, UNM, HNW, n_env, eps, T, EB, BP, BMB, D)
    d = tl.arange(0, D)
    xa = (h0 * r * tl.load(HNW + d)[None, :]).to(tl.bfloat16)
    xb = (x * r * tl.load(HNW + D + d)[None, :]).to(tl.bfloat16)
    cb = tl.arange(0, HBH)
    o = tl.arange(0, NO)
    w12 = d[:, None] * HBH + cb[None, :]
    out = tl.zeros([BMB, NO], dtype=tl.float32) + tl.load(BOUT + o)[None, :]
    for j in tl.range(HNCH, num_stages=STAGES):
        w, bb = HW12 + j * 4 * D * HBH, HB12 + j * 2 * HBH
        u = tl.dot(xa, tl.load(w + w12, cache_modifier=".cg"))
        u = tl.dot(xb, tl.load(w + D * HBH + w12, cache_modifier=".cg"), u) + tl.load(bb + cb)[None, :]
        z = tl.dot(xa, tl.load(w + 2 * D * HBH + w12, cache_modifier=".cg"))
        z = tl.dot(xb, tl.load(w + 3 * D * HBH + w12, cache_modifier=".cg"), z) + tl.load(bb + HBH + cb)[None, :]
        m3 = tl.load(HM3 + j * HBH * NO + cb[:, None] * NO + o[None, :], cache_modifier=".cg")
        out = tl.dot((u * z * _sigmoid(z)).to(tl.bfloat16), m3, out)
    dest = tl.load(DEST + b[:, None] * NO + o[None, :])
    tl.store(Y + env[:, None] * yd + dest, out, mask=live[:, None] & (dest >= 0))


@triton.jit
def _slot_table(P, E: tl.constexpr, GB: tl.constexpr, K: tl.constexpr):
    """A table of an environment's body slots (GB 16, K) at P, for every environment of a tile: (E GB, 16, K)."""
    t = tl.load(P + (tl.arange(0, GB)[:, None, None] * 16 + tl.arange(0, 16)[None, :, None]) * K + tl.arange(0, K)[None, None, :])
    return tl.reshape(tl.broadcast_to(tl.reshape(t, [1, GB, 16, K]), [E, GB, 16, K]), [E * GB, 16, K])


@triton.jit
def _token_scale(h0, x, xa, xb, TOB, UNE, ENQ, eps, E: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BP: tl.constexpr):
    """The one-hot (E BP / 16, 16, T) bf16 of every group of 16 body slots (a slot's token) and the norm's scale r (E BP /
    16, 16) fp32 of every body slot, from the token rows h0, x (E T, D) fp32 and their bf16 copies xa, xb: mean(v^2) of
    every token with every body from |v_t|^2 + 2 <v_t, e_b> + |e_b|^2 (``_token_readout``), its reciprocal root copied
    to the bodies' rows in two bf16 terms."""
    GB: tl.constexpr = BP // 16
    NG: tl.constexpr = E * GB
    d = tl.arange(0, D)
    bq = tl.arange(0, BP)
    slot = (tl.arange(0, NG) % GB)[:, None] * 16 + tl.arange(0, 16)[None, :]  # the body slot of every row of every group (NG, 16)
    sel = tl.where(tl.load(TOB + slot)[:, :, None] == tl.arange(0, T)[None, None, :], 1.0, 0.0).to(tl.bfloat16)
    vv = tl.dot(xa, tl.load(UNE + d[:, None] * BP + bq[None, :]))
    vv = tl.dot(xb, tl.load(UNE + D * BP + d[:, None] * BP + bq[None, :]), vv)
    vv = (tl.sum(h0 * h0, 1) + tl.sum(x * x, 1))[:, None] + 2.0 * vv + tl.load(ENQ + bq)[None, :]
    rt = tl.rsqrt(tl.maximum(vv, 0.0) / (2 * D) + eps)  # (E T, BP): of token t with the embedding of body b
    rhi = rt.to(tl.bfloat16)
    rg = tl.dot(sel, _env_cols(rhi, E, GB, T, BP))
    rg = tl.dot(sel, _env_cols((rt - rhi.to(tl.float32)).to(tl.bfloat16), E, GB, T, BP), rg)
    return sel, tl.sum(tl.where(slot[:, :, None] == bq[None, None, :], rg, 0.0), 2)  # a slot's own column


@triton.jit
def _token_readout(h0, x, TOB, UNE, ENQ, CQ, HW12, HB12, HM3, BOUT, eps, E: tl.constexpr, T: tl.constexpr, D: tl.constexpr, HNCH: tl.constexpr,
                   HBH: tl.constexpr, NO: tl.constexpr, BP: tl.constexpr, STAGES: tl.constexpr):  # fmt: skip
    """The merged tokens' read-out from the token rows of a tile of E whole environments (in the whole-network kernel,
    ``kernels.rf_block._net_kernel``, and the training passes' ``_head_tok_fwd_kernel``): h0, x (E T, D) fp32 the tokens'
    features on the embedding and the last block -> the head outputs of the tile's body slots (E BP, NO) fp32, the slots
    of an environment side by side (BP a multiple of 16).

    A body b of token t reads v = (h0_t + e0_b, x_t + e1_b), normalized over both halves: y = v r_b g with r_b =
    (mean(v^2) + eps)^(-1/2). The read-out's first products are linear in y, W y = r_b (W (g (h0_t, x_t)) + W (g e_b)):
    the first term is computed per token (16 rows an environment instead of its 32 body slots, and the weights fetched
    for a tile of 128 token rows), the second is a constant of the body. HW12 (HNCH, 4, D, HBH) bf16 holds the SwiGLU's
    value and gate columns with the norm's weight g multiplied in, CQ (HNCH, 2, BP, HBH) fp32 the bodies' constants
    (``token_head_tensors``). A token's chunk reaches its bodies' rows by a one-hot product per group of 16 body slots
    (TOB (BP,) each slot's token, -1: none) against the tokens of the group's environment, the body's constant its
    initial value; the chunk is rounded to bf16 on the way (the reference rounds the same products' results to bf16).
    The norm's scale comes the same way (``_token_scale``): the inner products one product with the bodies' embeddings
    as columns (UNE (2, D, BP) bf16, ENQ (BP,) their squared norms). From there on as ``_head_fwd_kernel``, per body
    slot: the SwiGLU's nonlinearity and the product with HM3."""
    GB: tl.constexpr = BP // 16
    NG: tl.constexpr = E * GB
    d = tl.arange(0, D)
    cb = tl.arange(0, HBH)
    o = tl.arange(0, NO)
    xa = h0.to(tl.bfloat16)
    xb = x.to(tl.bfloat16)
    sel, r2 = _token_scale(h0, x, xa, xb, TOB, UNE, ENQ, eps, E, T, D, BP)
    r = r2[:, :, None]
    w12 = d[:, None] * HBH + cb[None, :]
    out = tl.zeros([NG * 16, NO], dtype=tl.float32) + tl.load(BOUT + o)[None, :]
    for j in tl.range(HNCH, num_stages=STAGES):
        w, b, cq = HW12 + j * 4 * D * HBH, HB12 + j * 2 * HBH, CQ + j * 2 * BP * HBH
        u = tl.dot(xa, tl.load(w + w12, cache_modifier=".cg"))
        u = tl.dot(xb, tl.load(w + D * HBH + w12, cache_modifier=".cg"), u)
        z = tl.dot(xa, tl.load(w + 2 * D * HBH + w12, cache_modifier=".cg"))
        z = tl.dot(xb, tl.load(w + 3 * D * HBH + w12, cache_modifier=".cg"), z)
        u = tl.dot(sel, _env_cols(u.to(tl.bfloat16), E, GB, T, HBH), _slot_table(cq, E, GB, HBH))  # (NG, 16, HBH): per body slot from here on
        z = tl.dot(sel, _env_cols(z.to(tl.bfloat16), E, GB, T, HBH), _slot_table(cq + BP * HBH, E, GB, HBH))
        u = r * u + tl.load(b + cb)[None, None, :]
        z = r * z + tl.load(b + HBH + cb)[None, None, :]
        m3 = tl.load(HM3 + j * HBH * NO + cb[:, None] * NO + o[None, :], cache_modifier=".cg")
        out = tl.dot(tl.reshape((u * z * _sigmoid(z)).to(tl.bfloat16), [NG * 16, HBH]), m3, out)
    return out


@triton.jit
def _head_tok_fwd_kernel(H0, H1, Y, yd, TOB, UNE, ENQ, CQ, HW12, HB12, HM3, BOUT, DEST, n_env, eps, T: tl.constexpr, E: tl.constexpr,
                         BP: tl.constexpr, D: tl.constexpr, HNCH: tl.constexpr, HBH: tl.constexpr, NO: tl.constexpr, STAGES: tl.constexpr):  # fmt: skip
    """The merged tokens' read-out of the environments [E pid, E pid + E) from the tokens' rows (``_token_readout``, which
    describes the tensors), the training passes' forward kernel: H0, H1 (N T, D) fp32 -> Y (N, out); DEST (BP, NO)."""
    i = tl.arange(0, E * T)
    env = (tl.program_id(0) * E + i // T).to(tl.int64)
    live = (env < n_env)[:, None]
    row = env * T + i % T
    d = tl.arange(0, D)
    h0 = tl.load(H0 + row[:, None] * D + d[None, :], mask=live, other=0.0)
    x = tl.load(H1 + row[:, None] * D + d[None, :], mask=live, other=0.0)
    out = _token_readout(h0, x, TOB, UNE, ENQ, CQ, HW12, HB12, HM3, BOUT, eps, E, T, D, HNCH, HBH, NO, BP, STAGES)
    ib = tl.arange(0, E * BP)
    envb = (tl.program_id(0) * E + ib // BP).to(tl.int64)
    dest = tl.load(DEST + (ib % BP)[:, None] * NO + tl.arange(0, NO)[None, :])
    tl.store(Y + envb[:, None] * yd + dest, out, mask=(envb < n_env)[:, None] & (dest >= 0))


@triton.jit
def _head_tok_bwd_kernel(H0, H1, H0R, H1R, DY, DH0, DH1, yd, TOB, UNE, UNT, ENQ, CQ, HW12, HB12, HW12T, HM3T, DEST, n_env, eps,
                         T: tl.constexpr, E: tl.constexpr, BP: tl.constexpr, D: tl.constexpr, HNCH: tl.constexpr, HBH: tl.constexpr,
                         NO: tl.constexpr, STAGES: tl.constexpr):  # fmt: skip
    """The backward pass of ``_head_tok_fwd_kernel`` on a tile: DY (N, out) the outputs' gradient -> DH0, DH1 (N T, D) the
    gradients of the tokens' features on the two layers (every token row of the tile's environments is written). The
    forward pass's products are recomputed chunk by chunk; per body slot the gradients du, dz of a chunk's value and gate
    columns (as ``_deltas``: HM3T (HNCH, NO, HBH) bf16 takes the outputs' gradient to the chunk's columns); a one-hot
    product per group of 16 token rows against the body slots of the group's environment sums r_b du, r_b dz (bf16)
    over a token's bodies; the token rows' gradient accumulates their products with HW12T (HNCH, 4, HBH, D) bf16, the
    forward pass's tiles transposed. The norm's scale: with q_b = sum over the SwiGLU's columns of <u_b - b_u, r_b du_b>
    + <z_b - b_z, r_b dz_b> (summed in the loop; r_b^2 times the sum over the columns before the scale), dS_b = -q_b r_b
    / (4 D) is the gradient of S_b = |v_t|^2 + 2 <v_t, e_b> + |e_b|^2, so a token gets 2 v_t sum_b dS_b + 2 sum_b dS_b e_b
    over its bodies: the second term one more one-hot product, with the embeddings' rows UNT (BP, 2 D) bf16, on the
    token rows' accumulators. H0R and H1R are H0 and H1 again: what is read after the loop is read through them, so
    that it is not held in registers through it."""
    BM: tl.constexpr = E * T
    GB: tl.constexpr = BP // 16
    NG: tl.constexpr = E * GB
    GT: tl.constexpr = T // 16
    NT: tl.constexpr = E * GT
    i = tl.arange(0, BM)
    env = (tl.program_id(0) * E + i // T).to(tl.int64)
    live = (env < n_env)[:, None]
    d = tl.arange(0, D)
    row = (env * T + i % T)[:, None] * D + d[None, :]
    bq = tl.arange(0, BP)
    cb = tl.arange(0, HBH)
    o = tl.arange(0, NO)
    h0 = tl.load(H0 + row, mask=live, other=0.0)
    x = tl.load(H1 + row, mask=live, other=0.0)
    xa = h0.to(tl.bfloat16)
    xb = x.to(tl.bfloat16)
    sel, r2 = _token_scale(h0, x, xa, xb, TOB, UNE, ENQ, eps, E, T, D, BP)
    r = r2[:, :, None]
    # a one-hot (NT, 16, BP) of every group of 16 token rows: the body slots of each of its tokens
    tokg = (tl.arange(0, NT) % GT)[:, None] * 16 + tl.arange(0, 16)[None, :]
    selt = tl.where(tokg[:, :, None] == tl.load(TOB + bq)[None, None, :], 1.0, 0.0)
    seltb = selt.to(tl.bfloat16)
    ib = tl.arange(0, NG * 16)
    envb = (tl.program_id(0) * E + ib // BP).to(tl.int64)
    dest = tl.load(DEST + (ib % BP)[:, None] * NO + o[None, :])
    dout = tl.load(DY + envb[:, None] * yd + dest, mask=(envb < n_env)[:, None] & (dest >= 0), other=0.0).to(tl.bfloat16)
    dout = tl.reshape(dout, [NG, 16, NO])
    w12 = d[:, None] * HBH + cb[None, :]
    w12t = cb[:, None] * D + d[None, :]
    q = tl.zeros([NG, 16], dtype=tl.float32)
    acc0 = tl.zeros([BM, D], dtype=tl.float32)
    acc1 = tl.zeros([BM, D], dtype=tl.float32)
    for j in tl.range(HNCH, num_stages=STAGES):
        w, b, cq = HW12 + j * 4 * D * HBH, HB12 + j * 2 * HBH, CQ + j * 2 * BP * HBH
        u = tl.dot(xa, tl.load(w + w12, cache_modifier=".cg"))
        u = tl.dot(xb, tl.load(w + D * HBH + w12, cache_modifier=".cg"), u)
        z = tl.dot(xa, tl.load(w + 2 * D * HBH + w12, cache_modifier=".cg"))
        z = tl.dot(xb, tl.load(w + 3 * D * HBH + w12, cache_modifier=".cg"), z)
        u = r * tl.dot(sel, _env_cols(u.to(tl.bfloat16), E, GB, T, HBH), _slot_table(cq, E, GB, HBH))  # (NG, 16, HBH): per body slot
        z = r * tl.dot(sel, _env_cols(z.to(tl.bfloat16), E, GB, T, HBH), _slot_table(cq + BP * HBH, E, GB, HBH))
        zb = z + tl.load(b + HBH + cb)[None, None, :]
        sg = _sigmoid(zb)
        m3t = tl.load(HM3T + j * NO * HBH + o[None, :, None] * HBH + cb[None, None, :], cache_modifier=".cg")
        ds = tl.dot(dout, tl.broadcast_to(m3t, [NG, NO, HBH])) * (sg * r)
        du = ds * zb  # r_b times the gradients of the value and gate columns
        dz = ds * (u + tl.load(b + cb)[None, None, :]) * (1.0 + zb * (1.0 - sg))
        q += tl.sum(u * du + z * dz, 2)  # u, z here without their biases: r (U, Z)
        du = tl.dot(seltb, _env_cols(tl.reshape(du.to(tl.bfloat16), [E * BP, HBH]), E, GT, BP, HBH))  # (NT, 16, HBH): per token
        dz = tl.dot(seltb, _env_cols(tl.reshape(dz.to(tl.bfloat16), [E * BP, HBH]), E, GT, BP, HBH))
        du = tl.reshape(du, [BM, HBH]).to(tl.bfloat16)
        dz = tl.reshape(dz, [BM, HBH]).to(tl.bfloat16)
        wt = HW12T + j * 4 * HBH * D
        acc0 = tl.dot(du, tl.load(wt + w12t, cache_modifier=".cg"), acc0)
        acc1 = tl.dot(du, tl.load(wt + HBH * D + w12t, cache_modifier=".cg"), acc1)
        acc0 = tl.dot(dz, tl.load(wt + 2 * HBH * D + w12t, cache_modifier=".cg"), acc0)
        acc1 = tl.dot(dz, tl.load(wt + 3 * HBH * D + w12t, cache_modifier=".cg"), acc1)
    ds = q * r2 * (-0.5 / D)  # (NG, 16): twice dS of every body slot (q holds r^2 times the sum)
    ds = tl.reshape(tl.broadcast_to(tl.reshape(ds, [E, 1, 1, BP]), [E, GT, 1, BP]), [NT, 1, BP])
    a = selt * ds  # (NT, 16, BP): twice dS of the bodies of every token
    ss = tl.reshape(tl.sum(a, 2), [BM])[:, None]
    a = a.to(tl.bfloat16)
    unt = UNT + bq[None, :, None] * 2 * D + d[None, None, :]
    acc0 = tl.reshape(tl.dot(a, tl.broadcast_to(tl.load(unt), [NT, BP, D]), tl.reshape(acc0, [NT, 16, D])), [BM, D])
    acc1 = tl.reshape(tl.dot(a, tl.broadcast_to(tl.load(unt + D), [NT, BP, D]), tl.reshape(acc1, [NT, 16, D])), [BM, D])
    tl.store(DH0 + row, acc0 + tl.load(H0R + row, mask=live, other=0.0) * ss, mask=live)
    tl.store(DH1 + row, acc1 + tl.load(H1R + row, mask=live, other=0.0) * ss, mask=live)


@triton.jit
def _deltas(xa, xb, dout, HW12, HB12, HM3T, j, D: tl.constexpr, HBH: tl.constexpr, NO: tl.constexpr):
    """Chunk j of the SwiGLU in the backward pass: the value and gate columns recomputed, their gradients du, dz (bf16)
    from the head outputs' gradient, and <u - b_u, du> + <z - b_z, dz> per row (the norm's scalar, see _head_bwd_kernel)."""
    d = tl.arange(0, D)
    cb = tl.arange(0, HBH)
    o = tl.arange(0, NO)
    w12 = d[:, None] * HBH + cb[None, :]
    w, bb = HW12 + j * 4 * D * HBH, HB12 + j * 2 * HBH
    bu = tl.load(bb + cb)[None, :]
    bz = tl.load(bb + HBH + cb)[None, :]
    wua = tl.load(w + w12, cache_modifier=".cg")
    wub = tl.load(w + D * HBH + w12, cache_modifier=".cg")
    wza = tl.load(w + 2 * D * HBH + w12, cache_modifier=".cg")
    wzb = tl.load(w + 3 * D * HBH + w12, cache_modifier=".cg")
    u = tl.dot(xa, wua)
    u = tl.dot(xb, wub, u) + bu
    z = tl.dot(xa, wza)
    z = tl.dot(xb, wzb, z) + bz
    sg = _sigmoid(z)
    ds = tl.dot(dout, tl.load(HM3T + j * NO * HBH + o[:, None] * HBH + cb[None, :], cache_modifier=".cg"))
    du = (ds * z * sg).to(tl.bfloat16)
    dz = (ds * u * sg * (1.0 + z * (1.0 - sg))).to(tl.bfloat16)
    return du, dz, tl.sum((u - bu) * du.to(tl.float32) + (z - bz) * dz.to(tl.float32), 1), wua, wub, wza, wzb


@triton.jit
def _token_grad(acc, k, r, g, HR, UNMR, DH, tk, b, env, live, n_env, T: tl.constexpr, EB: tl.constexpr, BP: tl.constexpr,
                BMB: tl.constexpr, D: tl.constexpr):  # fmt: skip
    """One half's gradient: the norm's backward pass dv = r g acc - v k on the bodies' rows (v read here: the token's
    features from HR plus the read-out embedding from UNMR), the bodies of each token summed by a one-hot (token rows,
    body slots) product, stored in the tile's token rows of DH."""
    d = tl.arange(0, D)
    v = tl.load(HR + (env * T + tl.where(tk >= 0, tk, 0))[:, None] * D + d[None, :], mask=live[:, None], other=0.0)
    v += tl.load(UNMR + b[:, None] * 2 * D + d[None, :])
    dv = (r * g[None, :] * acc - v * k).to(tl.bfloat16)
    TT: tl.constexpr = EB * T
    tr = tl.arange(0, TT)
    i = tl.arange(0, BMB)
    sel = ((i // BP)[None, :] == (tr // T)[:, None]) & (tk[None, :] == (tr % T)[:, None]) & live[None, :]
    tenv = (tl.program_id(0) * EB + tr // T).to(tl.int64)
    tl.store(DH + (tenv * T + tr % T)[:, None] * D + d[None, :], tl.dot(tl.where(sel, 1.0, 0.0).to(tl.bfloat16), dv), mask=(tenv < n_env)[:, None])


@triton.jit
def _head_bwd_kernel(H0, H1, H0R, H1R, UNMR, TOBR, DY, DH0, DH1, yd, TOB, UNM, HNW, HW12, HB12, HM3T, DEST, n_env, eps, T: tl.constexpr,
                     EB: tl.constexpr, BP: tl.constexpr, BMB: tl.constexpr, D: tl.constexpr, HNCH: tl.constexpr, HBH: tl.constexpr,
                     NO: tl.constexpr, STAGES: tl.constexpr):  # fmt: skip
    """The read-out's backward pass on a tile: DY (N, out) the outputs' gradient -> DH0, DH1 (N T, D) the gradients of the
    tokens' features on the two layers (every token row of the tile's environments is written; a token without a body, as
    a register, gets zeros). The tensors are the backward pass's own, in its chunks of HBH columns: HW12, HB12 as the
    forward pass's (a chunk's tiles are fetched once: the gradient's products take their transposes); HM3T (HNCH, NO,
    HBH) bf16 the forward pass's HM3 transposed (a head output's gradient -> the chunk's columns).

    The norm y = v r g, r = (mean(v^2) + eps)^(-1/2) over both halves, has dv = r g dy - v r^3 sum(v g dy) / (2 D), and
    sum(v g dy) = <y, dy> / r = (<u - b_u, du> + <z - b_z, dz>) / r over the SwiGLU's columns, summed in the loop: after
    it, each half's gradient needs only its own accumulator and its own input. H0R, H1R, UNMR and TOBR are H0, H1, UNM
    and TOB again: what is read after the loop is read through them, so that it is not held in registers through it."""
    h0, x, r, b, env, tok, live = _inputs(H0, H1, TOB, UNM, HNW, n_env, eps, T, EB, BP, BMB, D)
    d = tl.arange(0, D)
    xa = (h0 * r * tl.load(HNW + d)[None, :]).to(tl.bfloat16)
    xb = (x * r * tl.load(HNW + D + d)[None, :]).to(tl.bfloat16)
    o = tl.arange(0, NO)
    dest = tl.load(DEST + b[:, None] * NO + o[None, :])
    dout = tl.load(DY + env[:, None] * yd + dest, mask=live[:, None] & (dest >= 0), other=0.0).to(tl.bfloat16)
    ydy = tl.zeros([BMB], dtype=tl.float32)
    acc0 = tl.zeros([BMB, D], dtype=tl.float32)
    acc1 = tl.zeros([BMB, D], dtype=tl.float32)
    for j in tl.range(HNCH, num_stages=STAGES):
        du, dz, s, wua, wub, wza, wzb = _deltas(xa, xb, dout, HW12, HB12, HM3T, j, D, HBH, NO)
        ydy += s
        acc0 = tl.dot(du, tl.trans(wua), acc0)
        acc1 = tl.dot(du, tl.trans(wub), acc1)
        acc0 = tl.dot(dz, tl.trans(wza), acc0)
        acc1 = tl.dot(dz, tl.trans(wzb), acc1)
    k = r * r * (ydy / (2 * D))[:, None]
    _token_grad(acc0, k, r, tl.load(HNW + d), H0R, UNMR, DH0, tok, b, env, live, n_env, T, EB, BP, BMB, D)
    _token_grad(acc1, k, r, tl.load(HNW + D + d), H1R, UNMR + D, DH1, tl.load(TOBR + b), b, env, live, n_env, T, EB, BP, BMB, D)


def _cfg(config, backward: bool = False):
    n = len(HEAD_CFG)
    return dict(zip(HEAD_CFG, config[n:] if backward else config[:n]))


@torch.library.custom_op("ndm_gym::rf_head_fwd", mutates_args=())
def rf_head_fwd(h0: torch.Tensor, h1: torch.Tensor, tensors: list[torch.Tensor], out_size: int, eps: float, config: list[int]) -> torch.Tensor:
    """The read-out (``_head_fwd_kernel``) on the tokens' features h0, h1 (N, T, D) fp32 -> y (N, out_size) fp32; tensors:
    ``pack_head``'s (the shared four, the forward pass's four, the backward pass's three); config: the forward's HEAD_CFG, then the
    backward's."""
    k = _cfg(config)
    N = h0.shape[0]
    y = h0.new_empty(N, out_size)
    tob, unm, hnw, dest, hw12, hb12, hm3, bout = tensors[:8]
    _head_fwd_kernel[(triton.cdiv(N, k["EB"]),)](h0, h1, y, y.stride(0), tob, unm, hnw, hw12, hb12, hm3, bout, dest, N, eps,
                                                 T=k["T"], EB=k["EB"], BP=k["BP"], BMB=k["BMB"], D=k["D"], HNCH=k["HNCH"], HBH=k["HBH"],
                                                 NO=k["NO"], STAGES=k["STAGES"], num_warps=k["num_warps"], maxnreg=k["maxnreg"] or None)  # fmt: skip
    return y


@rf_head_fwd.register_fake
def _(h0, h1, tensors, out_size, eps, config):
    return h0.new_empty(h0.shape[0], out_size)


@torch.library.custom_op("ndm_gym::rf_head_bwd", mutates_args=())
def rf_head_bwd(dy: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, tensors: list[torch.Tensor], eps: float,
                config: list[int]) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """The read-out's backward pass (``_head_bwd_kernel``): dy (N, out_size) fp32 -> the gradients of h0 and h1."""
    k = _cfg(config, backward=True)
    N = h0.shape[0]
    dh0, dh1 = torch.empty_like(h0), torch.empty_like(h1)
    tob, unm, hnw, dest, bw12, bb12, bm3t = *tensors[:4], *tensors[8:]
    _head_bwd_kernel[(triton.cdiv(N, k["EB"]),)](h0, h1, h0, h1, unm, tob, dy, dh0, dh1, dy.stride(0), tob, unm, hnw, bw12, bb12, bm3t, dest,
                                                 N, eps, T=k["T"], EB=k["EB"], BP=k["BP"], BMB=k["BMB"], D=k["D"], HNCH=k["HNCH"], HBH=k["HBH"],
                                                 NO=k["NO"], STAGES=k["STAGES"], num_warps=k["num_warps"], maxnreg=k["maxnreg"] or None)  # fmt: skip
    return dh0, dh1


@rf_head_bwd.register_fake
def _(dy, h0, h1, tensors, eps, config):
    return torch.empty_like(h0), torch.empty_like(h1)


def _setup_context(ctx, inputs, output):
    h0, h1, tensors, out_size, eps, config = inputs
    ctx.save_for_backward(h0, h1, *tensors)
    ctx.eps, ctx.config = eps, config
    ctx.set_materialize_grads(False)


def _backward(ctx, dy):
    h0, h1, *tensors = ctx.saved_tensors
    if dy is None:
        return None, None, [None] * len(tensors), None, None, None
    dh0, dh1 = rf_head_bwd(dy.float().contiguous(), h0, h1, tensors, ctx.eps, ctx.config)
    ng = ctx.needs_input_grad
    return dh0 if ng[0] else None, dh1 if ng[1] else None, [None] * len(tensors), None, None, None


rf_head_fwd.register_autograd(_backward, setup_context=_setup_context)


def _tok_cfg(config, backward: bool = False):
    n = len(TOK_CFG)
    k = dict(zip(TOK_CFG, config[n:] if backward else config[:n]))
    return k, k.pop("maxnreg") or None


@torch.library.custom_op("ndm_gym::rf_head_tok_fwd", mutates_args=())
def rf_head_tok_fwd(h0: torch.Tensor, h1: torch.Tensor, tensors: list[torch.Tensor], out_size: int, eps: float,
                    config: list[int]) -> torch.Tensor:  # fmt: skip
    """The merged tokens' read-out from the tokens' rows (``_head_tok_fwd_kernel``) on the tokens' features h0, h1 (N, T,
    D) fp32 -> y (N, out_size) fp32; tensors: ``pack_token_head``'s; config: the forward's TOK_CFG, then the backward's."""
    k, maxnreg = _tok_cfg(config)
    N = h0.shape[0]
    y = h0.new_empty(N, out_size)
    tob, dest, une, enq, cq, hw12, hb12, hm3, bout = tensors[:9]
    _head_tok_fwd_kernel[(triton.cdiv(N, k["E"]),)](h0, h1, y, y.stride(0), tob, une, enq, cq, hw12, hb12, hm3, bout, dest, N, eps, **k,
                                                    maxnreg=maxnreg)  # fmt: skip
    return y


@rf_head_tok_fwd.register_fake
def _(h0, h1, tensors, out_size, eps, config):
    return h0.new_empty(h0.shape[0], out_size)


@torch.library.custom_op("ndm_gym::rf_head_tok_bwd", mutates_args=())
def rf_head_tok_bwd(dy: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, tensors: list[torch.Tensor], eps: float,
                    config: list[int]) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """The backward pass of ``rf_head_tok_fwd`` (``_head_tok_bwd_kernel``): dy (N, out_size) fp32 -> the gradients of h0
    and h1."""
    k, maxnreg = _tok_cfg(config, backward=True)
    N = h0.shape[0]
    dh0, dh1 = torch.empty_like(h0), torch.empty_like(h1)
    tob, dest, une, enq, cq, hw12, hb12, _, _, unt, hw12t, hm3t = tensors
    _head_tok_bwd_kernel[(triton.cdiv(N, k["E"]),)](h0, h1, h0, h1, dy, dh0, dh1, dy.stride(0), tob, une, unt, enq, cq, hw12, hb12, hw12t, hm3t,
                                                    dest, N, eps, **k, maxnreg=maxnreg)  # fmt: skip
    return dh0, dh1


@rf_head_tok_bwd.register_fake
def _(dy, h0, h1, tensors, eps, config):
    return torch.empty_like(h0), torch.empty_like(h1)


def _tok_backward(ctx, dy):
    h0, h1, *tensors = ctx.saved_tensors
    if dy is None:
        return None, None, [None] * len(tensors), None, None, None
    dh0, dh1 = rf_head_tok_bwd(dy.float().contiguous(), h0, h1, tensors, ctx.eps, ctx.config)
    ng = ctx.needs_input_grad
    return dh0 if ng[0] else None, dh1 if ng[1] else None, [None] * len(tensors), None, None, None


rf_head_tok_fwd.register_autograd(_tok_backward, setup_context=_setup_context)


def head_tables(net, block_m: int = 64, chunk: int = 32, bwd_block_m: int = 128, bwd_chunk: int = 32, warps: int | None = None,
                bwd_warps: int | None = None, maxnreg: int = 0, bwd_maxnreg: int = 0, token_rows: bool | None = None,
                tok_block_m: int = 128) -> dict:
    """What does not depend on the parameters: each body slot's token and the column of y of its head outputs
    (``RigidFormerNet._readout``'s layout: the root's outputs, then each joint output o of the J joints at root_out + o J
    + j), and the two kernels' dimensions and launch options (rows of a tile block_m and bwd_block_m, 16 a warp unless
    warps or bwd_warps say otherwise; the SwiGLU's columns of a chunk, chunk and bwd_chunk; maxnreg and bwd_maxnreg a
    thread's registers, 0 the assembler's choice); refused shapes raise. The defaults are the fastest measured on an
    A100: the forward pass in tiles of 64 rows, two programs on a multiprocessor; the
    backward pass, which holds two accumulators, in tiles of 128 rows. ``token_rows``: merged tokens read out from the
    tokens' rows (``_head_tok_fwd_kernel`` and ``_head_tok_bwd_kernel``, tiles of tok_block_m token rows, 16 a warp, one
    chunk size, launched with all of a thread's registers: for pointers that are not known to be aligned the assembler
    picked 32 for the backward pass); None: where the shapes allow it (merged tokens, tokens a multiple of 16 and not
    more than the body slots, tiles of whole environments); True or False ask for it or for the kernels per body."""
    B, D, T = net.n_body, net.embed.weight.shape[0], net.n_tok + net.n_reg
    hid, ro, jo = net.head_w3.in_features, net.root_head.out_features, net.joint_head.out_features
    p2 = lambda n: n > 0 and n & (n - 1) == 0  # noqa: E731
    BP = 1 << max(0, B - 1).bit_length()
    tiles = all(m % BP == 0 and p2(m // BP) and 32 <= m <= 128 for m in (block_m, bwd_block_m))  # 2, 4 or 8 warps of 16 rows
    chunks = all(p2(c) and c >= 16 and hid % c == 0 for c in (chunk, bwd_chunk))
    if not (p2(T) and T >= 16 and p2(D) and chunks and tiles and tuple(net.head_layers) == (0, len(net.blocks))):
        raise ValueError(f"the fused read-out supports a power of two of tokens (>= 16), a width that is a power of two, a SwiGLU width that is a "
                         f"multiple of the chunks ({chunk}, {bwd_chunk}: powers of two from 16), tiles of 32 to 128 rows of whole environments "
                         f"and the read-out on the embedding and the last block; got {T} tokens, width {D}, SwiGLU width {hid}, tiles of "
                         f"{block_m} and {bwd_block_m} rows with {BP} body slots, read-out layers {tuple(net.head_layers)}")  # fmt: skip
    NO = max(16, 1 << max(0, ro + jo - 1).bit_length())
    dev = net.embed.weight.device
    ar = lambda n, s=0: torch.arange(s, s + n, dtype=torch.int32)  # noqa: E731
    dest = torch.full((BP, NO), -1, dtype=torch.int32)
    dest[0, :ro] = ar(ro)
    for b in range(1, B):
        dest[b, ro : ro + jo] = ro + ar(jo) * NUM_J + int(net.joint_of_body[b - 1])
    tob = torch.full((BP,), -1, dtype=torch.int32)
    tob[:B] = net.tok_of_body.cpu().to(torch.int32) if net.merge != "none" else ar(B)
    cfg = lambda m, c, w, reg: dict(T=T, EB=m // BP, BP=BP, BMB=m, D=D, HNCH=hid // c, HBH=c, NO=NO, STAGES=2,  # noqa: E731
                                    num_warps=m // 16 if w is None else w, maxnreg=reg)  # fmt: skip
    fwd, bwd = cfg(block_m, chunk, warps, maxnreg), cfg(bwd_block_m, bwd_chunk, bwd_warps, bwd_maxnreg)
    allowed = (net.merge != "none" and T % 16 == 0 and T <= BP and tok_block_m % T == 0 and p2(tok_block_m // T) and 32 <= tok_block_m <= 128
               and chunk == bwd_chunk)  # fmt: skip
    if token_rows and not allowed:
        raise ValueError(f"the read-out from the tokens' rows takes merged tokens, a multiple of 16 of tokens that is not more than the body "
                         f"slots, tiles of 32 to 128 rows of whole environments and one chunk size; got merge {net.merge}, {T} tokens, {BP} "
                         f"body slots, tiles of {tok_block_m} rows, chunks {chunk} and {bwd_chunk}")  # fmt: skip
    tok = dict(T=T, E=tok_block_m // T, BP=BP, D=D, HNCH=hid // chunk, HBH=chunk, NO=NO, STAGES=2, num_warps=tok_block_m // 16, maxnreg=255)
    return dict(tob=tob.to(dev), dest=dest.to(dev), config=[fwd[k] for k in HEAD_CFG] + [bwd[k] for k in HEAD_CFG], out_size=ro + NUM_J * jo,
                token_rows=allowed if token_rows is None else bool(token_rows), tok_config=2 * [tok[k] for k in TOK_CFG])  # fmt: skip


def pack_head(net, tables: dict) -> list[torch.Tensor]:
    """The read-out's parameters as the kernels read them: TOB, UNM, HNW and DEST, then the forward pass's HW12, HB12, HM3
    and BOUT, then the backward pass's HW12, HB12 and HM3T in its own chunks; made from the network's current
    parameters. The SwiGLU's output projection and the heads are two linear maps in a row: the kernels multiply by their
    product (the reference rounds the projection's output to bf16 in between), hidden column -> head output."""
    k, kb = _cfg(tables["config"]), _cfg(tables["config"], backward=True)
    D, NO, BP, B = k["D"], k["NO"], k["BP"], net.n_body
    ro, jo = net.root_head.out_features, net.joint_head.out_features
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731

    def w12(c):  # the SwiGLU's weights in chunks of c columns: value or gate, chunk, column, half of the input, input channel
        n = net.head_w12.out_features // (2 * c)
        return net.head_w12.weight.float().view(2, n, c, 2, D), net.head_w12.bias.float().view(2, n, c).permute(1, 0, 2).contiguous(), n

    w3 = net.head_w3.weight.float()  # (D, hid)
    wout = w3.new_zeros(D, NO)
    wout[:, :ro] = net.root_head.weight.float().t()
    wout[:, ro : ro + jo] = net.joint_head.weight.float().t()
    bout = w3.new_zeros(NO)
    bout[:ro] = net.root_head.bias.float()
    bout[ro : ro + jo] = net.joint_head.bias.float()
    m3 = w3.t() @ wout  # (hid, NO)
    unm = w3.new_zeros(BP, 2 * D)
    if net.merge != "none":
        unm[:B] = (net.unmerge_emb * net.shared).float()
    w, hb12, n = w12(k["HBH"])
    hw12 = bf(w.permute(1, 0, 3, 4, 2).reshape(n, 4, D, k["HBH"]))
    fwd = [hw12, hb12, bf(m3.view(n, k["HBH"], NO)), bout + net.head_w3.bias.float() @ wout]
    w, bb12, n = w12(kb["HBH"])
    bw12 = hw12 if kb["HBH"] == k["HBH"] else bf(w.permute(1, 0, 3, 4, 2).reshape(n, 4, D, kb["HBH"]))
    bwd = [bw12, bb12, bf(m3.t().reshape(NO, n, kb["HBH"]).permute(1, 0, 2))]
    return [tables["tob"], unm, net.head_norm.weight.float().contiguous(), tables["dest"], *fwd, *bwd]


def token_head_tensors(hw12: torch.Tensor, hnw: torch.Tensor, unm: torch.Tensor) -> list[torch.Tensor]:
    """``_token_readout``'s tensors from the per-body read-out's: hw12 (HNCH, 4, D, HBH) fp32 the SwiGLU's value and gate
    columns on the two halves of its input (value on the embedding, value on the last block, gate, gate), hnw (2 D,) the
    norm's weight, unm (BP, 2 D) the body slots' read-out embeddings -> HW12 with the norm's weight multiplied in (bf16),
    UNE (2, D, BP) bf16 the embeddings as columns, ENQ (BP,) their squared norms, CQ (HNCH, 2, BP, HBH) fp32 the value and
    gate columns of each body's embedding (the weights in fp32)."""
    n, _, D, hbh = hw12.shape
    wg = hw12.float().view(n, 2, 2, D, hbh) * hnw.float().view(1, 1, 2, D, 1)  # chunk, value or gate, half, channel, column
    e = unm.float().view(-1, 2, D)
    return [wg.reshape(n, 4, D, hbh).to(torch.bfloat16).contiguous(), e.permute(1, 2, 0).to(torch.bfloat16).contiguous(),
            e.square().sum((1, 2)).contiguous(), torch.einsum("jvhdc,bhd->jvbc", wg, e).contiguous()]  # fmt: skip


def pack_token_head(net, tables: dict) -> list[torch.Tensor]:
    """The read-out's parameters as the token-row kernels read them (``head_tables``' token_rows): TOB and DEST, then
    ``token_head_tensors``' UNE, ENQ and CQ, HW12 with the norm's weight multiplied in, HB12, HM3 and BOUT as
    ``pack_head``'s, then the backward pass's UNT (BP, 2 D) bf16 the body slots' read-out embeddings as rows, HW12T (HNCH,
    4, HBH, D) bf16 HW12's tiles transposed and HM3T (HNCH, NO, HBH) bf16; made from the network's current parameters,
    in fp32 whatever the caller's autocast."""
    k, _ = _tok_cfg(tables["tok_config"])
    D, NO, BP, B, c = k["D"], k["NO"], k["BP"], net.n_body, k["HBH"]
    ro, jo = net.root_head.out_features, net.joint_head.out_features
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
    with torch.autocast("cuda", enabled=False):
        n = net.head_w12.out_features // (2 * c)
        hw12 = net.head_w12.weight.float().view(2, n, c, 2, D).permute(1, 0, 3, 4, 2).reshape(n, 4, D, c)  # chunk, (value or gate, half), channel
        hb12 = net.head_w12.bias.float().view(2, n, c).permute(1, 0, 2).contiguous()
        w3 = net.head_w3.weight.float()  # (D, hid)
        wout = w3.new_zeros(D, NO)
        wout[:, :ro] = net.root_head.weight.float().t()
        wout[:, ro : ro + jo] = net.joint_head.weight.float().t()
        bout = w3.new_zeros(NO)
        bout[:ro] = net.root_head.bias.float()
        bout[ro : ro + jo] = net.joint_head.bias.float()
        m3 = w3.t() @ wout  # (hid, NO)
        unm = w3.new_zeros(BP, 2 * D)
        unm[:B] = (net.unmerge_emb * net.shared).float()
        hw12g, une, enq, cq = token_head_tensors(hw12, net.head_norm.weight, unm)
        return [tables["tob"], tables["dest"], une, enq, cq, hw12g, hb12, bf(m3.view(n, c, NO)), bout + net.head_w3.bias.float() @ wout,
                bf(unm), hw12g.transpose(-1, -2).contiguous(), bf(m3.t().reshape(NO, n, c).permute(1, 0, 2))]  # fmt: skip


def rf_readout(net, h0: torch.Tensor, h1: torch.Tensor, tables: dict) -> torch.Tensor:
    """``net._readout(h0, h1)`` (a ``RigidFormerNet`` with frozen parameters) by the fused kernels as under bf16 autocast,
    differentiable in h0 and h1 ((N, T, D) fp32, the tokens' features on the embedding and the last block): (N, out_dim)
    fp32."""
    if h0.dtype != torch.float32 or h1.dtype != torch.float32:
        raise ValueError(f"the tokens' features must be fp32, got {h0.dtype} and {h1.dtype}")
    if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != torch.bfloat16:
        raise ValueError(f"the fused kernels compute the read-out as under bf16 autocast, not {torch.get_autocast_dtype('cuda')}")
    if tables["token_rows"]:  # merged tokens, from the tokens' rows
        with torch.no_grad():
            tensors = pack_token_head(net, tables)
        return rf_head_tok_fwd(h0.contiguous(), h1.contiguous(), tensors, tables["out_size"], net.head_norm.eps, tables["tok_config"])
    with torch.no_grad():
        tensors = pack_head(net, tables)
    return rf_head_fwd(h0.contiguous(), h1.contiguous(), tensors, tables["out_size"], net.head_norm.eps, tables["config"])
