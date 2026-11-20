"""SuperTrack's world-model window loss (``algos.supertrack.wm_window_terms``) as Triton kernels with a hand-written backward
pass, for training (train.py --st_wm_fused): per sample, the L1 (or, ``squared``, L2) errors of every body's world position,
link-origin linear velocity, rotation (the rotation vector of Rt^T Rh, as ``core.rotations.so3_log`` computes it) and angular
velocity between the forward kinematics of the predicted state and the target's, each summed over the bodies and the three
components: (N, 4), differentiable in the predicted state's six parts (the target's are data: no gradient, as the default).

Forward, two kernels. The first walks the tree of kernels/fk.py (its registered constants, ``fk.register``) for the
prediction and the target together, one sample per lane, with only the open branches of the two trees in registers; as soon
as a body's two states are computed it adds the body's position, velocity and angular velocity errors and writes one row of
32 values (27 used): those errors' derivatives with respect to the body's outputs (their signs; squared, twice the
differences), M = Rt^T Rh and Rt. The second runs over the bodies in a loop (one body's code) and, from M and Rt, the
rotation term and its derivative with respect to the body's predicted rotation, Rt (d|r|/dM)^T through so3_log's branches (the
largest of the quaternion's four estimates, its normalization and sign, the small-angle limit, atan2 of the half angle),
written in place of M. Backward, one kernel: kernels/fk.py's, each body's output adjoint its row's first 18 values times the
sample's term gradients, so it recomputes only the prediction's rotations and angular velocities along the open path, as
fk.py's does. A body's positions, velocities and the prediction's rotations never go to memory; per body one row of 32 values
is written twice and read twice (the default writes both trees' bodies' states and reads them back in several kernels,
forward and backward).

Why two forward kernels and one row per body: Triton's coalescing pass costs every memory access of a kernel about the size
of the whole kernel. With the rotation's chain unrolled over the 30 bodies and every value its own access, the forward kernel
took 32 minutes to compile for sm_80 (one CPU core of a busy node); with the row as one access per body and the chain in the
loop, 83 s, and the rotation kernel 1 s (fk.py's forward: 64 s there). Once per tree and loss, then from Triton's cache.

The half angle is atan2 of two non-negative numbers; Triton's interpreter has no atan2, so both the compiled kernels and the
interpreted ones take Cephes' single-precision atan on the ratio of the smaller to the larger (relative error about 2e-7, as
libdevice's atan2f). fp32 throughout: equal to the default up to float rounding (tests/test_st_wm_fused.py checks values and
gradients against wm_window_terms, and both against float64). No atomics: the same inputs give the same bits.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import fk
from .fk import (
    BLOCK,
    WARPS,
    _add3,
    _add_adjoint,
    _adjoint,
    _cross,
    _ld3,
    _ld9,
    _launchable,
    _on_device,
    _recompute,
    _rows,
    _st3,
    _st9,
    _times_mat,
    _times_vec,
    _turn,
    _tree,
)

INTERPRETED = fk.INTERPRETED
PI_2, PI_4, TAN_PI_8 = tl.constexpr(1.5707963267948966), tl.constexpr(0.7853981633974483), tl.constexpr(0.41421356237309503)  # the kernels' globals


@triton.jit
def _atan2_pos(y, x):
    """atan2(y, x) for y >= 0 and x >= 0: Cephes' atanf (the ratio of the smaller to the larger, reduced at tan(pi/8)
    about pi/4), pi/2 less it where y > x; 0 at (0, 0), as torch.atan2."""
    big, small = tl.maximum(x, y), tl.minimum(x, y)
    t = tl.where(big > 0.0, tl.div_rn(small, tl.where(big > 0.0, big, 1.0)), 0.0)
    red = t > TAN_PI_8
    z = tl.where(red, tl.div_rn(t - 1.0, t + 1.0), t)
    zz = z * z
    a = (((8.05374449538e-2 * zz - 1.38776856032e-1) * zz + 1.99777106478e-1) * zz - 3.33329491539e-1) * zz * z + z
    a = tl.where(red, a + PI_4, a)
    return tl.where(y > x, PI_2 - a, a)


@triton.jit
def _bitrev5(j):
    """j's five bits in reverse order: the column that a (B, 32) tile joined or split in halves five times holds at j."""
    return ((j & 1) << 4) | ((j & 2) << 2) | (j & 4) | ((j & 8) >> 2) | ((j & 16) >> 4)


@triton.jit
def _st_row(ptr, row, m, vals, K: tl.constexpr, zero):
    """vals (K <= 32 tensors (BLOCK,)) at ptr[row + k], k < K, in one store: a (BLOCK, 32) tile joined from them in halves,
    zeros past K, its columns in the order the joins leave them. One memory access in place of K: the compiler's
    coalescing pass costs the kernel's operations per access, and the forward kernel has many of both."""
    v = vals
    for _ in tl.static_range(K, 32):
        v = v + (zero,)
    for h in tl.static_range(5):
        w = ()
        for i in tl.static_range(len(v) // 2):
            w = w + (tl.join(v[2 * i], v[2 * i + 1]),)
        v = w
    cols = _bitrev5(tl.arange(0, 32))
    tl.store(ptr + row[:, None] + cols[None, :], tl.reshape(v[0], (zero.shape[0], 32)), mask=m[:, None] & (cols[None, :] < K))


@triton.jit
def _ld_row(ptr, row, m, K: tl.constexpr):
    """_st_row's row back: ptr[row + k], k < K, in one load, split in halves into 32 tensors (BLOCK,) (zeros past K)."""
    cols = _bitrev5(tl.arange(0, 32))
    t = tl.load(ptr + row[:, None] + cols[None, :], mask=m[:, None] & (cols[None, :] < K), other=0.0)
    v = (tl.reshape(t, (t.shape[0], 2, 2, 2, 2, 2)),)
    for h in tl.static_range(5):
        w = ()
        for i in tl.static_range(len(v)):
            w = w + tl.split(v[i])
        v = w
    return v


@triton.jit
def _sgn(x):
    """torch.sgn of a real: -1, 0 or 1 (abs's gradient)."""
    return tl.where(x > 0.0, 1.0, tl.where(x < 0.0, -1.0, 0.0))


@triton.jit
def _rel(rt, rh):
    """M = Rt^T Rh (row-major) from the target's and the prediction's rotations rt, rh."""
    m = ()
    for k in tl.static_range(9):
        m = m + (rt[k // 3] * rh[k % 3] + rt[3 + k // 3] * rh[3 + k % 3] + rt[6 + k // 3] * rh[6 + k % 3],)
    return m


@triton.jit
def _rot_term(m, rt, SQUARED: tl.constexpr):
    """The rotation term of one body from M = Rt^T Rh (``_rel``) and the target's rotation rt (row-major): the sum of |r_k|
    (squared: r_k^2) for r = so3_log(M) (core.rotations: mat_to_quat, the candidate of the largest of the four diagonal
    estimates, normalized, w >= 0; quat_to_rotvec, its small-angle branch below 1e-6), and that sum's gradient with respect to
    Rh, Rt gM, by so3_log's chain backwards on the same branch, as autograd differentiates it."""
    # the four diagonal estimates, the largest (the first of equals, as argmax) and its candidate
    tw, tx = 1.0 + m[0] + m[4] + m[8], 1.0 + m[0] - m[4] - m[8]
    ty, tz = 1.0 - m[0] + m[4] - m[8], 1.0 - m[0] - m[4] + m[8]
    qw, qx = tl.sqrt_rn(tl.maximum(tw, 1e-10)) / 2, tl.sqrt_rn(tl.maximum(tx, 1e-10)) / 2
    qy, qz = tl.sqrt_rn(tl.maximum(ty, 1e-10)) / 2, tl.sqrt_rn(tl.maximum(tz, 1e-10)) / 2
    is_w = (qw >= qx) & (qw >= qy) & (qw >= qz)
    is_x = (qx > qw) & (qx >= qy) & (qx >= qz)
    is_y = (qy > qw) & (qy > qx) & (qy >= qz)
    is_z = (qz > qw) & (qz > qx) & (qz > qy)
    d = tl.where(is_w, qw, tl.where(is_x, qx, tl.where(is_y, qy, qz)))
    t = tl.where(is_w, tw, tl.where(is_x, tx, tl.where(is_y, ty, tz)))
    a, b, c = m[7] - m[5], m[2] - m[6], m[3] - m[1]  # m21 - m12, m02 - m20, m10 - m01
    e, f, g = m[1] + m[3], m[2] + m[6], m[5] + m[7]  # m01 + m10, m02 + m20, m12 + m21
    x0 = tl.where(is_x, a, tl.where(is_y, b, c))  # each component's numerator where another estimate is the largest
    x1 = tl.where(is_w, a, tl.where(is_y, e, f))
    x2 = tl.where(is_w, b, tl.where(is_x, e, g))
    x3 = tl.where(is_w, c, tl.where(is_x, f, g))
    d4 = 4.0 * d
    c0 = tl.where(is_w, d, tl.div_rn(x0, d4))
    c1 = tl.where(is_x, d, tl.div_rn(x1, d4))
    c2 = tl.where(is_y, d, tl.div_rn(x2, d4))
    c3 = tl.where(is_z, d, tl.div_rn(x3, d4))
    nrm = tl.sqrt_rn(c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3)
    den = tl.maximum(nrm, 1e-12)  # F.normalize
    q0, q1, q2, q3 = tl.div_rn(c0, den), tl.div_rn(c1, den), tl.div_rn(c2, den), tl.div_rn(c3, den)
    sgn = tl.where(q0 < 0.0, -1.0, 1.0)
    q0, q1, q2, q3 = q0 * sgn, q1 * sgn, q2 * sgn, q3 * sgn
    # quat_to_rotvec (its own sign flip leaves q as it is)
    sh = tl.sqrt_rn(q1 * q1 + q2 * q2 + q3 * q3)
    half = _atan2_pos(sh, q0)
    small = sh < 1e-6
    q0c = tl.maximum(q0, 1e-6)
    she = sh + 1e-12
    scale = tl.where(small, tl.div_rn(2.0, q0c), tl.div_rn(2.0 * half, she))
    r0, r1, r2 = q1 * scale, q2 * scale, q3 * scale
    if SQUARED:
        value = r0 * r0 + r1 * r1 + r2 * r2
        g0, g1, g2 = 2.0 * r0, 2.0 * r1, 2.0 * r2
    else:
        value = tl.abs(r0) + tl.abs(r1) + tl.abs(r2)
        g0, g1, g2 = _sgn(r0), _sgn(r1), _sgn(r2)
    # backwards: r = v scale
    gs = g0 * q1 + g1 * q2 + g2 * q3
    gv0, gv1, gv2 = g0 * scale, g1 * scale, g2 * scale
    # scale: 2 / max(q0, 1e-6) below 1e-6, else 2 half / (sh + 1e-12) with half = atan2(sh, q0)
    gq0_small = tl.where(q0 >= 1e-6, -gs * tl.div_rn(2.0, q0c * q0c), 0.0)
    gh = tl.div_rn(2.0 * gs, she)
    hyp = q0 * q0 + sh * sh
    hyp = tl.where(hyp > 0.0, hyp, 1.0)
    gsh = tl.where(small, 0.0, -tl.div_rn(2.0 * half * gs, she * she) + tl.div_rn(gh * q0, hyp))
    gq0 = tl.where(small, gq0_small, -tl.div_rn(gh * sh, hyp))
    gn = tl.where(sh > 0.0, tl.div_rn(gsh, tl.where(sh > 0.0, sh, 1.0)), 0.0)  # |v|: v / |v|, nothing at 0
    gq1, gq2, gq3 = gv0 + gn * q1, gv1 + gn * q2, gv2 + gn * q3
    # the sign, then the normalization (q = c / max(|c|, 1e-12))
    gq0, gq1, gq2, gq3 = gq0 * sgn, gq1 * sgn, gq2 * sgn, gq3 * sgn
    gc0, gc1, gc2, gc3 = tl.div_rn(gq0, den), tl.div_rn(gq1, den), tl.div_rn(gq2, den), tl.div_rn(gq3, den)
    dot = tl.where(nrm >= 1e-12, -tl.div_rn(gq0 * c0 + gq1 * c1 + gq2 * c2 + gq3 * c3, den * den), 0.0)
    gnrm = tl.div_rn(dot, tl.where(nrm > 0.0, nrm, 1.0))
    gc0, gc1, gc2, gc3 = gc0 + gnrm * c0, gc1 + gnrm * c1, gc2 + gnrm * c2, gc3 + gnrm * c3
    # the candidate: the largest estimate d itself, the others x / (4 d)
    gd = tl.where(is_w, gc0, 0.0) + tl.where(is_x, gc1, 0.0) + tl.where(is_y, gc2, 0.0) + tl.where(is_z, gc3, 0.0)
    gx0, gx1 = tl.where(is_w, 0.0, tl.div_rn(gc0, d4)), tl.where(is_x, 0.0, tl.div_rn(gc1, d4))
    gx2, gx3 = tl.where(is_y, 0.0, tl.div_rn(gc2, d4)), tl.where(is_z, 0.0, tl.div_rn(gc3, d4))
    gd = gd - tl.where(is_w, 0.0, tl.div_rn(gc0 * c0, d)) - tl.where(is_x, 0.0, tl.div_rn(gc1 * c1, d))
    gd = gd - tl.where(is_y, 0.0, tl.div_rn(gc2 * c2, d)) - tl.where(is_z, 0.0, tl.div_rn(gc3 * c3, d))
    # d = sqrt(max(t, 1e-10)) / 2
    gt = tl.where(t >= 1e-10, tl.div_rn(gd, 4.0 * tl.sqrt_rn(tl.maximum(t, 1e-10))), 0.0)
    s0 = tl.where(is_w | is_x, 1.0, -1.0)  # the signs of m00, m11, m22 in the largest estimate
    s1 = tl.where(is_w | is_y, 1.0, -1.0)
    s2 = tl.where(is_w | is_z, 1.0, -1.0)
    ga = tl.where(is_w, gx1, tl.where(is_x, gx0, 0.0))  # the numerators' gradients, by the combination each is
    gb = tl.where(is_w, gx2, tl.where(is_y, gx0, 0.0))
    gc = tl.where(is_w, gx3, tl.where(is_z, gx0, 0.0))
    ge = tl.where(is_x, gx2, tl.where(is_y, gx1, 0.0))
    gf = tl.where(is_x, gx3, tl.where(is_z, gx1, 0.0))
    gg = tl.where(is_y, gx3, tl.where(is_z, gx2, 0.0))
    gm = (s0 * gt, ge - gc, gb + gf, gc + ge, s1 * gt, gg - ga, gf - gb, ga + gg, s2 * gt)
    # M = Rt^T Rh: Rh gets Rt gM
    y = ()
    for k in tl.static_range(9):
        y = y + (rt[k // 3 * 3] * gm[k % 3] + rt[k // 3 * 3 + 1] * gm[3 + k % 3] + rt[k // 3 * 3 + 2] * gm[6 + k % 3],)
    return value, y


@triton.jit
def _child(a, q, qd, OFFP: tl.constexpr, OFFR: tl.constexpr, AXIS: tl.constexpr, CROSS: tl.constexpr, i: tl.constexpr):
    """Body i's state (position, rotation, velocity, angular velocity) from its parent's a and its joint's angle and rate:
    kernels/fk.py's forward step."""
    pb = _add3(a[0], _times_vec(a[1], OFFP, i))
    e = _times_mat(a[1], OFFR, i)
    u = _times_vec(e, AXIS, i)
    wb = (a[3][0] + u[0] * qd, a[3][1] + u[1] * qd, a[3][2] + u[2] * qd)
    vb = _add3(a[2], _cross(a[3], (pb[0] - a[0][0], pb[1] - a[0][1], pb[2] - a[0][2])))  # the lever arm as forward has it, b - a
    return pb, _turn(e, u, tl.cos(q), tl.sin(q), CROSS, AXIS, i), vb, wb


@triton.jit
def _body_terms(h, t, acc, DY, n, m, zero, b: tl.constexpr, NB: tl.constexpr, SQUARED: tl.constexpr):
    """Body b's position, velocity and angular velocity errors (predicted state h, target t) added to acc; into its row of
    DY (N, NB, 32): their derivatives, M = Rt^T Rh and Rt, for the rotation kernel."""
    dp = (h[0][0] - t[0][0], h[0][1] - t[0][1], h[0][2] - t[0][2])
    dv = (h[2][0] - t[2][0], h[2][1] - t[2][1], h[2][2] - t[2][2])
    dw = (h[3][0] - t[3][0], h[3][1] - t[3][1], h[3][2] - t[3][2])
    if SQUARED:
        ep = dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]
        ev = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]
        ew = dw[0] * dw[0] + dw[1] * dw[1] + dw[2] * dw[2]
        sp, sv, sw = (2.0 * dp[0], 2.0 * dp[1], 2.0 * dp[2]), (2.0 * dv[0], 2.0 * dv[1], 2.0 * dv[2]), (2.0 * dw[0], 2.0 * dw[1], 2.0 * dw[2])
    else:
        ep = tl.abs(dp[0]) + tl.abs(dp[1]) + tl.abs(dp[2])
        ev = tl.abs(dv[0]) + tl.abs(dv[1]) + tl.abs(dv[2])
        ew = tl.abs(dw[0]) + tl.abs(dw[1]) + tl.abs(dw[2])
        sp, sv, sw = (_sgn(dp[0]), _sgn(dp[1]), _sgn(dp[2])), (_sgn(dv[0]), _sgn(dv[1]), _sgn(dv[2])), (_sgn(dw[0]), _sgn(dw[1]), _sgn(dw[2]))
    _st_row(DY, n * (32 * NB) + 32 * b, m, sp + sv + sw + _rel(t[1], h[1]) + t[1], 27, zero)
    return acc[0] + ep, acc[1] + ev, acc[2] + ew


# the sizes and strides are not specialized on (one compiled kernel per tree and loss, whatever views come in), as fk.py's
@triton.jit(
    do_not_specialize=("N", "sp", "sr", "sv", "sw", "sq", "sqd", "tp", "tr", "tv", "tw", "tq", "tqd"),
    do_not_specialize_on_alignment=("P", "R", "V", "W", "Q", "QD", "PT", "RT", "VT", "WT", "QT", "QDT", "E", "DY"),
)
def _wm_fwd_kernel(P, R, V, W, Q, QD, PT, RT, VT, WT, QT, QDT, E, DY, N, sp, sr, sv, sw, sq, sqd, tp, tr, tv, tw, tq, tqd,
                   PARENT: tl.constexpr, JOINT: tl.constexpr, OFFP: tl.constexpr, OFFR: tl.constexpr, AXIS: tl.constexpr, CROSS: tl.constexpr,
                   NB: tl.constexpr, NJ: tl.constexpr, SQUARED: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    """The position, velocity and angular velocity terms E[:, (0, 1, 3)] of the predicted states (rows of P, R, V, W, Q, QD)
    against the targets' (PT, ...), and every body's row of DY (N, NB, 32): those terms' derivatives, M and Rt."""
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    hat = ((_ld3(P, n * sp, m), _ld9(R, n * sr, m), _ld3(V, n * sv, m), _ld3(W, n * sw, m)),)
    tgt = ((_ld3(PT, n * tp, m), _ld9(RT, n * tr, m), _ld3(VT, n * tv, m), _ld3(WT, n * tw, m)),)
    acc = _body_terms(hat[0], tgt[0], (zero, zero, zero), DY, n, m, zero, 0, NB, SQUARED)
    q, qd, qt, qdt = _ld_row(Q, n * sq, m, NJ), _ld_row(QD, n * sqd, m, NJ), _ld_row(QT, n * tq, m, NJ), _ld_row(QDT, n * tqd, m, NJ)
    for i in tl.static_range(1, NB):
        hat = hat + (_child(hat[PARENT[i]], q[JOINT[i]], qd[JOINT[i]], OFFP, OFFR, AXIS, CROSS, i),)
        tgt = tgt + (_child(tgt[PARENT[i]], qt[JOINT[i]], qdt[JOINT[i]], OFFP, OFFR, AXIS, CROSS, i),)
        acc = _body_terms(hat[i], tgt[i], acc, DY, n, m, zero, i, NB, SQUARED)
    tl.store(E + n * 4, acc[0], mask=m)
    tl.store(E + n * 4 + 1, acc[1], mask=m)
    tl.store(E + n * 4 + 3, acc[2], mask=m)


@triton.jit(do_not_specialize=("N",), do_not_specialize_on_alignment=("E", "DY"))
def _wm_rot_kernel(E, DY, N, NB: tl.constexpr, SQUARED: tl.constexpr, BLOCK: tl.constexpr):
    """The rotation term E[:, 2] from every body's M and Rt in DY (N, NB, 32), over the bodies in a loop (one body's code:
    tl.range, not unrolled), and its derivative with respect to the body's predicted rotation in place of M."""
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    acc = zero
    for b in tl.range(0, NB):
        row = n * (32 * NB) + 32 * b
        v = _ld_row(DY + 9, row, m, 18)
        mm, rt = (), ()
        for k in tl.static_range(9):
            mm, rt = mm + (v[k],), rt + (v[9 + k],)
        value, y = _rot_term(mm, rt, SQUARED)
        _st_row(DY + 9, row, m, y, 9, zero)
        acc = acc + value
    tl.store(E + n * 4 + 2, acc, mask=m)


@triton.jit
def _gather_terms(DY, g4, n, m, NB: tl.constexpr, ups, CHILD: tl.constexpr, UPOS: tl.constexpr, b: tl.constexpr):
    """Body b's adjoint (gp, gR, gv, gw): its stored derivatives times the sample's term gradients g4 (position, velocity,
    rotation, angular velocity), plus what its children passed up (``ups[UPOS[child]]``)."""
    d = _ld_row(DY, n * (32 * NB) + 32 * b, m, 18)
    gr = ()
    for k in tl.static_range(9):
        gr = gr + (d[9 + k] * g4[2],)
    g = ((d[0] * g4[0], d[1] * g4[0], d[2] * g4[0]), gr, (d[3] * g4[1], d[4] * g4[1], d[5] * g4[1]), (d[6] * g4[3], d[7] * g4[3], d[8] * g4[3]))
    for c in tl.static_range(len(CHILD[b])):
        g = _add_adjoint(g, ups[UPOS[CHILD[b][c]]])
    return g


@triton.jit(
    do_not_specialize=("N", "sr", "sw", "sq", "sqd"),
    do_not_specialize_on_alignment=("G", "DY", "R", "W", "Q", "QD", "DP", "DR", "DV", "DW", "DQ", "DQD"),
)
def _wm_bwd_kernel(G, DY, R, W, Q, QD, DP, DR, DV, DW, DQ, DQD, N, sr, sw, sq, sqd, PARENT: tl.constexpr, JOINT: tl.constexpr,
                   OFFP: tl.constexpr, OFFR: tl.constexpr, AXIS: tl.constexpr, CROSS: tl.constexpr, KS: tl.constexpr, CHILD: tl.constexpr,
                   SCHED: tl.constexpr, POS: tl.constexpr, UPOS: tl.constexpr, NB: tl.constexpr, NJ: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    """_wm_fwd_kernel backwards: the terms' gradients G (N, 4) -> the gradients of the predicted state's p, R, v, w (DP, DV,
    DW (N, 3), DR (N, 3, 3)) and of its joints' angles and rates (DQ, DQD (N, NJ)): kernels/fk.py's backward kernel with the
    bodies' adjoints from DY and G."""
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    g4 = (tl.load(G + n * 4, mask=m, other=0.0), tl.load(G + n * 4 + 1, mask=m, other=0.0), tl.load(G + n * 4 + 2, mask=m, other=0.0),
          tl.load(G + n * 4 + 3, mask=m, other=0.0))  # fmt: skip
    states = ((_ld9(R, n * sr, m), _ld3(W, n * sw, m), zero, zero, zero),)
    ups = ()
    for k in tl.static_range(len(SCHED)):
        if SCHED[k][1] == 0:
            states = states + (_recompute(states, Q, QD, n, m, sq, sqd, PARENT, JOINT, OFFR, AXIS, CROSS, POS, SCHED[k][0]),)
        else:
            g = _gather_terms(DY, g4, n, m, NB, ups, CHILD, UPOS, SCHED[k][0])
            dq, dqd, up = _adjoint(g, states[POS[PARENT[SCHED[k][0]]]], states[POS[SCHED[k][0]]], zero, OFFP, OFFR, AXIS, CROSS, KS, SCHED[k][0])
            tl.store(DQ + n * NJ + JOINT[SCHED[k][0]], dq, mask=m)
            tl.store(DQD + n * NJ + JOINT[SCHED[k][0]], dqd, mask=m)
            ups = ups + (up,)
    g = _gather_terms(DY, g4, n, m, NB, ups, CHILD, UPOS, 0)
    _st3(DP, n * 3, m, g[0])
    _st9(DR, n * 9, m, g[1])
    _st3(DV, n * 3, m, g[2])
    _st3(DW, n * 3, m, g[3])


@torch.library.custom_op("ndm_gym::st_wm_terms", mutates_args=())
def st_wm_terms(p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, pt: torch.Tensor,
                Rt: torch.Tensor, vt: torch.Tensor, wt: torch.Tensor, qt: torch.Tensor, qdt: torch.Tensor, tree: str,
                squared: bool) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """``wm_window_terms`` of the registered ``tree``: the predicted state p, v, w (N, 3), R (N, 3, 3), q, qd (N, J) against the
    target's pt, ..., qdt, fp32 -> the terms (N, 4) and every body's row (N, NB, 32): the terms' derivatives, which the
    backward reads (the first 18 values)."""
    t = _tree(tree)
    _launchable(p, R, v, w, q, qd, pt, Rt, vt, wt, qt, qdt)
    N, nb = p.shape[0], t["NB"]
    out = (p.new_empty(N, 4), p.new_empty(N, nb, 32))
    if N:
        (p, sp), (R, sr), (v, sv), (w, sw), (q, sq), (qd, sqd) = (_rows(x) for x in (p, R, v, w, q, qd))
        (pt, tp), (Rt, tr), (vt, tv), (wt, tw), (qt, tq), (qdt, tqd) = (_rows(x) for x in (pt, Rt, vt, wt, qt, qdt))
        consts = {k: t[k] for k in ("PARENT", "JOINT", "OFFP", "OFFR", "AXIS", "CROSS", "NB", "NJ")}
        if t["NJ"] > 32:
            raise ValueError(f"kernels.wm_loss: at most 32 joints (a row of the kernels' tiles), got {t['NJ']}")
        with _on_device(p):
            _wm_fwd_kernel[(triton.cdiv(N, BLOCK),)](p, R, v, w, q, qd, pt, Rt, vt, wt, qt, qdt, *out, N, sp, sr, sv, sw, sq, sqd, tp, tr, tv, tw,
                                                     tq, tqd, **consts, SQUARED=bool(squared), BLOCK=BLOCK, num_warps=WARPS)  # fmt: skip
            _wm_rot_kernel[(triton.cdiv(N, BLOCK),)](*out, N, NB=nb, SQUARED=bool(squared), BLOCK=BLOCK, num_warps=WARPS)
    return out


@st_wm_terms.register_fake
def _(p, R, v, w, q, qd, pt, Rt, vt, wt, qt, qdt, tree, squared):
    N, nb = p.shape[0], _tree(tree)["NB"]
    return p.new_empty(N, 4), p.new_empty(N, nb, 32)


@torch.library.custom_op("ndm_gym::st_wm_terms_bwd", mutates_args=())
def st_wm_terms_bwd(G: torch.Tensor, DY: torch.Tensor, R: torch.Tensor, w: torch.Tensor, q: torch.Tensor, qd: torch.Tensor,
                    tree: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """st_wm_terms' backward pass: the terms' gradients G (N, 4), the derivatives it stored and the predicted state's root
    rotation, angular velocity and joints -> the gradients of the predicted p, R, v, w, q and qd."""
    t = _tree(tree)
    N, nb, nj = R.shape[0], t["NB"], t["NJ"]
    if tuple(G.shape) != (N, 4) or tuple(DY.shape) != (N, nb, 32):
        raise ValueError(f"kernels.wm_loss: G (N, 4) and DY (N, {nb}, 32) for N = {N}, got {tuple(G.shape)} and {tuple(DY.shape)}")
    _launchable(G, DY, R, w, q, qd)
    out = (R.new_empty(N, 3), R.new_empty(N, 3, 3), R.new_empty(N, 3), R.new_empty(N, 3), R.new_empty(N, nj), R.new_empty(N, nj))
    if N:
        (R, sr), (w, sw), (q, sq), (qd, sqd) = (_rows(x) for x in (R, w, q, qd))
        consts = {k: t[k] for k in ("PARENT", "JOINT", "OFFP", "OFFR", "AXIS", "CROSS", "KS", "CHILD", "SCHED", "POS", "UPOS", "NB", "NJ")}
        with _on_device(R):
            _wm_bwd_kernel[(triton.cdiv(N, BLOCK),)](G.contiguous(), DY.contiguous(), R, w, q, qd, *out, N, sr, sw, sq, sqd, **consts,
                                                     BLOCK=BLOCK, num_warps=WARPS)  # fmt: skip
    return out


@st_wm_terms_bwd.register_fake
def _(G, DY, R, w, q, qd, tree):
    N, nj = R.shape[0], _tree(tree)["NJ"]
    return R.new_empty(N, 3), R.new_empty(N, 3, 3), R.new_empty(N, 3), R.new_empty(N, 3), R.new_empty(N, nj), R.new_empty(N, nj)


def _terms_setup(ctx, inputs, output):
    p, R, v, w, q, qd, pt, Rt, vt, wt, qt, qdt, tree, squared = inputs
    _, DY = output
    ctx.save_for_backward(DY, R, w, q, qd)
    ctx.tree = tree


def _terms_backward(ctx, gE, gDY):
    DY, R, w, q, qd = ctx.saved_tensors
    if gE is None:
        gE = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    return (*st_wm_terms_bwd(gE, DY, R, w, q, qd, ctx.tree), None, None, None, None, None, None, None, None)


st_wm_terms.register_autograd(_terms_backward, setup_context=_terms_setup)


def window_terms(tree: str, S_hat: tuple, S: tuple, squared: bool = False) -> torch.Tensor:
    """``algos.supertrack.wm_window_terms`` by the kernels for the registered ``tree`` (``fk.register``): the predicted states
    S_hat and the targets S, each (p, R, v, w, q, qd) of one batch shape, fp32 -> the four terms (..., 4); the targets get no
    gradient. Anything else raises."""
    nj, lead = _tree(tree)["NJ"], S_hat[0].shape[:-1]
    shapes = ((*lead, 3), (*lead, 3, 3), (*lead, 3), (*lead, 3), (*lead, nj), (*lead, nj))
    for name, X in (("S_hat", S_hat), ("S", S)):
        if len(X) != 6 or any(tuple(x.shape) != s for x, s in zip(X, shapes)):
            raise ValueError(f"kernels.wm_loss: {name}'s p, R, v, w, q, qd of the shapes {shapes}, got {[tuple(x.shape) for x in X]}")
        if any(x.dtype != torch.float32 for x in X):
            raise TypeError(f"kernels.wm_loss: fp32 states only, got {name} {[str(x.dtype) for x in X]}")
    flat = tuple(x.reshape(-1, *s[len(lead) :]) for X in (S_hat, S) for x, s in zip(X, shapes))
    E, _ = st_wm_terms(*flat[:6], *(x.detach() for x in flat[6:]), tree, bool(squared))
    return E.view(*lead, 4)


__all__ = ["INTERPRETED", "st_wm_terms", "window_terms"]
