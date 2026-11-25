"""The G1 world models' step outside their network as two Triton kernels, for inference (``g1_infer.inference_copy(fused=True)``),
as ``hoi_fused`` does for HDMI's scenes.

``rf_features`` / ``mlp_features``: the network's raw input features from the robot's state, the setpoints and the
normalized sim context in one kernel: for the RigidFormer (``RigidFormerNet.inputs``) the forward kinematics of the 30
bodies, unrolled in registers with the kinematic tree as compile-time constants and run directly in the root's heading
frame, every body's token features (position, 6-D rotation, velocities, its joint's q, qd, setpoint, PD error, clipped PD
torque and calibration offset), the global context and the rotary positions; for the MLP the computed part of
``invariant_features`` (height, tilt, body-frame velocities). ``integrate``: the robot's next state from the de-normalized
outputs, both parameterizations (``AccelWorldModel.step`` with its substeps' position coefficient, ``IncrementWorldModel.step``),
in one kernel. Torch does the rest (the copies, the normalization, which it fuses with the network's first cast, and the
output de-normalization). Each program handles a block of samples, one per lane; the features are written feature-major
((D, N), coalesced) and read transposed. The math is that of ``RigidFormerNet.inputs``, ``invariant_features`` and the
world models' ``step`` in fp32, equal up to float rounding. Forward pass only.

``rf_inputs``: the RigidFormer's features again, differentiable in the state and the setpoints, for training
(``RigidFormerNet(fused_inputs=True)``, ``--rf_fused_inputs``: the policy's BPTT and the world-model fit's rollouts,
where the compiled kinematics and their autograd are hundreds of small kernels per step). Its two kernels loop over the
bodies with the kinematic tree in tables and every body's state in a scratch buffer, instead of unrolling the tree in
registers as the inference kernel does: the backward pass needs every body's state through its reverse sweep (more than
a thread's registers), and a looped kernel compiles in seconds where the unrolled one takes minutes. The backward kernel
runs the forward recursion again, then from the last body to the first adds to its parent's adjoint what a body passes up
(written by hand; tests/test_rf_inputs.py compares every gradient with autograd's).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from ..robots.g1 import NUM_J
from ..state import CONTROL_DT
from .hoi_fused import _TREES, _exp, _ld, _mm3, _mv3, _rows, _st, _yaw
from .rigidformer import GLOBAL_DIM, POS_SCALE, TOKEN_DIM, XI_OFFSET_SCALE

if TYPE_CHECKING:
    from .rigidformer import TreeKinematics

BLOCK, WARPS = 64, 2  # samples per program (one per lane) and warps, as hoi_fused


@triton.jit
def _g1_pose(out, n, N, m, i: tl.constexpr, b, TD: tl.constexpr, D_IN: tl.constexpr, POS: tl.constexpr):
    """Body i's position, 6-D rotation (the first two columns, row by row) and velocities (token columns TD i ...) and
    its rotary position (D_IN + 3 i)."""
    for k in tl.static_range(3):
        _st(out, n, N, m, TD * i + k, b[k])
        _st(out, n, N, m, D_IN + 3 * i + k, b[k] / POS)
        _st(out, n, N, m, TD * i + 3 + 2 * k, b[3 + 3 * k])
        _st(out, n, N, m, TD * i + 4 + 2 * k, b[4 + 3 * k])
    for k in tl.static_range(6):
        _st(out, n, N, m, TD * i + 9 + k, b[12 + k])


@triton.jit
def _g1_rf_features_kernel(P, R, V, W, Q, QD, A, XI, OUT, N, sp, sr, sv, sw, sq, sqd, sa, sx, PARENT: tl.constexpr, JOINT: tl.constexpr,
                           OFFP: tl.constexpr, OFFR: tl.constexpr, AXIS: tl.constexpr, Q0: tl.constexpr, ALPHA: tl.constexpr, KP: tl.constexpr,
                           KD: tl.constexpr, TMAX: tl.constexpr, NB: tl.constexpr, NJ: tl.constexpr, TD: tl.constexpr, D_IN: tl.constexpr,
                           POS: tl.constexpr, XOFF: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    p2 = _ld(P, n * sp, m, 2)
    r00, r01, r02 = _ld(R, n * sr, m, 0), _ld(R, n * sr, m, 1), _ld(R, n * sr, m, 2)
    r10, r11, r12 = _ld(R, n * sr, m, 3), _ld(R, n * sr, m, 4), _ld(R, n * sr, m, 5)
    r20, r21, r22 = _ld(R, n * sr, m, 6), _ld(R, n * sr, m, 7), _ld(R, n * sr, m, 8)
    v0, v1, v2 = _ld(V, n * sv, m, 0), _ld(V, n * sv, m, 1), _ld(V, n * sv, m, 2)
    w0, w1, w2 = _ld(W, n * sw, m, 0), _ld(W, n * sw, m, 1), _ld(W, n * sw, m, 2)
    c, s = _yaw(r00, r10)
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    # every joint's angle and rate first (one memory round trip), then the kinematic tree unrolled with its constants:
    # each body's position, rotation (row-major), link-origin velocity and angular velocity in the heading frame with the
    # origin at the pelvis's ground projection, stored as soon as computed so that only the open branches stay live
    qs, qds = (zero,), (zero,)
    for i in tl.static_range(1, NB):
        qs = qs + (tl.load(Q + n * sq + JOINT[i], mask=m, other=0.0),)
        qds = qds + (tl.load(QD + n * sqd + JOINT[i], mask=m, other=0.0),)
    states = ((zero, zero, p2, c * r00 + s * r10, c * r01 + s * r11, c * r02 + s * r12, c * r10 - s * r00, c * r11 - s * r01, c * r12 - s * r02,
               r20, r21, r22, c * v0 + s * v1, c * v1 - s * v0, v2, c * w0 + s * w1, c * w1 - s * w0, w2),)  # fmt: skip
    _g1_pose(OUT, n, N, m, 0, states[0], TD, D_IN, POS)
    for k in tl.static_range(TD - 15):  # the pelvis has no joint
        _st(OUT, n, N, m, 15 + k, zero)
    for i in tl.static_range(1, NB):
        a = states[PARENT[i]]
        o, q_, x = OFFP[i], OFFR[i], AXIS[i]
        b0 = a[0] + (a[3] * o[0] + a[4] * o[1] + a[5] * o[2])
        b1 = a[1] + (a[6] * o[0] + a[7] * o[1] + a[8] * o[2])
        b2 = a[2] + (a[9] * o[0] + a[10] * o[1] + a[11] * o[2])
        e = (a[3] * q_[0] + a[4] * q_[3] + a[5] * q_[6], a[3] * q_[1] + a[4] * q_[4] + a[5] * q_[7], a[3] * q_[2] + a[4] * q_[5] + a[5] * q_[8],
             a[6] * q_[0] + a[7] * q_[3] + a[8] * q_[6], a[6] * q_[1] + a[7] * q_[4] + a[8] * q_[7], a[6] * q_[2] + a[7] * q_[5] + a[8] * q_[8],
             a[9] * q_[0] + a[10] * q_[3] + a[11] * q_[6], a[9] * q_[1] + a[10] * q_[4] + a[11] * q_[7],
             a[9] * q_[2] + a[10] * q_[5] + a[11] * q_[8])  # fmt: skip
        cq, sn = tl.cos(qs[i]), tl.sin(qs[i])
        C = 1 - cq
        Ax = (cq + x[0] * x[0] * C, x[0] * x[1] * C - x[2] * sn, x[0] * x[2] * C + x[1] * sn,
              x[1] * x[0] * C + x[2] * sn, cq + x[1] * x[1] * C, x[1] * x[2] * C - x[0] * sn,
              x[2] * x[0] * C - x[1] * sn, x[2] * x[1] * C + x[0] * sn, cq + x[2] * x[2] * C)  # fmt: skip
        nw = (a[15] + (e[0] * x[0] + e[1] * x[1] + e[2] * x[2]) * qds[i], a[16] + (e[3] * x[0] + e[4] * x[1] + e[5] * x[2]) * qds[i],
              a[17] + (e[6] * x[0] + e[7] * x[1] + e[8] * x[2]) * qds[i])  # fmt: skip
        d0, d1, d2 = b0 - a[0], b1 - a[1], b2 - a[2]
        rb = _mm3(e, Ax)
        states = states + ((b0, b1, b2, rb[0], rb[1], rb[2], rb[3], rb[4], rb[5], rb[6], rb[7], rb[8], a[12] + (a[16] * d2 - a[17] * d1),
                            a[13] + (a[17] * d0 - a[15] * d2), a[14] + (a[15] * d1 - a[16] * d0), nw[0], nw[1], nw[2]),)  # fmt: skip
        _g1_pose(OUT, n, N, m, i, states[i], TD, D_IN, POS)
        # the body's parent joint: q, qd, setpoint, PD error q0 + offset + alpha u - q, PD torque / effort limit (clipped), offset
        # (the joint's constants indexed by JOINT[i] in place: a constexpr assigned to a variable becomes a tensor)
        u = tl.load(A + n * sa + JOINT[i], mask=m, other=0.0)
        off = tl.load(XI + n * sx + 1 + JOINT[i], mask=m, other=0.0)
        err = Q0[JOINT[i]] + XOFF * off + ALPHA[JOINT[i]] * u - qs[i]
        tau = tl.minimum(tl.maximum(KP[JOINT[i]] * err - KD[JOINT[i]] * qds[i], -TMAX[JOINT[i]]), TMAX[JOINT[i]]) / TMAX[JOINT[i]]
        _st(OUT, n, N, m, TD * i + 15, qs[i])
        _st(OUT, n, N, m, TD * i + 16, qds[i])
        _st(OUT, n, N, m, TD * i + 17, u)
        _st(OUT, n, N, m, TD * i + 18, err)
        _st(OUT, n, N, m, TD * i + 19, tau)
        _st(OUT, n, N, m, TD * i + 20, off)
    # the global context: friction and the torso's centre-of-mass offset (normalized xi)
    _st(OUT, n, N, m, TD * NB, tl.load(XI + n * sx, mask=m, other=0.0))
    for k in tl.static_range(3):
        _st(OUT, n, N, m, TD * NB + 1 + k, tl.load(XI + n * sx + 1 + NJ + k, mask=m, other=0.0))


SD = 18  # a body's state in the heading frame: position 3, rotation 9 (row-major), link-origin velocity 3, angular velocity 3
BODY_COLS = 20  # a body's constants (body_tables): parent offset 3, its rotation 9, joint axis 3, q0, alpha, kp, kd, effort limit


@triton.jit
def _g1_root(P, R, V, W, n, m, sp, sr, sv, sw, BLOCK: tl.constexpr):
    """The pelvis's state in its heading frame with the origin at its ground projection, the yaw's (c, s, norm) and the
    inputs it is made of (R row-major, v, w)."""
    r = (_ld(R, n * sr, m, 0), _ld(R, n * sr, m, 1), _ld(R, n * sr, m, 2), _ld(R, n * sr, m, 3), _ld(R, n * sr, m, 4), _ld(R, n * sr, m, 5),
         _ld(R, n * sr, m, 6), _ld(R, n * sr, m, 7), _ld(R, n * sr, m, 8))  # fmt: skip
    v = (_ld(V, n * sv, m, 0), _ld(V, n * sv, m, 1), _ld(V, n * sv, m, 2))
    w = (_ld(W, n * sw, m, 0), _ld(W, n * sw, m, 1), _ld(W, n * sw, m, 2))
    nrm = tl.sqrt_rn(r[0] * r[0] + r[3] * r[3] + 1e-9)  # _yaw, with its norm
    c, s = tl.div_rn(r[0], nrm), tl.div_rn(r[3], nrm)
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    st = (zero, zero, _ld(P, n * sp, m, 2), c * r[0] + s * r[3], c * r[1] + s * r[4], c * r[2] + s * r[5], c * r[3] - s * r[0],
          c * r[4] - s * r[1], c * r[5] - s * r[2], r[6], r[7], r[8], c * v[0] + s * v[1], c * v[1] - s * v[0], v[2],
          c * w[0] + s * w[1], c * w[1] - s * w[0], w[2])  # fmt: skip
    return st, c, s, nrm, r, v, w


@triton.jit
def _g1_tables(BODY, i):
    """Body i's constants: its offset in the parent's frame o, that frame's rotation Q (row-major), its joint's axis x and
    the joint's (q0, alpha, kp, kd, effort limit)."""
    t = BODY + i * 20
    o = (tl.load(t), tl.load(t + 1), tl.load(t + 2))
    q_ = (tl.load(t + 3), tl.load(t + 4), tl.load(t + 5), tl.load(t + 6), tl.load(t + 7), tl.load(t + 8), tl.load(t + 9), tl.load(t + 10),
          tl.load(t + 11))  # fmt: skip
    x = (tl.load(t + 12), tl.load(t + 13), tl.load(t + 14))
    jc = (tl.load(t + 15), tl.load(t + 16), tl.load(t + 17), tl.load(t + 18), tl.load(t + 19))
    return o, q_, x, jc


@triton.jit
def _g1_state(ST, i, n, N, m):
    """Body i's 18 values from a scratch buffer, stored value-major ((18 B, N): value k of body i in row 18 i + k), so
    that the samples of a warp read and write contiguous memory."""
    t = (18 * i).to(tl.int64) * N + n
    return (_ld(ST, t, m, 0), _ld(ST, t, m, N), _ld(ST, t, m, 2 * N), _ld(ST, t, m, 3 * N), _ld(ST, t, m, 4 * N), _ld(ST, t, m, 5 * N),
            _ld(ST, t, m, 6 * N), _ld(ST, t, m, 7 * N), _ld(ST, t, m, 8 * N), _ld(ST, t, m, 9 * N), _ld(ST, t, m, 10 * N), _ld(ST, t, m, 11 * N),
            _ld(ST, t, m, 12 * N), _ld(ST, t, m, 13 * N), _ld(ST, t, m, 14 * N), _ld(ST, t, m, 15 * N), _ld(ST, t, m, 16 * N),
            _ld(ST, t, m, 17 * N))  # fmt: skip


@triton.jit
def _g1_put(ST, i, n, N, m, b):
    t = (18 * i).to(tl.int64) * N + n
    for k in tl.static_range(18):
        tl.store(ST + t + k * N, b[k], mask=m)


@triton.jit
def _add18(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5], a[6] + b[6], a[7] + b[7], a[8] + b[8], a[9] + b[9],
            a[10] + b[10], a[11] + b[11], a[12] + b[12], a[13] + b[13], a[14] + b[14], a[15] + b[15], a[16] + b[16], a[17] + b[17])  # fmt: skip


@triton.jit
def _g1_joint(a, o, q_, x, qi, qdi):
    """One joint of the tree: the child's state from its parent's a, the child's constants and its joint's angle and rate
    (b = a + Ra o, Rb = Ra Q A(q), v = va + wa x (b - a), w = wa + Ra Q x qd); also e = Ra Q and A, for the reverse sweep."""
    b0 = a[0] + (a[3] * o[0] + a[4] * o[1] + a[5] * o[2])
    b1 = a[1] + (a[6] * o[0] + a[7] * o[1] + a[8] * o[2])
    b2 = a[2] + (a[9] * o[0] + a[10] * o[1] + a[11] * o[2])
    e = (a[3] * q_[0] + a[4] * q_[3] + a[5] * q_[6], a[3] * q_[1] + a[4] * q_[4] + a[5] * q_[7], a[3] * q_[2] + a[4] * q_[5] + a[5] * q_[8],
         a[6] * q_[0] + a[7] * q_[3] + a[8] * q_[6], a[6] * q_[1] + a[7] * q_[4] + a[8] * q_[7], a[6] * q_[2] + a[7] * q_[5] + a[8] * q_[8],
         a[9] * q_[0] + a[10] * q_[3] + a[11] * q_[6], a[9] * q_[1] + a[10] * q_[4] + a[11] * q_[7],
         a[9] * q_[2] + a[10] * q_[5] + a[11] * q_[8])  # fmt: skip
    cq, sn = tl.cos(qi), tl.sin(qi)
    C = 1 - cq
    Ax = (cq + x[0] * x[0] * C, x[0] * x[1] * C - x[2] * sn, x[0] * x[2] * C + x[1] * sn,
          x[1] * x[0] * C + x[2] * sn, cq + x[1] * x[1] * C, x[1] * x[2] * C - x[0] * sn,
          x[2] * x[0] * C - x[1] * sn, x[2] * x[1] * C + x[0] * sn, cq + x[2] * x[2] * C)  # fmt: skip
    d0, d1, d2 = b0 - a[0], b1 - a[1], b2 - a[2]
    rb = _mm3(e, Ax)
    b = (b0, b1, b2, rb[0], rb[1], rb[2], rb[3], rb[4], rb[5], rb[6], rb[7], rb[8], a[12] + (a[16] * d2 - a[17] * d1),
         a[13] + (a[17] * d0 - a[15] * d2), a[14] + (a[15] * d1 - a[16] * d0),
         a[15] + (e[0] * x[0] + e[1] * x[1] + e[2] * x[2]) * qdi, a[16] + (e[3] * x[0] + e[4] * x[1] + e[5] * x[2]) * qdi,
         a[17] + (e[6] * x[0] + e[7] * x[1] + e[8] * x[2]) * qdi)  # fmt: skip
    return b, e, Ax, cq, sn


@triton.jit
def _g1_pose_at(OUT, n, N, m, i, b, TD: tl.constexpr, D_IN: tl.constexpr, POS: tl.constexpr):
    """_g1_pose for a body index known at run time (64-bit column offsets)."""
    t, rp = (TD * i).to(tl.int64) * N + n, (D_IN + 3 * i).to(tl.int64) * N + n
    for k in tl.static_range(3):
        tl.store(OUT + t + k * N, b[k], mask=m)
        tl.store(OUT + rp + k * N, b[k] / POS, mask=m)
        tl.store(OUT + t + (3 + 2 * k) * N, b[3 + 3 * k], mask=m)
        tl.store(OUT + t + (4 + 2 * k) * N, b[4 + 3 * k], mask=m)
    for k in tl.static_range(6):
        tl.store(OUT + t + (9 + k) * N, b[12 + k], mask=m)


@triton.jit
def _g1_rf_inputs_kernel(P, R, V, W, Q, QD, A, XI, OUT, ST, PAR, JNT, BODY, N, sp, sr, sv, sw, sq, sqd, sa, sx, NB: tl.constexpr,
                         NJ: tl.constexpr, TD: tl.constexpr, D_IN: tl.constexpr, POS: tl.constexpr, XOFF: tl.constexpr,
                         BLOCK: tl.constexpr):  # fmt: skip
    """_g1_rf_features_kernel's features (OUT (D_IN + 3 NB, N), feature-major) by a loop over the bodies: PAR, JNT (NB,)
    each body's parent and joint, BODY (NB, 20) its constants (``body_tables``); ST (18 NB, N) scratch, every body's
    state, written here."""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    i0 = tl.zeros([1], dtype=tl.int32)  # body 0
    st, c, s, nrm, r, v, w = _g1_root(P, R, V, W, n, m, sp, sr, sv, sw, BLOCK)
    _g1_put(ST, i0, n, N, m, st)
    _g1_pose(OUT, n, N, m, 0, st, TD, D_IN, POS)
    for k in tl.static_range(TD - 15):  # the pelvis has no joint
        _st(OUT, n, N, m, 15 + k, zero)
    for i in range(1, NB):
        par, j = tl.load(PAR + i), tl.load(JNT + i)
        o, q_, x, jc = _g1_tables(BODY, i)
        qi, qdi = tl.load(Q + n * sq + j, mask=m, other=0.0), tl.load(QD + n * sqd + j, mask=m, other=0.0)
        b, e, Ax, cq, sn = _g1_joint(_g1_state(ST, par, n, N, m), o, q_, x, qi, qdi)
        _g1_put(ST, i, n, N, m, b)
        _g1_pose_at(OUT, n, N, m, i, b, TD, D_IN, POS)
        # the body's parent joint: q, qd, setpoint, PD error q0 + offset + alpha u - q, PD torque / effort limit (clipped), offset
        u = tl.load(A + n * sa + j, mask=m, other=0.0)
        off = tl.load(XI + n * sx + 1 + j, mask=m, other=0.0)
        err = jc[0] + XOFF * off + jc[1] * u - qi
        tau = tl.minimum(tl.maximum(jc[2] * err - jc[3] * qdi, -jc[4]), jc[4]) / jc[4]
        t = (TD * i).to(tl.int64) * N + n
        tl.store(OUT + t + 15 * N, qi, mask=m)
        tl.store(OUT + t + 16 * N, qdi, mask=m)
        tl.store(OUT + t + 17 * N, u, mask=m)
        tl.store(OUT + t + 18 * N, err, mask=m)
        tl.store(OUT + t + 19 * N, tau, mask=m)
        tl.store(OUT + t + 20 * N, off, mask=m)
    # the global context: friction and the torso's centre-of-mass offset (normalized xi)
    _st(OUT, n, N, m, TD * NB, tl.load(XI + n * sx, mask=m, other=0.0))
    for k in tl.static_range(3):
        _st(OUT, n, N, m, TD * NB + 1 + k, tl.load(XI + n * sx + 1 + NJ + k, mask=m, other=0.0))


@triton.jit
def _g1_adjoint_at(G, n, N, m, i, TD: tl.constexpr, D_IN: tl.constexpr, POS: tl.constexpr, BLOCK: tl.constexpr):
    """Body i's state adjoint from the features' adjoint G (feature-major): _g1_pose backwards. The rotation's third
    column is not a feature."""
    t, rp = (TD * i).to(tl.int64) * N + n, (D_IN + 3 * i).to(tl.int64) * N + n
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    return (_ld(G, t, m, 0) + _ld(G, rp, m, 0) / POS, _ld(G, t, m, N) + _ld(G, rp, m, N) / POS, _ld(G, t, m, 2 * N) + _ld(G, rp, m, 2 * N) / POS,
            _ld(G, t, m, 3 * N), _ld(G, t, m, 4 * N), zero, _ld(G, t, m, 5 * N), _ld(G, t, m, 6 * N), zero, _ld(G, t, m, 7 * N),
            _ld(G, t, m, 8 * N), zero, _ld(G, t, m, 9 * N), _ld(G, t, m, 10 * N), _ld(G, t, m, 11 * N), _ld(G, t, m, 12 * N),
            _ld(G, t, m, 13 * N), _ld(G, t, m, 14 * N))  # fmt: skip


@triton.jit
def _g1_rf_inputs_bwd_kernel(G, P, R, V, W, Q, QD, A, XI, ST, AD, DP, DR, DV, DW, DQ, DQD, DA, PAR, JNT, BODY, N, sp, sr, sv, sw, sq, sqd, sa, sx,
                             NB: tl.constexpr, NJ: tl.constexpr, TD: tl.constexpr, D_IN: tl.constexpr, POS: tl.constexpr, XOFF: tl.constexpr,
                             BLOCK: tl.constexpr):  # fmt: skip
    """_g1_rf_inputs_kernel's reverse sweep: G (D_IN + 3 NB, N) the features' adjoint -> the adjoints of p, R, v, w (DP (N,
    3), DR (N, 9), DV, DW (N, 3)), of the joints' q, qd and of the setpoints (DQ, DQD, DA (N, NJ)); the sim context gets
    none. ST, AD (18 NB, N) scratch: every body's state (the forward recursion is run again) and what its children pass
    up to it (AD zeroed by the caller). From the last body to the first, a body's adjoint is its features' plus AD's, and
    it adds to its parent's through b = a + Ra o, Rb = Ra Q A(q), v = va + wa x (b - a) and w = wa + Ra Q x qd. A joint
    belongs to one body, so its three adjoints are complete, and stored, when its body is done; the clipped torque passes
    its adjoint on inside the limits only."""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    i0 = tl.zeros([1], dtype=tl.int32)  # body 0
    st, c, s, nrm, r, v, w = _g1_root(P, R, V, W, n, m, sp, sr, sv, sw, BLOCK)
    _g1_put(ST, i0, n, N, m, st)
    for i in range(1, NB):
        par, j = tl.load(PAR + i), tl.load(JNT + i)
        o, q_, x, jc = _g1_tables(BODY, i)
        qi, qdi = tl.load(Q + n * sq + j, mask=m, other=0.0), tl.load(QD + n * sqd + j, mask=m, other=0.0)
        b, e, Ax, cq, sn = _g1_joint(_g1_state(ST, par, n, N, m), o, q_, x, qi, qdi)
        _g1_put(ST, i, n, N, m, b)
    for k in range(1, NB):
        i = NB - k
        par, j = tl.load(PAR + i), tl.load(JNT + i)
        o, q_, x, jc = _g1_tables(BODY, i)
        qi, qdi = tl.load(Q + n * sq + j, mask=m, other=0.0), tl.load(QD + n * sqd + j, mask=m, other=0.0)
        a = _g1_state(ST, par, n, N, m)
        b, e, Ax, cq, sn = _g1_joint(a, o, q_, x, qi, qdi)
        g = _add18(_g1_adjoint_at(G, n, N, m, i, TD, D_IN, POS, BLOCK), _g1_state(AD, i, n, N, m))
        dAx = (x[0] * x[0] * sn - sn, x[0] * x[1] * sn - x[2] * cq, x[0] * x[2] * sn + x[1] * cq,
               x[1] * x[0] * sn + x[2] * cq, x[1] * x[1] * sn - sn, x[1] * x[2] * sn - x[0] * cq,
               x[2] * x[0] * sn - x[1] * cq, x[2] * x[1] * sn + x[0] * cq, x[2] * x[2] * sn - sn)  # fmt: skip
        ex0, ex1, ex2 = e[0] * x[0] + e[1] * x[1] + e[2] * x[2], e[3] * x[0] + e[4] * x[1] + e[5] * x[2], e[6] * x[0] + e[7] * x[1] + e[8] * x[2]
        d0, d1, d2 = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        # the angular velocity passes up unchanged and through the link-origin velocity's lever arm (d x its adjoint); the
        # lever arm d = Ra o gets that adjoint x wa, plus the position's adjoint
        tw0, tw1, tw2 = g[15] + (d1 * g[14] - d2 * g[13]), g[16] + (d2 * g[12] - d0 * g[14]), g[17] + (d0 * g[13] - d1 * g[12])
        dd0 = g[0] + (g[13] * a[17] - g[14] * a[16])
        dd1 = g[1] + (g[14] * a[15] - g[12] * a[17])
        dd2 = g[2] + (g[12] * a[16] - g[13] * a[15])
        # e = Ra Q gets Rb's adjoint times A^T and the angular velocity's adjoint (x) (x qd); A gets e^T times Rb's adjoint
        wq0, wq1, wq2 = g[15] * qdi, g[16] * qdi, g[17] * qdi
        eb = (g[3] * Ax[0] + g[4] * Ax[1] + g[5] * Ax[2] + wq0 * x[0], g[3] * Ax[3] + g[4] * Ax[4] + g[5] * Ax[5] + wq0 * x[1],
              g[3] * Ax[6] + g[4] * Ax[7] + g[5] * Ax[8] + wq0 * x[2],
              g[6] * Ax[0] + g[7] * Ax[1] + g[8] * Ax[2] + wq1 * x[0], g[6] * Ax[3] + g[7] * Ax[4] + g[8] * Ax[5] + wq1 * x[1],
              g[6] * Ax[6] + g[7] * Ax[7] + g[8] * Ax[8] + wq1 * x[2],
              g[9] * Ax[0] + g[10] * Ax[1] + g[11] * Ax[2] + wq2 * x[0], g[9] * Ax[3] + g[10] * Ax[4] + g[11] * Ax[5] + wq2 * x[1],
              g[9] * Ax[6] + g[10] * Ax[7] + g[11] * Ax[8] + wq2 * x[2])  # fmt: skip
        dq = ((e[0] * g[3] + e[3] * g[6] + e[6] * g[9]) * dAx[0] + (e[0] * g[4] + e[3] * g[7] + e[6] * g[10]) * dAx[1]
              + (e[0] * g[5] + e[3] * g[8] + e[6] * g[11]) * dAx[2]
              + (e[1] * g[3] + e[4] * g[6] + e[7] * g[9]) * dAx[3] + (e[1] * g[4] + e[4] * g[7] + e[7] * g[10]) * dAx[4]
              + (e[1] * g[5] + e[4] * g[8] + e[7] * g[11]) * dAx[5]
              + (e[2] * g[3] + e[5] * g[6] + e[8] * g[9]) * dAx[6] + (e[2] * g[4] + e[5] * g[7] + e[8] * g[10]) * dAx[7]
              + (e[2] * g[5] + e[5] * g[8] + e[8] * g[11]) * dAx[8])  # fmt: skip
        dqd = g[15] * ex0 + g[16] * ex1 + g[17] * ex2
        # the body's parent joint: q, qd, setpoint, PD error, clipped PD torque (the offset is the sim context's)
        u = tl.load(A + n * sa + j, mask=m, other=0.0)
        off = tl.load(XI + n * sx + 1 + j, mask=m, other=0.0)
        err = jc[0] + XOFF * off + jc[1] * u - qi
        pre = jc[2] * err - jc[3] * qdi
        t = (TD * i).to(tl.int64) * N + n
        tb = tl.where((pre > -jc[4]) & (pre < jc[4]), _ld(G, t, m, 19 * N) / jc[4], zero)
        eb_ = _ld(G, t, m, 18 * N) + jc[2] * tb
        tl.store(DQ + n * NJ + j, dq + _ld(G, t, m, 15 * N) - eb_, mask=m)
        tl.store(DQD + n * NJ + j, dqd + _ld(G, t, m, 16 * N) - jc[3] * tb, mask=m)
        tl.store(DA + n * NJ + j, _ld(G, t, m, 17 * N) + jc[1] * eb_, mask=m)
        # to the parent: its position, its rotation (e's adjoint times Q^T, and the lever arm's (x) o), its velocities
        up = (g[0], g[1], g[2],
              eb[0] * q_[0] + eb[1] * q_[1] + eb[2] * q_[2] + dd0 * o[0], eb[0] * q_[3] + eb[1] * q_[4] + eb[2] * q_[5] + dd0 * o[1],
              eb[0] * q_[6] + eb[1] * q_[7] + eb[2] * q_[8] + dd0 * o[2],
              eb[3] * q_[0] + eb[4] * q_[1] + eb[5] * q_[2] + dd1 * o[0], eb[3] * q_[3] + eb[4] * q_[4] + eb[5] * q_[5] + dd1 * o[1],
              eb[3] * q_[6] + eb[4] * q_[7] + eb[5] * q_[8] + dd1 * o[2],
              eb[6] * q_[0] + eb[7] * q_[1] + eb[8] * q_[2] + dd2 * o[0], eb[6] * q_[3] + eb[7] * q_[4] + eb[8] * q_[5] + dd2 * o[1],
              eb[6] * q_[6] + eb[7] * q_[7] + eb[8] * q_[8] + dd2 * o[2],
              g[12], g[13], g[14], tw0, tw1, tw2)  # fmt: skip
        _g1_put(AD, par, n, N, m, _add18(_g1_state(AD, par, n, N, m), up))
    # the root: the heading frame's rotation by (c, s) backwards, then the yaw's normalization
    g = _add18(_g1_adjoint_at(G, n, N, m, i0, TD, D_IN, POS, BLOCK), _g1_state(AD, i0, n, N, m))
    cb = g[3] * r[0] + g[4] * r[1] + g[5] * r[2] + g[6] * r[3] + g[7] * r[4] + g[8] * r[5] + g[12] * v[0] + g[13] * v[1] + g[15] * w[0] + g[16] * w[1]
    sb = g[3] * r[3] + g[4] * r[4] + g[5] * r[5] - g[6] * r[0] - g[7] * r[1] - g[8] * r[2] + g[12] * v[1] - g[13] * v[0] + g[15] * w[1] - g[16] * w[0]
    tl.store(DP + n * 3, zero, mask=m)
    tl.store(DP + n * 3 + 1, zero, mask=m)
    tl.store(DP + n * 3 + 2, g[2], mask=m)
    tl.store(DR + n * 9, c * g[3] - s * g[6] + tl.div_rn(cb * (1 - c * c) - sb * c * s, nrm), mask=m)
    tl.store(DR + n * 9 + 1, c * g[4] - s * g[7], mask=m)
    tl.store(DR + n * 9 + 2, c * g[5] - s * g[8], mask=m)
    tl.store(DR + n * 9 + 3, s * g[3] + c * g[6] + tl.div_rn(sb * (1 - s * s) - cb * c * s, nrm), mask=m)
    tl.store(DR + n * 9 + 4, s * g[4] + c * g[7], mask=m)
    tl.store(DR + n * 9 + 5, s * g[5] + c * g[8], mask=m)
    tl.store(DR + n * 9 + 6, g[9], mask=m)
    tl.store(DR + n * 9 + 7, g[10], mask=m)
    tl.store(DR + n * 9 + 8, g[11], mask=m)
    tl.store(DV + n * 3, c * g[12] - s * g[13], mask=m)
    tl.store(DV + n * 3 + 1, s * g[12] + c * g[13], mask=m)
    tl.store(DV + n * 3 + 2, g[14], mask=m)
    tl.store(DW + n * 3, c * g[15] - s * g[16], mask=m)
    tl.store(DW + n * 3 + 1, s * g[15] + c * g[16], mask=m)
    tl.store(DW + n * 3 + 2, g[17], mask=m)


@triton.jit
def _g1_mlp_features_kernel(P, R, V, W, OUT, N, sp, sr, sv, sw, BLOCK: tl.constexpr):
    """The computed part of invariant_features: height, tilt (6-D of Ht R), body-frame velocities (R^T v, R^T w)."""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    r00, r01, r02 = _ld(R, n * sr, m, 0), _ld(R, n * sr, m, 1), _ld(R, n * sr, m, 2)
    r10, r11, r12 = _ld(R, n * sr, m, 3), _ld(R, n * sr, m, 4), _ld(R, n * sr, m, 5)
    r20, r21, r22 = _ld(R, n * sr, m, 6), _ld(R, n * sr, m, 7), _ld(R, n * sr, m, 8)
    v0, v1, v2 = _ld(V, n * sv, m, 0), _ld(V, n * sv, m, 1), _ld(V, n * sv, m, 2)
    w0, w1, w2 = _ld(W, n * sw, m, 0), _ld(W, n * sw, m, 1), _ld(W, n * sw, m, 2)
    c, s = _yaw(r00, r10)
    _st(OUT, n, N, m, 0, _ld(P, n * sp, m, 2))
    _st(OUT, n, N, m, 1, c * r00 + s * r10)
    _st(OUT, n, N, m, 2, c * r01 + s * r11)
    _st(OUT, n, N, m, 3, c * r10 - s * r00)
    _st(OUT, n, N, m, 4, c * r11 - s * r01)
    _st(OUT, n, N, m, 5, r20)
    _st(OUT, n, N, m, 6, r21)
    _st(OUT, n, N, m, 7, r00 * v0 + r10 * v1 + r20 * v2)
    _st(OUT, n, N, m, 8, r01 * v0 + r11 * v1 + r21 * v2)
    _st(OUT, n, N, m, 9, r02 * v0 + r12 * v1 + r22 * v2)
    _st(OUT, n, N, m, 10, r00 * w0 + r10 * w1 + r20 * w2)
    _st(OUT, n, N, m, 11, r01 * w0 + r11 * w1 + r21 * w2)
    _st(OUT, n, N, m, 12, r02 * w0 + r12 * w1 + r22 * w2)


@triton.jit
def _g1_integrate_kernel(Y, P, R, V, W, Q, QD, NP, NR, NV, NW, NQ, NQD, N, sy, sp, sr, sv, sw, sq, sqd, dt, pc, ACCEL: tl.constexpr,
                         NJ: tl.constexpr, BLOCK: tl.constexpr, SEMI: tl.constexpr = 0):  # fmt: skip
    """The robot's next state from the de-normalized outputs y: accelerations in the root's frame and of the joints,
    integrated as AccelWorldModel.step (velocities gain dt a, positions dt (v + pc a), the rotation exp(dt (w + pc aw)) R),
    or frame-local increments as IncrementWorldModel.step; SEMI: the semi-implicit model's outputs as
    SemiImplicitIncrementWorldModel.integrate (velocity increments, position corrections on top of the drift
    dt ((1 - pc) v + pc v'), pc its drift weight)."""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    row = n * sy
    J0: tl.constexpr = 6 if ACCEL else 12
    qs, qds, y1, y2 = (), (), (), ()  # the joints' inputs first: one memory round trip
    for k in tl.static_range(NJ):
        qs = qs + (tl.load(Q + n * sq + k, mask=m, other=0.0),)
        qds = qds + (tl.load(QD + n * sqd + k, mask=m, other=0.0),)
        y1 = y1 + (_ld(Y, row, m, J0 + k),)
        if not ACCEL:
            y2 = y2 + (_ld(Y, row, m, J0 + NJ + k),)
    p = (_ld(P, n * sp, m, 0), _ld(P, n * sp, m, 1), _ld(P, n * sp, m, 2))
    r = (_ld(R, n * sr, m, 0), _ld(R, n * sr, m, 1), _ld(R, n * sr, m, 2), _ld(R, n * sr, m, 3), _ld(R, n * sr, m, 4), _ld(R, n * sr, m, 5),
         _ld(R, n * sr, m, 6), _ld(R, n * sr, m, 7), _ld(R, n * sr, m, 8))  # fmt: skip
    v = (_ld(V, n * sv, m, 0), _ld(V, n * sv, m, 1), _ld(V, n * sv, m, 2))
    w = (_ld(W, n * sw, m, 0), _ld(W, n * sw, m, 1), _ld(W, n * sw, m, 2))
    y = (_ld(Y, row, m, 0), _ld(Y, row, m, 1), _ld(Y, row, m, 2), _ld(Y, row, m, 3), _ld(Y, row, m, 4), _ld(Y, row, m, 5))
    if ACCEL:
        av, aw = _mv3(r, y[0], y[1], y[2]), _mv3(r, y[3], y[4], y[5])
        v1 = (v[0] + dt * av[0], v[1] + dt * av[1], v[2] + dt * av[2])
        w1 = (w[0] + dt * aw[0], w[1] + dt * aw[1], w[2] + dt * aw[2])
        p1 = (p[0] + dt * (v[0] + pc * av[0]), p[1] + dt * (v[1] + pc * av[1]), p[2] + dt * (v[2] + pc * av[2]))
        r1 = _mm3(_exp(dt * (w[0] + pc * aw[0]), dt * (w[1] + pc * aw[1]), dt * (w[2] + pc * aw[2])), r)
    elif SEMI:
        y3 = (_ld(Y, row, m, 6), _ld(Y, row, m, 7), _ld(Y, row, m, 8), _ld(Y, row, m, 9), _ld(Y, row, m, 10), _ld(Y, row, m, 11))
        c, s = _yaw(r[0], r[3])
        dv, dw = _mv3(r, y3[0], y3[1], y3[2]), _mv3(r, y3[3], y3[4], y3[5])
        v1 = (v[0] + dv[0], v[1] + dv[1], v[2] + dv[2])
        w1 = (w[0] + dw[0], w[1] + dw[1], w[2] + dw[2])
        ca, cb = dt * (1 - pc), dt * pc
        u = (ca * v[0] + cb * v1[0], ca * v[1] + cb * v1[1], ca * v[2] + cb * v1[2])  # the drift, in the world frame
        g = (c * u[0] + s * u[1], c * u[1] - s * u[0], u[2])  # in the heading frame
        e = (y[0] + g[0], y[1] + g[1], y[2] + g[2])
        p1 = (p[0] + (c * e[0] - s * e[1]), p[1] + (s * e[0] + c * e[1]), p[2] + e[2])
        uw = (ca * w[0] + cb * w1[0], ca * w[1] + cb * w1[1], ca * w[2] + cb * w1[2])
        gw = (r[0] * uw[0] + r[3] * uw[1] + r[6] * uw[2], r[1] * uw[0] + r[4] * uw[1] + r[7] * uw[2],
              r[2] * uw[0] + r[5] * uw[1] + r[8] * uw[2])  # R^T uw: in the body frame  # fmt: skip
        r1 = _mm3(r, _exp(y[3] + gw[0], y[4] + gw[1], y[5] + gw[2]))
    else:
        y3 = (_ld(Y, row, m, 6), _ld(Y, row, m, 7), _ld(Y, row, m, 8), _ld(Y, row, m, 9), _ld(Y, row, m, 10), _ld(Y, row, m, 11))
        c, s = _yaw(r[0], r[3])
        dv, dw = _mv3(r, y3[0], y3[1], y3[2]), _mv3(r, y3[3], y3[4], y3[5])
        p1 = (p[0] + (c * y[0] - s * y[1]), p[1] + (s * y[0] + c * y[1]), p[2] + y[2])
        r1 = _mm3(r, _exp(y[3], y[4], y[5]))
        v1 = (v[0] + dv[0], v[1] + dv[1], v[2] + dv[2])
        w1 = (w[0] + dw[0], w[1] + dw[1], w[2] + dw[2])
    for k in tl.static_range(3):
        tl.store(NP + n * 3 + k, p1[k], mask=m)
        tl.store(NV + n * 3 + k, v1[k], mask=m)
        tl.store(NW + n * 3 + k, w1[k], mask=m)
    for k in tl.static_range(9):
        tl.store(NR + n * 9 + k, r1[k], mask=m)
    for k in tl.static_range(NJ):
        if ACCEL:
            q1 = qs[k] + dt * (qds[k] + pc * y1[k])
            qd1 = qds[k] + dt * y1[k]
        elif SEMI:
            qd1 = qds[k] + y2[k]
            q1 = qs[k] + (y1[k] + (dt * (1 - pc) * qds[k] + dt * pc * qd1))
        else:
            q1 = qs[k] + y1[k]
            qd1 = qds[k] + y2[k]
        tl.store(NQ + n * NJ + k, q1, mask=m)
        tl.store(NQD + n * NJ + k, qd1, mask=m)


@torch.library.custom_op("ndm_gym::g1_rf_features", mutates_args=())
def rf_features(p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, a: torch.Tensor,
                xi_n: torch.Tensor, tree: int, q0: list[float], alpha: list[float], kp: list[float], kd: list[float], tau_max: list[float],
                d_in: int) -> torch.Tensor:  # fmt: skip
    """RigidFormerNet.inputs' raw features (d_in of them: the bodies' tokens, then the global context), then the bodies'
    positions / POS_SCALE, feature-major: (d_in + 3 B, N). tree: a key of ``hoi_fused.kinematic_tree``; q0, alpha, kp,
    kd, tau_max: each joint's default pose, action scale, PD gains and effort limit (G1 order)."""
    N = p.shape[0]
    parent, joint, offp, offr, axis = _TREES[tree]
    nb = len(parent)
    if d_in != nb * TOKEN_DIM + GLOBAL_DIM:
        raise ValueError(f"g1_fused.rf_features: {nb} bodies of {TOKEN_DIM} features and {GLOBAL_DIM} context scalars, not {d_in} features")
    out = torch.empty(d_in + 3 * nb, N, device=p.device, dtype=torch.float32)
    st = [_rows(p, 3), _rows(R, 9), _rows(v, 3), _rows(w, 3), _rows(q, NUM_J), _rows(qd, NUM_J), _rows(a, NUM_J), _rows(xi_n, xi_n.shape[1])]
    _g1_rf_features_kernel[(triton.cdiv(N, BLOCK),)](p, R, v, w, q, qd, a, xi_n, out, N, *st, PARENT=parent, JOINT=joint, OFFP=offp, OFFR=offr,
                                                     AXIS=axis, Q0=tuple(q0), ALPHA=tuple(alpha), KP=tuple(kp), KD=tuple(kd), TMAX=tuple(tau_max),
                                                     NB=nb, NJ=NUM_J, TD=TOKEN_DIM, D_IN=d_in, POS=POS_SCALE, XOFF=XI_OFFSET_SCALE, BLOCK=BLOCK,
                                                     num_warps=WARPS)  # fmt: skip
    return out


@rf_features.register_fake
def _(p, R, v, w, q, qd, a, xi_n, tree, q0, alpha, kp, kd, tau_max, d_in):
    return p.new_empty(d_in + 3 * len(_TREES[tree][0]), p.shape[0])


def body_tables(kin: TreeKinematics, q0: torch.Tensor, alpha: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor, tau_max: torch.Tensor,
                device: torch.device) -> list[torch.Tensor]:
    """The kinematic tree and the joints' constants as rf_inputs' kernels read them: PAR, JNT (B,) int32 each body's parent
    and joint (the pelvis: 0, unused), BODY (B, 20) fp32 its offset in the parent's frame, that frame's rotation, its
    joint's axis and the joint's default pose, action scale, PD gains and effort limit (q0 ... tau_max in G1 joint order)."""
    nb = len(kin.parent)
    jnt = torch.tensor([max(j, 0) for j in kin.joint_idx], dtype=torch.int32)
    par = torch.tensor([max(k, 0) for k in kin.parent], dtype=torch.int32)
    joint = torch.stack([torch.as_tensor(t, dtype=torch.float32).cpu() for t in (q0, alpha, kp, kd, tau_max)], -1)[jnt.long()]
    joint[0, 4] = 1.0  # the pelvis has no joint: its row is not read
    body = torch.cat((kin.off_p.reshape(nb, 3).float().cpu(), kin.off_R.reshape(nb, 9).float().cpu(), kin.axes.reshape(nb, 3).float().cpu(),
                      joint), -1)  # fmt: skip
    return [par.to(device), jnt.to(device), body.contiguous().to(device)]


def _inputs_strides(p, R, v, w, q, qd, a, xi_n):
    return [_rows(p, 3), _rows(R, 9), _rows(v, 3), _rows(w, 3), _rows(q, NUM_J), _rows(qd, NUM_J), _rows(a, NUM_J), _rows(xi_n, xi_n.shape[1])]


@torch.library.custom_op("ndm_gym::g1_rf_inputs", mutates_args=())
def rf_inputs(p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, a: torch.Tensor,
              xi_n: torch.Tensor, tables: list[torch.Tensor], d_in: int) -> torch.Tensor:  # fmt: skip
    """rf_features' output ((d_in + 3 B, N), feature-major) by the looped kernel, differentiable in p, R, v, w, q, qd and
    a (its backward pass: rf_inputs_bwd); tables: ``body_tables``."""
    N, nb = p.shape[0], tables[0].shape[0]
    if d_in != nb * TOKEN_DIM + GLOBAL_DIM:
        raise ValueError(f"g1_fused.rf_inputs: {nb} bodies of {TOKEN_DIM} features and {GLOBAL_DIM} context scalars, not {d_in} features")
    out = torch.empty(d_in + 3 * nb, N, device=p.device, dtype=torch.float32)
    scratch = torch.empty(nb * SD, N, device=p.device, dtype=torch.float32)
    _g1_rf_inputs_kernel[(triton.cdiv(N, BLOCK),)](p, R, v, w, q, qd, a, xi_n, out, scratch, *tables, N, *_inputs_strides(p, R, v, w, q, qd, a, xi_n),
                                                   NB=nb, NJ=NUM_J, TD=TOKEN_DIM, D_IN=d_in, POS=POS_SCALE, XOFF=XI_OFFSET_SCALE, BLOCK=BLOCK,
                                                   num_warps=WARPS)  # fmt: skip
    return out


@rf_inputs.register_fake
def _(p, R, v, w, q, qd, a, xi_n, tables, d_in):
    return p.new_empty(d_in + 3 * tables[0].shape[0], p.shape[0])


@torch.library.custom_op("ndm_gym::g1_rf_inputs_bwd", mutates_args=())
def rf_inputs_bwd(g: torch.Tensor, p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, q: torch.Tensor, qd: torch.Tensor,
                  a: torch.Tensor, xi_n: torch.Tensor, tables: list[torch.Tensor], d_in: int) -> list[torch.Tensor]:  # fmt: skip
    """rf_inputs' backward pass (``_g1_rf_inputs_bwd_kernel``): g (d_in + 3 B, N) fp32, the adjoint of its output -> the
    adjoints of p, R, v, w, q, qd and a (the sim context gets none)."""
    N, nb = p.shape[0], tables[0].shape[0]
    if g.shape != (d_in + 3 * nb, N) or g.dtype != torch.float32:
        raise ValueError(f"g1_fused.rf_inputs_bwd: the adjoint must be fp32 ({d_in + 3 * nb}, {N}), got {g.dtype} {tuple(g.shape)}")
    out = [torch.empty(N, k, device=p.device, dtype=torch.float32) for k in (3, 9, 3, 3, NUM_J, NUM_J, NUM_J)]
    states = torch.empty(nb * SD, N, device=p.device, dtype=torch.float32)
    passed = torch.zeros(nb * SD, N, device=p.device, dtype=torch.float32)
    _g1_rf_inputs_bwd_kernel[(triton.cdiv(N, BLOCK),)](g.contiguous(), p, R, v, w, q, qd, a, xi_n, states, passed, *out, *tables, N,
                                                       *_inputs_strides(p, R, v, w, q, qd, a, xi_n), NB=nb, NJ=NUM_J, TD=TOKEN_DIM, D_IN=d_in,
                                                       POS=POS_SCALE, XOFF=XI_OFFSET_SCALE, BLOCK=BLOCK, num_warps=WARPS)  # fmt: skip
    out[1] = out[1].view(R.shape)
    return out


@rf_inputs_bwd.register_fake
def _(g, p, R, v, w, q, qd, a, xi_n, tables, d_in):
    return [t.new_empty(t.shape, dtype=torch.float32) for t in (p, R, v, w, q, qd, a)]


def _rf_inputs_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:8], *inputs[8])
    ctx.d_in = inputs[9]


def _rf_inputs_backward(ctx, g):
    saved = ctx.saved_tensors
    grads = rf_inputs_bwd(g, *saved[:8], list(saved[8:]), ctx.d_in)
    return (*grads, None, [None] * 3, None)  # none for the sim context, the tables and d_in


rf_inputs.register_autograd(_rf_inputs_backward, setup_context=_rf_inputs_setup)


@torch.library.custom_op("ndm_gym::g1_mlp_features", mutates_args=())
def mlp_features(p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The computed part of invariant_features (height, tilt, body-frame velocities), feature-major: (13, N)."""
    N = p.shape[0]
    out = torch.empty(13, N, device=p.device, dtype=torch.float32)
    _g1_mlp_features_kernel[(triton.cdiv(N, BLOCK),)](p, R, v, w, out, N, _rows(p, 3), _rows(R, 9), _rows(v, 3), _rows(w, 3), BLOCK=BLOCK,
                                                      num_warps=WARPS)  # fmt: skip
    return out


@mlp_features.register_fake
def _(p, R, v, w):
    return p.new_empty(13, p.shape[0])


@torch.library.custom_op("ndm_gym::g1_integrate", mutates_args=())
def integrate(y: torch.Tensor, p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, q: torch.Tensor, qd: torch.Tensor,
              accel: bool, pos_coef: float, semi: bool = False) -> list[torch.Tensor]:  # fmt: skip
    """y (N, 6 + J accelerations, or 12 + 2 J increments) the de-normalized fp32 outputs -> the robot's next
    [p, R, v, w, q, qd]; pos_coef: AccelWorldModel's position coefficient (s), unused for increments; semi: the increments
    are SemiImplicitIncrementWorldModel's (velocity increments, position corrections), pos_coef its drift weight."""
    if semi and accel:
        raise ValueError("g1_fused.integrate: semi takes increments, not accelerations")
    N = p.shape[0]
    width = (6 if accel else 12) + (1 if accel else 2) * NUM_J
    if y.stride(-1) != 1 or y.dtype != torch.float32 or y.shape[-1] != width:
        raise ValueError(f"g1_fused.integrate: y needs fp32, a unit last stride and {width} columns, got {y.dtype} {tuple(y.shape)}")
    nxt = [torch.empty(t.shape, device=t.device, dtype=t.dtype) for t in (p, R, v, w, q, qd)]
    st = [y.stride(0), _rows(p, 3), _rows(R, 9), _rows(v, 3), _rows(w, 3), _rows(q, NUM_J), _rows(qd, NUM_J)]
    if semi:
        _g1_integrate_kernel[(triton.cdiv(N, BLOCK),)](y, p, R, v, w, q, qd, *nxt, N, *st, CONTROL_DT, pos_coef, ACCEL=0, NJ=NUM_J, BLOCK=BLOCK,
                                                       SEMI=1, num_warps=WARPS)  # fmt: skip
        return nxt
    _g1_integrate_kernel[(triton.cdiv(N, BLOCK),)](y, p, R, v, w, q, qd, *nxt, N, *st, CONTROL_DT, pos_coef, ACCEL=int(accel), NJ=NUM_J, BLOCK=BLOCK,
                                                   num_warps=WARPS)  # fmt: skip
    return nxt


@integrate.register_fake
def _(y, p, R, v, w, q, qd, accel, pos_coef, semi=False):
    return [t.new_empty(t.shape) for t in (p, R, v, w, q, qd)]
