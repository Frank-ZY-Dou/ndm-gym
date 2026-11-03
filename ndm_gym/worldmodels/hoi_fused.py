"""The HOI NDM step outside its network as two Triton kernels, for inference (``HOIWorldModel.inference_copy(fused=True)``).

``rf_features`` / ``mlp_features``: the network's raw input features from the robot's and the object's state, the
actions and the context in one kernel: for the RigidFormer the forward kinematics of the 30 bodies, unrolled in registers
with the kinematic tree as compile-time constants and run directly in the robot's heading frame, the token features and
the rotary positions; for the MLP the invariant features and the object's and poses' features. ``integrate``: the
integration of the robot and the object from the de-normalized outputs, both parameterizations, and the feet's forces
turned from the root's heading frame to the world frame, in one kernel. Torch does the rest of ``inputs`` / ``in_norm``
(copies, normalization, which it fuses with the network's first cast) and the output de-normalization (fused with the
network's last cast). Compiled by torch.compile, the same computations take about 110 (RigidFormer) and 20 kernels per
step. Each program handles a block of samples, one per lane; the features are written feature-major ((D, N), coalesced)
and read transposed by the fused normalization. The math is that of ``HOIWorldModel.inputs`` and ``step``
(``_robot_next``, ``_object_next``, ``_forces``) in fp32, equal up to float rounding. Forward pass only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..robots.g1 import NUM_J
from ..state import CONTROL_DT
from ..tasks.hdmi import DECIMATION, NUM_ACTIONS
from .rigidformer import POS_SCALE

BLOCK, WARPS = 64, 2  # samples per program (one per lane) and warps: many small programs hide the memory latency


@triton.jit
def _ld(ptr, row, m, k):
    return tl.load(ptr + row + k, mask=m, other=0.0)


@triton.jit
def _st(out, n, N, m, col, val):
    """Feature ``col`` of the samples n, stored feature-major (out is (D, N))."""
    tl.store(out + col * N + n, val, mask=m)


@triton.jit
def _rel6(out, n, N, m, col, c, s, x0, x1, x2, r00, r01, r10, r11, r20, r21):
    """Heading-frame position (x - origin, rotated by the yaw) and 6-D rotation (first two columns of Ht R)."""
    _st(out, n, N, m, col + 0, c * x0 + s * x1)
    _st(out, n, N, m, col + 1, c * x1 - s * x0)
    _st(out, n, N, m, col + 2, x2)
    _st(out, n, N, m, col + 3, c * r00 + s * r10)
    _st(out, n, N, m, col + 4, c * r01 + s * r11)
    _st(out, n, N, m, col + 5, c * r10 - s * r00)
    _st(out, n, N, m, col + 6, c * r11 - s * r01)
    _st(out, n, N, m, col + 7, r20)
    _st(out, n, N, m, col + 8, r21)


@triton.jit
def _pose(out, n, N, m, i: tl.constexpr, b, D_IN: tl.constexpr, POS: tl.constexpr):
    """Body i's position, 6-D rotation and velocities (token columns 26 i ...) and its rotary position (D_IN + 3 i)."""
    for k in tl.static_range(3):
        _st(out, n, N, m, 26 * i + k, b[k])
        _st(out, n, N, m, D_IN + 3 * i + k, b[k] / POS)
        _st(out, n, N, m, 26 * i + 3 + 2 * k, b[3 + 3 * k])
        _st(out, n, N, m, 26 * i + 4 + 2 * k, b[4 + 3 * k])
    for k in tl.static_range(6):
        _st(out, n, N, m, 26 * i + 9 + k, b[12 + k])


@triton.jit
def _joint_features(out, U, CTX, n, N, m, su, sc, qs, JOINT: tl.constexpr, Q0: tl.constexpr, GAIN: tl.constexpr, AIDX: tl.constexpr,
                    STARTUP: tl.constexpr, NA: tl.constexpr, SUB: tl.constexpr, LO: tl.constexpr, HI: tl.constexpr):  # fmt: skip
    """Bodies LO..HI-1: their joint's offset (token column 17), the substeps' PD-target offsets d (18...) and PD errors
    q0 + offset + d - q (18 + SUB ...); the loads first."""
    offs, ds = (), ()
    for i in tl.static_range(LO, HI):
        offs = offs + (tl.load(CTX + n * sc + STARTUP + JOINT[i], mask=m, other=0.0),)
        for t in tl.static_range(SUB):
            ds = ds + (tl.load(U + n * su + t * NA + AIDX[JOINT[i]], mask=m, other=0.0) * GAIN[JOINT[i]],)
    for i in tl.static_range(LO, HI):
        _st(out, n, N, m, 26 * i + 17, offs[i - LO])
        for t in tl.static_range(SUB):
            _st(out, n, N, m, 26 * i + 18 + t, ds[(i - LO) * SUB + t])
            _st(out, n, N, m, 26 * i + 18 + SUB + t, Q0[JOINT[i]] + offs[i - LO] + ds[(i - LO) * SUB + t] - qs[i])


@triton.jit
def _yaw(r00, r10):
    """cos and sin of the heading of a rotation with first column (r00, r10, .), as yaw_mat."""
    nrm = tl.sqrt_rn(r00 * r00 + r10 * r10 + 1e-9)
    return tl.div_rn(r00, nrm), tl.div_rn(r10, nrm)


@triton.jit
def _rf_features_kernel(P, R, V, W, Q, QD, OP, OR, OV, OW, U, CTX, OUT, N, sp, sr, sv, sw, sq, sqd, sop, sor, sov, sow, su, sc,
                        PARENT: tl.constexpr, JOINT: tl.constexpr, OFFP: tl.constexpr, OFFR: tl.constexpr, AXIS: tl.constexpr,
                        Q0: tl.constexpr, GAIN: tl.constexpr, AIDX: tl.constexpr, NB: tl.constexpr, STARTUP: tl.constexpr, POSES: tl.constexpr,
                        HINGED: tl.constexpr, D_IN: tl.constexpr, NA: tl.constexpr, SUB: tl.constexpr, NJ: tl.constexpr, POS: tl.constexpr,
                        BLOCK: tl.constexpr):  # fmt: skip
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    p0, p1, p2 = _ld(P, n * sp, m, 0), _ld(P, n * sp, m, 1), _ld(P, n * sp, m, 2)
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
    _pose(OUT, n, N, m, 0, states[0], D_IN, POS)
    for k in tl.static_range(11):
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
        A = (cq + x[0] * x[0] * C, x[0] * x[1] * C - x[2] * sn, x[0] * x[2] * C + x[1] * sn,
             x[1] * x[0] * C + x[2] * sn, cq + x[1] * x[1] * C, x[1] * x[2] * C - x[0] * sn,
             x[2] * x[0] * C - x[1] * sn, x[2] * x[1] * C + x[0] * sn, cq + x[2] * x[2] * C)  # fmt: skip
        nw = (a[15] + (e[0] * x[0] + e[1] * x[1] + e[2] * x[2]) * qds[i], a[16] + (e[3] * x[0] + e[4] * x[1] + e[5] * x[2]) * qds[i],
              a[17] + (e[6] * x[0] + e[7] * x[1] + e[8] * x[2]) * qds[i])  # fmt: skip
        d0, d1, d2 = b0 - a[0], b1 - a[1], b2 - a[2]
        states = states + ((b0, b1, b2,
                            e[0] * A[0] + e[1] * A[3] + e[2] * A[6], e[0] * A[1] + e[1] * A[4] + e[2] * A[7], e[0] * A[2] + e[1] * A[5] + e[2] * A[8],
                            e[3] * A[0] + e[4] * A[3] + e[5] * A[6], e[3] * A[1] + e[4] * A[4] + e[5] * A[7], e[3] * A[2] + e[4] * A[5] + e[5] * A[8],
                            e[6] * A[0] + e[7] * A[3] + e[8] * A[6], e[6] * A[1] + e[7] * A[4] + e[8] * A[7], e[6] * A[2] + e[7] * A[5] + e[8] * A[8],
                            a[12] + (a[16] * d2 - a[17] * d1), a[13] + (a[17] * d0 - a[15] * d2), a[14] + (a[15] * d1 - a[16] * d0),
                            nw[0], nw[1], nw[2]),)  # fmt: skip
        _pose(OUT, n, N, m, i, states[i], D_IN, POS)
        _st(OUT, n, N, m, 26 * i + 15, qs[i])
        _st(OUT, n, N, m, 26 * i + 16, qds[i])
    # the joints' offsets, the substeps' PD-target offsets d and PD errors q0 + offset + d - q, in two halves whose
    # loads come first
    _joint_features(OUT, U, CTX, n, N, m, su, sc, qs, JOINT, Q0, GAIN, AIDX, STARTUP, NA, SUB, 1, 1 + (NB - 1) // 2)
    _joint_features(OUT, U, CTX, n, N, m, su, sc, qs, JOINT, Q0, GAIN, AIDX, STARTUP, NA, SUB, 1 + (NB - 1) // 2, NB)
    # the object's token (a hinge's: its base's pose, angle, rate, friction and damping), then the props'
    col = 26 * NB
    if HINGED:
        i0 = STARTUP + NJ + 4
        bx = tl.load(CTX + n * sc + i0, mask=m, other=0.0) - p0
        by = tl.load(CTX + n * sc + i0 + 1, mask=m, other=0.0) - p1
        bz = tl.load(CTX + n * sc + i0 + 2, mask=m, other=0.0)
        _rel6(OUT, n, N, m, col, c, s, bx, by, bz, tl.load(CTX + n * sc + i0 + 3, mask=m, other=0.0),
              tl.load(CTX + n * sc + i0 + 4, mask=m, other=0.0), tl.load(CTX + n * sc + i0 + 6, mask=m, other=0.0),
              tl.load(CTX + n * sc + i0 + 7, mask=m, other=0.0), tl.load(CTX + n * sc + i0 + 9, mask=m, other=0.0),
              tl.load(CTX + n * sc + i0 + 10, mask=m, other=0.0))  # fmt: skip
        _st(OUT, n, N, m, col + 9, tl.load(OP + n * sop, mask=m, other=0.0))
        _st(OUT, n, N, m, col + 10, tl.load(OR + n * sor, mask=m, other=0.0))
        _st(OUT, n, N, m, col + 11, tl.load(CTX + n * sc + STARTUP + NJ + 2, mask=m, other=0.0))
        _st(OUT, n, N, m, col + 12, tl.load(CTX + n * sc + STARTUP + NJ + 3, mask=m, other=0.0))
        _st(OUT, n, N, m, D_IN + 3 * NB + 0, (c * bx + s * by) / POS)
        _st(OUT, n, N, m, D_IN + 3 * NB + 1, (c * by - s * bx) / POS)
        _st(OUT, n, N, m, D_IN + 3 * NB + 2, bz / POS)
        col += 13
    else:
        ox, oy, oz = _ld(OP, n * sop, m, 0) - p0, _ld(OP, n * sop, m, 1) - p1, _ld(OP, n * sop, m, 2)
        _rel6(OUT, n, N, m, col, c, s, ox, oy, oz, _ld(OR, n * sor, m, 0), _ld(OR, n * sor, m, 1), _ld(OR, n * sor, m, 3), _ld(OR, n * sor, m, 4),
              _ld(OR, n * sor, m, 6), _ld(OR, n * sor, m, 7))  # fmt: skip
        ov0, ov1, ov2 = _ld(OV, n * sov, m, 0), _ld(OV, n * sov, m, 1), _ld(OV, n * sov, m, 2)
        ow0, ow1, ow2 = _ld(OW, n * sow, m, 0), _ld(OW, n * sow, m, 1), _ld(OW, n * sow, m, 2)
        _st(OUT, n, N, m, col + 9, c * ov0 + s * ov1)
        _st(OUT, n, N, m, col + 10, c * ov1 - s * ov0)
        _st(OUT, n, N, m, col + 11, ov2)
        _st(OUT, n, N, m, col + 12, c * ow0 + s * ow1)
        _st(OUT, n, N, m, col + 13, c * ow1 - s * ow0)
        _st(OUT, n, N, m, col + 14, ow2)
        _st(OUT, n, N, m, D_IN + 3 * NB + 0, (c * ox + s * oy) / POS)
        _st(OUT, n, N, m, D_IN + 3 * NB + 1, (c * oy - s * ox) / POS)
        _st(OUT, n, N, m, D_IN + 3 * NB + 2, oz / POS)
        col += 15
    for k in tl.static_range(HINGED, POSES):
        i0 = STARTUP + NJ + 4 + 12 * k
        x0 = tl.load(CTX + n * sc + i0, mask=m, other=0.0) - p0
        x1 = tl.load(CTX + n * sc + i0 + 1, mask=m, other=0.0) - p1
        x2 = tl.load(CTX + n * sc + i0 + 2, mask=m, other=0.0)
        _rel6(OUT, n, N, m, col, c, s, x0, x1, x2, tl.load(CTX + n * sc + i0 + 3, mask=m, other=0.0),
              tl.load(CTX + n * sc + i0 + 4, mask=m, other=0.0), tl.load(CTX + n * sc + i0 + 6, mask=m, other=0.0),
              tl.load(CTX + n * sc + i0 + 7, mask=m, other=0.0), tl.load(CTX + n * sc + i0 + 9, mask=m, other=0.0),
              tl.load(CTX + n * sc + i0 + 10, mask=m, other=0.0))  # fmt: skip
        tok = NB + 1 + k - HINGED
        _st(OUT, n, N, m, D_IN + 3 * tok + 0, (c * x0 + s * x1) / POS)
        _st(OUT, n, N, m, D_IN + 3 * tok + 1, (c * x1 - s * x0) / POS)
        _st(OUT, n, N, m, D_IN + 3 * tok + 2, x2 / POS)
        col += 9


@triton.jit
def _mlp_features_kernel(P, R, V, W, OP, OR, OV, OW, CTX, OUT, N, sp, sr, sv, sw, sop, sor, sov, sow, sc, STARTUP: tl.constexpr, POSES: tl.constexpr,
                         HINGED: tl.constexpr, NJ: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    """The MLP's computed features: height, tilt (6-D of Ht R), body-frame velocities (R^T v, R^T w), then a free
    object's pose and velocities and every pose (a fixed base's too) relative to the root, in its heading frame."""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    p0, p1, p2 = _ld(P, n * sp, m, 0), _ld(P, n * sp, m, 1), _ld(P, n * sp, m, 2)
    r00, r01, r02 = _ld(R, n * sr, m, 0), _ld(R, n * sr, m, 1), _ld(R, n * sr, m, 2)
    r10, r11, r12 = _ld(R, n * sr, m, 3), _ld(R, n * sr, m, 4), _ld(R, n * sr, m, 5)
    r20, r21, r22 = _ld(R, n * sr, m, 6), _ld(R, n * sr, m, 7), _ld(R, n * sr, m, 8)
    v0, v1, v2 = _ld(V, n * sv, m, 0), _ld(V, n * sv, m, 1), _ld(V, n * sv, m, 2)
    w0, w1, w2 = _ld(W, n * sw, m, 0), _ld(W, n * sw, m, 1), _ld(W, n * sw, m, 2)
    c, s = _yaw(r00, r10)
    _st(OUT, n, N, m, 0, p2)
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
    col = 13
    if not HINGED:
        _rel6(OUT, n, N, m, col, c, s, _ld(OP, n * sop, m, 0) - p0, _ld(OP, n * sop, m, 1) - p1, _ld(OP, n * sop, m, 2) - p2, _ld(OR, n * sor, m, 0),
              _ld(OR, n * sor, m, 1), _ld(OR, n * sor, m, 3), _ld(OR, n * sor, m, 4), _ld(OR, n * sor, m, 6), _ld(OR, n * sor, m, 7))  # fmt: skip
        ov0, ov1, ov2 = _ld(OV, n * sov, m, 0), _ld(OV, n * sov, m, 1), _ld(OV, n * sov, m, 2)
        ow0, ow1, ow2 = _ld(OW, n * sow, m, 0), _ld(OW, n * sow, m, 1), _ld(OW, n * sow, m, 2)
        _st(OUT, n, N, m, col + 9, c * ov0 + s * ov1)
        _st(OUT, n, N, m, col + 10, c * ov1 - s * ov0)
        _st(OUT, n, N, m, col + 11, ov2)
        _st(OUT, n, N, m, col + 12, c * ow0 + s * ow1)
        _st(OUT, n, N, m, col + 13, c * ow1 - s * ow0)
        _st(OUT, n, N, m, col + 14, ow2)
        col += 15
    for k in tl.static_range(POSES):
        i0 = STARTUP + NJ + 4 + 12 * k
        _rel6(OUT, n, N, m, col, c, s, tl.load(CTX + n * sc + i0, mask=m, other=0.0) - p0, tl.load(CTX + n * sc + i0 + 1, mask=m, other=0.0) - p1,
              tl.load(CTX + n * sc + i0 + 2, mask=m, other=0.0) - p2, tl.load(CTX + n * sc + i0 + 3, mask=m, other=0.0),
              tl.load(CTX + n * sc + i0 + 4, mask=m, other=0.0), tl.load(CTX + n * sc + i0 + 6, mask=m, other=0.0),
              tl.load(CTX + n * sc + i0 + 7, mask=m, other=0.0), tl.load(CTX + n * sc + i0 + 9, mask=m, other=0.0),
              tl.load(CTX + n * sc + i0 + 10, mask=m, other=0.0))  # fmt: skip
        col += 9


@triton.jit
def _exp(x, y, z):
    """so3_exp(elementwise=True): I + a K + b K K of the rotation vector (x, y, z), row-major."""
    t2 = x * x + y * y + z * z
    th = tl.sqrt_rn(t2 + 1e-12)
    small = t2 < 1e-8
    a = tl.where(small, 1 - t2 / 6, tl.sin(th) / th)
    b = tl.where(small, 0.5 - t2 / 24, (1 - tl.cos(th)) / (t2 + 1e-12))
    return (1 + b * (-z * z - y * y), -a * z + b * (y * x), a * y + b * (z * x),
            a * z + b * (x * y), 1 + b * (-z * z - x * x), -a * x + b * (z * y),
            -a * y + b * (x * z), a * x + b * (y * z), 1 + b * (-y * y - x * x))  # fmt: skip


@triton.jit
def _mm3(a, b):
    """Product of two row-major 3 x 3 matrices given as 9-tuples."""
    return (a[0] * b[0] + a[1] * b[3] + a[2] * b[6], a[0] * b[1] + a[1] * b[4] + a[2] * b[7], a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
            a[3] * b[0] + a[4] * b[3] + a[5] * b[6], a[3] * b[1] + a[4] * b[4] + a[5] * b[7], a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
            a[6] * b[0] + a[7] * b[3] + a[8] * b[6], a[6] * b[1] + a[7] * b[4] + a[8] * b[7], a[6] * b[2] + a[7] * b[5] + a[8] * b[8])  # fmt: skip


@triton.jit
def _mv3(r, x0, x1, x2):
    return (r[0] * x0 + r[1] * x1 + r[2] * x2, r[3] * x0 + r[4] * x1 + r[5] * x2, r[6] * x0 + r[7] * x1 + r[8] * x2)


@triton.jit
def _body_next(Y, row, m, k0: tl.constexpr, P, R, V, W, n, sp, sr, sv, sw, NP, NR, NV, NW, dt, ACCEL: tl.constexpr):
    """A free body's next (p, R, v, w) from the de-normalized outputs Y[row + k0 ...]: accelerations in its frame
    (semi-implicit Euler, AccelWorldModel.step with one substep) or increments (IncrementWorldModel.step)."""
    p = (_ld(P, n * sp, m, 0), _ld(P, n * sp, m, 1), _ld(P, n * sp, m, 2))
    r = (_ld(R, n * sr, m, 0), _ld(R, n * sr, m, 1), _ld(R, n * sr, m, 2), _ld(R, n * sr, m, 3), _ld(R, n * sr, m, 4), _ld(R, n * sr, m, 5),
         _ld(R, n * sr, m, 6), _ld(R, n * sr, m, 7), _ld(R, n * sr, m, 8))  # fmt: skip
    v = (_ld(V, n * sv, m, 0), _ld(V, n * sv, m, 1), _ld(V, n * sv, m, 2))
    w = (_ld(W, n * sw, m, 0), _ld(W, n * sw, m, 1), _ld(W, n * sw, m, 2))
    y = (_ld(Y, row, m, k0 + 0), _ld(Y, row, m, k0 + 1), _ld(Y, row, m, k0 + 2),
         _ld(Y, row, m, k0 + 3), _ld(Y, row, m, k0 + 4), _ld(Y, row, m, k0 + 5))  # fmt: skip
    if ACCEL:
        av, aw = _mv3(r, y[0], y[1], y[2]), _mv3(r, y[3], y[4], y[5])
        v1 = (v[0] + dt * av[0], v[1] + dt * av[1], v[2] + dt * av[2])
        w1 = (w[0] + dt * aw[0], w[1] + dt * aw[1], w[2] + dt * aw[2])
        p1 = (p[0] + dt * v1[0], p[1] + dt * v1[1], p[2] + dt * v1[2])
        r1 = _mm3(_exp(dt * w1[0], dt * w1[1], dt * w1[2]), r)
    else:
        y2 = (_ld(Y, row, m, k0 + 6), _ld(Y, row, m, k0 + 7), _ld(Y, row, m, k0 + 8), _ld(Y, row, m, k0 + 9), _ld(Y, row, m, k0 + 10),
              _ld(Y, row, m, k0 + 11))  # fmt: skip
        c, s = _yaw(r[0], r[3])
        dv, dw = _mv3(r, y2[0], y2[1], y2[2]), _mv3(r, y2[3], y2[4], y2[5])
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


@triton.jit
def _integrate_kernel(Y, P, R, V, W, Q, QD, OP, OR, OV, OW, NP, NR, NV, NW, NQ, NQD, NOP, NOR, NOV, NOW, NF, N, sy, sp, sr, sv, sw, sq, sqd,
                      sop, sor, sov, sow, dt, ACCEL: tl.constexpr, HINGED: tl.constexpr, NJ: tl.constexpr, ROBOT: tl.constexpr,
                      FEET: tl.constexpr, SUB: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
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
    _body_next(Y, row, m, 0, P, R, V, W, n, sp, sr, sv, sw, NP, NR, NV, NW, dt, ACCEL)
    for k in tl.static_range(NJ):
        if ACCEL:
            qd1 = qds[k] + dt * y1[k]
            q1 = qs[k] + dt * qd1
        else:
            q1 = qs[k] + y1[k]
            qd1 = qds[k] + y2[k]
        tl.store(NQ + n * NJ + k, q1, mask=m)
        tl.store(NQD + n * NJ + k, qd1, mask=m)
    if HINGED:
        a, rate = tl.load(OP + n * sop, mask=m, other=0.0), tl.load(OR + n * sor, mask=m, other=0.0)
        if ACCEL:
            rate1 = rate + dt * _ld(Y, row, m, ROBOT)
            a1 = a + dt * rate1
        else:
            a1 = a + _ld(Y, row, m, ROBOT)
            rate1 = rate + _ld(Y, row, m, ROBOT + 1)
        tl.store(NOP + n, a1, mask=m)
        tl.store(NOR + n, rate1, mask=m)
    else:
        _body_next(Y, row, m, ROBOT, OP, OR, OV, OW, n, sop, sor, sov, sow, NOP, NOR, NOV, NOW, dt, ACCEL)
    # the feet's forces (substep, foot, axis), predicted in the heading frame of the robot's root, in the world frame: H f
    c, s = _yaw(_ld(R, n * sr, m, 0), _ld(R, n * sr, m, 3))
    for k in tl.static_range(2 * SUB):
        fx, fy = _ld(Y, row, m, FEET + 3 * k), _ld(Y, row, m, FEET + 3 * k + 1)
        tl.store(NF + n * (6 * SUB) + 3 * k, c * fx - s * fy, mask=m)
        tl.store(NF + n * (6 * SUB) + 3 * k + 1, s * fx + c * fy, mask=m)
        tl.store(NF + n * (6 * SUB) + 3 * k + 2, _ld(Y, row, m, FEET + 3 * k + 2), mask=m)


def _rows(t: torch.Tensor, width: int) -> int:
    """Row stride of an (N, ...) tensor whose trailing dimensions hold ``width`` contiguous values; raises otherwise."""
    if t.shape[0] and (t[0].numel() != width or not t[0].is_contiguous()):
        raise ValueError(f"hoi_fused: expected rows of {width} contiguous values, got shape {tuple(t.shape)} strides {t.stride()}")
    return t.stride(0)


def _object_strides(so: list[torch.Tensor], hinged: bool) -> tuple[list[torch.Tensor], list[int]]:
    """The object's four tensors (a hinge's angle and rate repeated) and their row strides."""
    if hinged:
        return [so[0], so[1], so[0], so[0]], [_rows(so[0], 1), _rows(so[1], 1), so[0].stride(0), so[0].stride(0)]
    return list(so), [_rows(so[0], 3), _rows(so[1], 9), _rows(so[2], 3), _rows(so[3], 3)]


_TREES: dict[int, tuple] = {}


def kinematic_tree(kin) -> int:
    """Register a TreeKinematics' constants for rf_features (compile-time constants of its kernel); returns its key."""
    key = len(_TREES)
    r = lambda t, k: tuple(tuple(float(v) for v in row) for row in t.reshape(t.shape[0], k).tolist())  # noqa: E731
    _TREES[key] = (tuple(kin.parent), tuple(kin.joint_idx), r(kin.off_p, 3), r(kin.off_R, 9), r(kin.axes, 3))
    return key


@torch.library.custom_op("ndm_gym::hoi_rf_features", mutates_args=())
def rf_features(p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, so: list[torch.Tensor],
                u: torch.Tensor, ctx: torch.Tensor, tree: int, q0: list[float], gain: list[float], aidx: list[int], startup: int, poses: int,
                hinged: bool, d_in: int, d_out: int) -> torch.Tensor:  # fmt: skip
    """HOIRigidFormerNet.inputs' raw features without the context's scalars (d_in of them), then the tokens' positions
    / POS_SCALE, feature-major: (d_out, N). tree: a key of ``kinematic_tree``; q0, gain, aidx: each joint's default
    pose, action gain and action index."""
    N = p.shape[0]
    out = torch.empty(d_out, N, device=p.device, dtype=torch.float32)
    so4, sos = _object_strides(so, hinged)
    st = [_rows(p, 3), _rows(R, 9), _rows(v, 3), _rows(w, 3), _rows(q, NUM_J), _rows(qd, NUM_J)] + sos
    st += [_rows(u, DECIMATION * NUM_ACTIONS), _rows(ctx, ctx.shape[1])]
    parent, joint, offp, offr, axis = _TREES[tree]
    _rf_features_kernel[(triton.cdiv(N, BLOCK),)](p, R, v, w, q, qd, *so4, u, ctx, out, N, *st, PARENT=parent, JOINT=joint, OFFP=offp, OFFR=offr,
                                                  AXIS=axis, Q0=tuple(q0), GAIN=tuple(gain), AIDX=tuple(aidx), NB=len(parent), STARTUP=startup,
                                                  POSES=poses, HINGED=int(hinged), D_IN=d_in, NA=NUM_ACTIONS, SUB=DECIMATION, NJ=NUM_J, POS=POS_SCALE,
                                                  BLOCK=BLOCK, num_warps=WARPS)  # fmt: skip
    return out


@rf_features.register_fake
def _(p, R, v, w, q, qd, so, u, ctx, tree, q0, gain, aidx, startup, poses, hinged, d_in, d_out):
    return p.new_empty(d_out, p.shape[0])


@torch.library.custom_op("ndm_gym::hoi_mlp_features", mutates_args=())
def mlp_features(p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, so: list[torch.Tensor], ctx: torch.Tensor, startup: int,
                 poses: int, hinged: bool) -> torch.Tensor:  # fmt: skip
    """The computed part of HOIWorldModel.inputs for the MLP, feature-major: (13 + (0 if hinged else 15) + 9 poses, N)."""
    N = p.shape[0]
    out = torch.empty(13 + (0 if hinged else 15) + 9 * poses, N, device=p.device, dtype=torch.float32)
    so4, sos = _object_strides(so, hinged)
    st = [_rows(p, 3), _rows(R, 9), _rows(v, 3), _rows(w, 3)] + sos + [_rows(ctx, ctx.shape[1])]
    _mlp_features_kernel[(triton.cdiv(N, BLOCK),)](p, R, v, w, *so4, ctx, out, N, *st, STARTUP=startup, POSES=poses, HINGED=int(hinged), NJ=NUM_J,
                                                   BLOCK=BLOCK, num_warps=WARPS)  # fmt: skip
    return out


@mlp_features.register_fake
def _(p, R, v, w, so, ctx, startup, poses, hinged):
    return p.new_empty(13 + (0 if hinged else 15) + 9 * poses, p.shape[0])


@torch.library.custom_op("ndm_gym::hoi_integrate", mutates_args=())
def integrate(y: torch.Tensor, p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, q: torch.Tensor, qd: torch.Tensor,
              so: list[torch.Tensor], accel: bool, robot: int, feet: int) -> list[torch.Tensor]:  # fmt: skip
    """y (N, >= feet + 24) the de-normalized fp32 outputs -> the robot's next [p, R, v, w, q, qd], then the object's next
    state ([angle, rate] of a hinge), then the feet's forces (N, 4, 2, 3) in the world frame from their heading-frame
    outputs y[:, feet : feet + 24]."""
    N, hinged = p.shape[0], len(so) == 2
    nxt = [torch.empty(t.shape, device=t.device, dtype=t.dtype) for t in (p, R, v, w, q, qd)]
    nso = [torch.empty(t.shape, device=t.device, dtype=t.dtype) for t in so]
    nf = torch.empty(N, DECIMATION, 2, 3, device=p.device, dtype=torch.float32)
    so4, sos = _object_strides(so, hinged)
    nso4 = nso + nso if hinged else nso
    if y.stride(-1) != 1 or y.dtype != torch.float32 or y.shape[-1] < feet + 6 * DECIMATION:
        raise ValueError(f"hoi_fused.integrate: y needs fp32, a unit last stride and {feet + 6 * DECIMATION} columns, got {y.dtype} {tuple(y.shape)}")
    st = [y.stride(0), _rows(p, 3), _rows(R, 9), _rows(v, 3), _rows(w, 3), _rows(q, NUM_J), _rows(qd, NUM_J)] + sos
    _integrate_kernel[(triton.cdiv(N, BLOCK),)](y, p, R, v, w, q, qd, *so4, *nxt, *nso4, nf, N, *st, CONTROL_DT, ACCEL=int(accel),
                                                HINGED=int(hinged), NJ=NUM_J, ROBOT=robot, FEET=feet, SUB=DECIMATION, BLOCK=BLOCK,
                                                num_warps=WARPS)  # fmt: skip
    return nxt + nso + [nf]


@integrate.register_fake
def _(y, p, R, v, w, q, qd, so, accel, robot, feet):
    return [t.new_empty(t.shape) for t in (p, R, v, w, q, qd)] + [t.new_empty(t.shape) for t in so] + [p.new_empty(p.shape[0], DECIMATION, 2, 3)]
