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
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..robots.g1 import NUM_J
from ..state import CONTROL_DT
from .hoi_fused import _TREES, _exp, _ld, _mm3, _mv3, _rows, _st, _yaw
from .rigidformer import GLOBAL_DIM, POS_SCALE, TOKEN_DIM, XI_OFFSET_SCALE

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
                         NJ: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    """The robot's next state from the de-normalized outputs y: accelerations in the root's frame and of the joints,
    integrated as AccelWorldModel.step (velocities gain dt a, positions dt (v + pc a), the rotation exp(dt (w + pc aw)) R),
    or frame-local increments as IncrementWorldModel.step."""
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
              accel: bool, pos_coef: float) -> list[torch.Tensor]:  # fmt: skip
    """y (N, 6 + J accelerations, or 12 + 2 J increments) the de-normalized fp32 outputs -> the robot's next
    [p, R, v, w, q, qd]; pos_coef: AccelWorldModel's position coefficient (s), unused for increments."""
    N = p.shape[0]
    width = (6 if accel else 12) + (1 if accel else 2) * NUM_J
    if y.stride(-1) != 1 or y.dtype != torch.float32 or y.shape[-1] != width:
        raise ValueError(f"g1_fused.integrate: y needs fp32, a unit last stride and {width} columns, got {y.dtype} {tuple(y.shape)}")
    nxt = [torch.empty(t.shape, device=t.device, dtype=t.dtype) for t in (p, R, v, w, q, qd)]
    st = [y.stride(0), _rows(p, 3), _rows(R, 9), _rows(v, 3), _rows(w, 3), _rows(q, NUM_J), _rows(qd, NUM_J)]
    _g1_integrate_kernel[(triton.cdiv(N, BLOCK),)](y, p, R, v, w, q, qd, *nxt, N, *st, CONTROL_DT, pos_coef, ACCEL=int(accel), NJ=NUM_J, BLOCK=BLOCK,
                                                   num_warps=WARPS)  # fmt: skip
    return nxt


@integrate.register_fake
def _(y, p, R, v, w, q, qd, accel, pos_coef):
    return [t.new_empty(t.shape) for t in (p, R, v, w, q, qd)]
