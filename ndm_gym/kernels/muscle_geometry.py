"""Fused muscle geometry of MuscleVAE's character for the body-token world model's features
(``worldmodels.muscle.MuscleRigidFormerNet(tokens="body", fused_features=True)``, ``--rf_fused_features``), in Triton:
from the bodies' frames, every muscle's length and length rate, and the force and the torque about each body's origin
that given muscle forces exert through the anchors, with the backward passes of both. The same functions as
``robots.muscles.MuscleModel``'s ``lengths``, ``length_rates`` of ``anchor_velocities``, and ``body_wrench`` of
``anchor_forces`` (each anchor pushing its body0 only), up to float rounding.

A program holds one block of muscles of one environment. A muscle's anchors (at most K = 8) are walked in order with the
last two kept in registers (``tl.static_range``): an anchor is w0 (R_b0 x0 + p_b0) + w1 (R_b1 x1 + p_b1), w1 = 0 unless
it is blended between two bodies, and its velocity the same map with R' = [w]x R and v; a segment's length rate is its
unit direction times its ends' relative velocity. The wrench's forward pass writes each anchor's force f (u_out - u_in)
and its moment about its body0's origin, the backward passes each anchor's position and velocity gradients (B, A, 6);
per-body kernels then sum them over the body's anchors or points. No atomics: every sum has a fixed order, so the same
inputs give the same bits. fp32 throughout; inputs and outputs batch-major.

Frames (``frames``): (B, NB, 24) fp32, per body R (9, row-major), p (3), R' (9), v (3). The wrench reads R and p only.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

K = 8  # anchors per muscle at most (MuscleVAE's table: 2 to 8)
BM = 128  # muscles per program
FW = 24  # floats per body frame


def frames(p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The kernels' body frames (B, NB, 24) from positions (..., NB, 3), rotations (..., NB, 3, 3), velocities and
    angular velocities (..., NB, 3); differentiable (PyTorch)."""
    rate = torch.cross(w.unsqueeze(-1).expand_as(R), R, dim=-2)  # [w]x R: column j is w x R[:, j]
    f = torch.cat((R.flatten(-2), p, rate.flatten(-2), v), -1)
    return f.reshape(-1, f.shape[-2], FW).contiguous()


class MuscleTables:
    """The kernels' index tables of one muscle model, on its device: per muscle its first anchor and anchor count; per
    anchor its two points (body, local point, weight; the second's weight 0 unless blended); per body the points on it
    (anchor, slot) and the anchors whose body0 it is, padded with -1."""

    def __init__(self, mm):
        t = mm.table
        A, M, NB = mm.num_anchors, mm.num_muscles, mm.num_bodies
        count = np.asarray(t.anchor_count, dtype=np.int64)
        if count.max() > K:
            raise ValueError(f"muscles with up to {K} anchors are supported, the table has {count.max()}")
        w0 = np.asarray(t.anchor_weight0, dtype=np.float64)
        blended = w0 < 1.0
        b0, b1 = np.asarray(t.anchor_body0, dtype=np.int64), np.asarray(t.anchor_body1, dtype=np.int64)
        x0, x1 = np.asarray(t.anchor_local0, dtype=np.float64), np.asarray(t.anchor_local1, dtype=np.float64)
        b1 = np.where(blended, b1, b0)
        x1 = np.where(blended[:, None], x1, 0.0)
        w1 = np.where(blended, 1.0 - w0, 0.0)
        points = [[] for _ in range(NB)]  # per body: (anchor, slot)
        for a in range(A):
            points[b0[a]].append((a, 0))
            if blended[a]:
                points[b1[a]].append((a, 1))
        anchors = [np.flatnonzero(b0 == b) for b in range(NB)]
        self.bp = triton.next_power_of_2(max(len(p) for p in points))
        self.ba = triton.next_power_of_2(max(len(a) for a in anchors))
        pa = np.full((NB, self.bp), -1, dtype=np.int64)
        px = np.zeros((NB, self.bp, 3))
        pw = np.zeros((NB, self.bp))
        for b, ps in enumerate(points):
            for i, (a, s) in enumerate(ps):
                pa[b, i], px[b, i], pw[b, i] = a, (x0[a] if s == 0 else x1[a]), (w0[a] if s == 0 else w1[a])
        al = np.full((NB, self.ba), -1, dtype=np.int64)
        for b, a in enumerate(anchors):
            al[b, : len(a)] = a
        dev = mm.device
        i32 = lambda x: torch.as_tensor(x, dtype=torch.int32, device=dev).contiguous()  # noqa: E731
        f32 = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev).contiguous()  # noqa: E731
        start = np.cumsum(count) - count
        self.A, self.M, self.NB = A, M, NB
        # in the order the custom operators take them
        self.tensors = [i32(start), i32(count), i32(b0), i32(b1), f32(x0), f32(x1), f32(w0), f32(w1), i32(pa), f32(px), f32(pw), i32(al)]


@triton.jit
def _map(Fp, b, mask, x0, x1, x2, w, OFF: tl.constexpr):
    """w (T_b x + t_b) for the bodies b (BM,) of one environment's frames Fp: T, t the rotation and position (OFF 0) or
    their time derivatives (OFF 12)."""
    base = Fp + b * 24 + OFF
    y0 = tl.load(base + 0, mask=mask, other=0.0) * x0 + tl.load(base + 1, mask=mask, other=0.0) * x1 + tl.load(base + 2, mask=mask, other=0.0) * x2
    y1 = tl.load(base + 3, mask=mask, other=0.0) * x0 + tl.load(base + 4, mask=mask, other=0.0) * x1 + tl.load(base + 5, mask=mask, other=0.0) * x2
    y2 = tl.load(base + 6, mask=mask, other=0.0) * x0 + tl.load(base + 7, mask=mask, other=0.0) * x1 + tl.load(base + 8, mask=mask, other=0.0) * x2
    y0 = y0 + tl.load(base + 9, mask=mask, other=0.0)
    y1 = y1 + tl.load(base + 10, mask=mask, other=0.0)
    y2 = y2 + tl.load(base + 11, mask=mask, other=0.0)
    return w * y0, w * y1, w * y2


@triton.jit
def _anchor(Fp, a, mask, AB0, AB1, AX0, AX1, AW0, AW1, OFF: tl.constexpr):
    """Anchors a (BM,): position (OFF 0) or velocity (OFF 12), w0 (T_b0 x0 + t_b0) + w1 (T_b1 x1 + t_b1)."""
    b0 = tl.load(AB0 + a, mask=mask, other=0)
    b1 = tl.load(AB1 + a, mask=mask, other=0)
    w0 = tl.load(AW0 + a, mask=mask, other=0.0)
    w1 = tl.load(AW1 + a, mask=mask, other=0.0)
    p0, p1, p2 = _map(Fp, b0, mask, tl.load(AX0 + 3 * a, mask=mask, other=0.0), tl.load(AX0 + 3 * a + 1, mask=mask, other=0.0),
                      tl.load(AX0 + 3 * a + 2, mask=mask, other=0.0), w0, OFF)  # fmt: skip
    q0, q1, q2 = _map(Fp, b1, mask, tl.load(AX1 + 3 * a, mask=mask, other=0.0), tl.load(AX1 + 3 * a + 1, mask=mask, other=0.0),
                      tl.load(AX1 + 3 * a + 2, mask=mask, other=0.0), w1, OFF)  # fmt: skip
    return p0 + q0, p1 + q1, p2 + q2


@triton.jit
def _unit(dx, dy, dz, ok):
    """Length of (dx, dy, dz) and its unit direction (zeros where not ok)."""
    ln = tl.sqrt(dx * dx + dy * dy + dz * dz)
    inv = tl.where(ok, 1.0 / tl.where(ok, ln, 1.0), 0.0)
    return ln, dx * inv, dy * inv, dz * inv


@triton.jit
def _len_fwd_kernel(F, L, DL, MSTART, MCOUNT, AB0, AB1, AX0, AX1, AW0, AW1, M, NB: tl.constexpr, K: tl.constexpr, BM: tl.constexpr):
    env = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    mm = m < M
    start = tl.load(MSTART + m, mask=mm, other=0)
    cnt = tl.load(MCOUNT + m, mask=mm, other=0)
    Fp = F + env * (NB * 24)
    length = tl.zeros([BM], dtype=tl.float32)
    rate = tl.zeros([BM], dtype=tl.float32)
    qx = tl.zeros([BM], dtype=tl.float32)
    qy = tl.zeros([BM], dtype=tl.float32)
    qz = tl.zeros([BM], dtype=tl.float32)
    ux = tl.zeros([BM], dtype=tl.float32)
    uy = tl.zeros([BM], dtype=tl.float32)
    uz = tl.zeros([BM], dtype=tl.float32)
    for k in tl.static_range(K):
        ok = mm & (k < cnt)
        a = tl.where(ok, start + k, 0)
        px, py, pz = _anchor(Fp, a, ok, AB0, AB1, AX0, AX1, AW0, AW1, 0)
        vx, vy, vz = _anchor(Fp, a, ok, AB0, AB1, AX0, AX1, AW0, AW1, 12)
        if k > 0:
            ln, ex, ey, ez = _unit(px - qx, py - qy, pz - qz, ok)
            length += tl.where(ok, ln, 0.0)
            rate += ex * (vx - ux) + ey * (vy - uy) + ez * (vz - uz)
        qx, qy, qz, ux, uy, uz = px, py, pz, vx, vy, vz
    tl.store(L + env * M + m, length, mask=mm)
    tl.store(DL + env * M + m, rate, mask=mm)


@triton.jit
def _len_bwd_kernel(F, GL, GDL, GA, MSTART, MCOUNT, AB0, AB1, AX0, AX1, AW0, AW1, M, A, NB: tl.constexpr, K: tl.constexpr, BM: tl.constexpr):
    """Each anchor's position and velocity gradients (GA (B, A, 6)) from the lengths' and rates' gradients: a segment
    d = q - p of length l and unit e takes gL e + gR (dv - e (e . dv)) / l on d and gR e on dv = v_q - v_p."""
    env = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    mm = m < M
    start = tl.load(MSTART + m, mask=mm, other=0)
    cnt = tl.load(MCOUNT + m, mask=mm, other=0)
    gl = tl.load(GL + env * M + m, mask=mm, other=0.0)
    gr = tl.load(GDL + env * M + m, mask=mm, other=0.0)
    Fp = F + env * (NB * 24)
    Gp = GA + env * A * 6
    z = tl.zeros([BM], dtype=tl.float32)
    qx, qy, qz, ux, uy, uz = z, z, z, z, z, z  # the previous anchor's position and velocity
    cx, cy, cz, cvx, cvy, cvz = z, z, z, z, z, z  # the previous anchor's gradient from the segment that ends at it
    qa = tl.zeros([BM], dtype=tl.int32)
    for k in tl.static_range(K):
        ok = mm & (k < cnt)
        a = tl.where(ok, start + k, 0)
        px, py, pz = _anchor(Fp, a, ok, AB0, AB1, AX0, AX1, AW0, AW1, 0)
        vx, vy, vz = _anchor(Fp, a, ok, AB0, AB1, AX0, AX1, AW0, AW1, 12)
        gx, gy, gz, gvx, gvy, gvz = z, z, z, z, z, z
        if k > 0:
            ln, ex, ey, ez = _unit(px - qx, py - qy, pz - qz, ok)
            dvx, dvy, dvz = vx - ux, vy - uy, vz - uz
            edv = ex * dvx + ey * dvy + ez * dvz
            inv = tl.where(ok, 1.0 / tl.where(ok, ln, 1.0), 0.0)
            gx = tl.where(ok, gl * ex + gr * (dvx - ex * edv) * inv, 0.0)
            gy = tl.where(ok, gl * ey + gr * (dvy - ey * edv) * inv, 0.0)
            gz = tl.where(ok, gl * ez + gr * (dvz - ez * edv) * inv, 0.0)
            gvx, gvy, gvz = gr * ex, gr * ey, gr * ez
            # the previous anchor is complete: the segment that ends at it (carried) minus this one
            pok = mm & (k - 1 < cnt)
            pb = Gp + qa * 6
            tl.store(pb + 0, cx - gx, mask=pok)
            tl.store(pb + 1, cy - gy, mask=pok)
            tl.store(pb + 2, cz - gz, mask=pok)
            tl.store(pb + 3, cvx - gvx, mask=pok)
            tl.store(pb + 4, cvy - gvy, mask=pok)
            tl.store(pb + 5, cvz - gvz, mask=pok)
        last = ok & (k == cnt - 1)  # the muscle's last anchor: only the segment that ends at it
        lb = Gp + a * 6
        tl.store(lb + 0, gx, mask=last)
        tl.store(lb + 1, gy, mask=last)
        tl.store(lb + 2, gz, mask=last)
        tl.store(lb + 3, gvx, mask=last)
        tl.store(lb + 4, gvy, mask=last)
        tl.store(lb + 5, gvz, mask=last)
        qx, qy, qz, ux, uy, uz = px, py, pz, vx, vy, vz
        cx, cy, cz, cvx, cvy, cvz = gx, gy, gz, gvx, gvy, gvz
        qa = a


@triton.jit
def _wr_fwd_kernel(F, FORCE, WA, MSTART, MCOUNT, AB0, AB1, AX0, AX1, AW0, AW1, M, A, NB: tl.constexpr, K: tl.constexpr, BM: tl.constexpr):
    """Each anchor's force f (u_out - u_in) and its moment r x f about its body0's origin (WA (B, A, 6))."""
    env = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    mm = m < M
    start = tl.load(MSTART + m, mask=mm, other=0)
    cnt = tl.load(MCOUNT + m, mask=mm, other=0)
    f = tl.load(FORCE + env * M + m, mask=mm, other=0.0)
    Fp = F + env * (NB * 24)
    Wp = WA + env * A * 6
    z = tl.zeros([BM], dtype=tl.float32)
    qx, qy, qz = z, z, z  # the previous anchor's position
    ix, iy, iz = z, z, z  # the unit of the segment that ends at the previous anchor (zero at the first)
    qa = tl.zeros([BM], dtype=tl.int32)
    for k in tl.static_range(K + 1):
        ok = mm & (k < cnt)
        a = tl.where(ok, start + k, 0)
        px, py, pz = _anchor(Fp, a, ok, AB0, AB1, AX0, AX1, AW0, AW1, 0)
        if k > 0:
            _, ex, ey, ez = _unit(px - qx, py - qy, pz - qz, ok)  # zero past the last anchor
            pok = mm & (k - 1 < cnt)
            fx, fy, fz = f * (ex - ix), f * (ey - iy), f * (ez - iz)
            b0 = tl.load(AB0 + qa, mask=pok, other=0)
            rx = qx - tl.load(Fp + b0 * 24 + 9, mask=pok, other=0.0)
            ry = qy - tl.load(Fp + b0 * 24 + 10, mask=pok, other=0.0)
            rz = qz - tl.load(Fp + b0 * 24 + 11, mask=pok, other=0.0)
            pb = Wp + qa * 6
            tl.store(pb + 0, fx, mask=pok)
            tl.store(pb + 1, fy, mask=pok)
            tl.store(pb + 2, fz, mask=pok)
            tl.store(pb + 3, ry * fz - rz * fy, mask=pok)
            tl.store(pb + 4, rz * fx - rx * fz, mask=pok)
            tl.store(pb + 5, rx * fy - ry * fx, mask=pok)
            ix, iy, iz = ex, ey, ez
        qx, qy, qz = px, py, pz
        qa = a


@triton.jit
def _wr_bwd_kernel(F, FORCE, GW, GA, GFORCE, MSTART, MCOUNT, AB0, AB1, AX0, AX1, AW0, AW1, M, A, NB: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr):  # fmt: skip
    """The wrench's backward pass: GW (B, NB, 6), the body forces' and torques' gradients -> each anchor's position
    gradient (GA (B, A, 6), the velocity part zero) and the muscle forces' gradients (GFORCE (B, M)). An anchor's force
    f_j = F (u_out - u_in) takes g_j = gF + gT x r_j (r_j its position minus its body0's origin) and its position
    f_j x gT (its body0's origin minus the same, summed by the caller as -F_b x gT_b); F takes g_j . (u_out - u_in); a
    segment from q to p takes F (g_q - g_p) on its unit e and (g_e - e (e . g_e)) / l on its vector p - q."""
    env = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    mm = m < M
    start = tl.load(MSTART + m, mask=mm, other=0)
    cnt = tl.load(MCOUNT + m, mask=mm, other=0)
    f = tl.load(FORCE + env * M + m, mask=mm, other=0.0)
    Fp = F + env * (NB * 24)
    Gw = GW + env * (NB * 6)
    Gp = GA + env * A * 6
    z = tl.zeros([BM], dtype=tl.float32)
    gforce = z
    qx, qy, qz = z, z, z  # the previous anchor q: position, g = gF + gT x r, gT
    qgx, qgy, qgz = z, z, z
    qtx, qty, qtz = z, z, z
    qa = tl.zeros([BM], dtype=tl.int32)
    sx, sy, sz = z, z, z  # the unit of the segment that ends at q (zero at the first anchor)
    cx, cy, cz = z, z, z  # that segment's vector gradient, which q takes with a plus sign
    for k in tl.static_range(K + 1):
        ok = mm & (k < cnt)
        a = tl.where(ok, start + k, 0)
        px, py, pz = _anchor(Fp, a, ok, AB0, AB1, AX0, AX1, AW0, AW1, 0)
        b0 = tl.load(AB0 + a, mask=ok, other=0)
        rx = px - tl.load(Fp + b0 * 24 + 9, mask=ok, other=0.0)
        ry = py - tl.load(Fp + b0 * 24 + 10, mask=ok, other=0.0)
        rz = pz - tl.load(Fp + b0 * 24 + 11, mask=ok, other=0.0)
        gtx = tl.load(Gw + b0 * 6 + 3, mask=ok, other=0.0)
        gty = tl.load(Gw + b0 * 6 + 4, mask=ok, other=0.0)
        gtz = tl.load(Gw + b0 * 6 + 5, mask=ok, other=0.0)
        pgx = tl.load(Gw + b0 * 6 + 0, mask=ok, other=0.0) + gty * rz - gtz * ry
        pgy = tl.load(Gw + b0 * 6 + 1, mask=ok, other=0.0) + gtz * rx - gtx * rz
        pgz = tl.load(Gw + b0 * 6 + 2, mask=ok, other=0.0) + gtx * ry - gty * rx
        if k > 0:
            ln, ex, ey, ez = _unit(px - qx, py - qy, pz - qz, ok)  # zero past the last anchor
            pok = mm & (k - 1 < cnt)
            # anchor q is complete: its force F (e - s), the force's gradient, its position's direct gradient f x gT
            ux, uy, uz = ex - sx, ey - sy, ez - sz
            gforce += tl.where(pok, qgx * ux + qgy * uy + qgz * uz, 0.0)
            fx, fy, fz = f * ux, f * uy, f * uz
            hx, hy, hz = f * (qgx - pgx), f * (qgy - pgy), f * (qgz - pgz)  # the segment q -> p's unit gradient
            eh = ex * hx + ey * hy + ez * hz
            inv = tl.where(ok, 1.0 / tl.where(ok, ln, 1.0), 0.0)
            dx, dy, dz = (hx - ex * eh) * inv, (hy - ey * eh) * inv, (hz - ez * eh) * inv
            pb = Gp + qa * 6
            tl.store(pb + 0, fy * qtz - fz * qty + cx - dx, mask=pok)
            tl.store(pb + 1, fz * qtx - fx * qtz + cy - dy, mask=pok)
            tl.store(pb + 2, fx * qty - fy * qtx + cz - dz, mask=pok)
            tl.store(pb + 3, z, mask=pok)
            tl.store(pb + 4, z, mask=pok)
            tl.store(pb + 5, z, mask=pok)
            sx, sy, sz = ex, ey, ez
            cx, cy, cz = dx, dy, dz
        qx, qy, qz = px, py, pz
        qgx, qgy, qgz = pgx, pgy, pgz
        qtx, qty, qtz = gtx, gty, gtz
        qa = a
    tl.store(GFORCE + env * M + m, gforce, mask=mm)


@triton.jit
def _frames_grad_kernel(GA, GF, PA, PX, PW, A, NB: tl.constexpr, BP: tl.constexpr):
    """The frames' gradients (B, NB, 24) from the anchors' (GA (B, A, 6)): per body the sum over its points of w gA (x, 1)^T
    (rotation, position) and w gV (x, 1)^T (rotation rate, velocity), each half one (4, 4) reduction."""
    env = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    i = tl.arange(0, BP)
    c = tl.arange(0, 4)
    a = tl.load(PA + b * BP + i)
    ok = a >= 0
    a = tl.where(ok, a, 0)
    w = tl.load(PW + b * BP + i)
    xe = tl.load(PX + (b * BP + i)[:, None] * 3 + c[None, :], mask=c[None, :] < 3, other=1.0)  # (BP, 4): the point x, then 1
    gp = GA + env * A * 6 + a * 6
    row, col = c[:, None], c[None, :]
    off = tl.where(col < 3, 3 * row + col, 9 + row)  # R's (i, j) at 3 i + j, the translation's i at 9 + i
    out = GF + (env * NB + b) * 24
    for h in tl.static_range(2):  # position, then velocity
        g = tl.load(gp[:, None] + 3 * h + c[None, :], mask=ok[:, None] & (c[None, :] < 3), other=0.0) * w[:, None]  # (BP, 4)
        t = tl.sum(g[:, :, None] * xe[:, None, :], axis=0)  # (4, 4): sum of g_i (x, 1)_j
        tl.store(out + 12 * h + off, t, mask=row < 3)


@triton.jit
def _body_sum_kernel(WA, OUT, AL, A, NB: tl.constexpr, BA: tl.constexpr):
    """Per body the sum of its anchors' (force, moment) (WA (B, A, 6)) -> OUT (B, NB, 6)."""
    env = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    i = tl.arange(0, BA)
    c = tl.arange(0, 8)
    a = tl.load(AL + b * BA + i)
    ok = a >= 0
    a = tl.where(ok, a, 0)
    v = tl.load(WA + env * A * 6 + a[:, None] * 6 + c[None, :], mask=ok[:, None] & (c[None, :] < 6), other=0.0)
    tl.store(OUT + (env * NB + b) * 6 + c, tl.sum(v, axis=0), mask=c < 6)


def _grid(B: int, M: int):
    return (B, triton.cdiv(M, BM))


@torch.library.custom_op("ndm_gym::muscle_lengths", mutates_args=())
def muscle_lengths(frames: torch.Tensor, tables: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Muscle lengths and their time derivatives (B, M) of frames (B, NB, 24) (``MuscleTables.tensors``)."""
    B, NB, _ = frames.shape
    M = tables[0].numel()
    length, rate = frames.new_empty(B, M), frames.new_empty(B, M)
    _len_fwd_kernel[_grid(B, M)](frames, length, rate, *tables[:8], M, NB=NB, K=K, BM=BM)
    return length, rate


@muscle_lengths.register_fake
def _(frames, tables):
    return frames.new_empty(frames.shape[0], tables[0].shape[0]), frames.new_empty(frames.shape[0], tables[0].shape[0])


@torch.library.custom_op("ndm_gym::muscle_lengths_bwd", mutates_args=())
def muscle_lengths_bwd(g_length: torch.Tensor, g_rate: torch.Tensor, frames: torch.Tensor, tables: list[torch.Tensor]) -> torch.Tensor:
    """The frames' gradient (B, NB, 24) from the lengths' and rates' gradients (B, M)."""
    B, NB, _ = frames.shape
    M, A = tables[0].numel(), tables[2].numel()
    ga = frames.new_empty(B, A, 6)  # each anchor belongs to one muscle, which writes it
    _len_bwd_kernel[_grid(B, M)](frames, g_length, g_rate, ga, *tables[:8], M, A, NB=NB, K=K, BM=BM)
    gf = frames.new_empty(B, NB, FW)
    _frames_grad_kernel[(B, NB)](ga, gf, tables[8], tables[9], tables[10], A, NB=NB, BP=tables[8].shape[1])
    return gf


@muscle_lengths_bwd.register_fake
def _(g_length, g_rate, frames, tables):
    return torch.empty_like(frames)


def _lengths_setup(ctx, inputs, output):
    frames, tables = inputs
    ctx.save_for_backward(frames, *tables)


def _lengths_backward(ctx, g_length, g_rate):
    frames, *tables = ctx.saved_tensors
    return muscle_lengths_bwd(g_length.contiguous(), g_rate.contiguous(), frames, tables), [None] * len(tables)


muscle_lengths.register_autograd(_lengths_backward, setup_context=_lengths_setup)


@torch.library.custom_op("ndm_gym::muscle_wrench", mutates_args=())
def muscle_wrench(frames: torch.Tensor, force: torch.Tensor, tables: list[torch.Tensor]) -> torch.Tensor:
    """Per body (B, NB, 6) the force and the torque about its origin that the muscle forces (B, M) exert through the
    anchors of frames (B, NB, 24), each anchor pushing its body0."""
    B, NB, _ = frames.shape
    M, A = tables[0].numel(), tables[2].numel()
    wa = frames.new_empty(B, A, 6)
    _wr_fwd_kernel[_grid(B, M)](frames, force, wa, *tables[:8], M, A, NB=NB, K=K, BM=BM)
    out = frames.new_empty(B, NB, 6)
    _body_sum_kernel[(B, NB)](wa, out, tables[11], A, NB=NB, BA=tables[11].shape[1])
    return out


@muscle_wrench.register_fake
def _(frames, force, tables):
    return frames.new_empty(frames.shape[0], frames.shape[1], 6)


@torch.library.custom_op("ndm_gym::muscle_wrench_bwd", mutates_args=())
def muscle_wrench_bwd(g_wrench: torch.Tensor, wrench: torch.Tensor, frames: torch.Tensor, force: torch.Tensor,
                      tables: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:  # fmt: skip
    """The frames' (B, NB, 24) and the muscle forces' (B, M) gradients from the wrench's (B, NB, 6)."""
    B, NB, _ = frames.shape
    M, A = tables[0].numel(), tables[2].numel()
    ga, g_force = frames.new_empty(B, A, 6), frames.new_empty(B, M)
    _wr_bwd_kernel[_grid(B, M)](frames, force, g_wrench, ga, g_force, *tables[:8], M, A, NB=NB, K=K, BM=BM)
    gf = frames.new_empty(B, NB, FW)
    _frames_grad_kernel[(B, NB)](ga, gf, tables[8], tables[9], tables[10], A, NB=NB, BP=tables[8].shape[1])
    # the bodies' origins: T_b = sum_j (x_j - p_b) x f_j takes -F_b x gT_b on p_b
    gf[..., 9:12] -= torch.cross(wrench[..., :3], g_wrench[..., 3:], dim=-1)
    return gf, g_force


@muscle_wrench_bwd.register_fake
def _(g_wrench, wrench, frames, force, tables):
    return torch.empty_like(frames), torch.empty_like(force)


def _wrench_setup(ctx, inputs, output):
    frames, force, tables = inputs
    ctx.save_for_backward(output, frames, force, *tables)


def _wrench_backward(ctx, g_wrench):
    wrench, frames, force, *tables = ctx.saved_tensors
    gf, g_force = muscle_wrench_bwd(g_wrench.contiguous(), wrench, frames, force, tables)
    return gf, g_force, [None] * len(tables)


muscle_wrench.register_autograd(_wrench_backward, setup_context=_wrench_setup)


__all__ = ["K", "MuscleTables", "frames", "muscle_lengths", "muscle_wrench"]
