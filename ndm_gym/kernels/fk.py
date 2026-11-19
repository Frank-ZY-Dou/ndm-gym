"""The forward kinematics of an articulation (``core.kinematics.ArticulationKinematics.forward``) as two Triton kernels with
a hand-written backward pass, for training (``ArticulationKinematics(fused=True)``, train.py --fk_fused): the world
positions, rotations, link-origin linear and angular velocities of all bodies or of some, from the root's position,
rotation, linear and angular velocity and the joints' angles and rates, differentiable in all six.

The tree is static: every body's parent, joint, offset in its parent's frame (position o, rotation Q) and hinge axis x are
compile-time constants of the kernels (``register``), and so are the output slots of the bodies asked for, so the loops
over the bodies unroll at compile time and the products with constants leave out the zero ones (for finite inputs the
same values). A program holds a block of samples, one per lane. The forward kernel walks the tree from the root, b = a +
Ra o, Rb = Ra Q A(q), v = va + wa x (b - a), w = wa + Ra Q x qd, with A(q) = c I + s [x]x + (1 - c) x x^T the rotation by q
about x (c, s: its cosine and sine), and stores a body's outputs as soon as they are computed, so that only the open
branches stay in registers. The backward kernel visits the tree depth first, larger subtrees first: on the way down it
runs the recursion of the rotations and angular velocities again, on the way up the adjoint of each body, its outputs'
adjoint plus what its children passed up, which gives its joint's angle and rate their gradients (each joint moves one
body) and passes up to its parent through the four relations above; what reaches the root is the gradient of p, R, v and
w. Only the bodies on the current path keep their rotations in registers. No scratch memory, no atomics: the same inputs
give the same bits. fp32 throughout; equal to ``forward`` up to float rounding (tests/test_fk_fused.py checks the values
and gradients, on the CPU through Triton's interpreter and on a GPU).

Unrolled, the kernels take long to compile (Triton's coalescing pass, in the number of loads and stores times the number of
operations): offline for sm_80 on one CPU core, the forward 19 s for the G1's tracked bodies and 38 s for all of them, the
backward 2.2 and 4.4 min. Once per tree and output set; then from Triton's cache.
"""

from __future__ import annotations

import contextlib
import hashlib
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from ..core.kinematics import ArticulationKinematics

BLOCK, WARPS = 64, 2  # samples per program (one per lane) and warps, as worldmodels/g1_fused.py
TINY = 2.0**-126  # constants below fp32's normal range compile as fp64: they are zero here, as the GPU's arithmetic flushes them


@triton.jit
def _ld3(ptr, row, m):
    return tl.load(ptr + row, mask=m, other=0.0), tl.load(ptr + row + 1, mask=m, other=0.0), tl.load(ptr + row + 2, mask=m, other=0.0)


@triton.jit
def _ld9(ptr, row, m):
    return _ld3(ptr, row, m) + _ld3(ptr, row + 3, m) + _ld3(ptr, row + 6, m)


@triton.jit
def _st3(ptr, row, m, x):
    for k in tl.static_range(3):
        tl.store(ptr + row + k, x[k], mask=m)


@triton.jit
def _st9(ptr, row, m, x):
    for k in tl.static_range(9):
        tl.store(ptr + row + k, x[k], mask=m)


@triton.jit
def _add3(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


@triton.jit
def _add9(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4], a[5] + b[5], a[6] + b[6], a[7] + b[7], a[8] + b[8])


@triton.jit
def _add_adjoint(a, b):
    return _add3(a[0], b[0]), _add9(a[1], b[1]), _add3(a[2], b[2]), _add3(a[3], b[3])


@triton.jit
def _cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


# A body's constants are read in place, M[i][k]: a constexpr tuple assigned to a variable, or passed to a function on its
# own, is not a constexpr any more, so the functions below take the tables and the body's index i.
@triton.jit
def _lin3(x0, x1, x2, c0: tl.constexpr, c1: tl.constexpr, c2: tl.constexpr):
    """x0 c0 + x1 c1 + x2 c2 for constants c, without the terms of the zero ones (all zero: x2 c2)."""
    if c0 != 0.0:
        y = x0 * c0
        if c1 != 0.0:
            y = y + x1 * c1
        if c2 != 0.0:
            y = y + x2 * c2
    elif c1 != 0.0:
        y = x1 * c1
        if c2 != 0.0:
            y = y + x2 * c2
    else:
        y = x2 * c2
    return y


@triton.jit
def _times_vec(r, V: tl.constexpr, i: tl.constexpr):
    """r V[i]: a row-major 3 x 3 r times body i's constant vector."""
    return (_lin3(r[0], r[1], r[2], V[i][0], V[i][1], V[i][2]), _lin3(r[3], r[4], r[5], V[i][0], V[i][1], V[i][2]),
            _lin3(r[6], r[7], r[8], V[i][0], V[i][1], V[i][2]))  # fmt: skip


@triton.jit
def _times_mat(r, M: tl.constexpr, i: tl.constexpr):
    """r M[i]: a row-major 3 x 3 r times body i's constant row-major 3 x 3 matrix."""
    out = ()
    for k in tl.static_range(9):
        out = out + (_lin3(r[k // 3 * 3], r[k // 3 * 3 + 1], r[k // 3 * 3 + 2], M[i][k % 3], M[i][3 + k % 3], M[i][6 + k % 3]),)
    return out


@triton.jit
def _turn(e, u, c, s, CROSS: tl.constexpr, AXIS: tl.constexpr, i: tl.constexpr):
    """e A(q) = c e + s e K + (1 - c) u x^T, row-major: e = Ra Q turned by body i's joint, x its axis, K = [x]x (CROSS[i]),
    u = e x, c and s the cosine and sine of the angle."""
    C = 1 - c
    out = ()
    for k in tl.static_range(9):
        y = c * e[k]
        if CROSS[i][k % 3] != 0.0 or CROSS[i][3 + k % 3] != 0.0 or CROSS[i][6 + k % 3] != 0.0:
            y = y + s * _lin3(e[k // 3 * 3], e[k // 3 * 3 + 1], e[k // 3 * 3 + 2], CROSS[i][k % 3], CROSS[i][3 + k % 3], CROSS[i][6 + k % 3])
        if AXIS[i][k % 3] != 0.0:
            y = y + C * u[k // 3] * AXIS[i][k % 3]
        out = out + (y,)
    return out


@triton.jit
def _put(OP, OR, OV, OW, n, m, s: tl.constexpr, NO: tl.constexpr, b):
    """Body state b (position, rotation, velocity, angular velocity) into output slot s of the samples n: OP, OV, OW
    (N, NO, 3), OR (N, NO, 3, 3)."""
    _st3(OP, n * (3 * NO) + 3 * s, m, b[0])
    _st9(OR, n * (9 * NO) + 9 * s, m, b[1])
    _st3(OV, n * (3 * NO) + 3 * s, m, b[2])
    _st3(OW, n * (3 * NO) + 3 * s, m, b[3])


@triton.jit
def _recompute(states, Q, QD, n, m, sq, sqd, PARENT: tl.constexpr, JOINT: tl.constexpr, OFFR: tl.constexpr, AXIS: tl.constexpr,
               CROSS: tl.constexpr, POS: tl.constexpr, b: tl.constexpr):  # fmt: skip
    """Body b's rotation and angular velocity from its parent's (``states[POS[parent]]``), and its joint's cosine, sine
    and rate, which its adjoint reads."""
    a = states[POS[PARENT[b]]]
    qi = tl.load(Q + n * sq + JOINT[b], mask=m, other=0.0)
    qdi = tl.load(QD + n * sqd + JOINT[b], mask=m, other=0.0)
    c, s = tl.cos(qi), tl.sin(qi)
    e = _times_mat(a[0], OFFR, b)
    u = _times_vec(e, AXIS, b)
    return _turn(e, u, c, s, CROSS, AXIS, b), (a[1][0] + u[0] * qdi, a[1][1] + u[1] * qdi, a[1][2] + u[2] * qdi), c, s, qdi


@triton.jit
def _gather(GP, GR, GV, GW, n, m, NO: tl.constexpr, ups, SLOT: tl.constexpr, CHILD: tl.constexpr, UPOS: tl.constexpr, b: tl.constexpr,
            zero):  # fmt: skip
    """Body b's adjoint (gp, gR, gv, gw): its outputs' (slot SLOT[b]; none if negative) plus what its children passed up
    (``ups[UPOS[child]]``)."""
    if SLOT[b] >= 0:
        g = (_ld3(GP, n * (3 * NO) + 3 * SLOT[b], m), _ld9(GR, n * (9 * NO) + 9 * SLOT[b], m), _ld3(GV, n * (3 * NO) + 3 * SLOT[b], m),
             _ld3(GW, n * (3 * NO) + 3 * SLOT[b], m))  # fmt: skip
        for c in tl.static_range(len(CHILD[b])):
            g = _add_adjoint(g, ups[UPOS[CHILD[b][c]]])
    elif len(CHILD[b]) > 0:
        g = ups[UPOS[CHILD[b][0]]]
        for c in tl.static_range(1, len(CHILD[b])):
            g = _add_adjoint(g, ups[UPOS[CHILD[b][c]]])
    else:
        g = ((zero, zero, zero), (zero, zero, zero, zero, zero, zero, zero, zero, zero), (zero, zero, zero), (zero, zero, zero))
    return g


@triton.jit
def _adjoint(g, a, b, zero, OFFP: tl.constexpr, OFFR: tl.constexpr, AXIS: tl.constexpr, CROSS: tl.constexpr, KS: tl.constexpr,
             i: tl.constexpr):  # fmt: skip
    """Body i's joint backwards. g = (gp, gR, gv, gw): the body's adjoint; a: its parent's (rotation, angular velocity, ...);
    b: its own (..., cos q, sin q, qd). Returns the gradients of the joint's angle and rate and what goes up to the
    parent: (gp, gRa, gv, gwa)."""
    gp, gr, gv, gw = g[0], g[1], g[2], g[3]
    c, s, qdi = b[2], b[3], b[4]
    d = _times_vec(a[0], OFFP, i)
    e = _times_mat(a[0], OFFR, i)
    u = _times_vec(e, AXIS, i)
    # v = va + wa x d: wa gets d x gv, the lever arm d gets gv x wa; the position, pa + d: d gets gp
    gwa = _add3(gw, _cross(d, gv))
    gd = _add3(gp, _cross(gv, a[1]))
    # Rb = e A: e gets gR A^T = c gR + s gR K^T + (1 - c) (gR x) x^T; w = wa + (e x) qd: e gets (gw qd) x^T, qd gets gw . e x
    gx = _times_vec(gr, AXIS, i)
    C = 1 - c
    ge = ()
    for k in tl.static_range(9):
        y = c * gr[k]
        if CROSS[i][3 * (k % 3)] != 0.0 or CROSS[i][3 * (k % 3) + 1] != 0.0 or CROSS[i][3 * (k % 3) + 2] != 0.0:
            y = y + s * _lin3(gr[k // 3 * 3], gr[k // 3 * 3 + 1], gr[k // 3 * 3 + 2], CROSS[i][3 * (k % 3)], CROSS[i][3 * (k % 3) + 1],
                              CROSS[i][3 * (k % 3) + 2])  # fmt: skip
        if AXIS[i][k % 3] != 0.0:
            y = y + (C * gx[k // 3] + gw[k // 3] * qdi) * AXIS[i][k % 3]
        ge = ge + (y,)
    # A gets e^T gR = G; dA/dq = s (x x^T - I) + c K (KS[i], CROSS[i]): the angle's gradient s <G, KS> + c <G, K>
    ts, tc = zero, zero
    for k in tl.static_range(9):
        if KS[i][k] != 0.0 or CROSS[i][k] != 0.0:
            G = e[k // 3] * gr[k % 3] + e[3 + k // 3] * gr[3 + k % 3] + e[6 + k // 3] * gr[6 + k % 3]
            if KS[i][k] != 0.0:
                ts = ts + G * KS[i][k]
            if CROSS[i][k] != 0.0:
                tc = tc + G * CROSS[i][k]
    dq = s * ts + c * tc
    dqd = gw[0] * u[0] + gw[1] * u[1] + gw[2] * u[2]
    # e = Ra Q: Ra gets ge Q^T; d = Ra o: Ra gets gd o^T
    gra = ()
    for k in tl.static_range(9):
        y = _lin3(ge[k // 3 * 3], ge[k // 3 * 3 + 1], ge[k // 3 * 3 + 2], OFFR[i][3 * (k % 3)], OFFR[i][3 * (k % 3) + 1], OFFR[i][3 * (k % 3) + 2])
        if OFFP[i][k % 3] != 0.0:
            y = y + gd[k // 3] * OFFP[i][k % 3]
        gra = gra + (y,)
    return dq, dqd, (gp, gra, gv, gwa)


# the sizes, strides and pointers' alignments are not specialized on: one compiled kernel per tree and output set, whatever views
# of the replay's states come in (the accesses are one value per lane anyway)
@triton.jit(do_not_specialize=("N", "sp", "sr", "sv", "sw", "sq", "sqd"),
            do_not_specialize_on_alignment=("P", "R", "V", "W", "Q", "QD", "OP", "OR", "OV", "OW"))
def _fk_fwd_kernel(P, R, V, W, Q, QD, OP, OR, OV, OW, N, sp, sr, sv, sw, sq, sqd, PARENT: tl.constexpr, JOINT: tl.constexpr,
                   OFFP: tl.constexpr, OFFR: tl.constexpr, AXIS: tl.constexpr, CROSS: tl.constexpr, SLOT: tl.constexpr, NB: tl.constexpr,
                   NO: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    """The bodies' states from the root's (rows of P, R, V, W with strides sp, sr, sv, sw) and the joints' (Q, QD): body i
    into output slot SLOT[i] of OP, OR, OV, OW (N, NO, ...), unless SLOT[i] < 0."""
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    qs, qds = (zero,), (zero,)  # every joint's angle and rate first (one memory round trip), by body
    for i in tl.static_range(1, NB):
        qs = qs + (tl.load(Q + n * sq + JOINT[i], mask=m, other=0.0),)
        qds = qds + (tl.load(QD + n * sqd + JOINT[i], mask=m, other=0.0),)
    states = ((_ld3(P, n * sp, m), _ld9(R, n * sr, m), _ld3(V, n * sv, m), _ld3(W, n * sw, m)),)
    if SLOT[0] >= 0:
        _put(OP, OR, OV, OW, n, m, SLOT[0], NO, states[0])
    for i in tl.static_range(1, NB):
        a = states[PARENT[i]]
        pb = _add3(a[0], _times_vec(a[1], OFFP, i))
        e = _times_mat(a[1], OFFR, i)
        u = _times_vec(e, AXIS, i)
        wb = (a[3][0] + u[0] * qds[i], a[3][1] + u[1] * qds[i], a[3][2] + u[2] * qds[i])
        vb = _add3(a[2], _cross(a[3], (pb[0] - a[0][0], pb[1] - a[0][1], pb[2] - a[0][2])))  # the lever arm as forward has it, b - a
        states = states + ((pb, _turn(e, u, tl.cos(qs[i]), tl.sin(qs[i]), CROSS, AXIS, i), vb, wb),)
        if SLOT[i] >= 0:
            _put(OP, OR, OV, OW, n, m, SLOT[i], NO, states[i])


@triton.jit(do_not_specialize=("N", "sr", "sw", "sq", "sqd"),
            do_not_specialize_on_alignment=("GP", "GR", "GV", "GW", "R", "W", "Q", "QD", "DP", "DR", "DV", "DW", "DQ", "DQD"))
def _fk_bwd_kernel(GP, GR, GV, GW, R, W, Q, QD, DP, DR, DV, DW, DQ, DQD, N, sr, sw, sq, sqd, PARENT: tl.constexpr, JOINT: tl.constexpr,
                   OFFP: tl.constexpr, OFFR: tl.constexpr, AXIS: tl.constexpr, CROSS: tl.constexpr, KS: tl.constexpr, SLOT: tl.constexpr,
                   CHILD: tl.constexpr, SCHED: tl.constexpr, POS: tl.constexpr, UPOS: tl.constexpr, NO: tl.constexpr, NJ: tl.constexpr,
                   BLOCK: tl.constexpr):  # fmt: skip
    """_fk_fwd_kernel backwards: the outputs' adjoints GP, GR, GV, GW (N, NO, ...) -> the gradients of p, R, v, w (DP, DV,
    DW (N, 3), DR (N, 3, 3)) and of the joints' angles and rates (DQ, DQD (N, NJ)). SCHED: the depth-first visit, (body, 0)
    on the way down, (body, 1) on the way up; POS[b], UPOS[b]: where body b's state and what it passes up are kept."""
    n = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = n < N
    zero = tl.zeros([BLOCK], dtype=tl.float32)
    states = ((_ld9(R, n * sr, m), _ld3(W, n * sw, m), zero, zero, zero),)
    ups = ()
    for k in tl.static_range(len(SCHED)):
        if SCHED[k][1] == 0:
            states = states + (_recompute(states, Q, QD, n, m, sq, sqd, PARENT, JOINT, OFFR, AXIS, CROSS, POS, SCHED[k][0]),)
        else:
            g = _gather(GP, GR, GV, GW, n, m, NO, ups, SLOT, CHILD, UPOS, SCHED[k][0], zero)
            dq, dqd, up = _adjoint(g, states[POS[PARENT[SCHED[k][0]]]], states[POS[SCHED[k][0]]], zero, OFFP, OFFR, AXIS, CROSS, KS, SCHED[k][0])
            tl.store(DQ + n * NJ + JOINT[SCHED[k][0]], dq, mask=m)
            tl.store(DQD + n * NJ + JOINT[SCHED[k][0]], dqd, mask=m)
            ups = ups + (up,)
    g = _gather(GP, GR, GV, GW, n, m, NO, ups, SLOT, CHILD, UPOS, 0, zero)
    _st3(DP, n * 3, m, g[0])
    _st9(DR, n * 9, m, g[1])
    _st3(DV, n * 3, m, g[2])
    _st3(DW, n * 3, m, g[3])


# whether triton.jit made the kernels Triton's interpreter's (TRITON_INTERPRET=1 when this module was imported): then they
# run on CPU tensors too (the tests' CPU checks); compiled, on CUDA tensors only
INTERPRETED = not isinstance(_fk_fwd_kernel, triton.runtime.jit.JITFunction)
_TREES: dict[str, dict] = {}


def register(kin: ArticulationKinematics) -> str:
    """Make an ArticulationKinematics' tree the kernels' constants (its parents, joints, offsets, rotations and axes as fp32
    values, and the backward kernel's visit) and keep them under a key made of them, the same in every process; returns the
    key. A tree the kernels do not cover raises: one root, the first body, parents before their children, and every joint
    moving exactly one body."""
    parent, joint, nb, nj = tuple(kin.parent), tuple(kin.joint_idx), len(kin.parent), int(kin.q_lo.numel())
    if parent[0] != -1 or joint[0] != -1 or any(not 0 <= par < i for i, par in enumerate(parent) if i):
        raise ValueError(f"kernels.fk: one root, the first body, and every parent before its children; parents {parent}")
    if sorted(joint[1:]) != list(range(nj)):
        raise ValueError(f"kernels.fk: every one of the {nj} joints must move exactly one body; joints of the bodies {joint}")

    def table(t: torch.Tensor, k: int) -> tuple:
        rows = t.detach().to("cpu", torch.float32).reshape(nb, k).tolist()
        return tuple(tuple(0.0 if abs(x) < TINY else float(x) for x in row) for row in rows)

    offp, offr, axis = table(kin.off_p, 3), table(kin.off_R, 9), table(kin.axes, 3)
    key = "tree-" + hashlib.sha256(repr((parent, joint, offp, offr, axis, nj)).encode()).hexdigest()[:16]
    if key in _TREES:  # another object of the same tree (a device of the multi-GPU layouts): compiled functions keep their guards
        return key
    cross = tuple((0.0, -x[2], x[1], x[2], 0.0, -x[0], -x[1], x[0], 0.0) for x in axis)  # [x]x
    ks = tuple(tuple(x[r] * x[c] - (r == c) for r in range(3) for c in range(3)) for x in axis)  # x x^T - I
    # the backward kernel's depth-first visit, larger subtrees first: fewer adjoints wait for their parent meanwhile
    size = [1] * nb
    for b in range(nb - 1, 0, -1):
        size[parent[b]] += size[b]
    child = tuple(tuple(sorted((c for c in range(nb) if parent[c] == b), key=lambda c: (-size[c], c))) for b in range(nb))
    sched = []

    def visit(b: int) -> None:
        for c in child[b]:
            sched.append((c, 0))
            visit(c)
            sched.append((c, 1))

    visit(0)
    pos, upos = [0] * nb, [-1] * nb
    for k, (b, up) in enumerate(sched):
        if up:
            upos[b] = sum(u for _, u in sched[:k])
        else:
            pos[b] = 1 + sum(1 - u for _, u in sched[:k])
    _TREES[key] = dict(PARENT=parent, JOINT=joint, OFFP=offp, OFFR=offr, AXIS=axis, CROSS=cross, KS=ks, CHILD=child, SCHED=tuple(sched),
                       POS=tuple(pos), UPOS=tuple(upos), NB=nb, NJ=nj)  # fmt: skip
    return key


def _tree(key: str) -> dict:
    if key not in _TREES:
        raise KeyError(f"kernels.fk: tree {key} is not registered in this process (kernels.fk.register)")
    return _TREES[key]


def _rows(t: torch.Tensor) -> tuple[torch.Tensor, int]:
    """t (N, ...) with each row's values contiguous (copied otherwise) and its row stride."""
    if t.shape[0] and not t[0].is_contiguous():
        t = t.contiguous()
    return t, t.stride(0)


def _launchable(*ts: torch.Tensor) -> None:
    if any(t.device != ts[0].device or t.dtype != torch.float32 for t in ts):
        raise ValueError(f"kernels.fk: fp32 tensors on one device, got {[(str(t.dtype), str(t.device)) for t in ts]}")
    if ts[0].device.type != "cuda" and not INTERPRETED:
        raise RuntimeError(f"kernels.fk: the kernels run on CUDA tensors (on the CPU only through Triton's interpreter, TRITON_INTERPRET=1 "
                           f"before this module is imported), got tensors on {ts[0].device}")  # fmt: skip


def _on_device(t: torch.Tensor):
    """Triton launches on the current CUDA device: the tensors' (the multi-GPU layouts make it so already)."""
    return torch.cuda.device(t.device) if t.is_cuda else contextlib.nullcontext()


def _slots(nb: int, bodies: list[int]) -> tuple:
    if not bodies or any(not 0 <= b < nb for b in bodies) or len(set(bodies)) != len(bodies):
        raise ValueError(f"kernels.fk: the output bodies must be distinct indices in [0, {nb}), got {bodies}")
    return tuple(bodies.index(b) if b in bodies else -1 for b in range(nb))


@torch.library.custom_op("ndm_gym::fk_fused", mutates_args=())
def fk_fused(p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, tree: str,
             bodies: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """``ArticulationKinematics.forward`` of the registered ``tree`` for the bodies ``bodies``: p, v, w (N, 3), R (N, 3, 3),
    q, qd (N, J), fp32 -> positions (N, B, 3), rotations (N, B, 3, 3), linear and angular velocities (N, B, 3)."""
    t = _tree(tree)
    slot = _slots(t["NB"], bodies)
    _launchable(p, R, v, w, q, qd)
    N, no = p.shape[0], len(bodies)
    out = (p.new_empty(N, no, 3), p.new_empty(N, no, 3, 3), p.new_empty(N, no, 3), p.new_empty(N, no, 3))
    if N:
        (p, sp), (R, sr), (v, sv), (w, sw), (q, sq), (qd, sqd) = (_rows(x) for x in (p, R, v, w, q, qd))
        consts = {k: t[k] for k in ("PARENT", "JOINT", "OFFP", "OFFR", "AXIS", "CROSS", "NB")}
        with _on_device(p):
            _fk_fwd_kernel[(triton.cdiv(N, BLOCK),)](p, R, v, w, q, qd, *out, N, sp, sr, sv, sw, sq, sqd, **consts, SLOT=slot, NO=no, BLOCK=BLOCK,
                                                     num_warps=WARPS)  # fmt: skip
    return out


@fk_fused.register_fake
def _(p, R, v, w, q, qd, tree, bodies):
    N, no = p.shape[0], len(bodies)
    return p.new_empty(N, no, 3), p.new_empty(N, no, 3, 3), p.new_empty(N, no, 3), p.new_empty(N, no, 3)


@torch.library.custom_op("ndm_gym::fk_fused_bwd", mutates_args=())
def fk_fused_bwd(gp: torch.Tensor, gR: torch.Tensor, gv: torch.Tensor, gw: torch.Tensor, R: torch.Tensor, w: torch.Tensor, q: torch.Tensor,
                 qd: torch.Tensor, tree: str, bodies: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                                            torch.Tensor]:  # fmt: skip
    """fk_fused's backward pass: the adjoints of its four outputs and the root's rotation and angular velocity, the
    joints' angles and rates it was called with -> the gradients of p, R, v, w, q and qd."""
    t = _tree(tree)
    slot = _slots(t["NB"], bodies)
    N, no, nj = R.shape[0], len(bodies), t["NJ"]
    shapes = ((N, no, 3), (N, no, 3, 3), (N, no, 3), (N, no, 3))
    if any(tuple(g.shape) != s for g, s in zip((gp, gR, gv, gw), shapes)):
        raise ValueError(f"kernels.fk: the outputs' adjoints must have the shapes {shapes}, got {[tuple(g.shape) for g in (gp, gR, gv, gw)]}")
    _launchable(gp, gR, gv, gw, R, w, q, qd)
    out = (R.new_empty(N, 3), R.new_empty(N, 3, 3), R.new_empty(N, 3), R.new_empty(N, 3), R.new_empty(N, nj), R.new_empty(N, nj))
    if N:
        (R, sr), (w, sw), (q, sq), (qd, sqd) = (_rows(x) for x in (R, w, q, qd))
        consts = {k: t[k] for k in ("PARENT", "JOINT", "OFFP", "OFFR", "AXIS", "CROSS", "KS", "CHILD", "SCHED", "POS", "UPOS", "NJ")}
        with _on_device(R):
            _fk_bwd_kernel[(triton.cdiv(N, BLOCK),)](gp.contiguous(), gR.contiguous(), gv.contiguous(), gw.contiguous(), R, w, q, qd, *out, N, sr,
                                                     sw, sq, sqd, **consts, SLOT=slot, NO=no, BLOCK=BLOCK, num_warps=WARPS)  # fmt: skip
    return out


@fk_fused_bwd.register_fake
def _(gp, gR, gv, gw, R, w, q, qd, tree, bodies):
    N, nj = R.shape[0], _tree(tree)["NJ"]
    return R.new_empty(N, 3), R.new_empty(N, 3, 3), R.new_empty(N, 3), R.new_empty(N, 3), R.new_empty(N, nj), R.new_empty(N, nj)


def _fk_setup(ctx, inputs, output):
    p, R, v, w, q, qd, tree, bodies = inputs
    ctx.save_for_backward(R, w, q, qd)
    ctx.tree, ctx.bodies = tree, bodies


def _fk_backward(ctx, gp, gR, gv, gw):
    R, w, q, qd = ctx.saved_tensors
    return (*fk_fused_bwd(gp, gR, gv, gw, R, w, q, qd, ctx.tree, ctx.bodies), None, None)


fk_fused.register_autograd(_fk_backward, setup_context=_fk_setup)


def forward_kinematics(tree: str, bodies: list[int], p: torch.Tensor, R: torch.Tensor, v: torch.Tensor, w: torch.Tensor, q: torch.Tensor,
                       qd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """``ArticulationKinematics.forward`` by the kernels for the bodies ``bodies`` of the registered ``tree``: positions
    (..., B, 3), rotations (..., B, 3, 3), linear and angular velocities (..., B, 3) from p, v, w (..., 3), R (..., 3, 3)
    and q, qd (..., J) of one batch shape, all fp32; anything else raises."""
    nj, lead = _tree(tree)["NJ"], p.shape[:-1]
    shapes = ((*lead, 3), (*lead, 3, 3), (*lead, 3), (*lead, 3), (*lead, nj), (*lead, nj))
    if any(tuple(t.shape) != s for t, s in zip((p, R, v, w, q, qd), shapes)):
        raise ValueError(f"kernels.fk: p, R, v, w, q, qd of the shapes {shapes}, got {[tuple(t.shape) for t in (p, R, v, w, q, qd)]}")
    if any(t.dtype != torch.float32 for t in (p, R, v, w, q, qd)):
        raise TypeError(f"kernels.fk: fp32 inputs only, got {[str(t.dtype) for t in (p, R, v, w, q, qd)]}")
    no = len(bodies)
    P, Rs, V, W = fk_fused(p.reshape(-1, 3), R.reshape(-1, 3, 3), v.reshape(-1, 3), w.reshape(-1, 3), q.reshape(-1, nj), qd.reshape(-1, nj), tree,
                           list(bodies))  # fmt: skip
    return P.view(*lead, no, 3), Rs.view(*lead, no, 3, 3), V.view(*lead, no, 3), W.view(*lead, no, 3)


__all__ = ["INTERPRETED", "fk_fused", "forward_kinematics", "register"]
