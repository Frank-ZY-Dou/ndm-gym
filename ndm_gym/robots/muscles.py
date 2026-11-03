"""MuscleVAE's muscle model (Feng, Xu and Liu, SIGGRAPH Asia 2023) for a batch of environments, in torch.

From the body poses: the anchor positions (blended between two bodies near a joint), the polyline muscle lengths and
their rates, the Hill-type curves (rigid tendon, no pennation), the muscle-space PD force clipped to the range the
3CC-r fatigue model allows, the activation by inverting the Hill model, the 3CC-r fatigue update, and the anchor forces
summed into one wrench per body. Re-implemented from the paper (Sec. 3, App. A) and the equations of the released code,
wangchek/MuscleVAE_ @ 54c001c, which has no license: nothing is copied from it. References are to that code: BE =
ModifyODESrc/VclSimuBackend.pyx, CHR = Muscle/muscle_character.py, ENV = MuscleVAECore/Env/muscle_env.py.

Conventions: positions in the world frame, orientations as rotation matrices (v_world = R v_body), each body's origin at
its center of mass (the origin of an ODE body, as in their simulator; the converted character keeps it). Per-muscle
tensors are (..., M), per-anchor (..., A, 3), per-body (..., N, 3), for any leading batch shape. Everything at step time
is a fixed sequence of elementwise ops and gathers with precomputed indices: no Python loop over muscles or anchors, no
host synchronization, no matmul (so TF32 matmul settings cannot round it), and the same result on every run for the same
inputs. That holds for the muscle model only: the GPU physics it drives (MuJoCo-Warp, ndm_gym.sim.rigid.muscle_sim) is not
run-to-run deterministic.

Parameter sets (MUSCLE_MODELS), both on the converted table's kp = f0 x manual_muscle_kp4.txt (ENV:101) and
f0max = 1.5 x MuscleVAE_force.txt (ENV:103), with r = 2 and L_D = L_R = 50 (ENV:111-115) at 120 Hz:

* ``released`` (the default) is the released code's behavior, so that its pretrained models run in this port:
  kd = 2 sqrt(kp) (CHR:128-129); fatigue rates F = 0.1, R = 0.02 (0.05 and 0.01 times config.yml's
  fatigue_adjust_ratio 2, ENV:111-113); length rates divided by fps instead of multiplied, (l - l_prev) / 120 (BE:4725-
  4726), which keeps the force-velocity factor within about 1e-3 of 1 and makes the damping term 14,400 times smaller
  than kd (l - l_prev) / dt; and zero velocity in the Hill inversion, because CHR:221 overwrites the previous length
  before CHR:237 reads it.
* ``paper``: kd = 0.1 kp, F = 0.01, R = 0.002 (App. C.1); length rates (l - l_prev) / dt (Eq. 24), also in the
  Hill inversion (Eq. 5).

One substep (ENV:373-381, :meth:`MuscleModel.substep`): lengths from the body poses, the PD force within the bounds of
the current fatigue state, applied at the anchors; the activation that produces it; the fatigue update with that
activation; then the physics step (the caller's). Resets draw the fatigue state as MuscleVAE does
(:meth:`MuscleModel.random_fatigue`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch


NUM_PARTS = 5  # fatigue state: trunk and four limbs (B.4)


@dataclass(frozen=True)
class MuscleTable:
    """Static muscle data (numpy). The anchors of a muscle are contiguous and in path order, from origin to insertion."""

    names: tuple[str, ...]  # (M,)
    num_bodies: int
    anchor_count: np.ndarray  # (M,) int, >= 2
    anchor_body0: np.ndarray  # (A,) int: the body the anchor moves with and its force acts on (the nearer one if blended)
    anchor_local0: np.ndarray  # (A, 3) position in body0's frame
    anchor_body1: np.ndarray  # (A,) int: the other body of a blended anchor, body0 if not blended
    anchor_local1: np.ndarray  # (A, 3) position in body1's frame
    anchor_weight0: np.ndarray  # (A,) blend weight of body0, in (0, 1]; 1 if not blended
    rest_length: np.ndarray  # (M,) length in the T-pose (l_Tpose)
    lm: np.ndarray  # (M,) normalized length: (l / rest_length - lt) / lm
    lt: np.ndarray  # (M,)
    f0: np.ndarray  # (M,) MASS's maximum isometric force: weights of the part fatigue state
    f0max: np.ndarray  # (M,) the Hill model's maximum force
    kp: np.ndarray  # (M,) PD stiffness
    part: np.ndarray  # (M,) int, body part 0..NUM_PARTS-1

    def __post_init__(self):
        M, A = len(self.names), int(np.sum(self.anchor_count))
        shapes = {"anchor_count": (M,), "anchor_body0": (A,), "anchor_local0": (A, 3), "anchor_body1": (A,), "anchor_local1": (A, 3),
                  "anchor_weight0": (A,), **{k: (M,) for k in ("rest_length", "lm", "lt", "f0", "f0max", "kp", "part")}}  # fmt: skip
        for k, shape in shapes.items():
            if np.shape(getattr(self, k)) != shape:
                raise ValueError(f"MuscleTable.{k} has shape {np.shape(getattr(self, k))}, expected {shape}")
        if M == 0 or np.any(self.anchor_count < 2):
            raise ValueError("every muscle needs at least two anchors")
        for k in ("anchor_body0", "anchor_body1"):
            b = getattr(self, k)
            if np.any(b < 0) or np.any(b >= self.num_bodies):
                raise ValueError(f"MuscleTable.{k} indexes a body outside 0..{self.num_bodies - 1}")
        w = self.anchor_weight0
        if np.any(~(w > 0)) or np.any(w > 1):
            raise ValueError("anchor weights must be in (0, 1]")
        if np.any((w == 1) & (self.anchor_body1 != self.anchor_body0)):
            raise ValueError("an anchor with weight 1 has a second body")
        for k in ("rest_length", "lm", "f0", "f0max"):
            if np.any(~(getattr(self, k) > 0)):
                raise ValueError(f"MuscleTable.{k} must be positive")
        if np.any(~(self.kp >= 0)) or np.any(~np.isfinite(self.lt)):
            raise ValueError("MuscleTable.kp must be non-negative and lt finite")
        if np.any(self.part < 0) or np.any(self.part >= NUM_PARTS):
            raise ValueError(f"MuscleTable.part must be in 0..{NUM_PARTS - 1}")

    @property
    def num_muscles(self) -> int:
        return len(self.names)

    @property
    def num_anchors(self) -> int:
        return int(np.sum(self.anchor_count))


KD_RULES = ("2*sqrt(kp)", "0.1*kp")
RATES = ("delta/fps", "delta/dt")  # length rate: (l - l_prev) / sim_hz (the released code) or (l - l_prev) * sim_hz
INVERSION_RATES = ("zero", "pd")  # normalized length rate in the Hill inversion: 0 (the released code) or the PD's


@dataclass(frozen=True)
class MuscleParams:
    """The muscle model's rules and constants; the defaults are the released code's (MUSCLE_MODELS["released"])."""

    kd: str = "2*sqrt(kp)"
    fatigue_F: float = 0.1  # 1/s
    fatigue_R: float = 0.02  # 1/s
    fatigue_r: float = 2.0  # rest recovery multiplier (Eq. 27)
    LD: float = 50.0  # development factor (Eq. 28)
    LR: float = 50.0  # relaxation factor
    sim_hz: float = 120.0  # muscle and fatigue rate
    rate: str = "delta/fps"
    inversion_rate: str = "zero"
    gamma: float = 0.5  # Hill curves (Eq. 25)
    k_pe: float = 4.0
    e_mo: float = 0.6
    fpe_cap: float = 2.0  # passive force cap (BE:4718-4719)

    def __post_init__(self):
        for k, allowed in (("kd", KD_RULES), ("rate", RATES), ("inversion_rate", INVERSION_RATES)):
            if getattr(self, k) not in allowed:
                raise ValueError(f"MuscleParams.{k} = {getattr(self, k)!r}, expected one of {allowed}")
        for k in ("fatigue_F", "fatigue_R", "fatigue_r", "LD", "LR", "sim_hz", "gamma", "e_mo", "fpe_cap"):
            if not getattr(self, k) > 0:
                raise ValueError(f"MuscleParams.{k} must be positive")


MUSCLE_MODELS = {
    "released": MuscleParams(),
    "paper": MuscleParams(kd="0.1*kp", fatigue_F=0.01, fatigue_R=0.002, rate="delta/dt", inversion_rate="pd"),
}


def muscle_params(model: str) -> MuscleParams:
    if model not in MUSCLE_MODELS:
        raise KeyError(f"unknown muscle model {model!r}; known: {sorted(MUSCLE_MODELS)}")
    return MUSCLE_MODELS[model]


def damping_gain(kp, p: MuscleParams):
    """The PD damping kd of stiffnesses kp (numpy or torch) by the parameter set's rule."""
    if p.kd == "2*sqrt(kp)":
        return 2.0 * (torch.sqrt(kp) if isinstance(kp, torch.Tensor) else np.sqrt(kp))
    return 0.1 * kp


def hill_curves(l_norm: torch.Tensor, v_norm: torch.Tensor, p: MuscleParams) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Active force-length, force-velocity and passive force-length factors of the normalized length and its rate
    (Eq. 25, BE:4715-4731). Each force-velocity branch is evaluated on its own side of 0, so neither divides by zero."""
    d = l_norm - 1.0
    fl = torch.exp(-d * d / p.gamma)
    vp, vn = v_norm.clamp(min=0.0), v_norm.clamp(max=0.0)
    fv = torch.where(v_norm > 0.0, 1.5 + 0.5 * (-10.0 + vp) / (37.8 * vp + 10.0), (-10.0 - vn) / (-10.0 + vn * 5.0))
    # the passive force reaches its cap at d_cap; stretches beyond 2 d_cap give the cap without overflowing exp
    d_cap = p.e_mo / p.k_pe * math.log1p(p.fpe_cap * math.expm1(p.k_pe))
    fpe = ((torch.exp(p.k_pe * d.clamp(min=0.0, max=2.0 * d_cap) / p.e_mo) - 1.0) / (math.exp(p.k_pe) - 1.0)).clamp(max=p.fpe_cap)
    return fl, fv, fpe


def activation_bounds(MA: torch.Tensor, MR: torch.Tensor, p: MuscleParams) -> tuple[torch.Tensor, torch.Tensor]:
    """The next activation's range under the 3CC-r model, target load 0 and 1 (Eq. 34-35, BE:4734-4743)."""
    dt = 1.0 / p.sim_hz
    decay = (1.0 - dt * p.fatigue_F) * MA
    return (decay - dt * p.LR * MA).clamp(min=0.0), (decay + dt * p.LD * MR).clamp(max=1.0)


def pd_force(length, l_target, l_rate, kp, kd, f0max, fl, fv, fpe, lo, hi):
    """Muscle-space PD force, clipped to [0, inf) and then to the Hill force at activations [lo, hi] (Eq. 4, 6, 9, 10,
    BE:4732-4758). Returns the force, the PD force before any clipping, and the lower and upper bounds. The bounds are
    tested upper first, as the released code does (they cross only where the force-velocity factor is negative)."""
    f_pd = kp * (length - l_target) + kd * l_rate
    upper = f0max * (hi * fl * fv + fpe)
    lower = f0max * (lo * fl * fv + fpe)
    f = f_pd.clamp(min=0.0)
    f = torch.where(f > upper, upper, torch.where(f < lower, lower, f))
    return f, f_pd, lower, upper


def invert_hill(f, f0max, fl, fv, fpe, lo, hi) -> tuple[torch.Tensor, torch.Tensor]:
    """The activation whose Hill force is f (Eq. 5, BE:4834-4880): the active part of f over its maximum, clipped to
    [lo, hi] (upper bound first). Returns it and the value before the [lo, hi] clip."""
    a_unclipped = (f - fpe * f0max).clamp(min=0.0) / (fl * f0max * fv)
    a = torch.where(a_unclipped > hi, hi, a_unclipped)
    return torch.where(a < lo, lo, a), a_unclipped


def hill_force(a, f0max, fl, fv, fpe) -> torch.Tensor:
    """The Hill-type force of activation a (Eq. 22)."""
    return f0max * (a * fl * fv + fpe)


def fatigue_step(MA, MR, MF, a, p: MuscleParams) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward-Euler step of the 3CC-r model (Eq. 26-28) with the next activation a given (App. A.3.1, BE:3738-3792):
    the transfer C(u*) that leads from MA to a, with the target load u* recovered from a and clipped to [0, 1]; the rest
    recovery multiplier r applies while relaxing. Returns the next (MA, MR, MF) = (a, ...). With a in the bounds of
    :func:`activation_bounds` and MA + MR <= 1, MA + MR + MF is conserved."""
    dt = 1.0 / p.sim_hz
    decay = (1.0 - dt * p.fatigue_F) * MA
    relax = a <= decay
    develop = ~relax & (a <= decay + dt * p.LD * MR)
    u_relax = ((a - decay + dt * p.LR * MA) / (dt * p.LR)).clamp(0.0, 1.0)
    u_develop = ((a - decay + dt * p.LD * MA) / (dt * p.LD)).clamp(0.0, 1.0)
    C = torch.where(relax, p.LR * (u_relax - MA), torch.where(develop, p.LD * (u_develop - MA), p.LD * MR))
    Rr = torch.where(relax, torch.full_like(MA, p.fatigue_r * p.fatigue_R), torch.full_like(MA, p.fatigue_R))
    MR_next = MR - dt * C + dt * Rr * MF
    MF_next = MF + dt * p.fatigue_F * MA - dt * Rr * MF
    return a, MR_next, MF_next


FATIGUE_INIT_LOAD = 0.5  # target load of the initial fatigue curve (CHR:283, 340)
FATIGUE_INIT_TIMES = (20.0, 40.0, 10.0)  # s: the load's duration, the curve's length, the largest random time (CHR:282-283, 343)


def fatigue_init_curve(p: MuscleParams) -> np.ndarray:
    """The curve MuscleVAE draws initial fatigue states from (CHR:280-292, BE:4399-4441): one muscle from (MA, MR, MF) =
    (0, 1, 0) under the target load 0.5 for 20 s, then 0 until 40 s, by forward-Euler steps of the 3CC-r model at
    sim_hz with the target-load controller (BE:3694-3731): relaxing towards the load when MA >= load (rest recovery r R),
    else developing towards it from the resting pool, at most at the rate LD MR. Row k is the state before step k;
    (steps, 3) float64, with their step counts int(sim_hz / (1 / t))."""
    h, load_t, horizon, _ = 1.0 / p.sim_hz, *FATIGUE_INIT_TIMES
    steps, load_steps = int(p.sim_hz / (1.0 / horizon)), int(p.sim_hz / (1.0 / load_t))
    out = np.empty((steps, 3))
    MA, MR, MF = 0.0, 1.0, 0.0
    for k in range(steps):
        out[k] = MA, MR, MF
        load = FATIGUE_INIT_LOAD if k < load_steps else 0.0
        if MA >= load:
            C, Rr = p.LR * (load - MA), p.fatigue_r * p.fatigue_R
        elif MR > load - MA:
            C, Rr = p.LD * (load - MA), p.fatigue_R
        else:
            C, Rr = p.LD * MR, p.fatigue_R
        MA, MR, MF = MA + h * C - h * p.fatigue_F * MA, MR - h * C + h * Rr * MF, MF + h * p.fatigue_F * MA - h * Rr * MF
    return out


def _rows(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """The rows of x (K, ...) at an index of any shape, (*index.shape, ...): copies of whole rows."""
    return torch.index_select(x, 0, index.flatten()).unflatten(0, index.shape)


def _pad0(x: torch.Tensor) -> torch.Tensor:
    """x (K, ...) with a zero row K appended, the target of padding indices."""
    return torch.cat((x, x.new_zeros((1,) + x.shape[1:])), 0)


class MuscleStep(NamedTuple):
    """Everything one substep computes (:meth:`MuscleModel.substep`); fatigue values are the next substep's."""

    anchor_pos: torch.Tensor  # (..., A, 3)
    length: torch.Tensor  # (..., M)
    length_norm: torch.Tensor  # (..., M) normalized length
    rate: torch.Tensor  # (..., M) length rate in the PD
    rate_norm: torch.Tensor  # (..., M) normalized length rate in the force-velocity factor of the bounds
    fl: torch.Tensor  # (..., M) active force-length factor
    fv: torch.Tensor  # (..., M) force-velocity factor of the bounds
    fpe: torch.Tensor  # (..., M) passive force-length factor
    lower: torch.Tensor  # (..., M) force bounds
    upper: torch.Tensor  # (..., M)
    force_pd: torch.Tensor  # (..., M) PD force before clipping
    force: torch.Tensor  # (..., M) applied force (>= 0)
    activation_unclipped: torch.Tensor  # (..., M)
    activation: torch.Tensor  # (..., M) = the next MA
    MA: torch.Tensor  # (..., M) next fatigue state
    MR: torch.Tensor  # (..., M)
    MF: torch.Tensor  # (..., M)
    anchor_force: torch.Tensor  # (..., A, 3)
    body_force: torch.Tensor  # (..., N, 3)
    body_torque: torch.Tensor  # (..., N, 3) about each body's origin


class MuscleModel:
    """The muscle model of one character on one device, for any batch of environments.

    The index tensors built here (anchor points, polyline segments, per-body chunks of anchors) turn every step into a
    fixed sequence of row gathers and elementwise ops. Per-anchor, per-segment and per-body tensors are kept entity-major
    internally, (K, B, ...) for the B environments of the flattened batch, so that each gather copies whole rows; inputs
    and outputs are batch-major, (..., K, ...), the per-anchor outputs as views of the internal layout."""

    CHUNK = 16  # per-body sums go through chunks of this many anchors, which pads far less than one row per body

    def __init__(self, table: MuscleTable, params: MuscleParams | None = None, device: str | torch.device = "cpu", dtype=torch.float32):
        if not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating point type, got {dtype}")
        self.table, self.params = table, params if params is not None else MUSCLE_MODELS["released"]
        self.device, self.dtype = torch.device(device), dtype
        self.num_bodies, self.num_muscles, self.num_anchors = table.num_bodies, table.num_muscles, table.num_anchors
        flt = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype, device=self.device)  # noqa: E731
        idx = lambda x: torch.as_tensor(np.asarray(x, dtype=np.int64), device=self.device)  # noqa: E731
        count = np.asarray(table.anchor_count, dtype=np.int64)
        start = np.cumsum(count) - count
        A, M, N = self.num_anchors, self.num_muscles, self.num_bodies
        # anchor points: every anchor's point on body0, then the second point of each blended anchor (on body1)
        w0 = np.asarray(table.anchor_weight0, dtype=np.float64)
        blend = np.flatnonzero(w0 < 1.0)
        second = np.full(A, len(blend))  # each anchor's second point among the blended ones, len(blend) (a zero) if none
        second[blend] = np.arange(len(blend))
        self.point_body = idx(np.concatenate((table.anchor_body0, np.asarray(table.anchor_body1)[blend])))
        self.point_local = flt(np.concatenate((table.anchor_local0, np.asarray(table.anchor_local1)[blend])))[:, None, :]  # (P, 1, 3)
        self.point_weight = flt(np.concatenate((w0, 1.0 - w0[blend])))[:, None, None]
        self.second = idx(second)
        # segment s joins anchor seg_from[s] to anchor seg_from[s] + 1 of the same muscle; index S is a zero padding entry
        seg_from = np.concatenate([np.arange(s, s + c - 1) for s, c in zip(start, count)])
        S = len(seg_from)
        seg_start = np.cumsum(count - 1) - (count - 1)
        muscle_segs = np.full((M, int(count.max()) - 1), S)
        seg_out, seg_in = np.full(A, S), np.full(A, S)  # the segment leaving / reaching each anchor, S at the path's ends
        for m in range(M):
            k = np.arange(count[m] - 1)
            muscle_segs[m, k] = seg_start[m] + k
            seg_out[start[m] + k] = seg_start[m] + k
            seg_in[start[m] + k + 1] = seg_start[m] + k
        # per-body sums: each body's anchors in chunks of CHUNK (padded with A, a zero), then each body's chunks
        body0 = np.asarray(table.anchor_body0, dtype=np.int64)
        chunks, chunk_body = [], []
        for b in range(N):
            a = np.flatnonzero(body0 == b)
            for c in range(0, len(a), self.CHUNK):
                chunks.append(np.pad(a[c : c + self.CHUNK], (0, self.CHUNK - len(a[c : c + self.CHUNK])), constant_values=A))
                chunk_body.append(b)
        chunk_body = np.array(chunk_body)
        body_chunks = np.full((N, max(1, int(np.bincount(chunk_body, minlength=N).max()))), len(chunks))
        for b in range(N):
            c = np.flatnonzero(chunk_body == b)
            body_chunks[b, : len(c)] = c
        self.seg_from, self.seg_to, self.muscle_segs = idx(seg_from), idx(seg_from + 1), idx(muscle_segs)
        self.seg_out, self.seg_in = idx(seg_out), idx(seg_in)
        self.chunks, self.body_chunks = idx(np.array(chunks)), idx(body_chunks)
        self.anchor_muscle = idx(np.repeat(np.arange(M), count))
        self.anchor_count = idx(count)
        self.body0 = idx(body0)
        self.rest_length, self.lm, self.lt = flt(table.rest_length), flt(table.lm), flt(table.lt)
        self.f0max, self.kp = flt(table.f0max), flt(table.kp)
        self.kd = damping_gain(self.kp, self.params)
        # part fatigue state: each muscle weighted by f0 within its part (B.4, ENV:116-117, BE:4583-4609)
        part = np.asarray(table.part, dtype=np.int64)
        f0 = np.asarray(table.f0, dtype=np.float64)
        part_sum = np.bincount(part, weights=f0, minlength=NUM_PARTS)
        if np.any(part_sum == 0):
            raise ValueError(f"every body part needs a muscle; muscles per part: {np.bincount(part, minlength=NUM_PARTS).tolist()}")
        self.part_weight = flt(np.eye(NUM_PARTS)[part] * (f0 / part_sum[part])[:, None])  # (M, NUM_PARTS)
        self.part = idx(part)
        self.fatigue_curve = flt(fatigue_init_curve(self.params))  # (steps, 3): MA, MR, MF

    # ---- layout and checks
    def _check_bodies(self, body_pos: torch.Tensor, body_rot: torch.Tensor | None = None):
        if body_pos.shape[-2:] != (self.num_bodies, 3):
            raise ValueError(f"body positions must be (..., {self.num_bodies}, 3), got {tuple(body_pos.shape)}")
        if body_rot is not None and (body_rot.shape[-3:] != (self.num_bodies, 3, 3) or body_rot.shape[:-3] != body_pos.shape[:-2]):
            raise ValueError(f"body rotations must be (..., {self.num_bodies}, 3, 3) with the positions' batch shape, got {tuple(body_rot.shape)}")

    def _check_muscles(self, batch: torch.Size | None = None, **xs: torch.Tensor):
        for k, x in xs.items():
            if x.shape[-1:] != (self.num_muscles,) or (batch is not None and x.shape[:-1] != batch):
                want = tuple(batch) + (self.num_muscles,) if batch is not None else f"(..., {self.num_muscles})"
                raise ValueError(f"{k} must be {want}, got {tuple(x.shape)}")

    def _entity_major(self, x: torch.Tensor, rows: int, name: str, batch: torch.Size | None = None) -> tuple[torch.Tensor, torch.Size]:
        """(..., rows, 3) -> (rows, B, 3), and the batch shape (which must be ``batch`` if given)."""
        if x.shape[-2:] != (rows, 3) or (batch is not None and x.shape[:-2] != batch):
            raise ValueError(f"{name} must be ({'...' if batch is None else ', '.join(map(str, batch))}, {rows}, 3), got {tuple(x.shape)}")
        return x.reshape(-1, rows, 3).transpose(0, 1).contiguous(), x.shape[:-2]

    @staticmethod
    def _batch_major(x: torch.Tensor, batch: torch.Size) -> torch.Tensor:
        """(K, B, C) -> (..., K, C), a view."""
        return x.transpose(0, 1).reshape(batch + (x.shape[0], x.shape[2]))

    # ---- geometry, entity-major
    def _anchors(self, frames: torch.Tensor) -> torch.Tensor:
        """Anchor positions (A, B, 3) from body frames (N, B, 12): row-major rotation and origin."""
        g = _rows(frames, self.point_body)  # (P, B, 12): each point's body frame
        rot, x = g[..., :9].unflatten(-1, (3, 3)), self.point_local
        points = torch.addcmul(torch.addcmul(torch.addcmul(g[..., 9:], rot[..., 0], x[..., 0:1]), rot[..., 1], x[..., 1:2]), rot[..., 2], x[..., 2:3])
        points = points * self.point_weight
        return points[: self.num_anchors] + _rows(_pad0(points[self.num_anchors :]), self.second)

    def _segments(self, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Each polyline segment's vector (S, B, 3) and length (S, B)."""
        d = _rows(anchors, self.seg_to) - _rows(anchors, self.seg_from)
        return d, torch.sqrt((d * d).sum(-1))

    def _lengths(self, seg_len: torch.Tensor) -> torch.Tensor:
        """Muscle lengths (B, M) from segment lengths (S, B)."""
        return _rows(_pad0(seg_len), self.muscle_segs).sum(1).t().contiguous()

    def _anchor_forces(self, seg: torch.Tensor, seg_len: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """Anchor forces (A, B, 3) from the segments and the muscle forces (B, M)."""
        e = _pad0(seg / seg_len.unsqueeze(-1))
        return (_rows(e, self.seg_out) - _rows(e, self.seg_in)) * _rows(force.t().contiguous(), self.anchor_muscle).unsqueeze(-1)

    def _wrench(self, anchors: torch.Tensor, anchor_force: torch.Tensor, body_pos: torch.Tensor) -> torch.Tensor:
        """Per-body force and torque about the origin (N, B, 6) from anchors and their forces (A, B, 3), origins (N, B, 3)."""
        r0, r1, r2 = (anchors - _rows(body_pos, self.body0)).unbind(-1)
        f0, f1, f2 = anchor_force.unbind(-1)
        wrench = _pad0(torch.stack((f0, f1, f2, r1 * f2 - r2 * f1, r2 * f0 - r0 * f2, r0 * f1 - r1 * f0), -1))  # (A + 1, B, 6): force, r x force
        return _rows(_pad0(_rows(wrench, self.chunks).sum(1)), self.body_chunks).sum(1)

    # ---- geometry
    def anchor_positions(self, body_pos: torch.Tensor, body_rot: torch.Tensor) -> torch.Tensor:
        """World anchor positions (..., A, 3) from body origins (..., N, 3) and rotations (..., N, 3, 3): a blended anchor
        is w0 (R_0 x_0 + p_0) + (1 - w0) (R_1 x_1 + p_1) (Eq. 18, BE:2905-2970), any other R_0 x_0 + p_0."""
        self._check_bodies(body_pos, body_rot)
        frames = torch.cat((body_rot.flatten(-2), body_pos), -1).reshape(-1, self.num_bodies, 12).transpose(0, 1).contiguous()
        return self._batch_major(self._anchors(frames), body_pos.shape[:-2])

    def lengths(self, anchor_pos: torch.Tensor) -> torch.Tensor:
        """Muscle lengths (..., M): the polyline through the anchors (Eq. 19, BE:3019-3031)."""
        anchors, batch = self._entity_major(anchor_pos, self.num_anchors, "anchor positions")
        return self._lengths(self._segments(anchors)[1]).reshape(batch + (self.num_muscles,))

    def anchor_forces(self, anchor_pos: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """Force vectors at the anchors (..., A, 3) for muscle forces (..., M): f times the unit vector to the next anchor
        plus the unit vector to the previous one, one of them at the path's ends (Eq. 20, BE:3597-3663)."""
        anchors, batch = self._entity_major(anchor_pos, self.num_anchors, "anchor positions")
        self._check_muscles(batch, force=force)
        return self._batch_major(self._anchor_forces(*self._segments(anchors), force.reshape(-1, self.num_muscles)), batch)

    def body_wrench(self, anchor_pos: torch.Tensor, anchor_force: torch.Tensor, body_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per body (..., N, 3): the sum of the forces at its anchors and of their moments about its origin. Each anchor
        pushes only its body0, the nearer body of a blended anchor, at the anchor's blended position (CHR:218-220,
        BE:2882-2899, dBodyAddForceAtPos)."""
        anchors, batch = self._entity_major(anchor_pos, self.num_anchors, "anchor positions")
        forces, _ = self._entity_major(anchor_force, self.num_anchors, "anchor forces", batch)
        origins, _ = self._entity_major(body_pos, self.num_bodies, "body positions", batch)
        wrench = self._batch_major(self._wrench(anchors, forces, origins), batch)
        return wrench[..., :3], wrench[..., 3:]

    def anchor_velocities(self, body_pos: torch.Tensor, body_rot: torch.Tensor, body_vel: torch.Tensor, body_angvel: torch.Tensor) -> torch.Tensor:
        """World anchor velocities (..., A, 3) from the bodies' poses, origin velocities and angular velocities (..., N, 3):
        the time derivative of :meth:`anchor_positions`, sum of w (v + omega x R x) over an anchor's points, which is
        that map with R replaced by its derivative [omega]x R and p by v."""
        self._check_bodies(body_pos, body_rot)
        self._check_bodies(body_vel), self._check_bodies(body_angvel)
        rot_rate = torch.cross(body_angvel.unsqueeze(-1).expand_as(body_rot), body_rot, dim=-2)  # column j: omega x R[:, j]
        return self.anchor_positions(body_vel, rot_rate)

    def length_rates(self, anchor_pos: torch.Tensor, anchor_vel: torch.Tensor) -> torch.Tensor:
        """Time derivatives of the muscle lengths (..., M) from the anchors' positions and velocities (..., A, 3): over each
        segment, its ends' relative velocity along it."""
        anchors, batch = self._entity_major(anchor_pos, self.num_anchors, "anchor positions")
        vel, _ = self._entity_major(anchor_vel, self.num_anchors, "anchor velocities", batch)
        seg, seg_len = self._segments(anchors)
        dv = _rows(vel, self.seg_to) - _rows(vel, self.seg_from)
        return self._lengths((seg * dv).sum(-1) / seg_len).reshape(batch + (self.num_muscles,))

    def path_points(self, anchor_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per muscle (..., M, 3): the point halfway along its polyline (by arc length), and the unit directions of its
        first and last segments (from the origin towards the insertion)."""
        anchors, batch = self._entity_major(anchor_pos, self.num_anchors, "anchor positions")
        seg, seg_len = self._segments(anchors)  # (S, B, 3), (S, B)
        L = _rows(_pad0(seg_len), self.muscle_segs)  # (M, K, B): each muscle's segments, zeros past its last
        cum = L.cumsum(1)
        half = 0.5 * cum[:, -1:]
        k = (cum < half).sum(1, keepdim=True)  # (M, 1, B): the segment that holds the midpoint
        seg_k = torch.gather(self.muscle_segs.unsqueeze(-1).expand_as(L), 1, k).squeeze(1)  # (M, B)
        t = (half - torch.gather(cum - L, 1, k)).squeeze(1) / torch.gather(L, 1, k).squeeze(1)
        rows = lambda x, i: torch.gather(x, 0, i.unsqueeze(-1).expand(-1, -1, x.shape[-1]))  # noqa: E731  x (S + 1, B, C), i (M, B)
        mid = rows(_rows(anchors, self.seg_from), seg_k) + t.unsqueeze(-1) * rows(seg, seg_k)
        unit = seg / seg_len.unsqueeze(-1)
        last = self.muscle_segs.gather(1, (self.anchor_count - 2).unsqueeze(-1)).squeeze(-1)  # (M,)
        first, last = unit[self.muscle_segs[:, 0]], unit[last]  # (M, B, 3)
        return tuple(self._batch_major(x, batch) for x in (mid, first, last))

    def muscle_bodies(self) -> list[set[int]]:
        """The bodies each muscle's anchors move with (both bodies of a blended anchor)."""
        t, out, a = self.table, [], 0
        for c in np.asarray(t.anchor_count):
            out.append({int(b) for b in np.concatenate((t.anchor_body0[a : a + c], t.anchor_body1[a : a + c]))})
            a += c
        return out

    # ---- muscle dynamics
    def normalized_length(self, length: torch.Tensor) -> torch.Tensor:
        """(l / l_Tpose - lt) / lm (Eq. 23, BE:4707-4714); also their scaled muscle length (BE:3225-3245)."""
        return (length / self.rest_length - self.lt) / self.lm

    def target_length(self, action: torch.Tensor) -> torch.Tensor:
        """PD target lengths (1 + a) l_Tpose of actions (..., M) (Eq. 11, ENV:431-433)."""
        self._check_muscles(action=action)
        return self.rest_length * (1.0 + action)

    def rates(self, length: torch.Tensor, length_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Length rate (the PD's) and normalized length rate (the force-velocity factor's) from the previous substep's
        lengths: the change times sim_hz, or divided by it with ``rate="delta/fps"`` (the released code, BE:4725-4726)."""
        self._check_muscles(length=length, length_prev=length_prev)
        dl = length - length_prev
        dln = self.normalized_length(length) - self.normalized_length(length_prev)
        hz = self.params.sim_hz
        return (dl / hz, dln / hz) if self.params.rate == "delta/fps" else (dl * hz, dln * hz)

    def part_fatigue(self, MA: torch.Tensor, MR: torch.Tensor, MF: torch.Tensor) -> torch.Tensor:
        """The fatigue state (..., 15): f0-weighted means of MA, MR and MF over each body part, in that order (B.4,
        CHR:333-337, motion_utils.character_muscle_state)."""
        self._check_muscles(MA=MA, MR=MR, MF=MF)
        return torch.cat([(x.unsqueeze(-1) * self.part_weight).sum(-2) for x in (MA, MR, MF)], -1)

    def random_fatigue(self, batch: tuple[int, ...], generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MuscleVAE's random initial fatigue state (ENV:302-305; mode 0 of CHR:339-356, BE:4448-4502): for each body
        part a time uniform in [0, 10) s and the state of :func:`fatigue_init_curve` at step int(t / (1 / sim_hz)); every
        muscle takes its part's state. Returns (MA, MR, MF), each (*batch, M)."""
        t = FATIGUE_INIT_TIMES[2] * torch.rand(tuple(batch) + (NUM_PARTS,), generator=generator, dtype=torch.float64, device=self.device)
        state = self.fatigue_curve[(t / (1.0 / self.params.sim_hz)).long()]  # (*batch, parts, 3)
        return tuple(x.contiguous() for x in state[..., self.part, :].unbind(-1))

    def substep(self, body_pos, body_rot, length_prev, length_target, MA, MR, MF) -> MuscleStep:
        """One muscle substep from the body poses (..., N, 3) and (..., N, 3, 3), the previous substep's lengths, the PD
        targets and the fatigue state (..., M) (ENV:373-381): the anchor forces and wrenches to apply before the physics
        step, and the next fatigue state."""
        p, M = self.params, self.num_muscles
        self._check_bodies(body_pos, body_rot)
        batch = body_pos.shape[:-2]
        self._check_muscles(batch, length_prev=length_prev, length_target=length_target, MA=MA, MR=MR, MF=MF)
        frames = torch.cat((body_rot.flatten(-2), body_pos), -1).reshape(-1, self.num_bodies, 12).transpose(0, 1).contiguous()
        flat = [x.reshape(-1, M) for x in (length_prev, length_target, MA, MR, MF)]
        length_prev, length_target, MA, MR, MF = flat
        anchors = self._anchors(frames)
        seg, seg_len = self._segments(anchors)
        length = self._lengths(seg_len)
        length_norm = self.normalized_length(length)
        rate, rate_norm = self.rates(length, length_prev)
        fl, fv, fpe = hill_curves(length_norm, rate_norm, p)
        lo, hi = activation_bounds(MA, MR, p)
        force, force_pd, lower, upper = pd_force(length, length_target, rate, self.kp, self.kd, self.f0max, fl, fv, fpe, lo, hi)
        fv_inv = torch.ones_like(fv) if p.inversion_rate == "zero" else fv  # the released code: fv(0) = 1 exactly
        activation, activation_unclipped = invert_hill(force, self.f0max, fl, fv_inv, fpe, lo, hi)
        MA_next, MR_next, MF_next = fatigue_step(MA, MR, MF, activation, p)
        anchor_force = self._anchor_forces(seg, seg_len, force)
        wrench = self._batch_major(self._wrench(anchors, anchor_force, frames[..., 9:].contiguous()), batch)
        per_muscle = [x.reshape(batch + (M,)) for x in (length, length_norm, rate, rate_norm, fl, fv, fpe, lower, upper, force_pd, force,
                                                         activation_unclipped, activation, MA_next, MR_next, MF_next)]  # fmt: skip
        return MuscleStep(self._batch_major(anchors, batch), *per_muscle, self._batch_major(anchor_force, batch), wrench[..., :3], wrench[..., 3:])
