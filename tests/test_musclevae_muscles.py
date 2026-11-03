"""MuscleVAE's muscle model (ndm_gym.robots.muscles) on random inputs, without data: each formula against an independent
reference written from the paper's equations (Feng, Xu and Liu, SIGGRAPH Asia 2023) and the rules MuscleParams
documents for the released model (wangchek/MuscleVAE_ @ 54c001c; BE = ModifyODESrc/VclSimuBackend.pyx, CHR =
Muscle/muscle_character.py, ENV = MuscleVAECore/Env/muscle_env.py), the inversion's round trip, the conservation of the
3CC-r compartments, the per-body wrench, the random initial fatigue, the two parameter sets, shapes and batching."""

import dataclasses
import math

import numpy as np
import pytest
import torch

from ndm_gym.core.rotations import so3_exp
from ndm_gym.robots.muscles import (
    MUSCLE_MODELS, NUM_PARTS, MuscleModel, MuscleParams, MuscleTable, activation_bounds, fatigue_init_curve, fatigue_step, hill_curves, hill_force,
    invert_hill, muscle_params, pd_force,
)  # fmt: skip

RELEASED, PAPER = MUSCLE_MODELS["released"], MUSCLE_MODELS["paper"]
N_BODIES = 5
DT = torch.float64
_g = torch.Generator().manual_seed(1234)
REF_POS = 0.3 * torch.randn(N_BODIES, 3, generator=_g, dtype=DT)  # the rest pose
REF_ROT = so3_exp(0.5 * torch.randn(N_BODIES, 3, generator=_g, dtype=DT))


def random_table(seed: int = 0, num_muscles: int = 9) -> MuscleTable:
    """Random muscles on N_BODIES bodies, about 40% of the anchors blended; rest lengths in the pose REF_POS, REF_ROT."""
    g = np.random.default_rng(seed)
    count = g.integers(2, 7, num_muscles)
    A = int(count.sum())
    body0 = g.integers(0, N_BODIES, A)
    blend = g.random(A) < 0.4
    body1 = np.where(blend, (body0 + g.integers(1, N_BODIES, A)) % N_BODIES, body0)
    local0 = 0.1 * g.normal(size=(A, 3))
    t = MuscleTable(
        names=tuple(f"m{i}" for i in range(num_muscles)), num_bodies=N_BODIES, anchor_count=count, anchor_body0=body0, anchor_local0=local0,
        anchor_body1=body1, anchor_local1=np.where(blend[:, None], 0.1 * g.normal(size=(A, 3)), local0),
        anchor_weight0=np.where(blend, 0.5 + 0.5 * g.random(A), 1.0), rest_length=np.ones(num_muscles),
        lm=g.choice([1.0, 1.2], num_muscles), lt=np.full(num_muscles, 0.2), f0=100 + 900 * g.random(num_muscles),
        f0max=1.5 * (100 + 1000 * g.random(num_muscles)), kp=(100 + 900 * g.random(num_muscles)) * g.choice([1.0, 1.5, 2.0], num_muscles),
        part=np.arange(num_muscles) % NUM_PARTS,
    )  # fmt: skip
    return dataclasses.replace(t, rest_length=ref_geometry(t, REF_POS.numpy(), REF_ROT.numpy(), np.zeros(num_muscles))[1])


def random_bodies(gen: torch.Generator, *batch: int, scale: float = 1.0):
    """Body poses near the rest pose: turned by about 0.2 rad, moved by about 3 cm (times scale)."""
    pos = REF_POS + 0.03 * scale * torch.randn(*batch, N_BODIES, 3, generator=gen, dtype=DT)
    return pos, so3_exp(0.2 * scale * torch.randn(*batch, N_BODIES, 3, generator=gen, dtype=DT)) @ REF_ROT


def random_fatigue(gen: torch.Generator, *shape: int):
    """(MA, MR, MF) >= 0 with MA + MR + MF = 1."""
    x = -torch.log(torch.rand(*shape, 3, generator=gen, dtype=DT))
    x = x / x.sum(-1, keepdim=True)
    return x[..., 0], x[..., 1], x[..., 2]


# ---- independent references, written from the paper's equations (Sec. 3, App. A) and the rules MuscleParams documents
# for the released model, vectorized over numpy arrays; each branch of a piecewise curve evaluated where it applies
FV_VMAX, FV_SHORTENING_CURVATURE = 10.0, 0.2  # force-velocity, shortening: (Vmax + v) / (Vmax - v / curvature)
FV_LENGTHENING_LIMIT, FV_LENGTHENING_SLOPE = 1.5, 37.8  # lengthening: limit - (limit - 1) (Vmax - v) / (Vmax + slope v)


def ref_curves(l_norm, v_norm, p: MuscleParams):
    """The Hill-type factors of the normalized fiber length and its rate (Eq. 25): the active force-length
    exp(-(l - 1)^2 / gamma); the force-velocity factor, 1 at rest, falling to 0 at the maximal shortening velocity and
    rising towards its lengthening limit; the passive force-length (exp(k_pe (l - 1) / e_mo) - 1) / (exp(k_pe) - 1) for
    a stretched fiber, up to its cap."""
    active = np.exp(-np.square(l_norm - 1.0) / p.gamma)
    shortening, lengthening = np.minimum(v_norm, 0.0), np.maximum(v_norm, 0.0)
    velocity = np.where(v_norm > 0.0,
                        FV_LENGTHENING_LIMIT - (FV_LENGTHENING_LIMIT - 1.0) * (FV_VMAX - lengthening) / (FV_VMAX + FV_LENGTHENING_SLOPE * lengthening),
                        (FV_VMAX + shortening) / (FV_VMAX - shortening / FV_SHORTENING_CURVATURE))  # fmt: skip
    passive = np.minimum(np.expm1(p.k_pe * np.maximum(l_norm - 1.0, 0.0) / p.e_mo) / np.expm1(p.k_pe), p.fpe_cap)
    return active, velocity, passive


def ref_lengths(length, prev, rest, lm, lt, p: MuscleParams):
    """The normalized fiber length (l / l_rest - l_t) / l_m, and the rates of the length and of the normalized length
    over a substep: the change times sim_hz (the paper's derivative) or, in the released code, divided by it."""

    def fiber(x):
        return (x / rest - lt) / lm

    per_substep = p.sim_hz if p.rate == "delta/dt" else 1.0 / p.sim_hz
    return fiber(length), (length - prev) * per_substep, (fiber(length) - fiber(prev)) * per_substep


def ref_transfer(load, MA, MR, p: MuscleParams):
    """The 3CC-r controller's transfer rate C from the resting to the active compartment for a target load u (Eq. 28):
    relaxing towards a load at or below the active fraction at L_R (u - MA); developing towards a higher one at
    L_D (u - MA), at most L_D MR when the resting pool cannot supply it."""
    return np.where(MA >= load, p.LR * (load - MA), np.where(MR > load - MA, p.LD * (load - MA), p.LD * MR))


def ref_bounds(MA, MR, p: MuscleParams):
    """The next activation's range (Eq. 34-35): the active fraction one explicit Euler step later, MA + h (C(u) - F MA),
    at the extreme target loads u = 0 and u = 1, within [0, 1]; states with MA + MR + MF = 1."""
    h = 1.0 / p.sim_hz

    def next_active(load):
        return MA + h * (ref_transfer(np.full_like(MA, load), MA, MR, p) - p.fatigue_F * MA)

    return np.clip(next_active(0.0), 0.0, 1.0), np.clip(next_active(1.0), 0.0, 1.0)


def ref_pd(length, prev, target, rest, lm, lt, kp, kd, f0max, MA, MR, p: MuscleParams):
    """The muscle-space PD force kp (l - l_target) + kd dl/dt (Eq. 4), which only pulls (at least 0), kept within the
    Hill forces f0max (a fL fV + fPE) at the activation bounds (Eq. 6, 9, 10), the upper bound applied first. Returns the
    force, the PD force, the lower and the upper bound."""
    l_norm, rate, rate_norm = ref_lengths(length, prev, rest, lm, lt, p)
    fl, fv, fpe = ref_curves(l_norm, rate_norm, p)
    lo, hi = ref_bounds(MA, MR, p)
    lower, upper = f0max * (lo * fl * fv + fpe), f0max * (hi * fl * fv + fpe)
    pd = kp * (length - target) + kd * rate
    pulling = np.maximum(pd, 0.0)
    return np.select([pulling > upper, pulling < lower], [upper, lower], pulling), pd, lower, upper


def ref_activation(force, length, prev, rest, lm, lt, f0max, MA, MR, p: MuscleParams):
    """The activation whose Hill force f0max (a fL fV + fPE) is ``force`` (Eq. 5, 22): the active part of the force over
    f0max fL fV, at least 0, then within the activation bounds, the upper applied first; the factors at the length and
    rate of (length, prev). Returns it and the value before the bounds."""
    l_norm, _, rate_norm = ref_lengths(length, prev, rest, lm, lt, p)
    fl, fv, fpe = ref_curves(l_norm, rate_norm, p)
    lo, hi = ref_bounds(MA, MR, p)
    a = np.maximum(force - f0max * fpe, 0.0) / (f0max * fl * fv)
    return np.select([a > hi, a < lo], [hi, lo], a), a


def ref_fatigue(MA, MR, MF, a, p: MuscleParams):
    """One explicit Euler step of the 3CC-r model (Eq. 26-28) that brings the active compartment to the activation a
    (App. A.3.1): the target load the controller turns into a, clipped to [0, 1], found on the branch a falls on
    (relaxing at or below the active fraction left after fatigue, (1 - h F) MA; developing up to what the resting pool
    supplies in a step; saturated beyond), its transfer, and the rest recovery r R while relaxing."""
    h = 1.0 / p.sim_hz
    remaining = (1.0 - h * p.fatigue_F) * MA
    relaxing = a <= remaining
    saturated = ~relaxing & (a > remaining + h * p.LD * MR)
    gain = np.where(relaxing, p.LR, p.LD)
    load = np.clip(MA + (a - remaining) / (h * gain), 0.0, 1.0)
    transfer = np.where(saturated, p.LD * MR, gain * (load - MA))
    recovery = np.where(relaxing, p.fatigue_r * p.fatigue_R, p.fatigue_R)
    return a, MR + h * (recovery * MF - transfer), MF + h * (p.fatigue_F * MA - recovery * MF)


def ref_fatigue_curve(p: MuscleParams) -> np.ndarray:
    """The initial-fatigue curve as MuscleVAE documents it: one muscle from rest, (MA, MR, MF) = (0, 1, 0), held at the
    target load 0.5 for 20 s and released for 20 s more, by explicit Euler steps of the 3CC-r model at sim_hz under its
    controller (:func:`ref_transfer`, the rest recovery r R while relaxing), each term of each rate times the step;
    the state before each step."""
    h, hold_steps, all_steps = 1.0 / p.sim_hz, round(20.0 * p.sim_hz), round(40.0 * p.sim_hz)
    state, rows = (0.0, 1.0, 0.0), []
    for step in range(all_steps):
        rows.append(state)
        active, resting, fatigued = state
        load = 0.5 if step < hold_steps else 0.0
        transfer = float(ref_transfer(load, active, resting, p))
        recovery = p.fatigue_r * p.fatigue_R if active >= load else p.fatigue_R
        state = (active + h * transfer - h * p.fatigue_F * active, resting - h * transfer + h * recovery * fatigued,
                 fatigued + h * p.fatigue_F * active - h * recovery * fatigued)  # fmt: skip
    return np.array(rows)


def ref_geometry(t: MuscleTable, pos: np.ndarray, rot: np.ndarray, force: np.ndarray):
    """Anchor positions, lengths, anchor forces and wrenches of one environment, anchor by anchor (BE:2905-2970,
    3019-3031, 3597-3663, 2882-2899)."""
    per_anchor = zip(t.anchor_body0, t.anchor_local0, t.anchor_body1, t.anchor_local1, t.anchor_weight0)
    anchors = np.array([w * (rot[b0] @ x0 + pos[b0]) + (1 - w) * (rot[b1] @ x1 + pos[b1]) for b0, x0, b1, x1, w in per_anchor])
    lengths, forces, start = [], np.zeros_like(anchors), 0
    for m, n in enumerate(t.anchor_count):
        p = anchors[start : start + n]
        lengths.append(sum(math.sqrt(((p[k] - p[k + 1]) ** 2).sum()) for k in range(n - 1)))
        for k in range(n):
            d = 0.0
            if k < n - 1:
                d = d + (p[k + 1] - p[k]) / np.linalg.norm(p[k + 1] - p[k])
            if k > 0:
                d = d + (p[k - 1] - p[k]) / np.linalg.norm(p[k - 1] - p[k])
            forces[start + k] = d * force[m]
        start += n
    body_force, body_torque = np.zeros((t.num_bodies, 3)), np.zeros((t.num_bodies, 3))
    for a, b in enumerate(t.anchor_body0):  # dBodyAddForceAtPos on body0
        body_force[b] += forces[a]
        body_torque[b] += np.cross(anchors[a] - pos[b], forces[a])
    return anchors, np.array(lengths), forces, body_force, body_torque


def close(a, b, tol=1e-12):
    torch.testing.assert_close(torch.as_tensor(np.asarray(a), dtype=DT), torch.as_tensor(np.asarray(b), dtype=DT), rtol=tol, atol=tol)


# ---- tests
def test_parameter_sets():
    rules = lambda p: (p.kd, p.fatigue_F, p.fatigue_R, p.rate, p.inversion_rate)  # noqa: E731
    assert rules(RELEASED) == ("2*sqrt(kp)", 0.1, 0.02, "delta/fps", "zero")
    assert rules(PAPER) == ("0.1*kp", 0.01, 0.002, "delta/dt", "pd")
    for p in (RELEASED, PAPER):
        assert (p.fatigue_r, p.LD, p.LR, p.sim_hz, p.gamma, p.k_pe, p.e_mo, p.fpe_cap) == (2.0, 50.0, 50.0, 120.0, 0.5, 4.0, 0.6, 2.0)
    assert muscle_params("released") is RELEASED and muscle_params("paper") is PAPER
    with pytest.raises(KeyError):
        muscle_params("odd")
    for bad in ({"kd": "kp"}, {"rate": "delta"}, {"inversion_rate": "same"}, {"fatigue_F": 0.0}, {"sim_hz": -1.0}):
        with pytest.raises(ValueError):
            MuscleParams(**bad)
    t = random_table()
    kp = torch.tensor(t.kp, dtype=DT)
    close(MuscleModel(t, RELEASED, dtype=DT).kd, 2.0 * torch.sqrt(kp))  # CHR:128-129
    close(MuscleModel(t, PAPER, dtype=DT).kd, 0.1 * kp)
    assert MuscleModel(t).params is RELEASED and MuscleModel(t).dtype == torch.float32


def test_table_validation():
    t = random_table()
    fields = {k: getattr(t, k) for k in t.__dataclass_fields__}
    for k, v in (("anchor_count", np.r_[t.anchor_count[:-1], 1]), ("anchor_body0", t.anchor_body0 + N_BODIES), ("lm", t.lm[:-1]),
                 ("anchor_weight0", t.anchor_weight0 * 0), ("rest_length", -t.rest_length), ("part", t.part + NUM_PARTS), ("kp", -t.kp)):  # fmt: skip
        with pytest.raises(ValueError):
            MuscleTable(**{**fields, k: v})
    w = t.anchor_weight0.copy()
    w[np.flatnonzero(t.anchor_body1 != t.anchor_body0)[0]] = 1.0  # a second body without a blend
    with pytest.raises(ValueError):
        MuscleTable(**{**fields, "anchor_weight0": w})
    with pytest.raises(ValueError):  # a body part without muscles
        MuscleModel(MuscleTable(**{**fields, "part": np.zeros_like(t.part)}))


def test_hill_curves():
    g = torch.Generator().manual_seed(1)
    ln = 0.3 + 2.2 * torch.rand(4000, generator=g, dtype=DT)  # below and above 1, past the passive cap (about 1.7)
    vn = 30 * torch.rand(4000, generator=g, dtype=DT) - 15  # both branches, below the zero of the shortening branch (-10)
    vn[:5] = 0.0
    for p in (RELEASED, PAPER):
        fl, fv, fpe = hill_curves(ln, vn, p)
        close(torch.stack((fl, fv, fpe), -1), np.stack(ref_curves(ln.numpy(), vn.numpy(), p), -1))
    assert (fpe == 2.0).any() and (fpe == 0.0).any() and (fv < 0).any()
    fl, fv, fpe = hill_curves(torch.tensor([1.0, 0.7], dtype=DT), torch.tensor([0.0, 0.0], dtype=DT), RELEASED)
    assert fl[0] == 1.0 and (fv == 1.0).all() and (fpe == 0.0).all()


def test_pd_force_and_activation():
    t = random_table(2)
    g = torch.Generator().manual_seed(2)
    S = 6000
    m = torch.randint(0, t.num_muscles, (S,), generator=g)
    rest, lm, lt = (torch.tensor(x, dtype=DT)[m] for x in (t.rest_length, t.lm, t.lt))
    f0max, kp = torch.tensor(t.f0max, dtype=DT)[m], torch.tensor(t.kp, dtype=DT)[m]
    length = rest * (0.6 + 0.9 * torch.rand(S, generator=g, dtype=DT))
    prev = length + 0.02 * torch.randn(S, generator=g, dtype=DT)
    target = rest * (0.6 + 0.9 * torch.rand(S, generator=g, dtype=DT))
    MA, MR, MF = random_fatigue(g, S)
    for p in (RELEASED, PAPER):
        kd = 2.0 * torch.sqrt(kp) if p is RELEASED else 0.1 * kp
        ln, dl, dln = (torch.as_tensor(x) for x in ref_lengths(*(x.numpy() for x in (length, prev, rest, lm, lt)), p))
        fl, fv, fpe = hill_curves(ln, dln, p)
        lo, hi = activation_bounds(MA, MR, p)
        close(torch.stack((lo, hi), -1), np.stack(ref_bounds(MA.numpy(), MR.numpy(), p), -1))
        f, f_pd, lower, upper = pd_force(length, target, dl, kp, kd, f0max, fl, fv, fpe, lo, hi)
        close(torch.stack((f, f_pd, lower, upper), -1), np.stack(ref_pd(*(x.numpy() for x in (length, prev, target, rest, lm, lt, kp, kd, f0max, MA, MR)), p), -1),
              1e-10)  # fmt: skip
        assert (f_pd < 0).any() and (f == upper).any() and (f == lower).any() and ((f > lower) & (f < upper)).any()  # every clip case
        # the activation, with the previous length the inversion sees: the current one in the released code (CHR:221, 237)
        inv_prev = length if p.inversion_rate == "zero" else prev
        fv_inv = torch.ones_like(fv) if p.inversion_rate == "zero" else fv
        a, a_unc = invert_hill(f, f0max, fl, fv_inv, fpe, lo, hi)
        close(torch.stack((a, a_unc), -1), np.stack(ref_activation(*(x.numpy() for x in (f, length, inv_prev, rest, lm, lt, f0max, MA, MR)), p), -1), 1e-10)
        # round trip: an activation inside the bounds reproduces the force through the Hill model
        inside = (a_unc > lo) & (a_unc < hi) & (f > fpe * f0max)
        assert inside.sum() > 100
        close(hill_force(a, f0max, fl, fv_inv, fpe)[inside], f[inside], 1e-10)


def test_fatigue_step():
    g = torch.Generator().manual_seed(3)
    MA, MR, MF = random_fatigue(g, 5000)
    for p in (RELEASED, PAPER):
        lo, hi = activation_bounds(MA, MR, p)
        a = lo + (hi - lo) * torch.rand(5000, generator=g, dtype=DT)
        a[:50], a[50:100] = lo[:50], hi[50:100]
        nxt = fatigue_step(MA, MR, MF, a, p)
        close(torch.stack(nxt, -1), np.stack(ref_fatigue(MA.numpy(), MR.numpy(), MF.numpy(), a.numpy(), p), -1))
        assert torch.equal(nxt[0], a)
        close(sum(nxt), MA + MR + MF)  # conserved within the bounds
        assert (a <= (1 - p.fatigue_F / p.sim_hz) * MA).any() and (a > (1 - p.fatigue_F / p.sim_hz) * MA).any()  # relaxing and developing
        assert all((x >= -1e-15).all() for x in nxt)
        # outside the bounds (not produced by the inversion): the clipped target load and the saturated branch, as theirs
        above = hi[:2500] + 0.3 * torch.rand(2500, generator=g, dtype=DT)
        a_out = torch.cat((above, (lo[2500:] - 0.3 * torch.rand(2500, generator=g, dtype=DT)).clamp(min=0)))
        nxt = fatigue_step(MA, MR, MF, a_out, p)
        close(torch.stack(nxt, -1), np.stack(ref_fatigue(MA.numpy(), MR.numpy(), MF.numpy(), a_out.numpy(), p), -1))


def test_geometry_and_wrench():
    t = random_table(4)
    model = MuscleModel(t, dtype=DT)
    g = torch.Generator().manual_seed(4)
    pos, rot = random_bodies(g, 3)
    force = 1000 * torch.rand(3, t.num_muscles, generator=g, dtype=DT)
    anchors = model.anchor_positions(pos, rot)
    lengths = model.lengths(anchors)
    anchor_force = model.anchor_forces(anchors, force)
    body_force, body_torque = model.body_wrench(anchors, anchor_force, pos)
    for e in range(3):
        ref = ref_geometry(t, pos[e].numpy(), rot[e].numpy(), force[e].numpy())
        for got, want in zip((anchors[e], lengths[e], anchor_force[e], body_force[e], body_torque[e]), ref):
            close(got, want, 1e-10)
    # the wrenches carry all the anchor forces and their moments: totals about the world origin
    close(body_force.sum(-2), anchor_force.sum(-2), 1e-10)
    close((body_torque + torch.cross(pos, body_force, dim=-1)).sum(-2), torch.cross(anchors, anchor_force, dim=-1).sum(-2), 1e-10)
    # a muscle's forces cancel when all of its anchors move with one body
    one = MuscleTable(**{**{k: getattr(t, k) for k in t.__dataclass_fields__}, "anchor_body0": np.zeros_like(t.anchor_body0),
                         "anchor_body1": np.zeros_like(t.anchor_body1), "anchor_weight0": np.ones_like(t.anchor_weight0)})  # fmt: skip
    m1 = MuscleModel(one, dtype=DT)
    a1 = m1.anchor_positions(pos, rot)
    f1, tau1 = m1.body_wrench(a1, m1.anchor_forces(a1, force), pos)
    close(f1, torch.zeros_like(f1), 1e-9)
    close(tau1, torch.zeros_like(tau1), 1e-9)


def test_blended_anchor_positions():
    """A blended anchor sits at w0 (R0 x0 + p0) + (1 - w0) (R1 x1 + p1); an unblended one moves rigidly with body0."""
    t = random_table(5)
    model = MuscleModel(t, dtype=DT)
    g = torch.Generator().manual_seed(5)
    pos, rot = random_bodies(g)
    anchors = model.anchor_positions(pos, rot)
    blend = t.anchor_body1 != t.anchor_body0
    assert blend.any() and (~blend).any()
    for a in range(t.num_anchors):
        b0, b1, w = int(t.anchor_body0[a]), int(t.anchor_body1[a]), float(t.anchor_weight0[a])
        x0, x1 = torch.tensor(t.anchor_local0[a]), torch.tensor(t.anchor_local1[a])
        want = w * (rot[b0] @ x0 + pos[b0]) + (1 - w) * (rot[b1] @ x1 + pos[b1]) if blend[a] else rot[b0] @ x0 + pos[b0]
        close(anchors[a], want)
    # a rigid motion of all bodies together moves every anchor rigidly (blends of points of one rigid frame)
    R, d = so3_exp(torch.randn(3, generator=g, dtype=DT)), torch.randn(3, generator=g, dtype=DT)
    moved = model.anchor_positions(pos @ R.T + d, R @ rot)
    close(moved, anchors @ R.T + d)


def test_substep():
    t = random_table(6)
    g = torch.Generator().manual_seed(6)
    pos, rot = random_bodies(g, 4)
    MA, MR, MF = random_fatigue(g, 4, t.num_muscles)
    for p in (RELEASED, PAPER):
        model = MuscleModel(t, p, dtype=DT)
        length = model.lengths(model.anchor_positions(pos, rot))
        prev = length * (1 + 0.05 * torch.randn(4, t.num_muscles, generator=g, dtype=DT))
        action = 0.3 * torch.randn(4, t.num_muscles, generator=g, dtype=DT)
        target = model.target_length(action)
        close(target, torch.tensor(t.rest_length) * (1.0 + action))  # ENV:431-433
        out = model.substep(pos, rot, prev, target, MA, MR, MF)
        assert torch.equal(out.length, length)
        for e in range(4):
            anchors, lengths, _, _, _ = ref_geometry(t, pos[e].numpy(), rot[e].numpy(), np.zeros(t.num_muscles))
            kd = 2.0 * np.sqrt(t.kp) if p is RELEASED else 0.1 * t.kp
            fat = (MA[e].numpy(), MR[e].numpy())
            f = ref_pd(lengths, prev[e].numpy(), target[e].numpy(), t.rest_length, t.lm, t.lt, t.kp, kd, t.f0max, *fat, p)[0]
            inv_prev = lengths if p.inversion_rate == "zero" else prev[e].numpy()  # CHR:221 overwrites the previous length before CHR:237
            a = ref_activation(f, lengths, inv_prev, t.rest_length, t.lm, t.lt, t.f0max, *fat, p)[0]
            nxt = np.stack(ref_fatigue(MA[e].numpy(), MR[e].numpy(), MF[e].numpy(), a, p), -1)
            _, _, forces, body_force, body_torque = ref_geometry(t, pos[e].numpy(), rot[e].numpy(), f)
            close(out.force[e], f, 1e-10)
            close(out.activation[e], a, 1e-10)
            close(torch.stack((out.MA[e], out.MR[e], out.MF[e]), -1), nxt, 1e-10)
            close(out.anchor_force[e], forces, 1e-9)
            close(out.body_force[e], body_force, 1e-9)
            close(out.body_torque[e], body_torque, 1e-9)
            close(out.anchor_pos[e], anchors, 1e-12)
    # the released rates are the change divided by 120, the paper's multiplied by it
    model = MuscleModel(t, RELEASED, dtype=DT)
    r, rn = model.rates(length, prev)
    close(r, (length - prev) / 120.0)
    r, rn = MuscleModel(t, PAPER, dtype=DT).rates(length, prev)
    close(r, (length - prev) * 120.0)
    close(rn, (model.normalized_length(length) - model.normalized_length(prev)) * 120.0)


def test_part_fatigue():
    t = random_table(7)
    model = MuscleModel(t, dtype=DT)
    g = torch.Generator().manual_seed(7)
    MA, MR, MF = random_fatigue(g, 2, t.num_muscles)
    out = model.part_fatigue(MA, MR, MF)
    assert out.shape == (2, 3 * NUM_PARTS)
    for e in range(2):
        want = np.zeros((3, NUM_PARTS))
        for m in range(t.num_muscles):  # BE:4367-4390 with the f0 weights of BE:4583-4609
            w = t.f0[m] / t.f0[t.part == t.part[m]].sum()
            for k, x in enumerate((MA, MR, MF)):
                want[k, t.part[m]] += x[e, m].item() * w
        close(out[e], want.reshape(-1))
    close(model.part_fatigue(torch.ones_like(MA), torch.zeros_like(MA), torch.zeros_like(MA))[:, :NUM_PARTS], torch.ones(2, NUM_PARTS))


def test_batching_shapes_and_dtypes():
    t = random_table(8)
    g = torch.Generator().manual_seed(8)
    pos, rot = random_bodies(g, 2, 3)
    MA, MR, MF = random_fatigue(g, 2, 3, t.num_muscles)
    model = MuscleModel(t, dtype=DT)
    length = model.lengths(model.anchor_positions(pos, rot))
    prev, target = 1.01 * length, model.target_length(0.1 * torch.randn(2, 3, t.num_muscles, generator=g, dtype=DT))
    out = model.substep(pos, rot, prev, target, MA, MR, MF)
    shapes = {"anchor_pos": (2, 3, t.num_anchors, 3), "force": (2, 3, t.num_muscles), "body_force": (2, 3, N_BODIES, 3),
              "body_torque": (2, 3, N_BODIES, 3)}
    for k, s in shapes.items():
        assert tuple(getattr(out, k).shape) == s, k
    for i in range(2):  # each environment alone gives the same numbers
        for j in range(3):
            one = model.substep(pos[i, j], rot[i, j], prev[i, j], target[i, j], MA[i, j], MR[i, j], MF[i, j])
            for k, v in one._asdict().items():
                torch.testing.assert_close(v, getattr(out, k)[i, j], rtol=1e-13, atol=1e-13, msg=k)
    f32 = MuscleModel(t, dtype=torch.float32).substep(*(x.float() for x in (pos, rot, prev, target, MA, MR, MF)))
    assert f32.force.dtype == torch.float32
    torch.testing.assert_close(f32.body_force.double(), out.body_force, rtol=1e-3, atol=1e-2)
    for bad in ((pos[..., :4, :], rot), (pos, rot[..., :2, :]), (pos, rot[0])):
        with pytest.raises(ValueError):
            model.anchor_positions(*bad)
    with pytest.raises(ValueError):
        model.substep(pos, rot, prev[..., :-1], target, MA, MR, MF)
    with pytest.raises(ValueError):
        model.anchor_forces(out.anchor_pos[..., :-1, :], out.force)
    with pytest.raises(ValueError):  # no broadcasting between environments
        model.body_wrench(out.anchor_pos, out.anchor_force[0], pos)
    with pytest.raises(ValueError):
        MuscleModel(t, dtype=torch.int64)


def test_gradients_are_finite():
    """The substep is differentiable everywhere it is evaluated (no division by zero in the unused branches)."""
    t = random_table(9)
    g = torch.Generator().manual_seed(9)
    pos, rot = random_bodies(g, 2)
    pos.requires_grad_(True)
    model = MuscleModel(t, PAPER, dtype=DT)
    length = model.lengths(model.anchor_positions(pos, rot)).detach()
    MA, MR, MF = random_fatigue(g, 2, t.num_muscles)
    out = model.substep(pos, rot, length, model.target_length(torch.zeros(2, t.num_muscles, dtype=DT)), MA, MR, MF)  # rate 0
    (out.body_force.square().sum() + out.activation.sum() + out.MR.sum()).backward()
    assert torch.isfinite(pos.grad).all() and pos.grad.abs().sum() > 0
    ln = torch.tensor([0.5, 1.0, 1.6, 30.0, 300.0], dtype=torch.float32, requires_grad=True)  # far past the passive cap: no overflow
    fl, fv, fpe = hill_curves(ln, torch.zeros_like(ln), PAPER)
    (fl + fpe).sum().backward()
    assert torch.isfinite(ln.grad).all() and (fpe[3:] == 2.0).all()


def test_random_initial_fatigue():
    """MuscleVAE's initial fatigue (ENV:302-305, CHR:339-356): the curve of one muscle under the target load 0.5 for 20 s
    then at rest (their 4800 steps at 120 Hz), and per body part the curve's state at a time uniform in [0, 10) s, shared
    by the part's muscles."""
    for p in (RELEASED, PAPER):
        curve = fatigue_init_curve(p)
        assert curve.shape == (4800, 3)
        np.testing.assert_array_equal(curve, ref_fatigue_curve(p))
        assert np.array_equal(curve[0], [0.0, 1.0, 0.0]) and np.abs(curve.sum(-1) - 1).max() < 1e-12 and (curve >= 0).all()
        assert abs(curve[1199, 0] - 0.5) < 2e-3 and curve[2401:, 0].max() < curve[2400, 0]  # loaded at 0.5 for 20 s, then relaxing
    assert fatigue_init_curve(RELEASED)[1199, 2] > 5 * fatigue_init_curve(PAPER)[1199, 2]  # F ten times the paper's
    t = random_table(10, num_muscles=40)
    model = MuscleModel(t, dtype=DT)
    MA, MR, MF = model.random_fatigue((300,), torch.Generator().manual_seed(10))
    assert MA.shape == (300, 40) and torch.allclose(MA + MR + MF, torch.ones_like(MA))
    curve = torch.tensor(fatigue_init_curve(RELEASED))
    state = torch.stack((MA, MR, MF), -1)
    for part in range(NUM_PARTS):
        m = np.flatnonzero(t.part == part)
        assert (state[:, m] == state[:, m[:1]]).all()  # one state per part
        k = (state[:, m[0], None, :] == curve[None, :1200]).all(-1)  # the curve's state at a step below 10 s
        assert (k.sum(-1) >= 1).all()
    a = model.random_fatigue((300,), torch.Generator().manual_seed(10))
    assert all(torch.equal(x, y) for x, y in zip(a, (MA, MR, MF)))
    assert MA.std(0).min() > 0 and MR.min() < 0.2 and MR.max() > 0.9  # from fresh to fatigued
