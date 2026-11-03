"""The muscle character's state convention and world models (ndm_gym.tasks.muscle_tracking, ndm_gym.worldmodels.muscle) on
the CPU, without MuscleVAE's assets: the backward-difference state of the simulator's poses and MuscleVAE's angular
velocity of it (against quaternion finite differences written here), the policy's observation against MuscleVAE's built
from the poses, both world models' integration round trip, their invariance to a turn and shift of the scene, and
SuperTrack's local terms against algos.supertrack's. With the converted muscle table: the muscle-token RigidFormer's
features against ndm_gym.robots.muscles, its relation classes, its start at zero, its invariance and its gradients; the
body-token RigidFormer's muscle wrench against the muscle model's substep, its pooling matrices and relation classes,
and the same start, invariance and gradients."""

import math
import os

import pytest
import torch

from ndm_gym.core.rotations import mat_to_quat, quat_to_mat, so3_exp
from ndm_gym.policies import musclevae_pretrained as mp
from ndm_gym.tasks import muscle_tracking as mt
from ndm_gym.worldmodels.muscle import IN_DIM, OUT_DIM, MuscleWorldModel, increments

DT, NB = mt.DT, mt.NB
HAVE_TABLE = os.path.isfile(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "musclevae", "muscles.json"))
needs_table = pytest.mark.skipif(not HAVE_TABLE, reason="needs MuscleVAE's converted muscle table (tools/convert_musclevae.py)")


def random_state(n: int, seed: int, dtype=torch.float64):
    """Body states (p, R, v, w) of n characters and their muscle states (n, 15)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)  # noqa: E731
    p = r(n, NB, 3) * 0.4 + torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    return (p, so3_exp(1.5 * r(n, NB, 3)), r(n, NB, 3), 3.0 * r(n, NB, 3)), torch.rand(n, 15, generator=g, dtype=dtype)


def turn(S, yaw: float, shift):
    """The scene turned by ``yaw`` about z and shifted horizontally by ``shift``."""
    c, s = math.cos(yaw), math.sin(yaw)
    Q = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=S[0].dtype)
    p, R, v, w = S
    return p @ Q.T + torch.tensor([shift[0], shift[1], 0.0], dtype=p.dtype), Q @ R, v @ Q.T, w @ Q.T


def quat_mul(a, b):  # wxyz
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True), aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def test_their_angular_velocity():
    """For q' = exp(dt w) q, MuscleVAE's angular velocity is the vector part of 2 (q' q^* - 1) / dt (sign-aligned), and
    rate_from_their inverts it; rates up to 3 rad per step (a backward difference, a log, turns by at most pi)."""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(256, 3, generator=g, dtype=torch.float64) * torch.tensor([0.1, 3.0, 20.0, 50.0], dtype=torch.float64).repeat(64)[:, None]
    n = w.norm(dim=-1, keepdim=True) * DT
    w = torch.where(n > 3.0, w * 3.0 / n, w)
    q = mat_to_quat(so3_exp(torch.randn(256, 3, generator=g, dtype=torch.float64)))
    q1 = mat_to_quat(so3_exp(DT * w) @ quat_to_mat(q))
    q1 = q1 * torch.where((q1 * q).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    conj = q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64)
    theirs = 2.0 * quat_mul(q1, conj)[..., 1:] / DT
    torch.testing.assert_close(mt.their_angular_velocity(w), theirs, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(mt.rate_from_their(theirs), w, rtol=1e-7, atol=1e-7)
    assert torch.equal(mt.their_angular_velocity(torch.zeros(1, 3)), torch.zeros(1, 3))


def sim_views(seed: int, n: int = 6):
    """MuscleSim-like views: poses before and after a step (quaternions xyzw) and the simulator's velocities."""
    S, _ = random_state(n, seed)
    w = 4.0 * torch.randn(n, NB, 3, generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float64)
    v = torch.randn(n, NB, 3, generator=torch.Generator().manual_seed(seed + 2), dtype=torch.float64)
    p0, R0 = S[0], S[1]
    p1, R1 = p0 + DT * v, so3_exp(DT * w) @ R0
    xyzw = lambda R: mat_to_quat(R)[..., [1, 2, 3, 0]]  # noqa: E731
    before, now = torch.cat((p0, xyzw(R0)), -1), torch.cat((p1, xyzw(R1)), -1)
    qd = torch.cat((torch.randn(n, NB, 3, dtype=torch.float64), 2.0 * torch.randn(n, NB, 3, dtype=torch.float64)), -1)
    return before, now, qd, v, w


def test_body_states_and_observation():
    """The state of the simulator's poses: backward-difference velocities, or the simulator's at a fresh reset; its
    policy observation is MuscleVAE's (their velocities of the poses, PolicyRunner.observe's), and positions follow
    from the velocities exactly."""
    before, now, qd, v, w = sim_views(3)
    fresh = torch.tensor([False, True, False, False, True, False])
    s = mt.body_states(now, qd, before, fresh)
    p, R, vs, ws = mt.unflatten_bodies(s)
    keep = ~fresh
    torch.testing.assert_close(vs[keep], v[keep], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(ws[keep], w[keep], rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(vs[fresh], qd[fresh][..., :3])
    torch.testing.assert_close(mt.their_angular_velocity(ws[fresh]), qd[fresh][..., 3:], rtol=1e-9, atol=1e-9)
    # positions and rotations follow from the velocities: p = p0 + dt v, R = exp(dt w) R0
    torch.testing.assert_close(p[keep], before[keep][..., :3] + DT * vs[keep], rtol=1e-9, atol=1e-9)
    R0 = quat_to_mat(before[..., [6, 3, 4, 5]])
    torch.testing.assert_close(R[keep], so3_exp(DT * ws[keep]) @ R0[keep], rtol=1e-7, atol=1e-7)
    # the policy's observation, as PolicyRunner.observe builds it from the simulator
    from ndm_gym.policies.muscle_policy import policy_observation

    F = torch.rand(6, 15, dtype=torch.float64)
    vel, avel = mp.velocities(now[..., :3], now[..., 3:], before[..., :3], before[..., 3:], DT)
    vel = torch.where(fresh[:, None, None], qd[..., :3], vel)
    avel = torch.where(fresh[:, None, None], qd[..., 3:], avel)
    theirs = torch.cat((mp.observation(now[..., :3], quat_to_mat(now[..., [6, 3, 4, 5]]), vel, avel), F), -1)
    torch.testing.assert_close(policy_observation((p, R, vs, ws), F), theirs, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_integration_round_trip(kind):
    """The targets of a transition, integrated, give back its next state: any for the increments; one whose velocities
    are its backward differences (the state convention) for the accelerations."""
    wm = MuscleWorldModel(kind, hidden=(32,)).double()
    S, F = random_state(8, 1)
    S2, F1 = random_state(8, 2)
    if kind == "accel":
        p, R, _, _ = S
        v1, w1 = S2[2], S2[3]
        S1 = (p + DT * v1, so3_exp(DT * w1) @ R, v1, w1)
    else:
        S1 = S2
    y = wm.targets(S, F, S1, F1)
    assert y.shape == (8, OUT_DIM[kind]) and wm.inputs(S, F, torch.zeros(8, 284, dtype=torch.float64)).shape == (8, IN_DIM)
    S1_, F1_ = wm.integrate(S, F, y)
    for a, b in zip(S1_ + (F1_,), S1 + (F1,)):
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-7)  # rotations through quaternions: about 5e-9 in double precision
    assert increments(S, F, S, F).abs().max() < 1e-12


@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_turn_and_shift_invariance(kind):
    """With random weights, a step of the scene turned about z and shifted is the step turned and shifted: the inputs
    are the same, the outputs turn with the root's heading."""
    torch.manual_seed(0)
    wm = MuscleWorldModel(kind, hidden=(64, 64)).double()
    with torch.no_grad():
        for prm in wm.net.parameters():
            prm.normal_(0.0, 0.1)
        wm.out_norm.var.uniform_(0.1, 2.0)
        wm.out_norm.mean.normal_(0.0, 0.5)
    S, F = random_state(5, 4)
    a = torch.randn(5, 284, dtype=torch.float64)
    S1, F1 = wm.step(S, F, a)
    St = turn(S, 1.1, (3.0, -2.0))
    torch.testing.assert_close(wm.inputs(St, F, a), wm.inputs(S, F, a), rtol=1e-9, atol=1e-9)
    S1t, F1t = wm.step(St, F, a)
    for x, y in zip(S1t + (F1t,), turn(S1, 1.1, (3.0, -2.0)) + (F1,)):
        torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-9)
    assert (S1[0] - S[0]).abs().max() > 1e-3 and (F1 - F).abs().max() > 1e-3  # the step moved


def rf_model(kind: str, seed: int = 0, random_weights: bool = True, tokens: str = "all") -> MuscleWorldModel:
    """A RigidFormer world model (muscle tokens, or body tokens only) in double precision, with random weights and
    output normalizer."""
    torch.manual_seed(seed)
    extra = dict(tokens="body", muscle_dim=16) if tokens == "body" else {}
    wm = MuscleWorldModel(kind, rigidformer=dict(dim=48, layers=2, heads=4, checkpoint="never", **extra)).double()
    if random_weights:
        with torch.no_grad():
            for prm in wm.net.parameters():
                prm.normal_(0.0, 0.1)
            wm.out_norm.var.uniform_(0.1, 2.0)
            wm.out_norm.mean.normal_(0.0, 0.5)
    return wm


def pose_state(n: int, seed: int):
    """Body states near the T-pose (small joint motions), where every muscle stays in its physical range, and muscle
    states (n, 15) with MA + MR + MF = 1 per part."""
    from ndm_gym.robots import musclevae_char as mc

    g = torch.Generator().manual_seed(seed)
    char = mc.load_character()
    q = torch.zeros(n, mc.JOINT_QPOS_DIM, dtype=torch.float64)
    for j, (adr, size) in enumerate(zip(mc.JOINT_QPOS_ADR, mc.JOINT_QPOS_SIZE)):
        if size == 4:
            q[:, adr : adr + 4] = mat_to_quat(so3_exp(0.15 * torch.randn(n, 3, generator=g, dtype=torch.float64)))
        else:
            q[:, adr] = 0.15 * torch.randn(n, generator=g, dtype=torch.float64)
    root_q = mat_to_quat(so3_exp(0.3 * torch.randn(n, 3, generator=g, dtype=torch.float64)))
    p, R = mc.forward_kinematics(torch.tensor([0.3, -0.2, 0.95], dtype=torch.float64).expand(n, 3), root_q, q, char)
    v, w = 0.5 * torch.randn(n, NB, 3, generator=g, dtype=torch.float64), torch.randn(n, NB, 3, generator=g, dtype=torch.float64)
    parts = torch.rand(n, 3, 5, generator=g, dtype=torch.float64)
    return (p, R, v, w), (parts / parts.sum(1, keepdim=True)).flatten(1)


@needs_table
def test_rigidformer_muscle_features():
    """The muscle tokens' features against ndm_gym.robots.muscles on poses near the T-pose: lengths, normalized lengths,
    the analytic length rates against finite differences of the lengths, the PD error, the first substep's force and
    activation with the part fatigue for every muscle, the Hill factors, the part fatigue, the path midpoints and end
    directions (heading frame, origin at the pelvis's ground projection); the body tokens' relative rotations."""
    from ndm_gym.robots import muscles as mu
    from ndm_gym.worldmodels.muscle import BODY_TOK, MUSCLE_TOK, heading

    wm = rf_model("accel", random_weights=False)
    net = wm.net
    S, F = pose_state(4, 3)
    a = 0.2 * torch.randn(4, 284, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    x = net.inputs(S, F, a)
    assert x.shape == (4, net.in_dim) and torch.isfinite(x).all()
    body = x[:, : NB * BODY_TOK].unflatten(-1, (NB, BODY_TOK))
    mus = x[:, NB * BODY_TOK : NB * BODY_TOK + 284 * MUSCLE_TOK].unflatten(-1, (284, MUSCLE_TOK))
    torch.testing.assert_close(x[:, -15:], F)
    p, R, v, w = S
    mm = net.muscles("cpu", torch.float64)
    P = mm.params
    anchors = mm.anchor_positions(p, R)
    length = mm.lengths(anchors)
    rest = mm.rest_length
    eps = 1e-6
    ahead, behind = (mm.lengths(mm.anchor_positions(p + e * v, so3_exp(e * w) @ R)) for e in (eps, -eps))
    dl = (ahead - behind) / (2 * eps)
    torch.testing.assert_close(mus[..., 0], length / rest)
    torch.testing.assert_close(mus[..., 1], mm.normalized_length(length))
    torch.testing.assert_close(mus[..., 2], dl / rest, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(mus[..., 3], a)
    torch.testing.assert_close(mus[..., 4], (length - mm.target_length(a)) / rest)
    MA, MR, MF = (F[:, k * 5 + net.part] for k in range(3))
    rate, rate_norm = mm.rates(length, length - dl / P.sim_hz)
    fl, fv, fpe = mu.hill_curves(mm.normalized_length(length), rate_norm, P)
    lo, hi = mu.activation_bounds(MA, MR, P)
    force = mu.pd_force(length, mm.target_length(a), rate, mm.kp, mm.kd, mm.f0max, fl, fv, fpe, lo, hi)[0]
    act = mu.invert_hill(force, mm.f0max, fl, torch.ones_like(fv), fpe, lo, hi)[0]
    for i, ref in enumerate((force / mm.f0max, act, fl, fv, fpe, MA, MR, MF)):
        torch.testing.assert_close(mus[..., 5 + i], ref, rtol=1e-6, atol=1e-6)
    H = heading(S)
    origin = torch.cat((p[:, :1, :2], torch.zeros_like(p[:, :1, 2:])), -1)
    mid, first, last = mm.path_points(anchors)
    to_h = lambda u: (H.transpose(-1, -2) @ u.unsqueeze(-1)).squeeze(-1)  # noqa: E731
    torch.testing.assert_close(mus[..., 13:16], to_h(mid - origin))
    torch.testing.assert_close(mus[..., 16:19], to_h(first))
    torch.testing.assert_close(mus[..., 19:22], to_h(last))
    torch.testing.assert_close(body[..., :3], to_h(p - origin))
    from ndm_gym.core.rotations import mat_to_6d
    from ndm_gym.robots.musclevae_char import BODY_PARENTS

    for b in (3, 12, 18):
        par = BODY_PARENTS[b]
        torch.testing.assert_close(body[:, b, 15:21], mat_to_6d(R[:, par].transpose(-1, -2) @ R[:, b]))
        torch.testing.assert_close(body[:, b, 21:24], (R[:, par].transpose(-1, -2) @ (w[:, b] - w[:, par]).unsqueeze(-1)).squeeze(-1))
    assert body[:, 0, 15:].abs().max() == 0  # the pelvis has no parent


@needs_table
def test_rigidformer_relation_classes():
    """Body pairs: the joints between them; a muscle and a body: from the muscle's nearest attached body; two muscles:
    between their nearest attached bodies; registers: one class (brute force over the muscle table)."""
    from ndm_gym.worldmodels.muscle import body_hops

    net = rf_model("accel", random_weights=False).net
    hops = body_hops()
    K = int(hops.max()) + 1
    bodies = net.muscles("cpu", torch.float64).muscle_bodies()
    rel = net.rel.numpy()
    assert (rel[:NB, :NB] == hops).all()
    for m in (0, 57, 140, 283):
        for b in (0, 5, 18):
            assert rel[NB + m, b] == rel[b, NB + m] == min(hops[c, b] for c in bodies[m]) + K
        for m2 in (3, 140, 200):
            assert rel[NB + m, NB + m2] == min(hops[c, d] for c in bodies[m] for d in bodies[m2]) + 2 * K
    assert (rel[NB + 284 :] == 3 * K).all() and (rel[:, NB + 284 :] == 3 * K).all() and net.rel_bias.shape[1] == 3 * K + 1


@needs_table
def test_rigidformer_body_tokens():
    """tokens="body": the body tokens' features are the muscle-token model's, then the force and torque about each
    body's origin that the muscle model's substep applies (the first substep's forces, every muscle with its part's
    fatigue, the previous lengths from the analytic length rates), in the heading frame; the muscle features are the
    first 13 of the muscle tokens'. The pooling matrices: the mean over a body's attached muscles, each part's f0
    weights shared equally by a muscle's bodies; the relation classes: the joints between bodies, registers one more."""
    from ndm_gym.worldmodels.muscle import BODY_TOK, MUSCLE_SCALARS, MUSCLE_TOK, WRENCH, body_hops, heading

    net_all, net = rf_model("accel", random_weights=False).net, rf_model("increment", random_weights=False, tokens="body").net
    S, F = pose_state(4, 6)
    a = 0.2 * torch.randn(4, 284, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    x_all, x = net_all.inputs(S, F, a), net.inputs(S, F, a)
    assert x.shape == (4, net.in_dim) and net.in_dim == NB * (BODY_TOK + WRENCH) + 284 * MUSCLE_SCALARS + 15 and torch.isfinite(x).all()
    body = x[:, : NB * (BODY_TOK + WRENCH)].unflatten(-1, (NB, BODY_TOK + WRENCH))
    torch.testing.assert_close(body[..., :BODY_TOK], x_all[:, : NB * BODY_TOK].unflatten(-1, (NB, BODY_TOK)), rtol=0, atol=0)
    mus_all = x_all[:, NB * BODY_TOK : NB * BODY_TOK + 284 * MUSCLE_TOK].unflatten(-1, (284, MUSCLE_TOK))
    mus = x[:, NB * (BODY_TOK + WRENCH) : -15].unflatten(-1, (284, MUSCLE_SCALARS))
    torch.testing.assert_close(mus, mus_all[..., :MUSCLE_SCALARS], rtol=0, atol=0)
    torch.testing.assert_close(x[:, -15:], F)
    p, R, v, w = S
    mm = net.muscles("cpu", torch.float64)
    length = mus[..., 0] * mm.rest_length
    dl = mus[..., 2] * mm.rest_length
    MA, MR, MF = (F[:, k * 5 + net.part] for k in range(3))
    step = mm.substep(p, R, length - dl / mm.params.sim_hz, mm.target_length(a), MA, MR, MF)
    torch.testing.assert_close(step.force / mm.f0max, mus[..., 5])
    to_h = lambda u: (heading(S).transpose(-1, -2) @ u.unsqueeze(-1)).squeeze(-1)  # noqa: E731
    torch.testing.assert_close(body[..., BODY_TOK : BODY_TOK + 3], to_h(step.body_force))
    torch.testing.assert_close(body[..., BODY_TOK + 3 :], to_h(step.body_torque))
    assert step.body_force.abs().max() > 1.0  # the muscles pull
    bodies = mm.muscle_bodies()
    count = [sum(b in s for s in bodies) for b in range(NB)]
    attach = torch.tensor([[1.0 / count[b] if b in bodies[m] else 0.0 for m in range(284)] for b in range(NB)], dtype=torch.float64)
    torch.testing.assert_close(net.attach, attach, rtol=1e-6, atol=0)  # float32 buffers
    part_body = torch.zeros(5, NB, dtype=torch.float64)
    for m, bs in enumerate(bodies):
        for b in bs:
            part_body[net.part[m], b] += net.part_weight[net.part[m], m] / len(bs)
    torch.testing.assert_close(net.part_body, part_body, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(net.part_body.sum(1), torch.ones(5, dtype=torch.float64), rtol=0, atol=1e-6)
    hops = torch.as_tensor(body_hops())
    K = int(hops.max()) + 1
    assert torch.equal(net.rel[:NB, :NB], hops) and (net.rel[NB:] == K).all() and (net.rel[:, NB:] == K).all()
    assert net.rel.shape == (NB + 2, NB + 2) and net.rel_bias.shape == (4, K + 1)
    with pytest.raises(ValueError):
        MuscleWorldModel("accel", rigidformer=dict(dim=48, layers=1, heads=4, muscle_dim=16))  # the encoders' width without them
    with pytest.raises(ValueError):
        MuscleWorldModel("accel", rigidformer=dict(dim=48, layers=1, heads=4, tokens="muscle"))


@needs_table
@pytest.mark.parametrize("tokens", ["all", "body"])
def test_rigidformer_recompute(tokens):
    """A step recomputed in the backward pass (checkpoint "always", and "frozen" with frozen parameters) has the
    gradients of one that keeps its activations."""
    S, F = pose_state(3, 7)
    a = (0.1 * torch.randn(3, 284, generator=torch.Generator().manual_seed(4), dtype=torch.float64)).requires_grad_(True)
    grads = {}
    for mode in ("never", "always", "frozen"):
        wm = rf_model("accel", seed=1, tokens=tokens)
        wm.net.checkpoint = mode
        if mode == "frozen":
            wm.requires_grad_(False)
        assert wm.net.recompute() == (mode != "never")
        v = S[2].clone().requires_grad_(True)
        S1, F1 = wm.step((S[0], S[1], v, S[3]), F, a)
        out = S1[0].square().sum() + S1[1].sum() + F1.square().sum()
        prm = [x for x in wm.net.parameters() if x.requires_grad]
        grads[mode] = torch.autograd.grad(out, [a, v] + prm)
    for mode in ("always", "frozen"):
        for g, g0 in zip(grads[mode], grads["never"]):
            torch.testing.assert_close(g, g0, rtol=1e-12, atol=1e-12)
    assert len(grads["always"]) == len(grads["never"]) > 2 and len(grads["frozen"]) == 2


@needs_table
@pytest.mark.parametrize("tokens", ["all", "body"])
@pytest.mark.parametrize("kind", ["accel", "increment"])
def test_rigidformer_start_invariance_and_gradients(kind, tokens):
    """At initialization the heads are zero, so a step integrates the output normalizer's mean, as the MLP's; with
    random weights a step of the scene turned about z and shifted is the step turned and shifted; the step's gradients
    reach the action, the state and every parameter."""
    wm0 = rf_model(kind, random_weights=False, tokens=tokens)
    S, F = pose_state(3, 5)
    a = 0.1 * torch.randn(3, 284, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    assert wm0.net(wm0.in_norm(wm0.inputs(S, F, a))).abs().max() == 0
    S1, F1 = wm0.step(S, F, a)
    S1m, F1m = wm0.integrate(S, F, wm0.out_norm.mean.expand(3, -1))
    assert all(torch.equal(x, y) for x, y in zip(S1 + (F1,), S1m + (F1m,)))
    wm = rf_model(kind, tokens=tokens)
    S1, F1 = wm.step(S, F, a)
    St = turn(S, -0.8, (1.5, 2.5))
    torch.testing.assert_close(wm.inputs(St, F, a), wm.inputs(S, F, a), rtol=1e-9, atol=1e-9)
    S1t, F1t = wm.step(St, F, a)
    for x, y in zip(S1t + (F1t,), turn(S1, -0.8, (1.5, 2.5)) + (F1,)):
        torch.testing.assert_close(x, y, rtol=1e-8, atol=1e-8)
    a = a.clone().requires_grad_(True)
    v = S[2].clone().requires_grad_(True)
    S1, F1 = wm.step((S[0], S[1], v, S[3]), F, a)
    (S1[0].square().sum() + F1.square().sum()).backward()
    assert a.grad.abs().max() > 0 and v.grad.abs().max() > 0 and torch.isfinite(a.grad).all()
    assert all(prm.grad is not None and torch.isfinite(prm.grad).all() for prm in wm.net.parameters())
