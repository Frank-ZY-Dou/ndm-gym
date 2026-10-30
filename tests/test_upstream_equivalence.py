"""The refactored package computes exactly what the upstream reproduction (third_party/supermimic) computes."""

import os

import pytest
import torch

from conftest import DEV, REPO, random_states, with_ndm_fk

EXACT = dict(rtol=0, atol=0)
# ndm_gym's FK writes the 3x3 products elementwise (fused by torch.compile) where upstream uses batched matmuls: the same
# arithmetic summed in another order, so the two agree up to float rounding
FK_ROUNDING = dict(rtol=1e-6, atol=1e-5)


@pytest.fixture(scope="session")
def old(upstream):
    kin = with_ndm_fk(upstream["kinematics"].G1Kinematics(DEV))
    motion = upstream["motion"].MotionLib.from_dir(os.path.join(REPO, "data", "lafan1_g1"), kin, DEV)
    mdp = upstream["mdp"].MDP(kin, motion, torch.tensor(upstream["robot"].default_joint_pos_list(), device=DEV))
    return kin, motion, mdp


def test_kinematics(old, kin, motion):
    k0 = old[0]
    assert (k0.body_names, k0.parent, k0.joint_idx, k0.tracked_idx) == (kin.body_names, kin.parent, kin.joint_idx, kin.tracked_idx)
    for name in ("axes", "off_p", "off_R", "q_lo", "q_hi"):
        torch.testing.assert_close(getattr(kin, name), getattr(k0, name), **EXACT)
    _, S, _ = random_states(motion, 64, seed=1)
    for all_bodies in (False, True):
        upstream = type(k0).forward(k0, *S, all_bodies=all_bodies)  # the upstream class's own FK
        for a, b in zip(kin.forward(*S, all_bodies=all_bodies), upstream):
            torch.testing.assert_close(a, b, **FK_ROUNDING)


def test_motion_library(old, motion):
    m0 = old[1]
    assert m0.names == motion.names and m0.num_frames == motion.num_frames and m0.num_bins == motion.num_bins
    for name in ("clip_len", "clip_start", "clip_end", "clip_id", "root_p", "root_R", "root_v", "root_w", "q", "qd", "body_p", "body_R",
                 "body_v", "body_w", "root_v_local", "root_w_local", "root_gz", "bin_clip", "bin_start", "frame_bin"):  # fmt: skip
        torch.testing.assert_close(getattr(motion, name), getattr(m0, name), **EXACT)
    f = torch.arange(0, motion.num_frames, 997)
    torch.testing.assert_close(motion.goal(f, 10), m0.goal(f, 10), **EXACT)


def test_task_heads(upstream, old, motion, mdp):
    from ndm_gym.state import INV_DIM, STATE_DIM, XI_DIM, flatten_state, invariant_features, normalize_xi, unflatten_state
    from ndm_gym.tasks import tracking

    v = upstream["mdp"]
    assert (STATE_DIM, INV_DIM, XI_DIM, tracking.OBS_DIM, tracking.CRITIC_EXTRA) == (v.STATE_DIM, v.INV_DIM, v.XI_DIM, v.OBS_DIM, v.CRITIC_EXTRA)
    assert tracking.MAX_STEP_REWARD == v.MAX_STEP_REWARD
    mdp0 = old[2]
    f, S, xi = random_states(motion, 64, seed=2)
    a, a_prev = torch.randn(64, 29), torch.randn(64, 29)
    x = flatten_state(S)
    torch.testing.assert_close(x, v.flatten_state(S), **EXACT)
    for p, q in zip(unflatten_state(x), v.unflatten_state(x)):
        torch.testing.assert_close(p, q, **EXACT)
    torch.testing.assert_close(invariant_features(S), v.invariant_features(S), **EXACT)
    torch.testing.assert_close(normalize_xi(xi), v.normalize_xi(xi), **EXACT)
    torch.testing.assert_close(mdp.obs(S, a_prev, xi, f), mdp0.obs(S, a_prev, xi, f), **EXACT)
    o = mdp.obs(S, a_prev, xi, f)
    torch.manual_seed(3)
    n1 = mdp.add_noise(o)
    torch.manual_seed(3)
    torch.testing.assert_close(n1, mdp0.add_noise(o), **EXACT)
    r1, t1 = mdp.reward(S, f + 1, a, a_prev, terms=True)
    r0, t0 = mdp0.reward(S, f + 1, a, a_prev, terms=True)
    torch.testing.assert_close(r1, r0, **EXACT)
    for k in t0:
        torch.testing.assert_close(t1[k], t0[k], **EXACT)
    torch.testing.assert_close(mdp.termination(S, f + 1), mdp0.termination(S, f + 1), **EXACT)
    torch.testing.assert_close(mdp.soft_termination(S, f + 1), mdp0.soft_termination(S, f + 1), **EXACT)
    torch.testing.assert_close(mdp.critic_features(S, f, normalize_xi(xi)), mdp0.critic_features(S, f, normalize_xi(xi)), **EXACT)


def test_networks(upstream, motion):
    from ndm_gym.policies.critic import Critic
    from ndm_gym.policies.tracking_policy import TrackingPolicy
    from ndm_gym.state import normalize_xi
    from ndm_gym.tasks.tracking import OBS_DIM
    from ndm_gym.worldmodels.increment import IncrementWorldModel

    n = upstream["networks"]
    torch.manual_seed(4)
    wm0, pi0, v0 = n.WorldModel(hidden=(64, 64)), n.Policy(OBS_DIM, 730, hist=10), n.Critic(300)
    torch.manual_seed(4)
    wm1, pi1, v1 = IncrementWorldModel(hidden=(64, 64)), TrackingPolicy(OBS_DIM, 730, hist=10), Critic(300)
    for m0, m1 in ((wm0, wm1), (pi0, pi1), (v0, v1)):
        s0, s1 = m0.state_dict(), m1.state_dict()
        assert list(s0) == list(s1)
        for k in s0:
            torch.testing.assert_close(s1[k], s0[k], **EXACT)
    f, S, xi = random_states(motion, 64, seed=5)
    S1 = motion.ref_state(f + 1)
    a = torch.randn(64, 29)
    torch.nn.init.normal_(wm0.net[-1].weight, std=0.05)
    wm0.in_norm.update(wm0.inputs(S, a, normalize_xi(xi)))
    wm0.out_norm.update(wm0.targets(S, S1))
    wm1.load_state_dict(wm0.state_dict())
    torch.testing.assert_close(wm1.inputs(S, a, normalize_xi(xi)), wm0.inputs(S, a, normalize_xi(xi)), **EXACT)
    torch.testing.assert_close(wm1.targets(S, S1), wm0.targets(S, S1), **EXACT)
    for p, q in zip(wm1.step(S, a, normalize_xi(xi)), wm0.step(S, a, normalize_xi(xi))):
        torch.testing.assert_close(p, q, **EXACT)
    torch.testing.assert_close(wm1.state_error(S1, S), wm0.state_error(S1, S), **EXACT)
    o_hist, g, x = torch.randn(64, 10, OBS_DIM), torch.randn(64, 730), torch.randn(64, 300)
    torch.testing.assert_close(pi1(o_hist, g), pi0(o_hist, g), **EXACT)
    torch.testing.assert_close(v1(x), v0(x), **EXACT)


def test_replay_buffer(upstream):
    from ndm_gym.buffer import FIELDS, ReplayBuffer

    B0 = upstream["buffer"]
    assert {k: v for k, v in FIELDS.items()} == B0.FIELDS
    T, N = 40, 16
    b1, b0 = ReplayBuffer(T, N, DEV), B0.ReplayBuffer(T, N, DEV)
    g = torch.Generator().manual_seed(6)
    for t in range(T + 7):  # wraps around
        tr = {k: (torch.randn((N,) + ((s,) if isinstance(s, int) else s), generator=g) if dt == torch.float32 else
                  torch.randint(0, 50, (N,), generator=g).to(dt)) for k, (s, dt) in FIELDS.items()}  # fmt: skip
        tr["done"] = torch.rand(N, generator=g) < 0.1
        b1.add(tr), b0.add(tr)
    torch.manual_seed(7)
    w1 = b1.sample_windows(32, 8)
    s1, h1 = b1.sample_starts(32, 10)
    torch.manual_seed(7)
    w0 = b0.sample_windows(32, 8)
    s0, h0 = b0.sample_starts(32, 10)
    for d1, d0 in ((w1, w0), (s1, s0), (h1, h0)):
        for k in d0:
            torch.testing.assert_close(d1[k], d0[k], **EXACT)
