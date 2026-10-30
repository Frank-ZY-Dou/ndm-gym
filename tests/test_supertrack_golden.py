"""Regression test of this repository's SuperTrack components on synthetic clips and states: they reproduce the outputs
in tests/data/supertrack_golden.pt, written on CPU by tests/data/make_supertrack_golden.py with this implementation.
SuperTrack has no public reference implementation; the components shared with SuperMimic and SuperMimic's updates are
compared with the upstream reproduction in test_upstream_equivalence and test_algo_equivalence."""

import os

import pytest
import torch

from conftest import REPO

TOL = dict(rtol=1e-6, atol=1e-6)


@pytest.fixture(scope="module")
def golden():
    return torch.load(os.path.join(REPO, "tests", "data", "supertrack_golden.pt"))


@pytest.fixture(scope="module")
def synthetic_motion(golden, kin):
    """The motion library of the synthetic clips stored with the golden outputs."""
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import Clip

    return MotionLib([Clip(**c) for c in golden["clips"]], kin, "cpu")


def test_accel_world_model(golden):
    from ndm_gym.state import flatten_state, normalize_xi, unflatten_state
    from ndm_gym.worldmodels.accel import AccelWorldModel

    wm = AccelWorldModel(hidden=(64, 64))
    wm.load_state_dict(golden["wm_state"])
    S, S1, u, xi_n = unflatten_state(golden["S"]), unflatten_state(golden["S1"]), golden["u"], normalize_xi(golden["xi"])
    torch.testing.assert_close(wm.inputs(S, u, xi_n), golden["wm_inputs"], **TOL)
    torch.testing.assert_close(wm.targets(S, S1), golden["wm_targets"], **TOL)
    torch.testing.assert_close(flatten_state(wm.step(S, u, xi_n)), golden["S_next"], **TOL)


def test_losses(golden, kin, synthetic_motion):
    from ndm_gym.algos.supertrack import LossBalancer, ReferenceLocal, local_features, track_terms, wm_window_terms
    from ndm_gym.state import unflatten_state

    S, S1, S_next, f = unflatten_state(golden["S"]), unflatten_state(golden["S1"]), unflatten_state(golden["S_next"]), golden["f"]
    for a, b in zip(local_features(*kin.forward(*S, all_bodies=True)), golden["local_S"]):
        torch.testing.assert_close(a, b, **TOL)
    ref = ReferenceLocal(synthetic_motion, kin).feats
    for i, (a, b) in enumerate(zip(ref, golden["ref_at_f"])):  # i = 3: angular velocities by central differences,
        torch.testing.assert_close(a[f], b, **(dict(rtol=1e-6, atol=1e-5) if i == 3 else TOL))  # which differ across builds by ~3e-6
    torch.testing.assert_close(track_terms(kin, S, ref, f), golden["track_terms"], **TOL)
    terms = wm_window_terms(kin, S_next, S1)
    torch.testing.assert_close(terms, golden["wm_window_terms"], **TOL)
    bal = LossBalancer(4)
    bal.calibrate(terms.mean(0), torch.ones(4))
    torch.testing.assert_close(bal.w, golden["bal_w"], **TOL)


def test_center_of_mass(kin, motion):
    """Whole-body CoM from torch FK (fixed links merged into their parents) equals MuJoCo's subtree CoM."""
    import mujoco
    import numpy as np

    from conftest import random_states
    from ndm_gym.core.rotations import mat_to_quat
    from ndm_gym.robots import g1

    m = mujoco.MjModel.from_xml_path(g1.MJCF_PATH)
    d = mujoco.MjData(m)
    root = m.body("pelvis").id
    _, S, _ = random_states(motion, 8, seed=11)
    c, _ = kin.com(*kin.forward(*S, all_bodies=True))
    jadr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
    for i in range(8):
        d.qpos[0:3] = S[0][i].numpy()
        d.qpos[3:7] = mat_to_quat(S[1][i]).numpy()
        for j, name in enumerate(g1.G1_JOINT_NAMES):
            d.qpos[jadr[name]] = S[4][i, j].item()
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        np.testing.assert_allclose(c[i].numpy(), d.subtree_com[root], rtol=0, atol=2e-6)
