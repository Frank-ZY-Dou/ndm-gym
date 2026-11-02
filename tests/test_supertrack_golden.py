"""Regression test of this repository's SuperTrack components on synthetic clips and states: they reproduce the outputs
in tests/data/supertrack_golden.pt, written on CPU by tests/data/make_supertrack_golden.py with this implementation, and
its full updates reproduce tests/data/supertrack_update_golden.pt (tests/data/make_supertrack_update_golden.py). Both
files pin the behavior of the code that wrote them. SuperTrack has no public reference implementation; the components
shared with SuperMimic and SuperMimic's updates are compared with the upstream reproduction in test_upstream_equivalence
and test_algo_equivalence."""

import importlib.util
import os

import pytest
import torch

from conftest import REPO

TOL = dict(rtol=1e-6, atol=1e-6)
# a full update runs 32 imagined steps and their backward pass: the build that wrote the file reproduces it bit for bit,
# the tolerance leaves room for other CPUs' kernels
UPDATE_TOL = dict(rtol=1e-5, atol=1e-6)


@pytest.fixture(scope="module")
def golden():
    return torch.load(os.path.join(REPO, "tests", "data", "supertrack_golden.pt"))


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


def test_loss_balancer_calibrated():
    """``calibrated`` (checked at every update without reading the device once set) follows ``ready``, also across
    loaded states."""
    from ndm_gym.algos.supertrack import LossBalancer

    a, b = LossBalancer(4), LossBalancer(4)
    assert not a.calibrated and not bool(a.ready)
    a.calibrate(torch.rand(4) + 0.1, torch.ones(4))
    assert a.calibrated and bool(a.ready)
    b.load_state_dict(a.state_dict())
    assert b.calibrated
    a.load_state_dict(LossBalancer(4).state_dict())
    assert not a.calibrated and not bool(a.ready)


def test_updates(golden):
    """Two world-model updates and two policy updates (imagination noise, the task's penalties and the failure penalty on;
    a replay with starts too close to their clip's end) return what tests/data/supertrack_update_golden.pt holds and leave
    the same loss weights and normalizers: ``updates`` of the script that wrote the file, run again on this code."""
    path = os.path.join(REPO, "tests", "data", "make_supertrack_update_golden.py")
    spec = importlib.util.spec_from_file_location("make_supertrack_update_golden", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    want = torch.load(gen.OUT)
    got = gen.updates(golden["clips"])
    for kind in ("wm_updates", "pi_updates"):
        assert len(got[kind]) == len(want[kind])
        for i, (a, b) in enumerate(zip(got[kind], want[kind])):
            assert len(a) == len(b)
            for j, (x, y) in enumerate(zip(a, b)):
                torch.testing.assert_close(x, y, equal_nan=True, **UPDATE_TOL, msg=lambda m: f"{kind}[{i}][{j}]: {m}")
    for kind in ("bal_wm", "bal_pi", "norms"):
        assert got[kind].keys() == want[kind].keys()
        for k in want[kind]:
            torch.testing.assert_close(got[kind][k], want[kind][k], **UPDATE_TOL, msg=lambda m: f"{kind}[{k}]: {m}")


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
