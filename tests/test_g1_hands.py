"""ndm_gym.robots.g1_hands and tools/build_g1_hand_task.py: the hand's mount on the wrist, the yaw helpers, the smoothing
before the lift, a frozen hinge and the walk's resampling."""

import os
import sys

import mujoco
import numpy as np
import pytest

from ndm_gym.robots import g1_hands

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from build_g1_hand_task import WALK_FPS, resample  # noqa: E402


@pytest.mark.parametrize("side", ["left", "right"])
def test_mount_rotation(side):
    """The fingers along the forearm (the wrist's x), the index's side up (its z), the palm toward the body's middle; a
    mirrored hand's axes are refused."""
    distal, radial = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    palmar = np.cross(distal, radial) * (1 if side == "left" else -1)  # a right palm faces down with the index to the left
    axes = np.stack((distal, radial, palmar), axis=1)
    rotation = g1_hands.mount_rotation(axes, side)
    np.testing.assert_allclose(rotation @ distal, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(rotation @ radial, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(rotation @ palmar, [0, -1 if side == "left" else 1, 0], atol=1e-12)
    with pytest.raises(ValueError):
        g1_hands.mount_rotation(axes * np.array([1, 1, -1]), side)


def test_yaw():
    for yaw in (-2.5, -0.3, 0.0, 1.2, 3.0):
        assert g1_hands.quat_yaw(np.array(g1_hands.yaw_quat(yaw))) == pytest.approx(yaw)
    tilt = np.array([np.cos(0.05), np.sin(0.05), 0.0, 0.0])  # a small roll keeps the yaw
    assert g1_hands.quat_yaw(g1_hands.pelvis_quat(0.7, tilt)) == pytest.approx(0.7, abs=1e-12)


def test_soften():
    """Smoothed before the lift, the solved joints from it on, a fade between."""
    rng = np.random.default_rng(0)
    solutions = rng.normal(size=(60, 3))
    out = g1_hands.soften(solutions, until=30)
    np.testing.assert_array_equal(out[30:], solutions[30:])
    assert np.abs(np.diff(out[:20], axis=0)).mean() < 0.5 * np.abs(np.diff(solutions[:20], axis=0)).mean()


def test_freeze():
    """A frozen hinge leaves its body where the joint at that angle put it."""
    quat = np.array([0.9, 0.1, 0.2, 0.3]) / np.linalg.norm([0.9, 0.1, 0.2, 0.3])  # unit, as the models' body orientations are
    xml = f"""<mujoco><worldbody><body name="a" pos="0.1 0.2 0.3" quat="{' '.join(map(str, quat))}">
             <joint name="h" axis="0.3 0.4 1" pos="0.05 -0.02 0.1"/><geom size="0.1" pos="0.2 0 0"/></body></worldbody></mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[0] = 0.7
    mujoco.mj_kinematics(model, data)
    g1_hands.freeze(spec, "h", "a", 0.7)
    frozen = spec.compile()
    frozen_data = mujoco.MjData(frozen)
    mujoco.mj_kinematics(frozen, frozen_data)
    assert frozen.njnt == 0
    np.testing.assert_allclose(frozen_data.geom_xpos, data.geom_xpos, atol=1e-12)
    np.testing.assert_allclose(frozen_data.geom_xmat, data.geom_xmat, atol=1e-12)


def test_resample():
    """The walk at another rate: its rows at whole frames, interpolated between them, unit quaternions."""
    rng = np.random.default_rng(1)
    quat = rng.normal(size=(10, 4))
    walk = np.concatenate((rng.normal(size=(10, 3)), quat / np.linalg.norm(quat, axis=1, keepdims=True), rng.normal(size=(10, 29))), axis=1)
    out = resample(walk, 2.0, 5, 1.0 / WALK_FPS)
    np.testing.assert_allclose(out[:, 7:], walk[2:7, 7:])
    np.testing.assert_allclose(np.linalg.norm(out[:, 3:7], axis=1), 1.0)
    half = resample(walk, 2.5, 1, 1.0 / WALK_FPS)
    np.testing.assert_allclose(half[0, :3], (walk[2, :3] + walk[3, :3]) / 2)
