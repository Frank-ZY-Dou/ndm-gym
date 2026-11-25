"""ndm_gym.robots.g1_hands and tools/build_g1_hand_task.py: the hand's mount on the wrist, the yaw helpers, the smoothing
before the lift, a frozen hinge and the walk's resampling."""

import os
import sys

import mujoco
import numpy as np
import pytest

from ndm_gym.robots import g1_hands
from ndm_gym.robots.g1 import MJCF_PATH as G1_MJCF

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


TEA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "spider", "processed", "gigahand", "inspire", "bimanual", "p36-tea")


@pytest.mark.skipif(not os.path.exists(os.path.join(TEA, "0", "trajectory_kinematic.npz")), reason="the tea clip (tools/download_hand_clips.py)")
@pytest.mark.skipif(not os.path.exists(G1_MJCF), reason="the G1's model (scripts/setup.sh)")
def test_build_and_compile(tmp_path, monkeypatch):
    """The standing G1 task from the tea clip's first 40 frames, on the CPU: its scene compiles, each hand is its arm's seven
    hinges (position servos of stiffness 1,000 on joints of armature 1) then its twelve finger joints, the G1's own shapes
    take no contacts, the kinematics follow MuJoCo's and train_hand builds the task."""
    import torch

    import build_g1_hand_task
    from ndm_gym.robots.hands import HandKinematics, hand_joints, hand_keypoints
    from ndm_gym.train_hand import build

    # the clip's directory and its three parents mirrored (the scene's meshes are relative to them), its trial cut short
    real, task = os.path.dirname(os.path.dirname(os.path.dirname(TEA))), tmp_path / "gigahand"
    for child in ("inspire", "bimanual", "p36-tea", "0"):
        task.mkdir()
        for entry in os.listdir(real):
            if entry != child:
                os.symlink(os.path.join(real, entry), task / entry)
        real, task = os.path.join(real, child), task / child
    task.mkdir()
    task = task.parent
    qpos = np.load(os.path.join(TEA, "0", "trajectory_kinematic.npz"))["qpos"][:40]
    np.savez(task / "0" / "trajectory_kinematic.npz", qpos=qpos)
    out = tmp_path / "g1"
    monkeypatch.setattr(sys, "argv", ["build_g1_hand_task.py", "--clip", str(task / "0"), "--out", str(out)])
    build_g1_hand_task.main()

    model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    joints, starts, palm = hand_joints(model)
    assert starts == [0, 19] and len(joints) == 38 == model.nu
    arms = [j for j in joints if j in palm]
    assert len(arms) == 14 and all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in arms)
    assert all(model.dof_armature[model.jnt_dofadr[j]] == 1.0 for j in arms)
    servos = [a for a in range(model.nu) if int(model.actuator_trnid[a, 0]) in palm]
    assert all(model.actuator_gainprm[a, 0] == 1000.0 and model.actuator_biasprm[a, 1] == -1000.0 for a in servos)
    g1_shapes = [g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.split("_")[0] in ("pelvis", "torso", "waist")]
    assert g1_shapes and all(model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 for g in g1_shapes)

    kin = HandKinematics(model, "cpu")
    assert kin.num_hands == 2 and kin.palm_translations == []
    data = mujoco.MjData(model)
    sites = [s for hand in hand_keypoints(model) for s in hand]
    q = kin.q_lo + (kin.q_hi - kin.q_lo) * torch.rand(3, kin.num_joints, generator=torch.Generator().manual_seed(0))
    keypoints, _ = kin.forward(q)
    for i in range(3):
        data.qpos[model.jnt_qposadr[joints]] = q[i].double().numpy()
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(keypoints[i].numpy(), data.site_xpos[sites], atol=1e-4)
    _, _, built = build(str(out / "0"), "cpu")
    assert built.ref.frames > 0 and built.kin.num_joints == 38
