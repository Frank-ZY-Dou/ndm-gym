"""Humanoid motion tracking on MuJoCo-Warp (robots.humanoid, tasks.humanoid_tracking, algos.humanoid_algos, train_humanoid,
evaluate_humanoid): the scene and its servos, the kinematics against MuJoCo's, the clips and the reference, the task at the
reference and its invariances, the agents' updates and the trainer's arguments without a simulator; on a GPU the
simulator's steps, the training loop with the systems options and the evaluation.

The tests that run on the CPU run on a small synthetic humanoid written by the tests (``toy``: a free pelvis, two legs of
four hinges with a mesh foot and a capsule foot, a torso hinge off its body's origin and a head without a joint; three short
clips in the LAFAN1 retarget's CSV layout), so that they need no download, and on the H1 (and the G1 where they compare with
MuJoCo) when its description and clips are downloaded. The tests of the H1's own values (its cited gains, Menagerie's foot,
the ankle's 40 N m) and the GPU's need the H1's files."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from typing import NamedTuple

import mujoco
import numpy as np
import pytest
import torch

from ndm_gym.algos.humanoid_algos import HUMANOID_ALGOS
from ndm_gym.algos.quadruped_algos import WINDOW_KEYS, replay_fields, world_model
from ndm_gym.buffer import ReplayBuffer
from ndm_gym.core.normalizer import RunningNorm
from ndm_gym.motion.sources import lafan1
from ndm_gym.robots.humanoid import ROBOTS, HumanoidKinematics, HumanoidSpec, load_scene
from ndm_gym.state import CONTROL_DT
from ndm_gym.tasks.humanoid_tracking import HumanoidTask, RewardCfg, build_reference, resample
from ndm_gym.train_humanoid import parse_args, select_clips

CLIPS = ["walk1_subject1", "run1_subject2"]
H1 = ROBOTS["h1"]
CUDA = "cuda:0"
HAVE_SIM = torch.cuda.is_available() and importlib.util.find_spec("mujoco_warp") is not None
GPU = pytest.mark.skipif(not HAVE_SIM, reason="needs a GPU and MuJoCo-Warp")


class Robot(NamedTuple):
    """A humanoid under test: its spec, the test clips, its mass (kg), its number of clips and a selection of them by name and
    glob with the clips it selects."""

    spec: HumanoidSpec
    clips: list[str]
    mass: float
    num_clips: int
    select: str
    selected: list[str]


def have(spec: HumanoidSpec) -> bool:
    """Whether the robot's description and test clips are downloaded."""
    return os.path.exists(spec.mjcf) and all(os.path.exists(os.path.join(spec.motions, c + ".csv")) for c in CLIPS)


def needs(name: str):
    spec = ROBOTS[name]
    return pytest.mark.skipif(not have(spec), reason=f"needs the {name.upper()}'s description and clips ({spec.download})")


NEEDS_H1 = needs("h1")
REAL = {"h1": Robot(H1, CLIPS, 51.6, 40, "walk1_*,run1_subject2", ["run1_subject2", "walk1_subject1", "walk1_subject2", "walk1_subject5"]),
        "g1": Robot(ROBOTS["g1"], CLIPS, 33.3, 40, "walk1_*,run1_subject2", ["run1_subject2", "walk1_subject1", "walk1_subject2", "walk1_subject5"])}  # fmt: skip

# ------------------------------------------------- the synthetic humanoid
TOY_FOOT = " ".join(f"{x} {y} {z}" for x in (-0.05, 0.12) for y in (-0.03, 0.03) for z in (-0.07, -0.04))


def _toy_leg(side: str, y: float) -> str:
    foot = ('<geom type="mesh" mesh="left_foot"/>' if side == "left" else '<geom type="capsule" fromto="-0.05 0 -0.05 0.12 0 -0.05" size="0.02"/>')  # fmt: skip
    return f"""
      <body name="{side}_hip_roll_link" pos="0 {y} -0.05">
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
        <joint name="{side}_hip_roll_joint" axis="1 0 0" range="-0.4 0.4" actuatorfrcrange="-100 100"/>
        <body name="{side}_hip_pitch_link" pos="0 0 -0.05">
          <inertial pos="0 0 -0.2" mass="2" diaginertia="0.03 0.03 0.005"/>
          <joint name="{side}_hip_pitch_joint" axis="0 1 0" range="-1.5 1.0" actuatorfrcrange="-100 100"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04"/>
          <body name="{side}_knee_link" pos="0 0 -0.4">
            <inertial pos="0 0 -0.2" mass="2.5" diaginertia="0.03 0.03 0.005"/>
            <joint name="{side}_knee_joint" axis="0 1 0" range="-0.1 2.0" actuatorfrcrange="-150 150"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.035"/>
            <body name="{side}_ankle_link" pos="0 0 -0.4">
              <inertial pos="0.03 0 -0.05" mass="1" diaginertia="0.003 0.003 0.001"/>
              <joint name="{side}_ankle_joint" axis="0 1 0" range="-0.8 0.6" actuatorfrcrange="-40 40"/>
              {foot}
            </body>
          </body>
        </body>
      </body>"""


TOY_MJCF = f"""<mujoco model="toy">
  <compiler angle="radian" autolimits="true"/>
  <asset>
    <mesh name="left_foot" vertex="{TOY_FOOT}"/>
  </asset>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="floating_base_joint"/>
      <inertial pos="0 0 0" mass="8" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.08 0.12 0.05"/>
      {_toy_leg("left", 0.1)}
      {_toy_leg("right", -0.1)}
      <body name="torso_link" pos="0 0 0.1" quat="0.99905 0 0.04362 0">
        <inertial pos="0 0 0.2" mass="10" diaginertia="0.2 0.2 0.05"/>
        <joint name="torso_joint" axis="0.1 0 1" pos="0.02 0 0" range="-1 1" actuatorfrcrange="-100 100"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.35" size="0.08"/>
        <body name="head_link" pos="0 0 0.45">
          <inertial pos="0 0 0" mass="2" diaginertia="0.01 0.01 0.01"/>
          <geom type="sphere" size="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
TOY_KINDS = ("hip_roll", "hip_pitch", "knee", "ankle")
TOY_JOINTS = tuple(f"{s}_{k}_joint" for s in ("left", "right") for k in TOY_KINDS) + ("torso_joint",)
# per joint kind: effort (N m, the MJCF's actuatorfrcrange), stiffness, damping
TOY_SERVOS = {"hip_roll": (100, 100, 2), "hip_pitch": (100, 100, 2), "knee": (150, 150, 3), "ankle": (40, 30, 1), "torso": (100, 200, 4)}
TOY_CLIPS = ["sway1_subject1", "step1_subject1"]
TOY_BENT = np.array([0.0, -0.3, 0.6, -0.3])  # a leg's hinges, the foot flat; the sole 0.934 m below the pelvis


def _quat_xyzw(yaw, pitch, roll) -> np.ndarray:
    """The rotation yaw, then pitch, then roll (z, y, x) as quaternions (..., 4), x, y, z, w (the CSV's order)."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(pitch / 2), np.sin(pitch / 2), np.cos(roll / 2), np.sin(roll / 2)
    return np.stack((sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy), -1)


def _toy_clip(kind: str, frames: int, phase: float = 0.0) -> np.ndarray:
    """A synthetic clip's rows at 30 fps (the root's position, its quaternion x, y, z, w, the joints), the feet about 2 cm
    into the floor as a retarget's may be: "sway", the robot standing, swaying and turning in place; "step", the robot
    moving forward at 0.15 m/s on its right leg while lifting its left foot."""
    t = np.arange(frames) / lafan1.FPS
    left, right = np.tile(TOY_BENT, (frames, 1)), np.tile(TOY_BENT, (frames, 1))
    if kind == "sway":
        roll = 0.03 * np.sin(2 * np.pi * t + phase)
        left[:, 0] += roll
        right[:, 0] -= roll
        yaw = 0.5 + 0.2 * t
        x, y = np.full(frames, 0.3), np.full(frames, -0.2)
        z = 0.915 + 0.01 * np.sin(2 * np.pi * 0.8 * t + phase)
    else:
        left[:, 1:] += np.outer(np.maximum(0.0, np.sin(2 * np.pi * 0.75 * t)), [-0.3, 0.6, -0.3])
        yaw = np.full(frames, -0.7)
        x, y = 0.15 * t * np.cos(yaw), 0.15 * t * np.sin(yaw)
        z = 0.92 + 0.005 * np.sin(2 * np.pi * t)
    quat = _quat_xyzw(yaw, 0.02 * np.sin(2 * np.pi * t), 0.02 * np.cos(np.pi * t))
    torso = 0.3 * np.sin(np.pi * t + phase)
    return np.concatenate((np.stack((x, y, z), -1), quat, left, right, torso[:, None]), -1)


@pytest.fixture(scope="module")
def toy(tmp_path_factory) -> Robot:
    """The synthetic humanoid: its MJCF and three clips (two of 2 s, one of 1.5 s) in a temporary directory."""
    root = tmp_path_factory.mktemp("toy")
    (root / "toy.xml").write_text(TOY_MJCF)
    motions = root / "motions"
    motions.mkdir()
    for name, rows in (
        ("sway1_subject1", _toy_clip("sway", 60)),
        ("sway1_subject2", _toy_clip("sway", 45, phase=1.0)),
        ("step1_subject1", _toy_clip("step", 60)),
    ):
        np.savetxt(motions / f"{name}.csv", rows, delimiter=",", fmt="%.6f")
    kind = [j.replace("left_", "").replace("right_", "").replace("_joint", "") for j in TOY_JOINTS]
    tracked = ("pelvis", "left_hip_pitch_link", "left_knee_link", "left_ankle_link", "right_hip_pitch_link", "right_knee_link", "right_ankle_link",
               "torso_link", "head_link")  # fmt: skip
    spec = HumanoidSpec("toy", str(root / "toy.xml"), str(motions), TOY_JOINTS, tracked, tuple(float(TOY_SERVOS[k][0]) for k in kind),
                        tuple(float(TOY_SERVOS[k][1]) for k in kind), tuple(float(TOY_SERVOS[k][2]) for k in kind), (0.05,) * len(kind),
                        (0.5,) * len(kind), "tests/test_humanoid_tracking.py (written by the tests)", ("left_ankle_link", "right_ankle_link"))  # fmt: skip
    return Robot(spec, TOY_CLIPS, 33.0, 3, "sway1_*,step1_subject1", ["step1_subject1", "sway1_subject1", "sway1_subject2"])


def _robot(request, name: str) -> Robot:
    return request.getfixturevalue("toy") if name == "toy" else REAL[name]


@pytest.fixture(scope="module", params=["toy", pytest.param("h1", marks=NEEDS_H1)])
def robot(request) -> Robot:
    """The synthetic humanoid, and the H1 when its files are downloaded."""
    return _robot(request, request.param)


@pytest.fixture(scope="module", params=["toy", pytest.param("h1", marks=NEEDS_H1), pytest.param("g1", marks=needs("g1"))])
def any_robot(request) -> Robot:
    """The synthetic humanoid, and the H1 and the G1 when their files are downloaded."""
    return _robot(request, request.param)


@pytest.fixture
def registered(robot, monkeypatch) -> Robot:
    """``robot`` in robots.humanoid.ROBOTS under its name, as train_humanoid.build looks robots up."""
    monkeypatch.setitem(ROBOTS, robot.spec.name, robot.spec)
    return robot


def short_clips(robot: Robot, n: int = 40):
    """The test clips cut to their first ``n`` frames at 30 fps (short clips keep the tests fast)."""
    clips = select_clips(lafan1.load_dir(robot.spec.motions, robot.spec.num_joints), robot.clips)
    return [c._replace(root_p=c.root_p[:n], root_quat=c.root_quat[:n], joints=c.joints[:n]) for c in clips]


def make_task(robot: Robot, dev: str):
    model = load_scene(robot.spec)
    kin = HumanoidKinematics(model, robot.spec, dev)
    return model, kin, HumanoidTask(kin, build_reference(short_clips(robot), kin, dev))


@pytest.fixture(scope="module")
def setup(robot):
    return make_task(robot, "cpu")


def test_scene_has_position_servos(robot):
    spec = robot.spec
    model = load_scene(spec)
    assert model.nu == spec.num_joints and float(model.geom_margin.max()) == 0 and model.opt.timestep == 0.002
    assert abs(float(model.body_mass.sum()) - robot.mass) < 0.1
    data = mujoco.MjData(model)
    data.qpos[2], data.qpos[3] = 1.1, 1.0
    data.ctrl[:] = data.qpos[7:] + 0.001
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, np.array(spec.kp) * 0.001, rtol=1e-6)  # at rest: kp times the offset
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], spec.effort)
    np.testing.assert_allclose(model.dof_armature[6:], spec.armature)
    np.testing.assert_allclose(model.dof_damping[6:], spec.damping)
    collides = model.geom_contype == 2  # the robot's collision geoms touch the floor only
    assert collides.any() and (model.geom_conaffinity[collides] == 1).all() and int((model.geom_contype == 1).sum()) == 1


def test_scene_refuses_another_layout(toy, tmp_path):
    """load_scene refuses a description whose joints are not the spec's in its order, and a missing one with its download."""
    import dataclasses

    swapped = dataclasses.replace(toy.spec, joints=toy.spec.joints[1:] + toy.spec.joints[:1])
    with pytest.raises(ValueError, match="expected a free pelvis"):
        load_scene(swapped)
    with pytest.raises(FileNotFoundError, match="written by the tests"):
        load_scene(dataclasses.replace(toy.spec, mjcf=str(tmp_path / "missing.xml")))


def test_kinematics_match_mujoco(any_robot):
    spec = any_robot.spec
    model = load_scene(spec)
    kin = HumanoidKinematics(model, spec, "cpu")
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    data.qpos[:3] = [0.1, 0.2, 1.0]
    q = rng.normal(size=4)
    data.qpos[3:7] = q / np.linalg.norm(q)
    data.qpos[7:] = rng.uniform(kin.q_lo.numpy(), kin.q_hi.numpy())
    data.qvel[:] = rng.normal(size=model.nv)
    mujoco.mj_forward(model, data)
    P, R = kin.forward(torch.tensor(data.qpos, dtype=torch.float32)[None])
    V, W = kin.velocities((P, R), torch.tensor(data.qvel, dtype=torch.float32)[None])
    vel = np.zeros((model.nbody - 1, 6))
    for b in range(1, model.nbody):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, vel[b - 1], 0)
    vel[:, 3:] += np.cross(vel[:, :3], data.xpos[1:] - data.xipos[1:])  # MuJoCo's at the centre of mass, ours at the origin
    np.testing.assert_allclose(P[0].numpy(), data.xpos[1:], atol=1e-5)
    np.testing.assert_allclose(R[0].numpy().reshape(-1, 9), data.xmat[1:], atol=1e-5)
    np.testing.assert_allclose(V[0].numpy(), vel[:, 3:], atol=1e-4)
    np.testing.assert_allclose(W[0].numpy(), vel[:, :3], atol=1e-4)
    assert kin.tracked.tolist()[0] == 0 and len(kin.tracked) == len(spec.tracked)


def test_clips_load_and_resample(robot):
    """The clips' columns (the quaternion x, y, z, w in the file, w, x, y, z loaded), their resampling to the control rate
    and their selection by names and globs."""
    spec = robot.spec
    clips = lafan1.load_dir(spec.motions, spec.num_joints)
    assert len(clips) == robot.num_clips and all(c.joints.shape[1] == spec.num_joints and c.fps == 30 for c in clips)
    with pytest.raises(ValueError, match="columns"):
        lafan1.load_dir(spec.motions, spec.num_joints + 1)
    rows = np.loadtxt(os.path.join(spec.motions, clips[0].name + ".csv"), delimiter=",")
    np.testing.assert_allclose(clips[0].root_quat[:, [1, 2, 3, 0]].numpy(), rows[:, 3:7], atol=1e-6)
    c = short_clips(robot, 31)[0]  # 1 s
    q = resample(c)
    assert (
        q.shape == (51, 7 + spec.num_joints)
        and torch.allclose(q[0, :3].float(), c.root_p[0])
        and torch.allclose(q[-1, 7:].float(), c.joints[-1], atol=1e-6)
    )
    assert torch.allclose(q[:, 3:7].norm(dim=-1), torch.ones(51, dtype=torch.float64)) and bool(((q[1:, 3:7] * q[:-1, 3:7]).sum(-1) > 0).all())
    assert torch.allclose(q[::5, 7:].float(), c.joints[::3], atol=1e-6)  # every 0.1 s a frame of the clip's
    assert [c.name for c in select_clips(clips, robot.select)] == robot.selected
    assert [c.name for c in select_clips(clips, robot.clips)] == sorted(robot.clips, key=[c.name for c in clips].index)
    assert select_clips(clips, None) is clips
    with pytest.raises(ValueError, match="no clip"):
        select_clips(clips, "nope")


def test_resample_keeps_the_quaternion_continuous():
    """A clip whose quaternion changes sign between frames (the same rotation) resamples to a continuous rotation, its
    first frame's w positive."""
    from ndm_gym.motion.sources import Clip

    t = torch.arange(10, dtype=torch.float32) / 30
    q = torch.stack((torch.cos(0.5 * t), torch.zeros(10), torch.zeros(10), torch.sin(0.5 * t)), -1)
    q[3::2] *= -1
    q = -q
    clip = Clip("flip", torch.zeros(10, 3), q, torch.zeros(10, 2), 30.0)
    out = resample(clip)
    assert float(out[0, 3]) > 0 and bool(((out[1:, 3:7] * out[:-1, 3:7]).sum(-1) > 0.99).all())
    yaw = 2 * torch.atan2(out[:, 6], out[:, 3])
    torch.testing.assert_close(yaw, torch.arange(out.shape[0], dtype=torch.float64) * CONTROL_DT, atol=1e-6, rtol=0)


def test_reference(setup):
    """The reference's clips, frames and velocities: the state's are finite differences of its poses, the bodies' those of
    their positions."""
    _, kin, task = setup
    ref = task.ref
    assert ref.num_clips == 2 and ref.frames == 2 * 66 and ref.clip_first.tolist() == [0, 66] and ref.clip_last.tolist() == [65, 131]
    f = torch.arange(1, 64)
    np.testing.assert_allclose(ref.qvel[f, :3], (ref.qpos[f + 1, :3] - ref.qpos[f - 1, :3]) / (2 * CONTROL_DT), atol=1e-4)
    np.testing.assert_allclose(ref.body_v[f], (ref.body_p[f + 1] - ref.body_p[f - 1]) / (2 * CONTROL_DT), atol=0.05)
    assert torch.equal(ref.next_frame(torch.tensor([64, 65, 66])), torch.tensor([65, 65, 67]))


def test_task_at_the_reference(setup):
    _, kin, task = setup
    ref = task.ref
    f = torch.arange(ref.frames)
    S = (ref.qpos[f], ref.qvel[f])
    fk, a = kin.forward(S[0]), torch.zeros(ref.frames, task.num_joints)
    assert task.obs(S, fk, a).shape == (ref.frames, task.obs_dim) and task.goal(S, fk, f, 3).shape == (ref.frames, 3 * task.goal_frame_dim)
    J, B = kin.num_joints, len(kin.spec.tracked)
    assert task.state_dim == 2 * (7 + J) - 1 and task.obs_dim == 1 + 6 + 6 + 3 * J + 3 * (B - 1)
    q = S[0][:, 7:]
    in_limits = ((q >= task.soft_lo) & (q <= task.soft_hi)).all(-1)
    assert in_limits.any()
    r = task.reward(S, fk, f, a, a)
    assert torch.allclose(r[in_limits], torch.full_like(r[in_limits], task.max_step_reward), atol=1e-5) and bool(
        (r <= task.max_step_reward + 1e-6).all()
    )
    e = task.errors(S, fk, f)
    assert all(float(v.abs().max()) < 1e-3 for k, v in e.items() if k != "root_ori_sq") and float(e["root_ori_sq"].max()) < 1e-5
    assert not task.termination(S, fk, f).any() and float(task.soft_termination(S, fk, f).max()) < 1e-3
    assert torch.allclose(task.setpoint(a, f), torch.clamp(ref.qpos[ref.next_frame(f), 7:], kin.q_lo, kin.q_hi))
    with_pos = HumanoidTask(kin, ref, RewardCfg(term_pos=1.0))  # the optional term beyond the G1's
    for shift, by_default in (((0.0, 0.0, -0.4), True), ((1.4, 0.0, 0.0), False)):  # fallen (the G1's terms), or far from the reference
        far = S[0].clone()
        far[:, :3] += torch.tensor(shift)
        fk = kin.forward(far)
        for t, ends in ((task, by_default), (with_pos, True)):
            assert bool(t.termination((far, S[1]), fk, f).all()) == ends and bool(t.termination((far, S[1]), fk, f).any()) == ends
            assert (float(t.soft_termination((far, S[1]), fk, f).min()) > 0.99) == ends
    m = task.step_metrics(task.errors((far, S[1]), fk, f))
    assert torch.allclose(m["mpjpe_g_mm"], torch.full((ref.frames,), 1400.0), atol=0.1) and float(m["mpjpe_l_mm"].max()) < 1


def test_observation_and_reward_are_heading_invariant(setup):
    """Turning and moving the robot and the reference together changes neither the observation nor the world model's
    inputs; turning and moving the robot alone horizontally leaves the bodies' re-anchored position term unchanged."""
    model, kin, task = setup
    torch.manual_seed(0)
    f = torch.arange(0, task.ref.frames, 5)
    qpos, qvel = task.ref.qpos[f].clone(), task.ref.qvel[f].clone()
    a = torch.rand(f.shape[0], task.num_joints)
    yaw = 1.1
    q_yaw = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    Rz = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    moved, mvel = qpos.clone(), qvel.clone()
    moved[:, :3] = qpos[:, :3] @ Rz.T + torch.tensor([1.0, -2.0, 0.0])
    w, v = q_yaw[0], q_yaw[1:]
    qv, qw = qpos[:, 4:7], qpos[:, 3:4]  # q_yaw * q
    moved[:, 3:4] = w * qw - (v * qv).sum(-1, keepdim=True)
    moved[:, 4:7] = w * qv + qw * v + torch.linalg.cross(v.expand_as(qv), qv)
    mvel[:, :3] = qvel[:, :3] @ Rz.T
    S, M = (qpos, qvel), (moved, mvel)
    assert torch.allclose(task.obs(S, kin.forward(qpos), a), task.obs(M, kin.forward(moved), a), atol=1e-4)
    assert float(task.reward_terms(M, kin.forward(moved), f)["body_pos"].max()) < 1e-8
    wm = world_model(model, "accel", hidden=(16,))
    u = qpos[:, 7:] + 0.1
    assert torch.allclose(wm.inputs(qpos, qvel, u), wm.inputs(moved, mvel, u), atol=1e-4)


def test_state_layout_and_world_model(setup):
    model, _, task = setup
    f = torch.arange(task.ref.frames - 1)
    S, S1 = (task.ref.qpos[f], task.ref.qvel[f]), (task.ref.qpos[f + 1], task.ref.qvel[f + 1])
    x = torch.cat(S, -1)
    assert x.shape == (len(f), task.state_dim) and all(torch.equal(a, b) for a, b in zip(task.unflatten(x), S, strict=True))
    J = task.num_joints
    assert replay_fields(task)["s"][0] == task.state_dim and replay_fields(task)["u"][0] == J
    torch.manual_seed(0)
    for kind in ("accel", "increment"):
        wm = world_model(model, kind, hidden=(32,))
        assert wm.lay.dt == CONTROL_DT and wm.lay.anchor == (0, 0) and wm.lay.nu == J
        wm.update_norms(*S, S[0][:, 7:], *S1)
        q1, v1 = wm.step(*S, S[0][:, 7:])
        assert q1.shape == S[0].shape and v1.shape == S[1].shape and torch.isfinite(q1).all()


def small(extra: list) -> list:
    argv = ["--goal_frames", "3", "--horizon", "4", "--wm_hidden", "32", "32", "--wm_batches", "2", "--wm_batch", "16", "--wm_window", "4",
            "--pi_batch", "16"] + extra  # fmt: skip
    return argv + ([] if "supertrack" in extra else ["--critic_epochs", "1", "--critic_minibatches", "2"])


@pytest.mark.parametrize("extra", [[], ["--algo", "supertrack", "--st_fail_w", "2"], ["--algo", "supertrack", "--st_task_weights"],
                                   ["--algo", "supertrack", "--st_terms", "g1", "--st_balance_w", "2", "--st_fail_w", "1"]],
                         ids=["supermimic", "supertrack", "supertrack_task_weights", "supertrack_g1_terms"])  # fmt: skip
def test_updates(setup, monkeypatch, extra):
    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    torch.manual_seed(0)
    model, kin, task = setup
    args = parse_args(small(extra))
    algo = HUMANOID_ALGOS[args.algo](args, model, kin, task, "cpu")
    assert algo.pi.net[-1].out_features == task.num_joints
    buffer = ReplayBuffer(16, 8, "cpu", fields=replay_fields(task), priorities=False)
    buffer.window_keys = WINDOW_KEYS
    f = torch.randint(0, task.ref.frames - 1, (8,))
    S, a_prev = (task.ref.qpos[f], task.ref.qvel[f]), torch.zeros(8, task.num_joints)
    for _ in range(16):
        a = algo.act(S, a_prev, f, noise=0.1)
        u, f1 = task.setpoint(a, f), task.ref.next_frame(f)
        S1 = (task.ref.qpos[f1], task.ref.qvel[f1])
        algo.update_wm_norms(S, S1, u)
        buffer.add({"s": torch.cat(S, -1), "s1": torch.cat(S1, -1), "u": u, "a_prev": a_prev, "f": f, "done": torch.zeros(8, dtype=torch.bool)})
        S, f, a_prev = S1, f1, a
    before = [p.clone() for p in algo.pi.parameters()]
    assert torch.isfinite(algo.wm_update(buffer))
    out = algo.policy_update(buffer)
    assert all(torch.isfinite(x) for x in out[:1] + out[2:]) and 0 < float(out[2]) <= 1 and 0 < float(out[4]) <= 1
    assert any(not torch.equal(a, b) for a, b in zip(before, algo.pi.parameters(), strict=True))
    if args.algo == "supertrack":
        e = algo._errors(S, kin.forward(S[0]), f)
        assert e.shape == (8, len(algo.TERMS)) and float(e.abs().max()) < 5e-3  # on the reference: no error (to float32 rounding of the angles)
        if args.st_task_weights:
            assert algo.step_worth == pytest.approx(5.0) and float(algo.bal.w[algo.TERMS.index("joint_pos")]) == 0
        if args.st_terms == "g1":  # the library G1's terms, calibrated, the balance terms weighted on top
            assert algo.TERMS == ("lpos", "lvel", "lrot", "lang", "hei", "up", "rpos", "rrot") and algo.step_worth == 8
            assert algo.term_mult.tolist() == [1, 1, 1, 1, 2, 2, 1, 1] and algo.bal.calibrated


@NEEDS_H1
def test_menagerie_feet():
    """--foot menagerie: the H1's ankle meshes stop colliding and Menagerie's three capsules per foot take their place, the
    sole at the mesh's height (0.070 m below the ankle), the masses unchanged; the default keeps the mesh."""

    def sole(model):  # the left foot's collision geoms and their lowest point below the ankle at the zero pose
        data = mujoco.MjData(model)
        data.qpos[2] = 1.5
        mujoco.mj_kinematics(model, data)
        ankle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link")
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == ankle and model.geom_contype[g]]
        low = []
        for g in geoms:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                m = model.geom_dataid[g]
                v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
                low.append((data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T)[:, 2].min())
            else:
                axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
                low.append(data.geom_xpos[g][2] - abs(axis[2]) * model.geom_size[g][1] - model.geom_size[g][0])
        return [int(model.geom_type[g]) for g in geoms], min(low) - data.xpos[ankle][2]

    mesh, caps = load_scene(H1), load_scene(H1, foot="menagerie")
    assert sole(mesh)[0] == [mujoco.mjtGeom.mjGEOM_MESH] and sole(caps)[0] == [mujoco.mjtGeom.mjGEOM_CAPSULE] * 3
    assert sole(caps)[1] == pytest.approx(-0.070, abs=1e-3) and sole(mesh)[1] == pytest.approx(-0.070, abs=2e-3)
    np.testing.assert_allclose(caps.body_mass, mesh.body_mass)
    assert int((caps.geom_contype == 1).sum()) == 1 and (caps.geom_conaffinity[caps.geom_contype == 2] == 1).all()
    with pytest.raises(ValueError):
        load_scene(ROBOTS["g1"], foot="menagerie")
    assert parse_args([]).foot == "mesh" and parse_args(["--foot", "menagerie"]).foot == "menagerie"
    with pytest.raises(SystemExit):
        parse_args(["--robot", "g1", "--foot", "menagerie"])


def test_ankle_effort_scale(robot):
    """--ankle_effort_scale: the ankles' force limits and action scale times the scale, every other joint as it was, in the
    scene both as the actuators' force limits and as the joints' actuator torque limits; 1 the robot's own."""
    from ndm_gym.robots.humanoid import with_ankle_effort

    own = robot.spec
    assert with_ankle_effort(own, 1.0) is own and parse_args([]).ankle_effort_scale == 1.0
    spec = with_ankle_effort(own, 2.0)
    ankle = np.array(["ankle" in j for j in own.joints])
    assert ankle.sum() == 2
    np.testing.assert_allclose(np.array(spec.effort)[ankle], 2 * np.array(own.effort)[ankle])
    np.testing.assert_allclose(np.array(spec.effort)[~ankle], np.array(own.effort)[~ankle])
    np.testing.assert_allclose(np.array(spec.action_scale)[ankle], 2 * np.array(own.action_scale)[ankle])
    model = load_scene(spec)
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], spec.effort)
    np.testing.assert_allclose(model.jnt_actfrcrange[1:, 1], spec.effort)
    with pytest.raises(ValueError):
        with_ankle_effort(own, 0.0)
    with pytest.raises(SystemExit):
        parse_args(["--ankle_effort_scale", "0"])


def test_training_arguments(toy):
    args = parse_args([])
    assert (args.robot, args.kp_scale, args.algo, args.clips) == ("h1", 1.0, "supermimic", None) and not hasattr(args, "kp")
    h1 = parse_args(["--config", "h1-supertrack"])
    assert (h1.robot, h1.algo, h1.iterations, h1.st_task_weights) == ("h1", "supertrack", 8000, True)
    assert (h1.no_sync, h1.action_bound, h1.bin_starts) == (True, 4.0, True)
    model = load_scene(toy.spec)
    wm = world_model(model, "accel", hidden=(16,), mean_velocity=True)  # --accel_mean_velocity
    assert float(wm.pos_lead) == pytest.approx(11 / 20) and not hasattr(world_model(model, "increment", hidden=(16,), mean_velocity=True), "pos_lead")
    assert (args.utd, args.term_pos, args.max_episode_steps, args.bin_starts, args.st_terms) == (1.0, float("inf"), 0, False, "task")
    for refused in (["--robot", "go2"], ["--joint_damping", "0.1"], ["--hist", "4"], ["--st_fail_w", "2"], ["--config", "go2-supertrack"],
                    ["--algo", "supertrack", "--st_balance_w", "2"], ["--fail_reset", "0.3"], ["--st_terms", "g1"], ["--utd", "0"],
                    ["--accel_mean_velocity"], ["--action_bound", "0"], ["--max_episode_steps", "-1"], ["--term_pos", "0"],
                    ["--bin_starts", "--fail_reset", "1.5"]):  # fmt: skip
        with pytest.raises(SystemExit):
            parse_args(refused)


# ------------------------------------------------- on a GPU: the simulator, the training loop and the evaluation
@pytest.fixture(scope="module")
def gpu_setup():
    if not have(H1):
        pytest.skip("needs the H1's description and clips (tools/download_h1.sh)")
    return make_task(REAL["h1"], CUDA)


@GPU
def test_simulator_steps_and_resets(gpu_setup):
    """The reference's own targets keep a world on its clip for a few steps; an ended world restarts at a start frame,
    and the masked reset writes the same starts' layout."""
    from ndm_gym.train_humanoid import make_sim

    model, kin, task = gpu_setup
    sim = make_sim(model, task, 64, CUDA)
    assert sim.substeps == 10 and sim.state().shape == (64, task.state_dim)
    first = task.ref.clip_first.repeat(32)
    sim.reset(torch.ones(64, dtype=torch.bool, device=CUDA), frames=first, noise=False)
    for _ in range(10):
        s1, term, done = sim.step(task.setpoint(torch.zeros(64, 19, device=CUDA), sim.f), reset=False)
        assert not term.any() and bool(torch.isfinite(s1).all())
    S1 = task.unflatten(s1)
    assert float(task.errors(S1, kin.forward(S1[0]), sim.f)["root_pos"].max()) < 0.1
    sim.qpos[:8, 2] -= 0.5  # fallen
    s1, term, done = sim.step(task.setpoint(torch.zeros(64, 19, device=CUDA), sim.f))
    assert bool(term[:8].all()) and bool((sim.f[:8] < task.ref.last[sim.f[:8]]).all())


@GPU
def test_pushes_do_not_depend_on_an_early_end(gpu_setup):
    """evaluate_quadruped.rollout_sums ends once every world has terminated; the remaining steps' pushes are still drawn,
    so that the next rollouts' pushes do not depend on how this policy did."""
    from ndm_gym.evaluate_quadruped import rollout_sums
    from ndm_gym.train_humanoid import make_sim

    model, kin, _ = gpu_setup
    task = HumanoidTask(kin, build_reference(short_clips(REAL["h1"], 400), kin, CUDA))  # 13 s: the end is checked every 100 steps
    sim, gen = make_sim(model, task, 8, CUDA), torch.Generator(device=CUDA).manual_seed(0)
    steps = int((task.ref.clip_last - task.ref.clip_first).max())
    _, term_at, _ = rollout_sums(sim, task, [None], 5, noise=True, push=(5.0, 10, gen))  # the reference's targets, pushed hard
    assert bool((term_at >= 0).all()) and int(term_at.max()) <= steps - 200  # every world down: the loop ended early
    want = torch.Generator(device=CUDA).manual_seed(0)
    for _ in range(sum(t % 10 == 0 for t in range(1, steps))):
        torch.rand(8, generator=want, device=CUDA)
    assert torch.equal(torch.rand(4, generator=gen, device=CUDA), torch.rand(4, generator=want, device=CUDA))


def test_clip_bins():
    from ndm_gym.sim.rigid.humanoid_sim import clip_bins

    b = clip_bins(torch.tensor([0, 120]), torch.tensor([119, 150]))  # clips of 120 and 31 frames: 3 bins and 1
    assert b.bin_start.tolist() == [0, 50, 100, 120] and b.bin_clip.tolist() == [0, 0, 0, 1] and b.num_bins == 4
    assert b.frame_bin[[0, 49, 50, 119, 120, 150]].tolist() == [0, 0, 1, 2, 3, 3]


@GPU
def test_simulator_episode_options(gpu_setup):
    """HumanoidSim's training options: episodes end after max_episode_steps, start frames are drawn by bins within the
    clips (never a clip's last frame), the bins' returns are updated from the ended episodes, and the G1's perturbation
    keeps a unit quaternion."""
    from ndm_gym.train_humanoid import make_sim

    model, kin, task = gpu_setup
    args = parse_args(["--max_episode_steps", "3", "--bin_starts", "--fail_reset", "0.3", "--reset_noise", "g1"])
    sim = make_sim(model, task, 64, CUDA, args=args)
    f = sim.sample_starts(4096)
    assert bool((f < task.ref.last[f]).all()) and bool((f >= 0).all())
    ended = torch.zeros(64, dtype=torch.bool, device=CUDA)
    for _ in range(3):
        _, term, done = sim.step_nosync(task.setpoint(torch.zeros(64, 19, device=CUDA), sim.f))
        ended |= done
        assert bool((sim.ep_step < 3).all())  # an episode ends at its third step at the latest, and restarts
    assert bool(ended.all()) and float(sim.bin_G.abs().sum()) > 0
    assert torch.allclose(sim.qpos[:, 3:7].norm(dim=-1), torch.ones(64, device=CUDA), atol=1e-5)


@GPU
@pytest.mark.parametrize("extra", [[], ["--algo", "supertrack", "--st_fail_w", "2", "--st_imag_noise", "0.2", "--st_task_weights"],
                                   ["--algo", "supertrack", "--st_terms", "g1", "--st_balance_w", "2", "--max_episode_steps", "5", "--bin_starts",
                                    "--fail_reset", "0.3", "--reset_noise", "g1", "--utd", "0.5", "--wm_lr", "1e-3"]],
                         ids=["supermimic", "supertrack", "supertrack_g1_options"])  # fmt: skip
def test_training_loop_with_the_systems_options(gpu_setup, monkeypatch, tmp_path, extra):
    """train_humanoid's loop with the systems options on two short clips: the graphs are captured, the log's rows hold the
    humanoid's columns and are finite, and the evaluation reads the checkpoint."""
    import ndm_gym.evaluate_humanoid as evaluate_humanoid
    import ndm_gym.train_humanoid as train_humanoid

    monkeypatch.setattr(torch, "compile", lambda f, **_: f)
    monkeypatch.setattr(RunningNorm, "no_sync", False)
    monkeypatch.setattr(train_humanoid, "build", lambda *a, **k: (*gpu_setup, CLIPS))
    run = tmp_path / "run"
    argv = [*small(extra), "--fused_adam", "--reset_masked", "--no_sync", "--collect_graph", "--compile_eager", "--update_graphs", "--num_envs", "32",
            "--buffer_steps", "16", "--warmup_steps", "8", "--wm_only_iters", "2", "--log_every", "4", "--ckpt_every", "4", "--keep_ckpts",
            "--iterations", "12", "--log_dir", str(run)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["train_humanoid", *argv])
    train_humanoid.main()
    with open(run / "train_log.csv") as f:
        header = f.readline().strip().split(",")
    assert "root_mm" in header and "mpjpe_l_mm" in header and "trunk_mm" not in header
    rows = np.loadtxt(run / "train_log.csv", delimiter=",", skiprows=1)
    assert rows[:, 0].tolist() == [4, 8, 12] and np.isfinite(rows[1:, header.index("wm_loss")]).all() and (rows[:, header.index("sim_r")] > 0).all()
    out = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate_humanoid", "--ckpt", str(run / "last.pt"), "--out", str(out), "--num_envs", "4", "--push", "1"])
    evaluate_humanoid.main()
    import json

    m = json.load(open(out / "metrics.json"))
    for key in ("policy", "reference_targets"):
        assert set(m[key]["clips"]) == set(CLIPS) and 0 <= m[key]["success_rate"] <= 1 and m[key]["mpjpe_g_mm"] > 0
        assert 0 < m[key]["norm_return"] <= 1 and "joint_deg" in m[key]
    assert "1" in m["push"]
    monkeypatch.setattr(sys, "argv", ["evaluate_humanoid", "--ckpts", str(run), "--out", str(out), "--ckpt_worlds", "8", "--success_return", "0.1"])
    evaluate_humanoid.main()
    c = json.load(open(out / "checkpoints.json"))
    assert [r["iteration"] for r in c["checkpoints"]] == [4, 8, 12] and c["success_return"] == 0.1
    # one pass (--run): the same numbers as the separate passes, eagerly and with the step as a CUDA graph
    for extra_eval in (["--eager"], []):
        one = tmp_path / f"one{len(extra_eval)}"
        monkeypatch.setattr(sys, "argv", ["evaluate_humanoid", "--run", str(run), "--out", str(one), "--num_envs", "4", "--push", "1", *extra_eval])
        evaluate_humanoid.main()
        m1, c1 = json.load(open(one / "metrics.json")), json.load(open(one / "checkpoints.json"))
        # (MuJoCo-Warp's steps are not bitwise reproducible from run to run: the policies' rollouts within a tolerance)
        assert m1["reference_targets"]["norm_return"] == pytest.approx(m["reference_targets"]["norm_return"], rel=1e-3)
        for r1, r0 in ((m1["policy"], m["policy"]), (m1["push"]["1"], m["push"]["1"])):
            assert r1["norm_return"] == pytest.approx(r0["norm_return"], abs=0.05) and r1["success_rate"] == pytest.approx(
                r0["success_rate"], abs=0.26
            )
        assert [r["iteration"] for r in c1["checkpoints"]] == [4, 8, 12]
    win = tmp_path / "windows"
    monkeypatch.setattr(sys, "argv", ["evaluate_humanoid", "--run", str(run), "--out", str(win), "--num_envs", "2", "--window_s", "0.3"])
    evaluate_humanoid.main()
    w = json.load(open(win / "windows.json"))
    assert set(w) >= {"policy", "reference_targets", "ckpt_000004.pt"} and w["policy"]["windows"] > 2 * 2  # windows of 15 steps
    assert 0 <= w["policy"]["window_success"] <= 1 and 0 < w["policy"]["window_return"] <= 1


def test_goal_anchored_at_the_reference(setup):
    """goal_anchor="reference": the goal does not see the pelvis' horizontal drift (the G1's goal), but its height; at the
    reference both anchors give the same goal."""
    _, kin, task = setup
    ref = task.ref
    f = torch.arange(ref.frames - 1)
    S = (ref.qpos[f], ref.qvel[f])
    at_ref = HumanoidTask(kin, ref, goal_anchor="reference")
    assert torch.allclose(at_ref.goal(S, kin.forward(S[0]), f, 3), task.goal(S, kin.forward(S[0]), f, 3), atol=1e-5)
    for shift, same in (((0.5, -0.3, 0.0), True), ((0.0, 0.0, -0.1), False)):
        moved = S[0].clone()
        moved[:, :3] += torch.tensor(shift)
        g0, g1 = at_ref.goal(S, kin.forward(S[0]), f, 3), at_ref.goal((moved, S[1]), kin.forward(moved), f, 3)
        assert torch.allclose(g0, g1, atol=1e-5) == same
    with pytest.raises(ValueError):
        HumanoidTask(kin, ref, goal_anchor="world")


def _joint(model, name: str) -> tuple[int, int, int]:
    """A joint's id, its qpos address and its dof address."""
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return j, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])


def test_target_range(any_robot):
    """--target_range: the servos' ctrlrange and the task's setpoint agree, the joint's range by default (the scene as it
    was), widened by effort / stiffness ("effort") or unlimited ("free"); the joints' stops, the actuators' force limits
    and the joints' actuator torque limits stay the robot's. At a joint's stop a target beyond it gives no proportional
    torque toward the stop with "joint", the full effort with "effort"; the world model's servo clips as MuJoCo's."""
    spec = any_robot.spec
    models = {r: load_scene(spec, target_range=r, kp_scale=1.5) for r in ("joint", "effort", "free")}
    kins = {r: HumanoidKinematics(m, spec, "cpu") for r, m in models.items()}
    base, kin = models["joint"], kins["joint"]
    kp, effort = 1.5 * np.array(spec.kp), np.array(spec.effort)
    np.testing.assert_array_equal(base.actuator_ctrlrange, base.jnt_range[1:])
    np.testing.assert_array_equal(base.actuator_ctrllimited, 1)
    assert torch.equal(kin.u_lo, kin.q_lo) and torch.equal(kin.u_hi, kin.q_hi)
    np.testing.assert_allclose(models["effort"].actuator_ctrlrange, base.jnt_range[1:] + np.stack((-effort / kp, effort / kp), -1), rtol=1e-12)
    assert not models["free"].actuator_ctrllimited.any() and bool(torch.isinf(kins["free"].u_lo).all() & torch.isinf(kins["free"].u_hi).all())
    for m in models.values():
        np.testing.assert_array_equal(m.jnt_range, base.jnt_range)
        np.testing.assert_array_equal(m.jnt_limited, base.jnt_limited)
        np.testing.assert_array_equal(m.actuator_forcerange, base.actuator_forcerange)
        np.testing.assert_array_equal(m.jnt_actfrcrange[1:], base.actuator_forcerange)
        np.testing.assert_array_equal(m.jnt_actfrclimited[1:], 1)
    raw = mujoco.MjModel.from_xml_path(spec.mjcf)  # the default scene's torque limits are the MJCF's own
    np.testing.assert_array_equal(base.jnt_actfrcrange, raw.jnt_actfrcrange)
    with pytest.raises(ValueError):
        load_scene(spec, target_range="wide")

    # the task's setpoint: a large action clamped to the scene's accepted targets
    clips = short_clips(any_robot._replace(clips=any_robot.clips[:1]), 20)
    f = torch.arange(4)
    for r, k in kins.items():
        task = HumanoidTask(k, build_reference(clips, k, "cpu"))
        free = task.ref.qpos[task.ref.next_frame(f), 7:] + task.scale * 50.0
        u = task.setpoint(torch.full((4, spec.num_joints), 50.0), f)
        hi = torch.tensor(models[r].actuator_ctrlrange[:, 1], dtype=torch.float32)
        assert torch.allclose(u, free) if r == "free" else torch.allclose(u, torch.minimum(free, hi.expand_as(free)))
        assert torch.equal(
            task.setpoint(torch.zeros(4, spec.num_joints), f), HumanoidTask(kin, task.ref).setpoint(torch.zeros(4, spec.num_joints), f)
        )

    # at the upper stop of the first leg's roll joint, a target beyond it
    name = next(n for n in spec.joints if "hip_roll" in n)
    i = spec.joints.index(name)
    for r, m in models.items():
        j, qa, da = _joint(m, name)
        data = mujoco.MjData(m)
        data.qpos[2], data.qpos[3] = 1.5, 1.0
        data.qpos[qa] = m.jnt_range[j, 1]
        data.ctrl[:] = data.qpos[7:]
        data.ctrl[i] = m.jnt_range[j, 1] + 10.0
        mujoco.mj_forward(m, data)
        want = 0.0 if r == "joint" else effort[i]
        assert data.qfrc_actuator[da] == pytest.approx(want, abs=1e-6)
        wm = world_model(m, "accel", hidden=(8,))
        qpos, qvel = torch.tensor(data.qpos, dtype=torch.float32)[None], torch.tensor(data.qvel, dtype=torch.float32)[None]
        force, _ = wm.servo(qpos, qvel, torch.tensor(data.ctrl, dtype=torch.float32)[None])
        np.testing.assert_allclose(force[0].numpy(), data.actuator_force, atol=1e-3)


def test_target_range_options(registered):
    """train_humanoid --target_range (default "joint"); the evaluation's scene is the checkpoint's (one from before the
    option: the joints' ranges), and refuses another; train_humanoid.build's task, setpoint and clips follow."""
    import argparse

    from ndm_gym.evaluate_humanoid import humanoid_setup

    name, clips = registered.spec.name, registered.clips
    assert parse_args([]).target_range == "joint" and parse_args(["--target_range", "effort"]).target_range == "effort"
    with pytest.raises(SystemExit):
        parse_args(["--target_range", "wide"])
    old = argparse.Namespace(robot=name)  # a checkpoint's arguments from before the option
    for targs, given, want in ((old, None, "joint"), (argparse.Namespace(robot=name, target_range="free"), None, "free"),
                               (argparse.Namespace(robot=name, target_range="effort"), "effort", "effort"), (None, "free", "free"), (None, None, "joint")):  # fmt: skip
        model, kin, task, names, algos, _ = humanoid_setup(targs, clips, "cpu", name, target_range=given)
        assert bool(model.actuator_ctrllimited.all()) == (want != "free")
        assert (float(model.actuator_ctrlrange[0, 1]) > float(model.jnt_range[1, 1])) == (want == "effort")
        assert sorted(names) == sorted(clips) and task.ref.num_clips == len(clips) and algos is HUMANOID_ALGOS
        assert torch.equal(
            task.setpoint(torch.full((1, kin.num_joints), 1e3), torch.zeros(1, dtype=torch.long))[0] == kin.u_hi, torch.isfinite(kin.u_hi)
        )
    with pytest.raises(SystemExit):
        humanoid_setup(argparse.Namespace(robot=name, target_range="effort"), clips, "cpu", name, target_range="joint")
    with pytest.raises(SystemExit, match="--robot"):  # a checkpoint's robot is its own
        humanoid_setup(argparse.Namespace(robot=name), clips, "cpu", "another")
    assert humanoid_setup(argparse.Namespace(robot=name), clips, "cpu")[3] == humanoid_setup(None, clips, "cpu", name)[3]


@NEEDS_H1
def test_ankle_effort_reaches_the_joint():
    """--ankle_effort_scale 2: the H1's ankle torque saturates at 80 N m as applied to the joint (qfrc_actuator), not at
    the MJCF's 40 (its actuatorfrcrange, now the scaled effort too); the robot's own ankle saturates at 40."""
    from ndm_gym.robots.humanoid import with_ankle_effort

    for scale in (1.0, 2.0):
        spec = with_ankle_effort(H1, scale)
        for target_range in ("joint", "effort"):
            model = load_scene(spec, target_range=target_range)
            j, qa, da = _joint(model, "left_ankle_joint")
            i = H1.joints.index("left_ankle_joint")
            data = mujoco.MjData(model)
            data.qpos[2], data.qpos[3] = 1.5, 1.0
            data.qpos[qa] = model.jnt_range[j, 0]
            data.qvel[da] = -40.0  # the servo's damping adds 80 N m upward
            data.ctrl[:] = data.qpos[7:]
            data.ctrl[i] = model.actuator_ctrlrange[i, 1]
            mujoco.mj_forward(model, data)
            assert data.actuator_force[i] == pytest.approx(40.0 * scale) and data.qfrc_actuator[da] == pytest.approx(40.0 * scale)
            np.testing.assert_allclose(model.jnt_actfrcrange[j], [-40.0 * scale, 40.0 * scale])


def test_selected_configurations_target_free():
    """The humanoids' selected configurations send their position targets unclamped (--target_range free, the library
    G1's); the new options' defaults are today's behaviour."""
    for name in ("h1-supertrack", "h1-supermimic"):
        args = parse_args(["--config", name])
        assert (args.target_range, args.gains, args.ref_ground) == ("free", "robot", "none")


def test_tracking_gains():
    """--gains tracking (the H1's): HumanoidVerse's and ProtoMotions' stiffness and damping per joint kind, the action scale
    0.25 effort / stiffness with them; the armature, the joints' damping, the efforts and the joints unchanged; "robot" is
    the robot's own; refused on another robot (the spec's values: no download needed)."""
    from ndm_gym.robots.humanoid import with_gains

    assert with_gains(H1, "robot") is H1 and parse_args([]).gains == "robot"
    spec = with_gains(H1, "tracking")
    want = {"hip_yaw": (200, 5), "hip_roll": (200, 5), "hip_pitch": (200, 5), "knee": (300, 6), "ankle": (40, 2), "torso": (300, 6),
            "shoulder_pitch": (100, 2), "shoulder_roll": (100, 2), "shoulder_yaw": (100, 2), "elbow": (100, 2)}  # fmt: skip
    kind = [j.replace("left_", "").replace("right_", "").replace("_joint", "") for j in H1.joints]
    assert spec.kp == tuple(float(want[k][0]) for k in kind) and spec.kd == tuple(float(want[k][1]) for k in kind)
    assert (spec.effort, spec.armature, spec.damping, spec.joints) == (H1.effort, H1.armature, H1.damping, H1.joints)
    np.testing.assert_allclose(spec.action_scale, 0.25 * np.array(H1.effort) / np.array(spec.kp))
    with pytest.raises(ValueError, match="H1"):
        with_gains(ROBOTS["g1"], "tracking")
    with pytest.raises(ValueError, match="gains"):
        with_gains(H1, "stiff")
    with pytest.raises(SystemExit):
        parse_args(["--robot", "g1", "--gains", "tracking"])


@NEEDS_H1
def test_tracking_gains_in_the_scene():
    """--gains tracking in the H1's scene: the servos' stiffness and damping, the armature and the ranges unchanged; the
    evaluation takes the checkpoint's gains (a checkpoint from before the option: the robot's)."""
    import argparse

    from ndm_gym.evaluate_humanoid import humanoid_setup
    from ndm_gym.robots.humanoid import with_gains

    spec = with_gains(H1, "tracking")
    model = load_scene(spec)
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], spec.kp)
    np.testing.assert_allclose(-model.actuator_biasprm[:, 2], spec.kd)
    np.testing.assert_allclose(model.dof_armature[6:], H1.armature)
    np.testing.assert_allclose(model.jnt_range, load_scene(H1).jnt_range)
    model, kin, task, *_ = humanoid_setup(argparse.Namespace(robot="h1", gains="tracking"), CLIPS[:1], "cpu", "h1")
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], spec.kp)
    np.testing.assert_allclose(task.scale.numpy(), spec.action_scale, rtol=1e-6)
    model, *_ = humanoid_setup(argparse.Namespace(robot="h1"), CLIPS[:1], "cpu", "h1")  # a checkpoint from before the option
    np.testing.assert_allclose(model.actuator_gainprm[:, 0], H1.kp)


def test_feet_lowest_matches_mujoco(any_robot):
    """HumanoidKinematics.feet_lowest: each foot's lowest collision point (the H1's ankle mesh by its convex hull, the G1's
    sole spheres, the synthetic humanoid's mesh and capsule) is MuJoCo's distance of the foot's geoms to the floor, at
    random poses."""
    spec = any_robot.spec
    model = load_scene(spec)
    kin = HumanoidKinematics(model, spec, "cpu")
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    rng = np.random.default_rng(0)
    for _ in range(5):
        data.qpos[:3] = [0.1, 0.2, 0.8]
        q = rng.normal(size=4)
        data.qpos[3:7] = q / np.linalg.norm(q)
        data.qpos[7:] = rng.uniform(kin.q_lo.numpy(), kin.q_hi.numpy())
        mujoco.mj_forward(model, data)
        got = kin.feet_lowest(kin.forward(torch.tensor(data.qpos, dtype=torch.float32)))
        for k, name in enumerate(spec.feet):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            want = min(
                mujoco.mj_geomDistance(model, data, g, floor, 10.0, None)
                for g in range(model.ngeom)
                if model.geom_bodyid[g] == b and model.geom_contype[g]
            )
            assert float(got[k]) == pytest.approx(want, abs=1e-5)


def test_reference_ground(robot):
    """--ref_ground: "clip" moves each clip's pelvis by one height, so that the 25th percentile of its supporting feet's lowest
    points is on the floor; "stance" adds a smooth correction that puts the supporting foot on the floor in stance, at most
    5 cm; the horizontal trajectory, the rotations and the joints are unchanged, the velocities and the bodies follow the
    corrected positions; "none" is the retarget's."""
    import dataclasses

    from ndm_gym.tasks.humanoid_tracking import GROUND_PERCENTILE, STANCE_MAX, ground_reference, supporting_feet

    spec = robot.spec
    model = load_scene(spec)
    kin = HumanoidKinematics(model, spec, "cpu")
    clips = select_clips(lafan1.load_dir(spec.motions, spec.num_joints), robot.clips)
    refs = {g: build_reference(clips, kin, "cpu", ground=g) for g in ("none", "clip", "stance")}
    none = refs["none"]
    for g, ref in refs.items():
        assert ref.frames == none.frames and torch.equal(ref.clip_first, none.clip_first)
        torch.testing.assert_close(ref.qpos[:, :2], none.qpos[:, :2]) and torch.testing.assert_close(ref.qpos[:, 3:], none.qpos[:, 3:])
        torch.testing.assert_close(ref.qvel[:, 3:], none.qvel[:, 3:])
        torch.testing.assert_close(ref.body_p[..., :2], none.body_p[..., :2])
        fk = kin.forward(ref.qpos)
        torch.testing.assert_close(ref.body_p, fk[0][:, kin.tracked])
        for c in range(ref.num_clips):
            a, b = int(ref.clip_first[c]), int(ref.clip_last[c]) + 1
            dz = ref.qpos[a:b, 2] - none.qpos[a:b, 2]
            f = torch.arange(a + 1, b - 1)
            torch.testing.assert_close(ref.qvel[f, 2], (ref.qpos[f + 1, 2] - ref.qpos[f - 1, 2]) / (2 * CONTROL_DT), atol=1e-4, rtol=0)
            if g == "none":
                assert float(dz.abs().max()) == 0
                continue
            low = kin.feet_lowest(fk)[a:b].double()
            sup = supporting_feet(low, fk[0][a:b, kin.feet].double())
            if g == "clip":
                assert float(dz.max() - dz.min()) < 1e-5
                assert float(torch.quantile(low[sup], GROUND_PERCENTILE / 100)) == pytest.approx(0.0, abs=1e-5)
            else:
                clip_dz = refs["clip"].qpos[a:b, 2] - none.qpos[a:b, 2]
                assert float((dz - clip_dz).abs().max()) <= STANCE_MAX + 1e-6
                lowest = torch.where(sup, low, torch.inf).min(-1).values[sup.any(-1)]
                low_clip = kin.feet_lowest(kin.forward(refs["clip"].qpos[a:b]))[sup.any(-1)].min(-1).values  # in stance: nearer the floor
                assert float(lowest.abs().median()) < 0.5 * float(low_clip.abs().median())
    with pytest.raises(ValueError, match="ground"):
        build_reference(clips, kin, "cpu", ground="floor")
    q = refs["none"].qpos[:10].double()
    assert ground_reference(q, kin, "none") is q
    with pytest.raises(ValueError, match="no feet"):
        ground_reference(q, HumanoidKinematics(model, dataclasses.replace(spec, feet=()), "cpu"), "clip")
    rising = q.clone()
    rising[:, 2] += torch.linspace(0, 1, 10, dtype=q.dtype)  # 5 m/s up: no foot supports
    with pytest.raises(ValueError, match="no supporting foot"):
        ground_reference(rising, kin, "clip")


def test_score_reference(registered):
    """The task follows its reference (goal, targets) and scores against score_ref when given (the evaluation's
    --score_ground): the reward, the errors and the termination of a state at the corrected reference measured against the
    original; evaluate_humanoid takes the checkpoint's --ref_ground and --score_ground overrides the scored one only."""
    import argparse

    from ndm_gym.evaluate_humanoid import humanoid_setup

    spec, names = registered.spec, registered.clips
    model = load_scene(spec)
    kin = HumanoidKinematics(model, spec, "cpu")
    clips = short_clips(registered)
    ref, orig = build_reference(clips, kin, "cpu", ground="clip"), build_reference(clips, kin, "cpu")
    same, scored = HumanoidTask(kin, ref), HumanoidTask(kin, ref, score_ref=orig)
    f = torch.arange(ref.frames)
    S = (ref.qpos, ref.qvel)
    fk = kin.forward(ref.qpos)
    a = torch.zeros(ref.frames, kin.num_joints)
    torch.testing.assert_close(scored.setpoint(a, f), same.setpoint(a, f))
    torch.testing.assert_close(scored.goal(S, fk, f, 3), same.goal(S, fk, f, 3))
    assert float(same.errors(S, fk, f)["root_height"].max()) < 1e-6
    dz = (ref.qpos[:, 2] - orig.qpos[:, 2]).abs()
    torch.testing.assert_close(scored.errors(S, fk, f)["root_height"], dz)
    assert bool((scored.reward(S, fk, f, a, a) < same.reward(S, fk, f, a, a)).all())
    with pytest.raises(ValueError, match="score_ref"):
        HumanoidTask(kin, ref, score_ref=build_reference(clips[:1], kin, "cpu"))
    targs = argparse.Namespace(robot=spec.name, ref_ground="clip", clips=names)
    _, _, t, *_ = humanoid_setup(targs, names, "cpu", spec.name)
    retarget = build_reference(select_clips(lafan1.load_dir(spec.motions, spec.num_joints), names), kin, "cpu")
    assert t.score_ref is t.ref and float((t.ref.qpos[:, 2] - retarget.qpos[:, 2]).min()) > 0
    _, _, t, *_ = humanoid_setup(targs, names, "cpu", spec.name, score_ground="none")
    assert t.score_ref is not t.ref and float((t.score_ref.qpos[:, 2] - t.ref.qpos[:, 2]).max()) < 0
    assert parse_args([]).ref_ground == "none" and parse_args(["--ref_ground", "stance"]).ref_ground == "stance"
    with pytest.raises(SystemExit):
        parse_args(["--ref_ground", "floor"])
