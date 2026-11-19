"""Unitree humanoids (the H1, and the G1 for reference) with joint position servos on a floor, for humanoid motion tracking
on MuJoCo-Warp (tasks.humanoid_tracking).

Each robot is its LAFAN1 retarget's description (tools/download_h1.sh, tools/download_data.sh) converted to MJCF with a free
pelvis (tools/urdf_to_mjcf.py), so the state is MuJoCo's: ``qpos`` (the pelvis' position and quaternion w, x, y, z, then the
joints in the retarget's order, ``HumanoidSpec.joints``) and ``qvel`` (the pelvis' world linear and body-frame angular
velocity, then the joints'). ``load_scene`` adds a floor and a position servo per joint about the policy's target, with the
gains of ``HumanoidSpec`` and the joint's effort limit as its force limit (the actuator's ``forcerange`` and the joint's
``actuatorfrcrange``, kept equal); the robot's collision geoms touch the floor and not each other. The position targets the
servo accepts (its ``ctrlrange``) are the joint's range by default (``target_range``, ``TARGET_RANGES``).

- **H1** (19 joints, 51.6 kg): the geometry, masses, inertias, joint ranges and efforts (torque limits) are the URDF's
  (Unitree's unitree_ros, ``robots/h1_description``, https://github.com/unitreerobotics/unitree_ros, BSD-3-Clause; the
  copy in the LAFAN1 Retargeting Dataset's ``robot_description/h1``). The servo gains are Unitree's own H1 training
  configuration's (unitree_rl_gym, ``legged_gym/envs/h1/h1_config.py``, https://github.com/unitreerobotics/unitree_rl_gym,
  BSD-3-Clause): stiffness 150 at the hips, 200 at the knees, 40 at the ankles, 300 at the torso, 150 at the shoulders and
  100 at the elbows (N m / rad), damping 2, 4, 2, 6, 2 and 2 (N m s / rad). The URDF has no armature or joint damping;
  each joint takes those of Unitree's own MuJoCo model of the H1 (unitree_mujoco, ``unitree_robots/h1/h1.xml``,
  https://github.com/unitreerobotics/unitree_mujoco, BSD-3-Clause; MuJoCo Menagerie's ``unitree_h1/h1.xml``,
  https://github.com/google-deepmind/mujoco_menagerie, has the same values): armature 0.1 kg m^2 and damping 1 N m s / rad.
  ``with_gains(spec, "tracking")`` (train_humanoid ``--gains tracking``, a departure, off by default) gives the stiffer, more
  damped gains of H1 motion-tracking projects instead, ``H1_TRACKING_GAINS``: stiffness 200 / 300 / 40 / 300 / 100 / 100 and
  damping 5 / 6 / 2 / 6 / 2 / 2 at the hips, knees, ankles, torso, shoulders and elbows, as HumanoidVerse's
  ``humanoidverse/config/robot/h1/h1.yaml`` (https://github.com/LeCAR-Lab/HumanoidVerse, MIT License) and ProtoMotions'
  ``protomotions/config/robot/h1.yaml`` (https://github.com/NVlabs/ProtoMotions at commit 4a905b9, NVIDIA License,
  non-commercial) set them; the values only, no code.
- **G1** (29 joints, 33.3 kg): the library's (robots.g1, BeyondMimic's actuator model): each motor's armature, stiffness and
  damping from a 10 Hz natural frequency and a damping ratio of 2 on it, and its effort limit.

The action offsets the position target about the reference pose by up to ``0.25 * effort / stiffness`` per joint, as the
G1's (BeyondMimic's) scale. ``HumanoidKinematics`` is the differentiable forward kinematics of every body, built from the
compiled model and shared by the reference, the simulator's states and the world model's predictions.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from ..core.kinematics import mat_mat, mat_vec
from ..core.rotations import axis_angle_to_mat, quat_to_mat
from . import g1

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data")
SIM_DT = 0.002  # the physics step


@dataclass(frozen=True)
class HumanoidSpec:
    name: str
    mjcf: str  # the description converted by tools/urdf_to_mjcf.py
    motions: str  # the LAFAN1 retarget's CSV clips
    joints: tuple[str, ...]  # qpos[7:] order: the retarget's
    tracked: tuple[str, ...]  # the tracked bodies, the pelvis first
    effort: tuple[float, ...]  # per joint: the motor's torque limit (N m)
    kp: tuple[float, ...]  # stiffness (N m / rad)
    kd: tuple[float, ...]  # damping (N m s / rad)
    armature: tuple[float, ...]  # reflected motor inertia (kg m^2)
    damping: tuple[float, ...]  # the joints' own damping (N m s / rad), beside the servos'
    download: str  # the script that fetches the clips and the description
    feet: tuple[str, ...] = ()  # the feet's bodies (their collision geoms: the reference's ground contact, tasks.humanoid_tracking)

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def action_scale(self) -> list[float]:
        return [0.25 * e / k for e, k in zip(self.effort, self.kp, strict=True)]


def _h1() -> HumanoidSpec:
    legs = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
    arms = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
    joints = legs[:5] + legs[5:] + ["torso_joint"] + arms
    # unitree_rl_gym's H1 configuration, per joint kind: (stiffness, damping); the URDF's efforts
    gains = {"hip_yaw": (150, 2), "hip_roll": (150, 2), "hip_pitch": (150, 2), "knee": (200, 4), "ankle": (40, 2), "torso": (300, 6),
             "shoulder_pitch": (150, 2), "shoulder_roll": (150, 2), "shoulder_yaw": (150, 2), "elbow": (100, 2)}  # fmt: skip
    effort = {"hip_yaw": 200, "hip_roll": 200, "hip_pitch": 200, "knee": 300, "ankle": 40, "torso": 200, "shoulder_pitch": 40,
              "shoulder_roll": 40, "shoulder_yaw": 18, "elbow": 18}  # fmt: skip
    kind = [j.replace("left_", "").replace("right_", "").replace("_joint", "") for j in joints]
    tracked = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_link", "right_hip_roll_link", "right_knee_link",
               "right_ankle_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "right_shoulder_roll_link",
               "right_elbow_link")  # fmt: skip  # the G1's tracked bodies that the H1 has (no wrists)
    return HumanoidSpec("h1", os.path.join(DATA, "h1_description", "h1.xml"), os.path.join(DATA, "lafan1_h1"), tuple(joints), tracked,
                        tuple(float(effort[k]) for k in kind), tuple(float(gains[k][0]) for k in kind), tuple(float(gains[k][1]) for k in kind),
                        (0.1,) * len(joints), (1.0,) * len(joints), "tools/download_h1.sh", ("left_ankle_link", "right_ankle_link"))  # fmt: skip


def _g1() -> HumanoidSpec:
    motors = [g1.motor_of(j) for j in g1.G1_JOINT_NAMES]
    return HumanoidSpec("g1", g1.MJCF_PATH, os.path.join(DATA, "lafan1_g1"), tuple(g1.G1_JOINT_NAMES), tuple(g1.TRACKED_BODIES),
                        tuple(m[0] for m in motors), tuple(g1.kp(m[2]) for m in motors), tuple(g1.kd(m[2]) for m in motors),
                        tuple(m[2] for m in motors), (0.0,) * len(motors), "tools/download_data.sh --no-usd",
                        ("left_ankle_roll_link", "right_ankle_roll_link"))  # fmt: skip


ROBOTS = {"h1": _h1(), "g1": _g1()}

# load_scene's ``target_range``: the position targets a servo accepts (its ctrlrange; the task's setpoint clamps to the same,
# HumanoidKinematics.u_lo / u_hi). "joint": the joint's range (a target at the joint's stop gives no proportional torque
# toward the stop, and near it less than the effort); "effort": the joint's range widened by effort / stiffness on each side
# (the full effort available toward either side at any angle within the range, no more); "free": unlimited, as the library
# G1's (sim.rigid.isaaclab_mjwarp.TrackingSim.step sends its targets unclamped). The joints' stops and the effort limits
# are the robot's in every case.
TARGET_RANGES = ("joint", "effort", "free")


# with_gains's servo gain sets, per joint kind: (stiffness N m / rad, damping N m s / rad). "robot": the spec's own (the
# H1's: unitree_rl_gym's locomotion gains, above). "tracking" (the H1 only): the stiffer, more damped gains several H1
# motion-tracking projects use, as two of them set them: HumanoidVerse, humanoidverse/config/robot/h1/h1.yaml
# (https://github.com/LeCAR-Lab/HumanoidVerse, commit 101492b, MIT License) and ProtoMotions, protomotions/config/robot/h1.yaml
# (https://github.com/NVlabs/ProtoMotions, commit 4a905b9, package version 2.0, NVIDIA License (non-commercial)); both give
# these values (the numbers only are taken, no code). Their shoulder entry covers the three shoulder joints.
H1_TRACKING_GAINS = {"hip_yaw": (200, 5), "hip_roll": (200, 5), "hip_pitch": (200, 5), "knee": (300, 6), "ankle": (40, 2), "torso": (300, 6),
                     "shoulder_pitch": (100, 2), "shoulder_roll": (100, 2), "shoulder_yaw": (100, 2), "elbow": (100, 2)}  # fmt: skip
GAINS = ("robot", "tracking")


def with_gains(spec: HumanoidSpec, gains: str) -> HumanoidSpec:
    """``spec`` with the servo gains ``gains`` (``GAINS``): "robot", ``spec`` itself (the robot's own: the H1's are
    unitree_rl_gym's), or "tracking", the H1's ``H1_TRACKING_GAINS`` (HumanoidVerse's and ProtoMotions'; a departure from the
    robot's values, train_humanoid --gains). Only the stiffness and the damping change: the armature, the joints' damping,
    the efforts and the ranges stay the robot's, and the action scale (``HumanoidSpec.action_scale``, 0.25 effort / stiffness)
    follows the stiffness in use."""
    if gains not in GAINS:
        raise ValueError(f"gains {gains!r}: one of {GAINS}")
    if gains == "robot":
        return spec
    if spec.name != "h1":
        raise ValueError(f"gains {gains!r}: the H1's option ({spec.name})")
    kind = [j.replace("left_", "").replace("right_", "").replace("_joint", "") for j in spec.joints]
    return dataclasses.replace(spec, kp=tuple(float(H1_TRACKING_GAINS[k][0]) for k in kind), kd=tuple(float(H1_TRACKING_GAINS[k][1]) for k in kind))


def with_ankle_effort(spec: HumanoidSpec, scale: float) -> HumanoidSpec:
    """``spec`` with its ankle joints' effort limits (the servos' force limits, and so their action scale) times ``scale``:
    a departure from the robot's values for diagnosis (train_humanoid --ankle_effort_scale; 1: ``spec`` itself).
    load_scene applies the limit both to the actuator (``forcerange``) and to the joint (``actuatorfrcrange``, which the
    MJCF sets to the URDF's effort and which MuJoCo applies to the joint's actuator torque after the actuator's own limit)."""
    if scale == 1.0:
        return spec
    if scale <= 0:
        raise ValueError(f"ankle effort scale {scale}: positive")
    effort = tuple(e * scale if "ankle" in j else e for j, e in zip(spec.joints, spec.effort, strict=True))
    return dataclasses.replace(spec, effort=effort)


# foot="menagerie": the H1's foot collision geoms of MuJoCo Menagerie's unitree_h1/h1.xml (classes foot1, foot2 and foot3;
# https://github.com/google-deepmind/mujoco_menagerie, the model's LICENSE: BSD-3-Clause, Unitree Robotics): capsules of
# radius 0.014 m along the sole and one across the toe, 6 cm between its ends, in the ankle link's frame (the URDF's, the
# sole at 0.070 m below the ankle as the mesh's)
H1_MENAGERIE_FOOT = (((-0.035, 0, -0.056), (0.02, 0, -0.045)), ((0.02, 0, -0.045), (0.115, 0, -0.056)), ((0.14, -0.03, -0.056), (0.14, 0.03, -0.056)))
FEET = ("mesh", "menagerie")


def load_scene(spec: HumanoidSpec, timestep: float = SIM_DT, kp_scale: float = 1.0, kd_scale: float = 1.0, foot: str = "mesh",
               target_range: str = "joint") -> mujoco.MjModel:
    """The robot of ``spec`` on a floor, each joint driven by a position servo (stiffness and damping ``spec``'s times the
    scales, its force limited to the joint's effort both as the actuator's ``forcerange`` and as the joint's
    ``actuatorfrcrange``, its control a position within ``target_range``: ``TARGET_RANGES``, the joint's range by default,
    widened by the joint's effort over the servo's stiffness with "effort", unlimited with "free"), the joints' armature and
    damping ``spec``'s. The robot's collision geoms collide with the floor only; the geoms' margins are zero (MuJoCo-Warp refuses a
    non-zero margin with multi-CCD). The solver's settings are those of the library's G1 on Isaac Lab
    (sim.rigid.isaaclab_mjwarp: pyramidal friction cones, implicitfast integration, which also keeps the servos' damping on
    the light distal links stable at the physics step).

    ``foot`` (the H1 only): "mesh", the URDF's ankle mesh as the foot's collision geom (MuJoCo collides its convex hull: a
    sole 3 cm wide; the default, as Unitree's unitree_mujoco model), or "menagerie", MuJoCo Menagerie's three capsules in its
    place (``H1_MENAGERIE_FOOT``: the toe 8.8 cm wide), a departure from the URDF's geometry."""
    if foot not in FEET:
        raise ValueError(f"foot {foot!r}: one of {FEET}")
    if target_range not in TARGET_RANGES:
        raise ValueError(f"target_range {target_range!r}: one of {TARGET_RANGES}")
    if foot != "mesh" and spec.name != "h1":
        raise ValueError(f"foot {foot!r}: the H1's option ({spec.name})")
    if not os.path.exists(spec.mjcf):
        raise FileNotFoundError(f"{spec.mjcf}: run {spec.download}")
    s = mujoco.MjSpec.from_file(spec.mjcf)
    if foot == "menagerie":
        for side in ("left", "right"):
            body = s.body(f"{side}_ankle_link")
            for g in body.geoms:
                if g.contype or g.conaffinity:  # the mesh's collision geom: visual only
                    g.contype, g.conaffinity = 0, 0
            for a, b in H1_MENAGERIE_FOOT:
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*a, *b], size=[0.014, 0, 0], density=0, contype=2, conaffinity=1,
                              rgba=[0.1, 0.1, 0.1, 0])  # fmt: skip
    s.option.timestep = timestep
    s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    for g in s.geoms:
        if g.contype or g.conaffinity:  # a collision geom (the floor not yet added): the floor's partner only
            g.contype, g.conaffinity = 2, 1
    s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=1, conaffinity=2, rgba=[0.5, 0.5, 0.5, 1])
    for j, name in enumerate(spec.joints):
        joint = s.joint(name)
        joint.armature = spec.armature[j]
        # the joint's own limit on its actuators' torque, the effort as the actuator's (the MJCF's actuatorfrcrange is the URDF's
        # effort: unchanged by default; with_ankle_effort's scaled effort would otherwise stay capped at the URDF's)
        joint.actfrcrange, joint.actfrclimited = [-spec.effort[j], spec.effort[j]], mujoco.mjtLimited.mjLIMITED_TRUE
        kp, kd = spec.kp[j] * kp_scale, spec.kd[j] * kd_scale
        ctrl = np.array(joint.range, dtype=np.float64)
        if target_range == "effort":
            ctrl = ctrl + np.array([-1.0, 1.0]) * spec.effort[j] / kp
        limited = mujoco.mjtLimited.mjLIMITED_FALSE if target_range == "free" else mujoco.mjtLimited.mjLIMITED_TRUE
        s.add_actuator(name=name.replace("_joint", ""), target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT, gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                       biastype=mujoco.mjtBias.mjBIAS_AFFINE, gainprm=[kp] + [0.0] * 9, biasprm=[0.0, -kp, -kd] + [0.0] * 7,
                       ctrlrange=ctrl, ctrllimited=limited,
                       forcerange=[-spec.effort[j], spec.effort[j]], forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE)  # fmt: skip
    model = s.compile()
    model.geom_margin[:] = 0.0
    model.dof_damping[6:] = spec.damping  # the joints' dofs follow the free root's six, in the spec's order (checked below)
    nq = 7 + spec.num_joints
    names = [model.joint(i).name for i in range(1, model.njnt)]
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or names != list(spec.joints) or model.nq != nq or model.nv != nq - 1:
        raise ValueError(f"{spec.mjcf}: expected a free pelvis and the joints {spec.joints} in that order, got {names}")
    return model


def _quat_mat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=np.float64))
    return R.reshape(3, 3)


class HumanoidKinematics:
    """Every body's world position and rotation from ``qpos``, and their velocities from ``qvel``, for a compiled model
    ``model`` of ``spec``'s robot (a free root and hinges; bodies without a joint ride on their parent)."""

    def __init__(self, model: mujoco.MjModel, spec: HumanoidSpec, device: str):
        self.spec, self.device = spec, device
        self.num_joints = spec.num_joints  # the size hand_fast.Collector reads
        joint_col = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n): 7 + i for i, n in enumerate(spec.joints)}
        self.parent, self.column, axes, anchors = [], [], [], []
        self.names = []
        for b in range(1, model.nbody):
            self.names.append(model.body(b).name)
            p = int(model.body_parentid[b])
            self.parent.append(p - 1)  # -1: the world
            if model.body_jntnum[b] == 0:
                self.column.append(-1)
                axes.append(np.zeros(3)), anchors.append(np.zeros(3))
                continue
            j = int(model.body_jntadr[b])
            if model.body_jntnum[b] != 1 or (model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE) != (b == 1):
                raise ValueError(f"body {model.body(b).name}: a free root and one hinge per other moving body expected")
            if b == 1:
                self.column.append(-2)  # the root
                axes.append(np.zeros(3)), anchors.append(np.zeros(3))
                continue
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or j not in joint_col:
                raise ValueError(f"body {model.body(b).name}: a hinge of the spec's joints expected")
            self.column.append(joint_col[j])
            axes.append(model.jnt_axis[j]), anchors.append(model.jnt_pos[j])
        if self.parent[0] != -1 or any(p < 0 for p in self.parent[1:]):
            raise ValueError("one tree under the root expected")
        f = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=device)  # noqa: E731
        self.body_pos, self.body_rot = f(model.body_pos[1:]), quat_to_mat(f(model.body_quat[1:]))
        self.axis, self.anchor = f(axes), f(anchors)
        self.has_anchor = [bool(np.any(a != 0)) for a in anchors]
        # the tracked bodies' indices, on the device (an index list would be copied from the host at every use, which a CUDA graph refuses)
        self.tracked = torch.tensor([self.names.index(n) for n in spec.tracked], dtype=torch.long, device=device)
        self.num_bodies = len(self.names)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in spec.joints]
        lo, hi = model.jnt_range[jids].T
        self.q_lo, self.q_hi = f(lo), f(hi)
        # the position targets the joints' servos accept (their ctrlrange; +-inf where unlimited: load_scene's target_range),
        # which the task's setpoint clamps to
        act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu) if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        if any(j not in act for j in jids):
            raise ValueError("a position servo per joint expected")
        limited = np.array([bool(model.actuator_ctrllimited[act[j]]) for j in jids])
        rng = model.actuator_ctrlrange[[act[j] for j in jids]]
        self.u_lo, self.u_hi = f(np.where(limited, rng[:, 0], -np.inf)), f(np.where(limited, rng[:, 1], np.inf))
        # the feet's collision geometry (spec.feet): points in their body's frame with a radius each, the lowest point of the
        # foot the lowest of (point height - radius): a mesh's convex hull vertices (radius 0), a sphere's centre, a capsule's
        # two ends (its radius); feet_lowest
        foot_of, pts, rad = [], [], []
        for k, name in enumerate(spec.feet):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != b or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                    continue
                Rg, pg, size = _quat_mat(model.geom_quat[g]), model.geom_pos[g], model.geom_size[g]
                t = model.geom_type[g]
                if t == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = int(model.geom_dataid[g])
                    v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
                    a = int(model.mesh_graphadr[mid])
                    if a >= 0:  # the convex hull's vertices (MuJoCo's, the geometry it collides)
                        v = v[model.mesh_graph[a + 2 + model.mesh_graph[a] : a + 2 + 2 * model.mesh_graph[a]]]
                    p, r = pg + v @ Rg.T, np.zeros(len(v))
                elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                    p, r = pg[None], np.array([size[0]])
                elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    p, r = pg + np.array([[0, 0, size[1]], [0, 0, -size[1]]]) @ Rg.T, np.array([size[0]] * 2)
                else:
                    raise ValueError(f"{name}: a foot geom of type {t} (mesh, sphere or capsule expected)")
                pts.append(p), rad.append(r), foot_of.append(np.full(len(p), k))
        if spec.feet:
            self.feet = torch.tensor([self.names.index(n) for n in spec.feet], dtype=torch.long, device=device)
            self.foot_pts, self.foot_rad = f(np.concatenate(pts)), f(np.concatenate(rad))
            self.foot_of = torch.tensor(np.concatenate(foot_of), dtype=torch.long, device=device)

    def feet_lowest(self, fk, chunk: int = 4096) -> torch.Tensor:
        """Per foot (spec.feet) the height of its collision geometry's lowest point (..., feet) at the poses ``fk``
        (``forward``; leading dimensions flattened in chunks of ``chunk`` poses)."""
        P, R = fk[0][..., self.feet, :], fk[1][..., self.feet, :, :]
        lead = P.shape[:-2]
        P, R = P.reshape(-1, *P.shape[-2:]), R.reshape(-1, *R.shape[-3:])
        out = []
        for i in range(0, P.shape[0], chunk):
            p, r = P[i : i + chunk, self.foot_of], R[i : i + chunk, self.foot_of]  # (n, points, 3), (n, points, 3, 3)
            z = p[..., 2] + (r[..., 2, :] * self.foot_pts).sum(-1) - self.foot_rad
            out.append(torch.full((z.shape[0], len(self.spec.feet)), float("inf"), device=z.device).scatter_reduce(1, self.foot_of.expand_as(z), z, "amin"))
        return torch.cat(out).reshape(*lead, len(self.spec.feet))

    def forward(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``qpos`` (..., nq) -> every body's position (..., B, 3) and rotation (..., B, 3, 3), the root's first."""
        p0, R0 = qpos[..., :3], quat_to_mat(qpos[..., 3:7] / qpos[..., 3:7].norm(dim=-1, keepdim=True))
        pos, rot = [p0], [R0]
        for i in range(1, self.num_bodies):
            pp, Rp = pos[self.parent[i]], rot[self.parent[i]]
            p, R = pp + mat_vec(Rp, self.body_pos[i]), mat_mat(Rp, self.body_rot[i])
            if self.column[i] >= 0:
                Rj = axis_angle_to_mat(self.axis[i], qpos[..., self.column[i]])
                if self.has_anchor[i]:
                    p = p + mat_vec(R, self.anchor[i] - mat_vec(Rj, self.anchor[i]))
                R = mat_mat(R, Rj)
            pos.append(p), rot.append(R)
        return torch.stack(pos, -2), torch.stack(rot, -3)

    def velocities(self, fk, qvel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Every body's world linear velocity (of its frame's origin) and angular velocity (..., B, 3), from its pose ``fk``
        (``forward``) and ``qvel`` (the root's world linear and body-frame angular velocity, then the joints')."""
        P, R = fk
        v, w = [qvel[..., :3]], [mat_vec(R[..., 0, :, :], qvel[..., 3:6])]
        for i in range(1, self.num_bodies):
            par = self.parent[i]
            wi, vi = w[par], v[par] + torch.cross(w[par], P[..., i, :] - P[..., par, :], dim=-1)
            if self.column[i] >= 0:  # the hinge's axis in the world (it turns about itself) times its rate
                axis_w = mat_vec(R[..., i, :, :], self.axis[i])
                qd = qvel[..., self.column[i] - 1 : self.column[i]]
                wi = wi + axis_w * qd
                if self.has_anchor[i]:  # a hinge off the body's origin moves the origin
                    a = P[..., i, :] + mat_vec(R[..., i, :, :], self.anchor[i])
                    vi = vi + torch.cross(axis_w * qd, P[..., i, :] - a, dim=-1)
            v.append(vi), w.append(wi)
        return torch.stack(v, -2), torch.stack(w, -2)
