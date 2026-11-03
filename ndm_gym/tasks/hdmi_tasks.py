"""HDMI's 13 human-object interaction tasks (LeCAR-Lab/HDMI @ 32282f6, cfg/task/G1/hdmi/*.yaml) as plain data.

The base tables are HDMI's cfg/task/base/hdmi-base.yaml (Y below). Each task adds its objects and contact geometry and
overrides rewards, terminations, reset ranges and randomization the way Hydra merges its YAML onto the base: a changed
entry keeps its place, a new one is appended. § numbers refer to docs/hdmi_task_spec.md. :mod:`ndm_gym.tasks.hdmi`
evaluates the rewards, terminations and observations; the reset ranges and the randomization are for the simulator.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

# HDMI's body and joint selections: regular expressions, matched in full against keypoint or joint names
UPPER_BODY = (".*_shoulder_pitch_link", ".*_elbow_link", ".*_wrist_yaw_link")  # Y:173
LOWER_BODY = (".*_hip_pitch_link", ".*_knee_link", ".*_ankle_roll_link")  # Y:182
VEL_BODIES = (
    "pelvis", ".*_hip_yaw_link", ".*_knee_link", ".*_ankle_roll_link", "torso_link", ".*_shoulder_pitch_link", ".*_elbow_link", ".*_wrist_yaw_link",
)  # Y:200, no hip pitch  # fmt: skip
TRACKED_JOINTS = ("waist_.*_joint", ".*_hip_.*_joint", ".*_knee_joint", ".*_shoulder_.*_joint", ".*_elbow_joint")  # Y:209, no ankles
FEET = ".*ankle_roll_link"


@dataclass(frozen=True)
class Term:
    """A reward term as HDMI's YAML declares it: ``kind`` is HDMI's reward class, the term adds ``weight`` x its value
    to its group; a disabled term is still evaluated (and logged) but not added (§5.1)."""

    kind: str
    weight: float
    enabled: bool = True
    params: dict = field(default_factory=dict)


def term(kind: str, weight: float, enabled: bool = True, **params) -> Term:
    return Term(kind, weight, enabled, params)


@dataclass(frozen=True)
class Termination:
    """A consecutive-step counter (TRM:12-32): fires once the term's error has been >= ``threshold`` for ``min_steps``
    steps in a row; ``params`` are the term's other arguments. Keyed by HDMI's termination class."""

    threshold: float
    min_steps: int
    params: dict = field(default_factory=dict)


# Y:170-231. HDMI's fifth group, debug, holds only disabled terms and is identically zero (§5.6), so it is left out.
BASE_REWARDS: dict[str, dict[str, Term]] = {
    "tracking": {
        "tracking_upper_body_pos": term("keypoint_pos_tracking_local_product", 0.5, body_names=UPPER_BODY, sigma=0.5),
        "tracking_upper_body_ori": term("keypoint_ori_tracking_local_product", 0.5, body_names=UPPER_BODY, sigma=1.0),
        "tracking_lower_body_pos": term("keypoint_pos_tracking_local_product", 0.5, body_names=LOWER_BODY, sigma=0.5),
        "tracking_lower_body_ori": term("keypoint_ori_tracking_local_product", 0.5, body_names=LOWER_BODY, sigma=1.0),
        "tracking_root_pos": term("keypoint_pos_tracking_product", 0.5, body_names=("pelvis",), sigma=0.5),
        "tracking_root_ori": term("keypoint_ori_tracking_product", 0.5, body_names=("pelvis",), sigma=0.5),
        "tracking_body_linvel": term("keypoint_lin_vel_tracking_product", 0.5, body_names=VEL_BODIES, sigma=0.5),
        "tracking_body_angvel": term("keypoint_ang_vel_tracking_product", 0.5, body_names=VEL_BODIES, sigma=2.5),
        "joint_pos_tracking_product": term("joint_pos_tracking_product", 0.5, joint_names=TRACKED_JOINTS, sigma=0.25),
        "joint_vel_tracking_product": term("joint_vel_tracking_product", 0.5, joint_names=TRACKED_JOINTS, sigma=2.5),
    },
    "object_tracking": {
        "object_pos_tracking": term("object_pos_tracking", 1.0, sigma=0.5),
        "object_ori_tracking": term("object_ori_tracking", 1.0, sigma=0.5),
        # frc_thres is a float in Y:220, so HDMI takes the force-magnitude branch (REW:400-401)
        "eef_contact_exp": term("eef_contact_exp", 1.0, pos_sigma=0.3, pos_tolerance=0.0, frc_sigma=40.0, frc_thres=10.0, per_axis=False, gain=5.0),
    },
    "loco": {
        "action_rate_l2": term("action_rate_l2", 0.1),
        "joint_vel_l2": term("joint_vel_l2", 5.0e-4, joint_names=(".*",)),
        "joint_pos_limits": term("joint_pos_limits", 10.0, joint_names=(".*",), soft_factor=0.9),
        "joint_torque_limits": term("joint_torque_limits", 0.01, joint_names=(".*",), soft_factor=0.6),
        "survival": term("survival", 1.0),
    },
    "feet": {
        "impact_force_l2": term("impact_force_l2", 1.0, body_names=(FEET,)),
        "feet_slip": term("feet_slip", 0.5, tolerance=0.0, body_names=(FEET,)),
        "feet_air_time": term("feet_air_time", 5.0, body_names=(FEET,), thres=0.5, soft_discount=1.0),
        "survival": term("survival", 1.0),
    },
}

# Y:259-269. cum_lost_contact_steps keeps its class default threshold 1.0 on its error 2 x lost (TRM:170, 190).
BASE_TERMINATIONS: dict[str, Termination] = {
    "cum_body_pos_error": Termination(0.5, 25, {"body_names": ("torso_link",)}),
    "cum_body_ori_error": Termination(1.2, 25, {"body_names": ("torso_link",)}),
    "cum_body_pos_error_local": Termination(0.5, 25, {"body_names": (".*",)}),
    "cum_body_ori_error_local": Termination(1.2, 25, {"body_names": (".*",)}),
    "cum_object_pos_error": Termination(0.5, 25),
    "cum_object_ori_error": Termination(0.8, 25),
    "cum_lost_contact_steps": Termination(1.0, 25, {"pos_thres": 0.2, "frc_thres": 1.0}),
}


def reward_table(disable: tuple[str, ...] = (), add: dict[str, dict[str, Term]] | None = None) -> dict[str, dict[str, Term]]:
    """The base reward table with terms disabled in place (HDMI's ``enabled: false``) and new terms appended to a group."""
    out = {g: dict(ts) for g, ts in BASE_REWARDS.items()}
    for name in disable:
        groups = [g for g, ts in out.items() if name in ts]
        if len(groups) != 1:
            raise KeyError(f"reward term {name!r} is in {len(groups)} groups, expected 1")
        out[groups[0]][name] = dataclasses.replace(out[groups[0]][name], enabled=False)
    for g, ts in (add or {}).items():
        for name, t in ts.items():
            if name in out[g]:
                raise KeyError(f"reward term {g}/{name} exists already")
            out[g][name] = t
    return out


def termination_table(add: dict[str, Termination] | None = None, **change: dict) -> dict[str, Termination]:
    """The base termination table with a term's threshold, min_steps or params changed in place and new terms appended."""
    out = dict(BASE_TERMINATIONS)
    for name, ch in change.items():
        t, ch = out[name], dict(ch)
        out[name] = Termination(ch.pop("threshold", t.threshold), ch.pop("min_steps", t.min_steps), {**t.params, **ch})
    for name, t in (add or {}).items():
        if name in out:
            raise KeyError(f"termination {name!r} exists already")
        out[name] = t
    return out


POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
BASE_POSE_RANGE = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
BASE_VELOCITY_RANGE = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.2, 0.2), "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5)}
BASE_OBJECT_POSE_RANGE = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)}


@dataclass(frozen=True)
class ResetCfg:
    """Reset noise (§2): uniform ranges keyed by POSE_KEYS (Y:74-88, 100-106; a missing key is (0, 0), CMD:95-98) and
    half-widths of the joint noise; HDMI zeroes the robot's pose and velocity noise in evaluation, not the object's."""

    pose_range: dict = field(default_factory=lambda: dict(BASE_POSE_RANGE))
    velocity_range: dict = field(default_factory=lambda: dict(BASE_VELOCITY_RANGE))
    joint_pos_noise: float = 0.1  # Y:90
    joint_vel_noise: float = 0.1  # Y:91
    object_pose_range: dict = field(default_factory=lambda: dict(BASE_OBJECT_POSE_RANGE))
    object_joint_pos_noise: float = 0.1  # CMD:356
    object_joint_vel_noise: float = 0.1  # CMD:357


@dataclass(frozen=True)
class ObjectRandomization:
    """The tracked object's randomization (§7): mass, dynamic friction, static/dynamic friction ratio and restitution at
    startup (HRN:12-83); a per-env per-axis scale of the whole asset at spawn, one scalar with ``homogeneous_scale``, the
    contact targets scaled alike (SPN:292-311, CMD:427-439); a hinge's armature at startup and its Coulomb friction and
    viscous damping at every reset (HRN:85-112)."""

    mass_range: tuple[float, float]
    dynamic_friction_range: tuple[float, float] = (0.3, 0.8)
    static_dynamic_friction_ratio_range: tuple[float, float] = (1.0, 2.0)
    restitution_range: tuple[float, float] = (0.0, 0.2)
    scale_range: tuple[float, float] = (0.9, 1.1)
    homogeneous_scale: bool = False
    hinge_armature_range: tuple[float, float] | None = None
    hinge_friction_range: tuple[float, float] | None = None
    hinge_damping_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class HDMITaskCfg:
    name: str  # HDMI's cfg/task/G1/hdmi/<name>.yaml
    hdmi_name: str  # HDMI's task name
    data_path: str  # HDMI's motion directory, relative to its repository
    clip_len: int  # frames at 50 fps (§8.1)
    robot_type: str  # HDMI's G1 variant (§3.4, §8.1)
    object_asset: str  # the tracked object; its root link is the tracked pose, its reference the clip body of that name (§9)
    object_body: str  # the object's body that carries the contact targets and filters the end effectors' contact sensors
    contact_target_offsets: tuple[tuple[float, float, float], ...]  # per end effector, object-body frame, before scaling (§8.2)
    randomization: ObjectRandomization
    contact_eef_bodies: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")  # Y:109
    contact_eef_offsets: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # eef-link frame (Y:110-112)
    contact_cols: int = 1  # columns of the clip's object_contact flags: 1 shared by all end effectors, or one each (§1.1)
    object_joint: str | None = None  # hinge of an articulated object (door, folding chair)
    extra_objects: tuple[str, ...] = ()  # static props: supports, platforms, a wall (§9)
    max_episode_length: int = 1000  # Y:14
    wrist_yaw_default: tuple[float, float] | None = None  # (left, right) default wrist-yaw angle instead of 0 (§3.3)
    wrist_kp: float | None = None  # stiffness of the six wrist joints instead of their motors' (§3.3)
    rewards: dict = field(default_factory=reward_table)
    terminations: dict = field(default_factory=termination_table)
    reset: ResetCfg = field(default_factory=ResetCfg)

    def __post_init__(self):
        E, nt, ne = len(self.contact_eef_bodies), len(self.contact_target_offsets), len(self.contact_eef_offsets)
        if not nt == ne == E:
            raise ValueError(f"{self.name}: {E} end effectors, {nt} target offsets and {ne} end-effector offsets")
        if self.contact_cols not in (1, E):
            raise ValueError(f"{self.name}: {self.contact_cols} contact columns for {E} end effectors")
        r = self.randomization
        if (self.object_joint is None) != (r.hinge_armature_range is None and r.hinge_friction_range is None and r.hinge_damping_range is None):
            raise ValueError(f"{self.name}: hinge randomization must be given exactly for a hinged object")

    @property
    def num_eefs(self) -> int:
        return len(self.contact_eef_bodies)


SPHERE_FEET = "g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule"
BOX_FEET = "g1_29dof_rubberhand-feet_box-eef_box-body_capsule"
NO_HAND = "g1_29dof_nohand-feet_sphere-eef_L-body_capsule"
WRIST_YAW = (-0.4, 0.4)
EEF_OFFSET = ((0.05, 0.0, 0.0), (0.05, 0.0, 0.0))
WIDE_POSE = {**BASE_POSE_RANGE, "x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.1, 0.1)}
FLAT_OBJECT = {**BASE_OBJECT_POSE_RANGE, "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0)}
OBJECT_STEPS_OFF = {"cum_object_pos_error": {"min_steps": 5000}, "cum_object_ori_error": {"min_steps": 5000}}  # never reached


def _hinged_rewards():
    """Doors and the folding chair: their root link is fixed, so the object's pose is not tracked, its hinge angle is."""
    return reward_table(disable=("object_pos_tracking", "object_ori_tracking"),
                        add={"object_tracking": {"object_joint_pos_tracking": term("object_joint_pos_tracking", 1.0, sigma=0.5)}})  # fmt: skip


_TASKS = (
    HDMITaskCfg(
        name="carry_and_place_bread_box", hdmi_name="G1CarryAndPlaceBreadBox", data_path="data/motion/data_for_sim/carry_and_place_bread_box-0829",
        clip_len=840, robot_type=SPHERE_FEET, object_asset="bread_box", object_body="bread_box", extra_objects=("support0", "support1"),
        contact_target_offsets=((0.0, 0.17, 0.14), (0.0, -0.17, 0.14)), contact_eef_offsets=EEF_OFFSET,
        wrist_yaw_default=WRIST_YAW, wrist_kp=4.0,
        reset=ResetCfg(object_pose_range={**BASE_OBJECT_POSE_RANGE, "x": (0.0, 0.0), "y": (0.0, 0.0)}),
        randomization=ObjectRandomization(mass_range=(1.2, 1.8)),
    ),
    HDMITaskCfg(
        name="carry_box_over_shoulder", hdmi_name="G1LiftBoxOverShoulder", data_path="data/motion/data_for_sim/lift_box_over_shoulder",
        clip_len=590, robot_type=SPHERE_FEET, object_asset="suitcase", object_body="suitcase",
        contact_target_offsets=((-0.1, 0.18, 0.15), (-0.1, -0.18, 0.15)), contact_eef_offsets=EEF_OFFSET,
        wrist_yaw_default=WRIST_YAW, wrist_kp=4.0,
        randomization=ObjectRandomization(mass_range=(1.2, 1.8)),
    ),
    HDMITaskCfg(
        name="move_foam", hdmi_name="G1TrackFoam", data_path="data/motion/data_for_sim/move_foam-climb-0825-2-0_580",
        clip_len=680, robot_type=BOX_FEET, object_asset="foam", object_body="foam", extra_objects=("stool_support",),
        contact_target_offsets=((0.0, 0.21, -0.1), (0.0, -0.21, -0.1)),
        wrist_yaw_default=WRIST_YAW, wrist_kp=4.0,
        randomization=ObjectRandomization(mass_range=(0.9, 1.2)),
    ),
    HDMITaskCfg(
        name="move_stool-climb", hdmi_name="G1TrackStool", data_path="data/motion/data_for_sim/move_stool-climb-0814",
        clip_len=833, robot_type=BOX_FEET, object_asset="stool", object_body="stool",
        contact_target_offsets=((0.0, 0.27, -0.05), (0.0, -0.27, -0.05)), contact_eef_offsets=EEF_OFFSET,
        wrist_yaw_default=WRIST_YAW,
        terminations=termination_table(add={"cum_body_z_error": Termination(0.3, 25, {"body_names": ("torso_link",)})}),
        randomization=ObjectRandomization(mass_range=(1.5, 3.0)),
    ),
    HDMITaskCfg(
        name="move_suitcase", hdmi_name="G1TrackSuitcase", data_path="data/motion/g1/omomo/sub1_suitcase_011",
        clip_len=472, robot_type=SPHERE_FEET, object_asset="suitcase", object_body="suitcase",
        contact_target_offsets=((-0.1, 0.18, 0.25), (-0.1, -0.18, 0.25)), contact_eef_offsets=EEF_OFFSET,
        max_episode_length=500, wrist_yaw_default=WRIST_YAW,
        randomization=ObjectRandomization(mass_range=(1.2, 1.8)),
    ),
    HDMITaskCfg(
        name="open_door-feet", hdmi_name="G1OpenDoorFeet", data_path="data/motion/data_for_sim/open_door-feet-0828",
        clip_len=928, robot_type=BOX_FEET, object_asset="door", object_body="door_panel", object_joint="door_joint",
        contact_eef_bodies=("left_ankle_roll_link",), contact_target_offsets=((0.0, -0.75, 0.1),), contact_eef_offsets=((0.05, 0.0, 0.0),),
        rewards=_hinged_rewards(), terminations=termination_table(**OBJECT_STEPS_OFF),
        reset=ResetCfg(pose_range=WIDE_POSE, object_pose_range=FLAT_OBJECT),
        randomization=ObjectRandomization(mass_range=(5.0, 20.0), hinge_armature_range=(0.0, 0.01), hinge_friction_range=(5.0, 10.0),
                                          hinge_damping_range=(1.0, 5.0)),
    ),
    HDMITaskCfg(
        name="open_foldchair-sit", hdmi_name="G1OpenFoldchairSit", data_path="data/motion/g1/multi_motion_0726/open_chair_and_sit_amass",
        clip_len=585, robot_type=BOX_FEET, object_asset="foldchair", object_body="seat_link", object_joint="foldchair_joint",
        contact_target_offsets=((0.28, -0.1, 0.02), (0.28, 0.1, 0.02)),
        rewards=_hinged_rewards(),
        terminations=termination_table(**OBJECT_STEPS_OFF, cum_lost_contact_steps={"min_steps": 200, "pos_thres": 0.2, "frc_thres": 1.0}),
        reset=ResetCfg(object_pose_range={**BASE_OBJECT_POSE_RANGE, "roll": (0.0, 0.0), "pitch": (0.0, 0.0)},
                       object_joint_pos_noise=0.0, object_joint_vel_noise=0.0),
        randomization=ObjectRandomization(mass_range=(2.0, 5.0), hinge_armature_range=(0.0, 0.01), hinge_friction_range=(5.0, 10.0),
                                          hinge_damping_range=(1.0, 5.0)),
    ),
    HDMITaskCfg(
        name="push_box", hdmi_name="G1PushBox", data_path="data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height",
        clip_len=792, robot_type=NO_HAND, object_asset="box", object_body="box",
        contact_target_offsets=((0.0, -0.2, 0.8), (0.0, 0.2, 0.8)), contact_eef_offsets=((0.1, 0.0, 0.0), (0.1, 0.0, 0.0)),
        randomization=ObjectRandomization(mass_range=(6.0, 12.0), dynamic_friction_range=(0.4, 0.6), static_dynamic_friction_ratio_range=(1.2, 2.0)),
    ),
    HDMITaskCfg(
        name="push_door-hand", hdmi_name="G1PushDoorHand", data_path="data/motion/data_for_sim/push_door-hand-0828",
        clip_len=573, robot_type=SPHERE_FEET, object_asset="door", object_body="door_panel", object_joint="door_joint",
        contact_eef_bodies=("right_wrist_yaw_link",), contact_target_offsets=((0.0, -0.6, 1.0),), contact_eef_offsets=((0.05, 0.0, 0.0),),
        rewards=_hinged_rewards(), terminations=termination_table(**OBJECT_STEPS_OFF),
        reset=ResetCfg(pose_range=WIDE_POSE, object_pose_range=FLAT_OBJECT),
        randomization=ObjectRandomization(mass_range=(5.0, 20.0), hinge_armature_range=(0.0, 0.01), hinge_friction_range=(0.1, 0.5),
                                          hinge_damping_range=(0.1, 1.0)),
    ),
    HDMITaskCfg(
        name="roll_ball-hand", hdmi_name="G1RollBall", data_path="data/motion/g1/multi_motion_0726/roll_ball_amass",
        clip_len=602, robot_type=SPHERE_FEET, object_asset="ball", object_body="ball",
        contact_target_offsets=((0.1, 0.0, 0.0), (0.1, 0.0, 0.0)),
        # the YAML writes frc_thres: 10, an int, so HDMI takes the per-axis force branch (REW:442-445, §5.3)
        rewards=reward_table(disable=("eef_contact_exp",), add={"object_tracking": {"eef_contact_exp_max": term(
            "eef_contact_exp_max", 1.0, pos_sigma=0.3, pos_tolerance=0.4, frc_sigma=40.0, frc_thres=10.0, per_axis=True)}}),
        terminations=termination_table(cum_lost_contact_steps={"min_steps": 500, "pos_thres": 0.4, "frc_thres": 2.0},
                                       cum_object_ori_error={"threshold": 5.0}),
        randomization=ObjectRandomization(mass_range=(2.0, 4.0), static_dynamic_friction_ratio_range=(1.2, 2.0), homogeneous_scale=True),
    ),
    HDMITaskCfg(
        name="topple_wood_board_and_cross", hdmi_name="G1ToppleWoodBoardAndCross", data_path="data/motion/data_for_sim/topple_wood_board_and_cross",
        clip_len=582, robot_type=SPHERE_FEET, object_asset="wood_board", object_body="wood_board", extra_objects=("platform0", "platform1", "wall0"),
        contact_target_offsets=((1.0, -0.17, 0.04), (1.0, -0.17, -0.04)), contact_eef_offsets=EEF_OFFSET,
        wrist_yaw_default=WRIST_YAW, wrist_kp=4.0,
        reset=ResetCfg(object_pose_range={**BASE_OBJECT_POSE_RANGE, "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}),
        randomization=ObjectRandomization(mass_range=(1.2, 1.8)),
    ),
    HDMITaskCfg(
        name="traverse_door", hdmi_name="G1TraverseDoor", data_path="data/motion/data_for_sim/traverse_door",
        clip_len=1463, robot_type=BOX_FEET, object_asset="door", object_body="door_panel", object_joint="door_joint",
        contact_eef_bodies=("right_wrist_yaw_link", "left_ankle_roll_link"), contact_target_offsets=((0.0, -0.6, 1.0), (0.0, -0.75, 0.1)),
        contact_eef_offsets=EEF_OFFSET, contact_cols=2,
        rewards=_hinged_rewards(), terminations=termination_table(**OBJECT_STEPS_OFF),
        reset=ResetCfg(pose_range=WIDE_POSE, object_pose_range=FLAT_OBJECT),
        randomization=ObjectRandomization(mass_range=(5.0, 20.0), hinge_armature_range=(0.0, 0.01), hinge_friction_range=(0.1, 0.5),
                                          hinge_damping_range=(0.1, 1.0)),
    ),
    HDMITaskCfg(
        # the YAML also enables self-collisions on the spawn config, where Isaac Lab most likely ignores it (§8.3)
        name="truman", hdmi_name="G1Truman", data_path="data/motion/data_for_sim/combo-2",
        clip_len=3352, robot_type=BOX_FEET, object_asset="stair", object_body="stair",
        contact_target_offsets=((0.0, -0.2, 0.65), (0.3, -0.2, 0.65)), contact_eef_offsets=EEF_OFFSET, contact_cols=2,
        max_episode_length=4000,
        rewards=reward_table(add={"loco": {"feet_stumble": term("feet_stumble", 1.0, body_names=(FEET,))}}),
        terminations=termination_table(add={"cum_body_z_error": Termination(0.2, 25, {"body_names": ("torso_link",)})},
                                       cum_lost_contact_steps={"min_steps": 1000}),
        reset=ResetCfg(pose_range={**BASE_POSE_RANGE, "x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}, object_pose_range=FLAT_OBJECT),
        randomization=ObjectRandomization(mass_range=(20.0, 40.0), dynamic_friction_range=(0.8, 1.2), scale_range=(0.98, 1.02)),
    ),
)
TASKS: dict[str, HDMITaskCfg] = {c.name: c for c in _TASKS}


def get(name: str) -> HDMITaskCfg:
    if name not in TASKS:
        raise KeyError(f"unknown HDMI task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]
