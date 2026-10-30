"""Unitree G1 (29 DoF), following BeyondMimic's actuator model.

The URDF is the one the LAFAN1 motions were retargeted to (g1_29dof_rev_1_0). Joint drives are
implicit PD with a 10 Hz natural frequency and damping ratio 2 on the reflected motor armature, and
the action scale is 0.25 * effort_limit / stiffness, as in BeyondMimic. The Isaac Lab asset built from
this spec lives in ``ndm_gym.sim.rigid.isaaclab_mjwarp``.
"""

from __future__ import annotations

import os
import re

from ..core.kinematics import ArticulationKinematics

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
URDF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof_rev_1_0.urdf")
MJCF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof.xml")
USD_DIR = os.path.join(DATA_DIR, "g1_usd")
USD_PATH = os.path.join(USD_DIR, "g1_29dof", "g1_29dof.usda")

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def kp(arm):
    return arm * NATURAL_FREQ**2


def kd(arm):
    return 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ


# Joint order of the LAFAN1 retarget CSVs (Unitree SDK order).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip
NUM_J = len(G1_JOINT_NAMES)

# BeyondMimic's tracked bodies; the anchor/root is the pelvis.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

# Per-joint (effort limit, velocity limit, armature) by motor type.
_M5020 = (25.0, 37.0, ARMATURE_5020)
_M4010 = (5.0, 22.0, ARMATURE_4010)
JOINT_MOTORS = {
    "hip_pitch": (88.0, 32.0, ARMATURE_7520_14),
    "hip_roll": (139.0, 20.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "knee": (139.0, 20.0, ARMATURE_7520_22),
    "ankle_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "ankle_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "waist_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "shoulder_pitch": _M5020,
    "shoulder_roll": _M5020,
    "shoulder_yaw": _M5020,
    "elbow": _M5020,
    "wrist_roll": _M5020,
    "wrist_pitch": _M4010,
    "wrist_yaw": _M4010,
}


def motor_of(joint_name: str) -> tuple[float, float, float]:
    key = joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")
    return JOINT_MOTORS[key]


def action_scale(joint_name: str) -> float:
    effort, _, arm = motor_of(joint_name)
    return 0.25 * effort / kp(arm)


DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def default_joint_pos_list() -> list[float]:
    out = []
    for name in G1_JOINT_NAMES:
        val = 0.0
        for pat, v in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pat, name):
                val = v
        out.append(val)
    return out


def kinematics(device: str, tracked: list[str] = TRACKED_BODIES) -> ArticulationKinematics:
    return ArticulationKinematics(MJCF_PATH, G1_JOINT_NAMES, tracked, device)
