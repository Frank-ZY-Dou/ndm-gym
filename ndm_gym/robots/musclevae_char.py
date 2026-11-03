"""MuscleVAE's muscle-actuated character (Feng, Xu and Liu, SIGGRAPH Asia 2023; wangchek/MuscleVAE_ @ 54c001c): 23 rigid
bodies, 22 joints (14 ball, 8 hinge), 61.4 kg, driven by 284 muscles through 1,212 anchors, simulated at 120 Hz and
controlled at 20 Hz.

MuscleVAE's repository has no license, so its files are never redistributed: tools/download_musclevae.sh fetches it at
the pinned commit into data/musclevae/src, and tools/convert_musclevae.py writes the assets this module loads into
data/musclevae/: character.xml (MJCF), character.json (bodies, joints and the conventions), muscles.json (the muscle
table) and clips/ (their walk, run and jump references at 20 Hz). The converter's docstring records every convention:
Z-up with the character facing +x, body origins at the bodies' centers of mass (their simulator's), body frames
aligned with the world in the T-pose; muscle forces are computed by :mod:`ndm_gym.robots.muscles`. The character's forward
kinematics and the per-axis Euler angles of its ball joints, which their simulator limits and MuJoCo cannot, are here too.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from ..core.rotations import axis_angle_to_mat, quat_to_mat
from .muscles import MUSCLE_MODELS, NUM_PARTS, MuscleModel, MuscleTable, damping_gain, muscle_params

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "musclevae")
MJCF_PATH = os.path.join(DATA_DIR, "character.xml")
CHARACTER_PATH = os.path.join(DATA_DIR, "character.json")
MUSCLES_PATH = os.path.join(DATA_DIR, "muscles.json")
CLIPS_DIR = os.path.join(DATA_DIR, "clips")

SOURCE_REPO = "https://github.com/wangchek/MuscleVAE_"
SOURCE_COMMIT = "54c001c913b156c0bba29e593a4a291e02d1f9ce"

SIM_HZ = 120  # physics, muscles and fatigue (their WorldAttr SimulateFPS, config.yml env_fps x env_substep)
CONTROL_HZ = 20  # policy (config.yml env_fps)
SUBSTEPS = SIM_HZ // CONTROL_HZ
JOINT_DAMPING = 10.0  # every joint (character JSON; App. C.1)
MASS = 61.4

# their body order, which is a depth-first order of the tree: body i is MuJoCo body i + 1, the root first
BODY_NAMES = (
    "pelvis", "rUpperLeg", "rLowerLeg", "rFoot", "FootThumbR", "FootPinkyR", "lUpperLeg", "lLowerLeg", "lFoot", "FootThumbL", "FootPinkyL",
    "lowerback", "torso", "Neck", "Head", "rClavicle", "rUpperArm", "rLowerArm", "rHand", "lClavicle", "lUpperArm", "lLowerArm", "lHand",
)  # fmt: skip
BODY_PARENTS = (-1, 0, 1, 2, 3, 3, 0, 6, 7, 8, 8, 0, 11, 12, 13, 12, 15, 16, 17, 12, 19, 20, 21)
# joint j moves body j + 1 relative to its parent; the BVH clips name the same joints
JOINT_NAMES = (
    "Character1_RightUpLeg", "Character1_RightLeg", "Character1_RightFoot", "FootThumbR", "FootPinkyR", "Character1_LeftUpLeg",
    "Character1_LeftLeg", "Character1_LeftFoot", "FootThumbL", "FootPinkyL", "Character1_Spine", "Character1_Spine1", "Character1_Neck",
    "Character1_Head", "Character1_RightShoulder", "Character1_RightArm", "Character1_RightForeArm", "Character1_RightHand",
    "Character1_LeftShoulder", "Character1_LeftArm", "Character1_LeftForeArm", "Character1_LeftHand",
)  # fmt: skip
HINGE_JOINTS = ("Character1_RightLeg", "FootThumbR", "FootPinkyR", "Character1_LeftLeg", "FootThumbL", "FootPinkyL", "Character1_RightForeArm",
                "Character1_LeftForeArm")  # fmt: skip
JOINT_TYPES = tuple("hinge" if n in HINGE_JOINTS else "ball" for n in JOINT_NAMES)
JOINT_QPOS_SIZE = tuple(1 if t == "hinge" else 4 for t in JOINT_TYPES)  # a ball joint's position is a quaternion (w, x, y, z)
JOINT_QPOS_ADR = tuple(int(a) for a in np.cumsum((0,) + JOINT_QPOS_SIZE[:-1]))  # in a clip's joint_qpos; MuJoCo's qpos adds 7
JOINT_QPOS_DIM = sum(JOINT_QPOS_SIZE)  # 64
JOINT_QVEL_SIZE = tuple(1 if t == "hinge" else 3 for t in JOINT_TYPES)
JOINT_QVEL_ADR = tuple(int(a) for a in np.cumsum((0,) + JOINT_QVEL_SIZE[:-1]))  # after the root's 6
JOINT_QVEL_DIM = sum(JOINT_QVEL_SIZE)  # 50
BALL_JOINTS = tuple(j for j, t in enumerate(JOINT_TYPES) if t == "ball")
NUM_BODIES, NUM_JOINTS, NUM_MUSCLES, NUM_ANCHORS = 23, 22, 284, 1212
HEAD = BODY_NAMES.index("Head")
PART_NAMES = ("trunk", "left_leg", "right_leg", "left_arm", "right_arm")  # the fatigue state's body parts, by insertion body
Y_UP_TO_Z_UP = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))  # theirs (x, y, z) -> ours (z, x, y): facing +x, left +y


def _read(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is missing: run tools/download_musclevae.sh, then python tools/convert_musclevae.py")
    with open(path) as f:
        return json.load(f)


def load_character(path: str = CHARACTER_PATH) -> dict:
    """The converted character description (character.json), checked against this module's constants."""
    char = _read(path)
    if char.get("source", {}).get("commit") != SOURCE_COMMIT:
        raise ValueError(f"{path} was not converted from MuscleVAE @ {SOURCE_COMMIT}; run python tools/convert_musclevae.py")
    bodies, joints = char["bodies"], char["joints"]
    if tuple(b["name"] for b in bodies) != BODY_NAMES or tuple(b["parent"] for b in bodies) != BODY_PARENTS:
        raise ValueError(f"{path}: bodies differ from ndm_gym.robots.musclevae_char.BODY_NAMES / BODY_PARENTS")
    if tuple(j["name"] for j in joints) != JOINT_NAMES or tuple(j["type"] for j in joints) != JOINT_TYPES:
        raise ValueError(f"{path}: joints differ from ndm_gym.robots.musclevae_char.JOINT_NAMES / JOINT_TYPES")
    if any(j["child"] != i + 1 or j["parent"] != BODY_PARENTS[i + 1] for i, j in enumerate(joints)):
        raise ValueError(f"{path}: joint j must move body j + 1 relative to its parent")
    return char


def load_muscle_table(path: str = MUSCLES_PATH) -> MuscleTable:
    """The converted muscle table (muscles.json) as a :class:`~ndm_gym.robots.muscles.MuscleTable`."""
    d = _read(path)
    if d.get("source", {}).get("commit") != SOURCE_COMMIT:
        raise ValueError(f"{path} was not converted from MuscleVAE @ {SOURCE_COMMIT}; run python tools/convert_musclevae.py")
    if tuple(d["bodies"]) != BODY_NAMES or tuple(d["parts"]) != PART_NAMES or len(PART_NAMES) != NUM_PARTS:
        raise ValueError(f"{path}: bodies or parts differ from ndm_gym.robots.musclevae_char")
    ms = d["muscles"]
    anchors = [a for m in ms for a in m["anchors"]]
    if len(ms) != NUM_MUSCLES or len(anchors) != NUM_ANCHORS:
        raise ValueError(f"{path}: {len(ms)} muscles and {len(anchors)} anchors, expected {NUM_MUSCLES} and {NUM_ANCHORS}")
    col = lambda key: np.array([m[key] for m in ms], dtype=np.float64)  # noqa: E731
    acol = lambda key: np.array([a[key] for a in anchors], dtype=np.float64)  # noqa: E731
    for name, params in MUSCLE_MODELS.items():  # the table's damping gains are the muscle model's
        kd = np.array([m["kd"][name] for m in ms])
        if not np.allclose(kd, damping_gain(col("kp"), params), rtol=1e-12, atol=0):
            raise ValueError(f"{path}: kd ({name}) differs from ndm_gym.robots.muscles; run python tools/convert_musclevae.py")
    return MuscleTable(
        names=tuple(m["name"] for m in ms), num_bodies=NUM_BODIES, anchor_count=np.array([len(m["anchors"]) for m in ms]),
        anchor_body0=acol("body0").astype(np.int64), anchor_local0=acol("local0"), anchor_body1=acol("body1").astype(np.int64),
        anchor_local1=acol("local1"), anchor_weight0=acol("weight0"), rest_length=col("rest_length"), lm=col("lm"), lt=col("lt"),
        f0=col("f0"), f0max=col("f0max"), kp=col("kp"), part=col("part").astype(np.int64),
    )  # fmt: skip


def muscle_model(model: str = "released", device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> MuscleModel:
    """The character's muscle model with the ``released`` (default) or ``paper`` parameters (ndm_gym.robots.muscles)."""
    return MuscleModel(load_muscle_table(), muscle_params(model), device, dtype)


# ---- kinematics
def _f(x, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=like.dtype, device=like.device)


def forward_kinematics(root_pos: torch.Tensor, root_quat: torch.Tensor, joint_qpos: torch.Tensor, char: dict | None = None):
    """Body positions (..., 23, 3) (their centers of mass) and rotations (..., 23, 3, 3) of the character's joint
    positions in the clips' layout: ``root_pos`` (..., 3), ``root_quat`` (..., 4) wxyz and ``joint_qpos`` (..., 64), as
    MuJoCo's forward kinematics of character.xml. Body frames are aligned with the world in the T-pose, so a body b
    below joint j with T-pose anchor a_j has R_b = R_p L_j and p_b = p_p + R_p (a_j - c_p) + R_b (c_b - a_j), with L_j
    the ball joint's rotation or the hinge's rotation about its axis and c the T-pose centers of mass."""
    char = load_character() if char is None else char
    com = _f([b["com_tpose"] for b in char["bodies"]], root_pos)
    pos, rot = [root_pos], [quat_to_mat(root_quat)]
    for j, joint in enumerate(char["joints"]):
        p, a, anchor = BODY_PARENTS[j + 1], JOINT_QPOS_ADR[j], _f(joint["anchor_tpose"], root_pos)
        if JOINT_TYPES[j] == "ball":
            local = quat_to_mat(joint_qpos[..., a : a + 4])
        else:
            angle = joint_qpos[..., a]
            local = axis_angle_to_mat(_f(joint["axis"], root_pos).expand(angle.shape + (3,)), angle)
        R = rot[p] @ local
        pos.append(pos[p] + rot[p] @ (anchor - com[p]) + R @ (com[j + 1] - anchor))
        rot.append(R)
    return torch.stack(pos, -2), torch.stack(rot, -3)


def ball_euler_angles(local: torch.Tensor) -> torch.Tensor:
    """The three angles their simulator limits on a ball joint (an ODE AMotor in Euler mode, BE:17727-17750 and ODE's
    amotor.cpp), of child rotations ``local`` (..., 3, 3) relative to the parent (our frame): axis 0, their x (our y),
    fixed in the child; axis 2, their z (our x), fixed in the parent; axis 1 their cross product; the reference vectors
    of the T-pose. Returns (..., 3), the angles about their x, y, z (character.json's euler_limits order), which satisfy
    local = Rot(z, a2) Rot(y, a1) Rot(x, a0) in their axes."""
    zero, one = torch.zeros_like(local[..., 0, 0]), torch.ones_like(local[..., 0, 0])  # on the device, no host copy
    ex, ey = torch.stack((one, zero, zero), -1), torch.stack((zero, one, zero), -1)
    ax0, ref1, ax2, ref2 = local[..., :, 1], local[..., :, 0], ex, ey
    ax1 = F.normalize(torch.cross(ax2, ax0, dim=-1), dim=-1)

    def dot(u, v):
        return (u * v).sum(-1)

    a0 = -torch.atan2(dot(ax2, torch.cross(ax0, ref1, dim=-1)), dot(ax2, ref1))
    a1 = -torch.atan2(dot(ax2, ax0), dot(ax2, torch.cross(ax0, ax1, dim=-1)))
    a2 = -torch.atan2(dot(ref2, ax1), dot(ref2, torch.cross(ax1, ax2, dim=-1)))
    return torch.stack((a0, a1, a2), -1)


def ball_euler_limits(char: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Their per-axis Euler limits (rad, about their x, y, z) of the ball joints in BALL_JOINTS order: lo, hi (14, 3)."""
    char = load_character() if char is None else char
    lim = [char["joints"][j]["euler_limits"] for j in BALL_JOINTS]
    return np.array([x["lo"] for x in lim]), np.array([x["hi"] for x in lim])
