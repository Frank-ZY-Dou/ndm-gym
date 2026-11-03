"""MuscleVAE's reference motions (their LAFAN1 walk, run and jump, retargeted to their muscle-actuated character) as
tools/convert_musclevae.py writes them: one ``<name>.npz`` per clip with

``root_pos`` (T, 3) and ``root_quat_wxyz`` (T, 4), the pose of the pelvis (its origin is its center of mass);
``joint_qpos`` (T, 64), the character's joint positions in its MJCF's qpos order after the free joint: for each joint
in ``ndm_gym.robots.musclevae_char.JOINT_NAMES`` order, a ball joint's rotation relative to its parent as a quaternion
(w, x, y, z) or a hinge's angle; ``fps``; and their dataset's own placement of the 23 bodies, ``body_pos`` (T, 23, 3) and
``body_quat_wxyz`` (T, 23, 4), read by :func:`load_placement`;

and a ``<name>.json`` beside it that describes the clip (its ``meta``). A clip's ``joints`` are these joint positions,
not one angle per joint as for the G1: ball joints need a motion library with ball-joint kinematics.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from ...robots.musclevae_char import BODY_NAMES, CLIPS_DIR, JOINT_NAMES, JOINT_QPOS_DIM, JOINT_TYPES
from . import Clip

ARRAYS = ("root_pos", "root_quat_wxyz", "joint_qpos", "fps", "body_pos", "body_quat_wxyz")


def _read(path: str) -> tuple[dict[str, torch.Tensor], float, dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} is missing: run tools/download_musclevae.sh, then python tools/convert_musclevae.py")
    with np.load(path, allow_pickle=False) as d:
        if sorted(d.files) != sorted(ARRAYS):
            raise ValueError(f"{path}: arrays {sorted(d.files)}, expected {sorted(ARRAYS)}")
        arr = {k: torch.tensor(d[k], dtype=torch.float32) for k in ARRAYS if k != "fps"}
        fps = float(d["fps"])
    with open(os.path.splitext(path)[0] + ".json") as f:
        meta = json.load(f)
    T, N = arr["root_pos"].shape[0], len(BODY_NAMES)
    shapes = {"root_pos": (T, 3), "root_quat_wxyz": (T, 4), "joint_qpos": (T, JOINT_QPOS_DIM), "body_pos": (T, N, 3), "body_quat_wxyz": (T, N, 4)}
    for k, shape in shapes.items():
        if tuple(arr[k].shape) != shape:
            raise ValueError(f"{path}: {k} has shape {tuple(arr[k].shape)}, expected {shape}")
    if (tuple(meta["joint_names"]), tuple(meta["joint_types"]), tuple(meta["body_names"])) != (JOINT_NAMES, JOINT_TYPES, BODY_NAMES):
        raise ValueError(f"{path}: joints or bodies differ from ndm_gym.robots.musclevae_char")
    if meta["quat_convention"] != "wxyz" or meta["fps"] != fps or meta["frames"] != T:
        raise ValueError(f"{path}: the sidecar's quat_convention, fps or frames differ from the arrays")
    return arr, fps, meta


def load_clip(path: str) -> Clip:
    arr, fps, meta = _read(path)
    name = os.path.splitext(os.path.basename(path))[0]
    return Clip(name, arr["root_pos"], arr["root_quat_wxyz"], arr["joint_qpos"], fps, meta=meta)


def load_placement(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Their dataset's placement of the bodies: positions (T, 23, 3) of the centers of mass, orientations (T, 23, 4) wxyz."""
    arr, _, _ = _read(path)
    return arr["body_pos"], arr["body_quat_wxyz"]


def load_dir(path: str = CLIPS_DIR) -> list[Clip]:
    """Every clip in ``path``; raises when there is none."""
    clips = [load_clip(f) for f in sorted(glob.glob(os.path.join(path, "*.npz")))]
    if not clips:
        raise FileNotFoundError(f"no clip in {path}: run tools/download_musclevae.sh, then python tools/convert_musclevae.py")
    return clips
