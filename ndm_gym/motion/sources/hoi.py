"""Human-object interaction clips for the G1 in one converted format, whatever their source (HDMI's
clips, OmniRetarget's retargeting of OMOMO): one ``<name>.npz`` per clip with

``root_pos`` (T, 3), ``root_quat_wxyz`` (T, 4), ``joint_pos`` (T, 29) in the G1's (Unitree SDK) joint order,
``object_pos`` (T, 3), ``object_quat_wxyz`` (T, 4), ``fps``, and optionally ``object_joint_pos`` (T, k) for an
articulated object, ``contact`` (T, c) reference contact flags, and ``extra_pos`` (T, n, 3) and ``extra_quat_wxyz``
(T, n, 4) for the scene's other objects;

and a ``<name>.json`` beside it that names the object (``object_name``), the other objects in their order
(``extra_objects``) and the source; it becomes the clip's ``meta``.

A source may also store its own reference: ``keypoint_pos`` (T, b, 3), ``keypoint_quat_wxyz`` (T, b, 4),
``keypoint_lin_vel`` and ``keypoint_ang_vel`` (T, b, 3) of the bodies named by the sidecar's ``keypoint_names``,
``joint_vel`` (T, 29) and ``object_joint_vel`` (T, k). The poses are checked but not loaded (forward kinematics
reproduces them); the velocities, which a source may not derive from its poses (HDMI's are not finite differences),
become the clip's ``src_vel``.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from ...robots.g1 import NUM_J
from . import Clip

REQUIRED = ("root_pos", "root_quat_wxyz", "joint_pos", "object_pos", "object_quat_wxyz", "fps")
OPTIONAL = ("object_joint_pos", "contact", "extra_pos", "extra_quat_wxyz", "keypoint_pos", "keypoint_quat_wxyz", "keypoint_lin_vel",
            "keypoint_ang_vel", "joint_vel", "object_joint_vel")  # fmt: skip


def load_clip(path: str) -> Clip:
    name = os.path.splitext(os.path.basename(path))[0]
    with np.load(path, allow_pickle=False) as d:
        missing = [k for k in REQUIRED if k not in d.files]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        unknown = sorted(set(d.files) - set(REQUIRED) - set(OPTIONAL))
        if unknown:
            raise ValueError(f"{path}: unknown arrays {unknown}")
        arr = {k: torch.tensor(d[k], dtype=torch.float32) for k in d.files if k != "fps"}
        fps = float(d["fps"])
    with open(os.path.splitext(path)[0] + ".json") as f:
        meta = json.load(f)
    T = arr["root_pos"].shape[0]
    extras = tuple(meta.get("extra_objects", ()))
    shapes = {"root_pos": (T, 3), "root_quat_wxyz": (T, 4), "joint_pos": (T, NUM_J), "object_pos": (T, 3), "object_quat_wxyz": (T, 4),
              "joint_vel": (T, NUM_J), "extra_pos": (T, len(extras), 3), "extra_quat_wxyz": (T, len(extras), 4)}  # fmt: skip
    if any(k.startswith("keypoint_") for k in arr):
        if "keypoint_names" not in meta:
            raise ValueError(f"{path}: keypoint arrays without keypoint_names")
        b = len(meta["keypoint_names"])
        shapes.update(keypoint_pos=(T, b, 3), keypoint_quat_wxyz=(T, b, 4), keypoint_lin_vel=(T, b, 3), keypoint_ang_vel=(T, b, 3))
    for k, shape in shapes.items():
        if k in arr and arr[k].shape != shape:
            raise ValueError(f"{path}: {k} has shape {tuple(arr[k].shape)}, expected {shape}")
    for k in ("object_joint_pos", "contact", "object_joint_vel"):
        if k in arr and (arr[k].dim() != 2 or arr[k].shape[0] != T):
            raise ValueError(f"{path}: {k} has shape {tuple(arr[k].shape)}, expected ({T}, k)")
    if extras and ("extra_pos" not in arr or "extra_quat_wxyz" not in arr):
        raise ValueError(f"{path}: the sidecar lists extra objects {list(extras)} without their poses")
    if not extras and ("extra_pos" in arr or "extra_quat_wxyz" in arr):
        raise ValueError(f"{path}: extra object poses without extra_objects in the sidecar")
    vel_keys = ("keypoint_lin_vel", "keypoint_ang_vel", "joint_vel")
    if 0 < sum(k in arr for k in vel_keys) < len(vel_keys):
        raise ValueError(f"{path}: the source's velocities need all of {list(vel_keys)}")
    if ("object_joint_vel" in arr) != ("object_joint_pos" in arr and "joint_vel" in arr):
        raise ValueError(f"{path}: object_joint_vel goes with object_joint_pos and the source's other velocities")
    src_vel = None
    if "joint_vel" in arr:
        src_vel = {"keypoint_names": list(meta["keypoint_names"]), **{k: arr[k] for k in vel_keys}}
        if "object_joint_vel" in arr:
            src_vel["object_joint_vel"] = arr["object_joint_vel"]
    return Clip(name, arr["root_pos"], arr["root_quat_wxyz"], arr["joint_pos"], fps, obj_name=meta["object_name"],
                obj_p=arr["object_pos"], obj_quat=arr["object_quat_wxyz"], obj_joint=arr.get("object_joint_pos"),
                contact=arr.get("contact"), extra_names=extras or None, extra_p=arr.get("extra_pos"),
                extra_quat=arr.get("extra_quat_wxyz"), meta=meta, src_vel=src_vel)  # fmt: skip


def load_dir(path: str, object_name: str | None = None) -> list[Clip]:
    """Every clip in ``path``, or only those with ``object_name``; raises when none is found."""
    clips = [load_clip(f) for f in sorted(glob.glob(os.path.join(path, "*.npz")))]
    if object_name is not None:
        clips = [c for c in clips if c.obj_name == object_name]
    if not clips:
        raise FileNotFoundError(f"no clip{'' if object_name is None else ' with object ' + object_name} in {path}")
    return clips
