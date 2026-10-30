"""Human-object interaction clips for the G1 in one converted format, whatever their source (HDMI's
clips, OmniRetarget's retargeting of OMOMO): one ``<name>.npz`` per clip with

``root_pos`` (T, 3), ``root_quat_wxyz`` (T, 4), ``joint_pos`` (T, 29) in the G1's (Unitree SDK) joint order,
``object_pos`` (T, 3), ``object_quat_wxyz`` (T, 4), ``fps``, and optionally ``object_joint_pos`` (T, k) for an
articulated object and ``contact`` (T, c) reference contact flags;

and a ``<name>.json`` beside it that names the object (``object_name``) and the source.
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


def load_clip(path: str) -> Clip:
    name = os.path.splitext(os.path.basename(path))[0]
    with np.load(path, allow_pickle=False) as d:
        missing = [k for k in REQUIRED if k not in d.files]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        arr = {k: torch.tensor(d[k], dtype=torch.float32) for k in d.files if k != "fps"}
        fps = float(d["fps"])
    with open(os.path.splitext(path)[0] + ".json") as f:
        meta = json.load(f)
    T = arr["root_pos"].shape[0]
    if arr["joint_pos"].shape != (T, NUM_J):
        raise ValueError(f"{path}: joint_pos has shape {tuple(arr['joint_pos'].shape)}, expected ({T}, {NUM_J})")
    for k in ("root_quat_wxyz", "object_pos", "object_quat_wxyz", "object_joint_pos", "contact"):
        if k in arr and arr[k].shape[0] != T:
            raise ValueError(f"{path}: {k} has {arr[k].shape[0]} frames, the root {T}")
    return Clip(name, arr["root_pos"], arr["root_quat_wxyz"], arr["joint_pos"], fps, obj_name=meta["object_name"],
                obj_p=arr["object_pos"], obj_quat=arr["object_quat_wxyz"], obj_joint=arr.get("object_joint_pos"),
                contact=arr.get("contact"))  # fmt: skip


def load_dir(path: str, object_name: str | None = None) -> list[Clip]:
    """Every clip in ``path``, or only those with ``object_name``; raises when none is found."""
    clips = [load_clip(f) for f in sorted(glob.glob(os.path.join(path, "*.npz")))]
    if object_name is not None:
        clips = [c for c in clips if c.obj_name == object_name]
    if not clips:
        raise FileNotFoundError(f"no clip{'' if object_name is None else ' with object ' + object_name} in {path}")
    return clips
