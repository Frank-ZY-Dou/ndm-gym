"""LAFAN1 clips retargeted to Unitree humanoids (HF ``lvhaidong/LAFAN1_Retargeting_Dataset``): folder ``g1/`` for the
29-DoF G1, ``h1/`` for the 19-DoF H1.

Each CSV row holds root position, root quaternion (xyzw) and the joint angles (29 for the G1, 19 for the H1) at 30 fps.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from . import Clip

FPS = 30.0


def load_dir(path: str, num_joints: int = 29) -> list[Clip]:
    """Every clip (``*.csv``) in ``path``, sorted by name, of a robot with ``num_joints`` joints (29: the G1)."""
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no LAFAN1 clips (*.csv) in {path}: tools/download_data.sh writes them to data/lafan1_g1")
    clips = []
    for f in files:
        raw = torch.tensor(np.loadtxt(f, delimiter=","), dtype=torch.float32)
        if raw.ndim != 2 or raw.shape[1] != 7 + num_joints:
            raise ValueError(f"{f}: {tuple(raw.shape)} rows, expected 7 + {num_joints} columns")
        p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7 : 7 + num_joints]
        quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
        clips.append(Clip(os.path.splitext(os.path.basename(f))[0], p, quat, j, FPS))
    return clips
