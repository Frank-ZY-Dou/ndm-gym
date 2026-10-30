"""LAFAN1 clips retargeted to the 29-DoF G1 (HF ``lvhaidong/LAFAN1_Retargeting_Dataset``, folder ``g1/``).

Each CSV row holds root position, root quaternion (xyzw) and 29 joint angles at 30 fps.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from . import Clip

FPS = 30.0


def load_dir(path: str) -> list[Clip]:
    clips = []
    for f in sorted(glob.glob(os.path.join(path, "*.csv"))):
        raw = torch.tensor(np.loadtxt(f, delimiter=","), dtype=torch.float32)
        p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
        quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
        clips.append(Clip(os.path.splitext(os.path.basename(f))[0], p, quat, j, FPS))
    return clips
