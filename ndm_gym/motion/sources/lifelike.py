"""Tencent's Labrador motion capture ("Lifelike agility and play in quadrupedal robots", Han et al., Nature Machine
Intelligence 2024) retargeted to the Unitree Go2 by tools/convert_lifelike.py, which writes per clip

``<clip>.npz``: ``qpos`` (n, 19), the Go2's trunk position and quaternion (w, x, y, z) and its 12 joints in the model's
order (robots.go2.JOINT_NAMES); ``contacts`` (n, 4), the planted feet (FL, FR, RL, RR); ``fps``; ``joint_names``. Each clip
starts at the origin facing +x.

The retarget is kinematic: no physics shaped it, so the reference may ask for a gait the robot cannot follow exactly.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from ...core.rotations import quat_slerp
from ...robots.go2 import JOINT_NAMES, NQ, REPO

DATA_DIR = os.path.join(REPO, "data", "lifelike_go2")


def clip_paths(clips: str | list[str] | None = None, root: str = DATA_DIR) -> list[str]:
    """The clips' files: ``clips`` a directory (every clip in it), or names, globs or paths, comma-separated in a string or
    a list; names and globs are looked up in ``root``. None: every clip in ``root``."""
    items = [root] if clips is None else [c for c in (clips.split(",") if isinstance(clips, str) else clips) if c]
    paths = []
    for item in items:
        if os.path.isdir(item):
            found = sorted(glob.glob(os.path.join(item, "*.npz")))
        else:
            pattern = item if os.path.dirname(item) else os.path.join(root, item)
            found = sorted(glob.glob(pattern if pattern.endswith(".npz") else pattern + ".npz"))
        if not found:
            raise FileNotFoundError(f"no clip matches {item!r} (in {root})")
        paths += [p for p in found if p not in paths]
    return paths


def clip_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_clip(path: str, dt: float) -> dict:
    """A clip resampled every ``dt`` seconds: ``qpos`` (T, 19) with the joints and the position linearly interpolated, the
    quaternion spherically and its sign made continuous; ``contacts`` (T, 4) of the nearest earlier frame; ``name``."""
    with np.load(path, allow_pickle=False) as d:
        qpos, contacts, fps = torch.tensor(d["qpos"], dtype=torch.float64), torch.tensor(d["contacts"]), float(d["fps"])
        names = [str(n) for n in d["joint_names"]]
    if names != JOINT_NAMES or qpos.ndim != 2 or qpos.shape[1] != NQ or qpos.shape[0] < 2 or not torch.isfinite(qpos).all():
        raise ValueError(f"{path}: qpos {tuple(qpos.shape)} with joints {names} is not the Go2's")
    quat = qpos[:, 3:7] / qpos[:, 3:7].norm(dim=-1, keepdim=True)
    flip = torch.cumprod(torch.where((quat[1:] * quat[:-1]).sum(-1) < 0, -1.0, 1.0), 0)
    quat[1:] *= flip.unsqueeze(-1)
    qpos[:, 3:7] = quat * torch.where(quat[:1, :1] < 0, -1.0, 1.0)
    x = torch.arange(0, int(round((qpos.shape[0] - 1) / (fps * dt))) + 1, dtype=torch.float64) * dt * fps
    i0 = x.floor().long().clamp(max=qpos.shape[0] - 2)
    t = (x - i0).clamp(0, 1)
    a, b = qpos[i0], qpos[i0 + 1]
    out = a + (b - a) * t[:, None]
    quat = quat_slerp(a[:, 3:7], b[:, 3:7], t)
    out[:, 3:7] = quat / quat.norm(dim=-1, keepdim=True)
    return {"name": clip_name(path), "qpos": out.float(), "contacts": contacts[i0 + (t > 0.999).long()]}
