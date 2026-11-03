"""Two-person clips retargeted to two G1s in their common world frame (InterHuman, ``tools/convert_interhuman.py``):
one ``<pair>.npz`` per clip with ``root_p`` (T, 2, 3), ``root_quat`` (T, 2, 4) wxyz, ``dof`` (T, 2, 29) in the order of
``dof_names``, ``fps`` and the conversion's ``record`` (JSON). Each file gives two clips, ``<pair>_p1`` and ``<pair>_p2``,
in this order, whose ``meta`` names their pair and side; the two-robot scene (``PairedTrackingSim``) plays them together.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from ...robots.g1 import G1_JOINT_NAMES
from . import Clip

ARRAYS = ("root_p", "root_quat", "dof", "dof_names", "fps", "record")


def load_pair(path: str) -> list[Clip]:
    pair = os.path.splitext(os.path.basename(path))[0]
    with np.load(path, allow_pickle=False) as d:
        if sorted(d.files) != sorted(ARRAYS):
            raise ValueError(f"{path}: arrays {sorted(d.files)}, expected {sorted(ARRAYS)}")
        names = [str(n) for n in d["dof_names"]]
        if sorted(names) != sorted(G1_JOINT_NAMES):
            raise ValueError(f"{path}: its joints are not the G1's 29")
        order = [names.index(n) for n in G1_JOINT_NAMES]
        p, quat, dof = (torch.tensor(d[k], dtype=torch.float32) for k in ("root_p", "root_quat", "dof"))
        fps, record = float(d["fps"]), json.loads(str(d["record"]))
    T = p.shape[0]
    for k, x, w in (("root_p", p, 3), ("root_quat", quat, 4), ("dof", dof, 29)):
        if x.shape != (T, 2, w) or not torch.isfinite(x).all():
            raise ValueError(f"{path}: {k} has shape {tuple(x.shape)}, expected ({T}, 2, {w}), finite")
    meta = {k: v for k, v in record.items() if k != "dof_names"}
    return [Clip(f"{pair}_p{s + 1}", p[:, s], quat[:, s], dof[:, s, order], fps, meta={"pair": pair, "side": s, **meta}) for s in range(2)]


def load(path: str) -> list[Clip]:
    """The pairs of one ``.npz`` or of every ``.npz`` in a directory (sorted by name), two clips each."""
    files = [path] if path.endswith(".npz") else sorted(glob.glob(os.path.join(path, "*.npz")))
    if not files:
        raise ValueError(f"no two-person clips (.npz) at {path}")
    return [c for f in files for c in load_pair(f)]
