"""The bimanual tracking task's demonstrations: ACT's scripted policies' episodes in its simulated ALOHA scenes with the whole
state kept, as tools/make_aloha_demos.py writes them, ``<data_dir>/<task>/demos.npz`` (ACT's own recordings hold no object
poses). A demonstration is the state after every control step of 20 ms (qpos, qvel; 401 frames with the start), the 16
position targets of every step and ACT's staged reward. The reference's targets are the demonstration's own: a frame's are
those that led to it, the first frame's those of the first step."""

from __future__ import annotations

import json
import os

import numpy as np

from ...robots.aloha_scenes import TASKS


def demo_path(data_dir: str, task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"task {task!r}, one of {sorted(TASKS)}")
    return os.path.join(data_dir, TASKS[task]["data"], "demos.npz")


def load_demo(path: str, demo: int) -> dict:
    """One demonstration: ``qpos`` (T + 1, nq), ``qvel`` (T + 1, nv), ``target`` (T + 1, nu), ``reward`` (T,) and the file's
    ``meta`` (its SHA-256, MuJoCo's version, whether it is the benchmark's set)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: the demonstrations are not there (python tools/make_aloha_demos.py, with MuJoCo 2.3.7)")
    with np.load(path) as z:
        n = z["qpos"].shape[0]
        if not 0 <= demo < n:
            raise ValueError(f"{path} has demonstrations 0 to {n - 1}, not {demo}")
        ctrl = z["ctrl"][demo].astype(np.float64)
        return {"qpos": z["qpos"][demo].astype(np.float64), "qvel": z["qvel"][demo].astype(np.float64),
                "target": np.concatenate([ctrl[:1], ctrl]), "reward": z["reward"][demo], "meta": json.loads(str(z["meta"]))}  # fmt: skip
