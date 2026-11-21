"""robomimic's demonstrations (Mandlekar et al., CoRL 2021): v1.5's ``demo_v15.hdf5``, one file per task and operator group,
as tools/download_robomimic.py writes them under ``<data_dir>/<task>/``.

A demonstration holds MuJoCo's flattened state at every control step (``states``: time, qpos, qvel), recorded at robosuite's
control frequency (20 Hz) after each step, its actions and its scene (``model_file``). The reference is the states' qpos and
qvel as they are: the arms', the grippers' and every object's, so the objects' trajectories are the demonstration's own.
"""

from __future__ import annotations

import json
import os

import numpy as np

CONTROL_FREQ = 20  # robosuite's control_freq of the v1.5 datasets


def demo_path(data_dir: str, task: str) -> str:
    return os.path.join(data_dir, task, "demo_v15.hdf5")


def demo_names(path: str) -> list[str]:
    """The demonstrations of a file in their numeric order (demo_0, demo_1, ...)."""
    import h5py

    with h5py.File(path, "r") as f:
        return sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))


def load_demo(path: str, demo: int | str) -> dict:
    """One demonstration: ``qpos`` (T, nq), ``qvel`` (T, nv) as float64 arrays, ``xml`` (its scene), ``env_name`` and the
    control frequency. ``demo``: its index (``demo_<i>``) or name."""
    import h5py

    name = demo if isinstance(demo, str) else f"demo_{demo}"
    with h5py.File(path, "r") as f:
        env = json.loads(f["data"].attrs["env_args"])
        freq = int(env["env_kwargs"]["control_freq"])
        if freq != CONTROL_FREQ:
            raise ValueError(f"{path}: control_freq {freq}, the task's control step is 1/{CONTROL_FREQ} s")
        g = f["data"][name]
        states, xml = np.asarray(g["states"][()], dtype=np.float64), g.attrs["model_file"]
    return {"states": states, "xml": xml if isinstance(xml, str) else xml.decode(), "env_name": env["env_name"], "control_freq": freq}


def split_states(states: np.ndarray, nq: int, nv: int) -> tuple[np.ndarray, np.ndarray]:
    """qpos and qvel of the flattened states (T, 1 + nq + nv): MuJoCo's time first."""
    if states.shape[-1] != 1 + nq + nv:
        raise ValueError(f"states of width {states.shape[-1]}, the scene's 1 + nq + nv is {1 + nq + nv}")
    return states[:, 1 : 1 + nq], states[:, 1 + nq : 1 + nq + nv]
