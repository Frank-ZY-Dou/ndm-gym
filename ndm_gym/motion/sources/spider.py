"""SPIDER's hand-object clips (github.com/facebookresearch/spider; the ``retarget/retarget_full`` dataset), as
tools/download_hand_clips.py fetches them: per clip a MuJoCo scene of the hand, the object and the floor, and

``<task>/<trial>/trajectory_kinematic.npz``: ``qpos`` (n, 25), the human demonstration retargeted to the hand by inverse
kinematics, in the scene's qpos order (the hand's 18 joints, then the object's position and quaternion (w, x, y, z)),
one frame every ``ref_dt`` seconds of ``<task>/task_info.json``.

The kinematic retarget is the tracking reference: no physics shaped it, so the fingers may float off or cut into the
object, as a retargeted humanoid motion's feet do with the floor.
"""

from __future__ import annotations

import json
import os

import mujoco
import numpy as np
import torch

from ...core.rotations import quat_slerp
from ...robots.xhand import scene_indices

SIM_DT = 0.01  # the scenes' physics step in SPIDER


def load_scene(trial_dir: str) -> mujoco.MjModel:
    """The clip's scene with SPIDER's solver settings for hands."""
    model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(os.path.normpath(trial_dir)), "scene.xml"))
    model.opt.timestep = SIM_DT
    model.opt.iterations, model.opt.ls_iterations = 20, 50
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model


def load_reference(trial_dir: str, model: mujoco.MjModel, dt: float) -> dict[str, torch.Tensor]:
    """The clip's kinematic reference resampled every ``dt`` seconds: the hand's joint positions ``q`` (T, 18) and the
    object's position ``p`` (T, 3) and quaternion ``quat`` (T, 4), linearly (the quaternion spherically) interpolated."""
    task_dir = os.path.dirname(os.path.normpath(trial_dir))
    with open(os.path.join(task_dir, "task_info.json")) as f:
        ref_dt = float(json.load(f)["ref_dt"])
    with np.load(os.path.join(trial_dir, "trajectory_kinematic.npz"), allow_pickle=False) as d:
        qpos = torch.tensor(d["qpos"], dtype=torch.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or qpos.shape[0] < 2 or not torch.isfinite(qpos).all():
        raise ValueError(f"{trial_dir}: qpos {tuple(qpos.shape)} does not fit the scene's {model.nq} coordinates")
    idx = scene_indices(model)
    x = torch.arange(0, int(round((qpos.shape[0] - 1) * ref_dt / dt)) + 1, dtype=torch.float64) * dt / ref_dt
    i0 = x.floor().long().clamp(max=qpos.shape[0] - 2)
    t = (x - i0).clamp(0, 1)
    a, b = qpos[i0], qpos[i0 + 1]
    lerp = a + (b - a) * t[:, None]
    o = idx["object_qpos"]
    quat = quat_slerp(a[:, o + 3 : o + 7], b[:, o + 3 : o + 7], t)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    return {"q": lerp[:, idx["qpos"]].float(), "p": lerp[:, o : o + 3].float(), "quat": quat.float()}
