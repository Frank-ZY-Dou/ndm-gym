"""MIMIC-MJX's reference clips (tools/download_mimic_mjx.py, ``<root>/<animal>/reference.h5``): stac-mjx's inverse kinematics
of each animal's recorded keypoints, MuJoCo's qpos and qvel of the body (robots.animals) per frame, in clips of a fixed
length one after the other (``ANIMALS[animal].clip_length``), at the recording's rate (``mocap_hz``)."""

from __future__ import annotations

import numpy as np
import torch

from ...core.rotations import quat_slerp


def load_clips(root: str, animal: str, clips: list[int] | None = None) -> dict[str, np.ndarray]:
    """The clips ``clips`` (indices; all by default) of ``animal``: ``qpos`` (C, T, nq) and ``qvel`` (C, T, nv) as float32, and
    ``clips`` (C,), their indices."""
    import h5py

    from ...robots.animals import ANIMALS

    T = ANIMALS[animal].clip_length
    with h5py.File(f"{root}/{animal}/reference.h5", "r") as f:
        n = f["qpos"].shape[0] // T
        idx = np.arange(n) if clips is None else np.asarray(clips, dtype=np.int64)
        if idx.size == 0 or idx.min() < 0 or idx.max() >= n:
            raise ValueError(f"{animal}: clips must be within 0..{n - 1}")
        qpos = np.stack([f["qpos"][i * T : (i + 1) * T] for i in idx]).astype(np.float32)
        qvel = np.stack([f["qvel"][i * T : (i + 1) * T] for i in idx]).astype(np.float32)
    return {"qpos": qpos, "qvel": qvel, "clips": idx}


def velocities_from_positions(model, qpos: torch.Tensor, dt: float) -> torch.Tensor:
    """The velocities (T, nv) of one clip's positions ``qpos`` (T, nq) sampled every ``dt`` s: per frame MuJoCo's velocity that
    takes it to the next frame in ``dt`` (mj_differentiatePos, a forward difference: the free root's linear velocity in the
    world's frame, its angular velocity in the body's), and the last frame's that of the frame before it. At the pinned
    revision the reference files' own ``qvel`` is not this: the root's linear velocity is 10 times it (the rat, the stick
    insect) or 0.1 times (the fly), and joint speeds are clipped at 20 rad/s (stac-mjx #135 fixed both after that revision)."""
    import mujoco

    q = qpos.detach().cpu().numpy().astype(np.float64)
    if q.ndim != 2 or q.shape[1] != model.nq or q.shape[0] < 2:
        raise ValueError(f"positions of shape (T >= 2, {model.nq}) expected, got {tuple(q.shape)}")
    v = np.zeros((q.shape[0], model.nv))
    for t in range(q.shape[0] - 1):
        mujoco.mj_differentiatePos(model, v[t], dt, q[t], q[t + 1])
    v[-1] = v[-2]
    return torch.from_numpy(v).to(qpos.dtype)


def resample(qpos: torch.Tensor, qvel: torch.Tensor, src_hz: float, dt: float, quat_addrs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """One clip (T, nq), (T, nv) recorded at ``src_hz`` resampled every ``dt`` s over its span: positions and velocities
    linearly, the quaternions at ``quat_addrs`` (qpos addresses of their w) by slerp."""
    T = qpos.shape[0]
    t = torch.arange(0.0, (T - 1) / src_hz + 1e-9, dt, dtype=torch.float64) * src_hz
    i0 = t.floor().long().clamp(max=T - 2)
    w = (t - i0).to(qpos.dtype).unsqueeze(-1)
    q = qpos[i0] + w * (qpos[i0 + 1] - qpos[i0])
    for a in quat_addrs:
        q[:, a : a + 4] = quat_slerp(qpos[i0, a : a + 4], qpos[i0 + 1, a : a + 4], w[:, 0])
    v = qvel[i0] + w * (qvel[i0 + 1] - qvel[i0])
    return q, v
