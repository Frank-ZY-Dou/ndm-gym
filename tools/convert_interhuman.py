"""Retarget one InterHuman two-person clip to two 29-DoF G1s in their common world frame (the two-G1 boxing example).

InterHuman (https://github.com/tr3e/InterGen; CC BY-NC-SA 4.0, not redistributed) stores each person's SMPL parameters
(z-up world, ``mocap_framerate``, ``trans``, ``root_orient``, ``pose_body``, ``betas``). Both people are resampled to 50 Hz,
each person's SMPL skeleton comes from their own shape (``betas``, the licensed neutral SMPL model), and each is
retargeted with HoloMotion's HoloRetarget (Newton IK; https://github.com/HorizonRobotics/HoloMotion, Apache-2.0) as
HoloBoxing does it (holoboxing/data/interhuman.py): the IK sees the person at the origin horizontally, and the pair's
horizontal positions are the people's pelvis positions scaled once, about their first-frame midpoint, by the G1's
height over the people's mean height, so that distance and facing between them are kept. HoloRetarget's G1 has other
feet than the simulated one, so each robot is then raised or lowered once, on the simulated model, until its lowest foot
point is on the floor at the median frame.

Runs in a separate environment with HoloRetarget's dependencies (numpy, scipy, mujoco 3.5, warp-lang 1.12, newton 1.0)
and HoloMotion's ``holoretarget`` and ``holosmpl.converters.smpl_to_body_poses`` on PYTHONPATH, on a CUDA GPU:
    PYTHONPATH=<holomotion subset> python tools/convert_interhuman.py --pkl <InterHuman>/motions/1566.pkl \
        --annot <InterHuman>/annots/1566.txt --smpl_model <SMPL_NEUTRAL.pkl> --robot_mjcf data/g1_description/g1_29dof.xml \
        --out <dir>/interhuman_1566.npz
Writes ``<out>`` (``ndm_gym.motion.sources.interhuman`` reads it) and ``<out>.json`` (the conversion's record).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from types import SimpleNamespace

import numpy as np

FPS = 50.0


class _ArrayUnpickler(pickle.Unpickler):
    """InterHuman's files hold NumPy arrays only; anything else is refused (they are downloaded, untrusted data)."""

    def find_class(self, module, name):
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(np.core.multiarray, name)
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return np.core.numeric._frombuffer
        raise pickle.UnpicklingError(f"refused {module}.{name} in an InterHuman file")


def read_pair(path: str):
    """The two people's (trans, root_orient, pose_body, betas) at the source rate, and the rate."""
    with open(path, "rb") as f:
        d = _ArrayUnpickler(f).load()
    T, fps = int(np.asarray(d["frames"]).reshape(-1)[0]), float(np.asarray(d["mocap_framerate"]).reshape(-1)[0])
    people = []
    for role in ("person1", "person2"):
        p = d[role]
        if str(p.get("gender", "neutral")).lower() not in ("neutral", "none"):
            raise ValueError(f"{path}: {role} is not neutral SMPL, unlike the neutral model used")
        out = {k: np.asarray(p[k], dtype=np.float64) for k in ("trans", "root_orient", "pose_body")}
        for k, w in (("trans", 3), ("root_orient", 3), ("pose_body", 63)):
            if out[k].shape != (T, w) or not np.isfinite(out[k]).all():
                raise ValueError(f"{path}: {role}.{k} has shape {out[k].shape}, expected ({T}, {w})")
        betas = np.asarray(p["betas"], dtype=np.float64)
        if betas.ndim == 2:
            if not np.allclose(betas, betas[0], atol=1e-6, rtol=0):
                raise ValueError(f"{path}: {role}'s shape changes over the clip")
            betas = betas[0]
        out["betas"] = betas
        people.append(out)
    return people, T, fps


def resample(person: dict, T: int, fps: float):
    """50 Hz: translation linearly, every joint's rotation by slerp (HoloBoxing's resample_person)."""
    from scipy.spatial.transform import Rotation, Slerp

    src = np.arange(T) / fps
    t = np.arange(int(np.floor(src[-1] * FPS + 1e-8)) + 1) / FPS
    trans = np.stack([np.interp(t, src, person["trans"][:, k]) for k in range(3)], -1)
    local = np.concatenate((person["root_orient"], person["pose_body"]), -1).reshape(T, 22, 3)
    rot = np.stack([Slerp(src, Rotation.from_rotvec(local[:, j]))(t).as_rotvec() for j in range(22)], 1)
    return trans, rot.reshape(len(t), 66)


def _array(x):
    """A licensed SMPL model's array, also when pickled as a Chumpy object (HoloRetarget's stand-in keeps its state)."""
    state = getattr(x, "__dict__", {}).get("_pickle_state")
    if isinstance(state, dict) and "x" in state:
        x = state["x"]
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float64)


class SmplShape:
    """Per-person SMPL skeleton (parents, rest offsets), rest pelvis joint and height from the shape coefficients."""

    def __init__(self, path: str):
        from holoretarget._engine_impl import _enable_chumpy_compat

        _enable_chumpy_compat()
        with open(path, "rb") as f:  # the user's licensed SMPL model (trusted); legacy files pickle Chumpy and SciPy objects
            m = pickle.load(f, encoding="latin1")
        self.regressor = _array(m["J_regressor"])
        self.template = _array(m["v_template"])
        self.dirs = _array(m["shapedirs"])
        self.parents = np.asarray(m["kintree_table"][0, :24], dtype=np.int32)
        self.parents[0] = -1

    def skeleton(self, betas: np.ndarray):
        if not 0 < len(betas) <= self.dirs.shape[-1]:
            raise ValueError(f"{len(betas)} shape coefficients for a model with {self.dirs.shape[-1]}")
        v = self.template + np.einsum("vci,i->vc", self.dirs[..., : len(betas)], betas)
        j = (self.regressor @ v)[:24]
        off = np.zeros_like(j, dtype=np.float32)
        off[1:] = j[1:] - j[self.parents[1:]]
        # the neutral mesh is y-up: its y extent is the person's height, whatever the clip's pose
        return SimpleNamespace(smpl_parents=self.parents, smpl_rest_parent_offsets=off), j[0], float(np.ptp(v[:, 1]))


def robot_height(mjcf: str) -> float:
    """The G1's height standing at zero joint angles: the z extent of its visual meshes (geom group 1)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    z = []
    for g in range(model.ngeom):
        if model.geom_group[g] != 1 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        m = int(model.geom_dataid[g])
        v = model.mesh_vert[model.mesh_vertadr[m] : model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        z.append((v @ data.geom_xmat[g].reshape(3, 3).T)[:, 2] + data.geom_xpos[g, 2])
    if not z:
        raise ValueError(f"{mjcf}: no visual meshes (geom group 1)")
    z = np.concatenate(z)
    return float(z.max() - z.min())


def foot_heights(mjcf: str, qpos: np.ndarray) -> np.ndarray:
    """(T,) the lowest point of the G1's foot collision spheres (ankle roll links) above the floor at each frame, on the
    simulated model (``mjcf``, whose joints follow qpos[7:]'s order)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    feet = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]
            and model.body(model.geom_bodyid[g]).name in ("left_ankle_roll_link", "right_ankle_roll_link")]  # fmt: skip
    if not feet:
        raise ValueError(f"{mjcf}: no colliding foot spheres on the ankle roll links")
    low = np.empty(len(qpos))
    for t, q in enumerate(qpos):
        data.qpos[:36] = q
        mujoco.mj_kinematics(model, data)
        low[t] = min(data.geom_xpos[g, 2] - model.geom_size[g, 0] for g in feet)
    return low


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="InterHuman motions/<id>.pkl")
    ap.add_argument("--annot", default=None, help="InterHuman annots/<id>.txt (kept in the record)")
    ap.add_argument("--smpl_model", required=True, help="the licensed neutral SMPL model (.pkl)")
    ap.add_argument("--robot_mjcf", required=True,
                    help="the simulated G1 (data/g1_description/g1_29dof.xml): each robot is raised or lowered once so that its "
                    "lowest foot point is on the floor at the median frame")  # fmt: skip
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--iterations", type=int, default=4, help="HoloRetarget's Newton iterations per frame (HoloBoxing's default)")
    a = ap.parse_args()

    from dataclasses import replace

    from holoretarget import DEFAULT_CONFIG, HoloRetargeter
    from holoretarget.schema import UNITREE_G1_29DOF_NAMES
    from holosmpl.converters.smpl_to_body_poses import SmplToBodyPosesConverter

    t0 = time.time()
    people, T, fps = read_pair(a.pkl)
    shape = SmplShape(a.smpl_model)
    cfg = replace(DEFAULT_CONFIG, newton_iterations=a.iterations)
    h_robot = robot_height(os.path.join(cfg.resolved_asset_root, cfg.robot, "g1_mocap_29dof.xml"))
    seqs, skels, pelvis, heights = [], [], [], []
    for p in people:
        trans, pose = resample(p, T, fps)
        skel, root_joint, h = shape.skeleton(p["betas"])
        seqs.append((trans + root_joint, pose))  # the pelvis joint's world position
        skels.append(skel), pelvis.append(trans + root_joint), heights.append(h)
    pelvis = np.stack(pelvis, 1)  # (T50, 2, 3)
    scale = h_robot / float(np.mean(heights))
    anchor = pelvis[0, :, :2].mean(0)
    pair_xy = scale * (pelvis[..., :2] - anchor)  # (T50, 2, 2): one scale and one anchor for both people
    qpos = np.empty((len(pelvis), 2, 36), dtype=np.float32)
    ground = []
    for role in range(2):
        rt = HoloRetargeter(cfg)
        rt.reset_sequence()
        rt._smpl_adapter = skels[role]  # this person's skeleton (HoloBoxing does the same on its own instances)
        conv = SmplToBodyPosesConverter(rt)
        for i, (tr, pose) in enumerate(zip(*seqs[role])):
            local = tr.copy()
            local[:2] = 0.0  # the IK sees the person at the origin horizontally; the pair's xy is set below
            q = rt.retarget_qpos_from_body_poses(conv(transl=local, global_orient_aa=pose[:3], body_pose_aa=pose[3:]))
            q[:2] = pair_xy[i, role]
            qpos[i, role] = q
        ground.append(rt.last_ground_offset_z)
    if not np.isfinite(qpos).all() or not np.allclose(np.linalg.norm(qpos[..., 3:7], axis=-1), 1.0, atol=1e-4):
        raise ValueError("HoloRetarget returned non-finite states or non-unit quaternions")
    expected = scale * (pelvis[:, 1, :2] - pelvis[:, 0, :2])
    xy_err = float(np.abs(qpos[:, 1, :2] - qpos[:, 0, :2] - expected).max())
    if xy_err > 1e-4:
        raise ValueError(f"the pair's relative position is off by {xy_err} m")
    # HoloRetarget's G1 has other feet than the simulated one: each robot's height is set once on the simulated model
    lift, feet = [], []
    for role in range(2):
        low = foot_heights(a.robot_mjcf, qpos[:, role].astype(np.float64))
        lift.append(-float(np.median(low)))
        qpos[:, role, 2] += lift[-1]
        low = low + lift[-1]
        feet.append({"min_m": round(float(low.min()), 4), "p95_m": round(float(np.percentile(low, 95)), 4)})
    record = {
        "source": os.path.basename(a.pkl), "source_frames": T, "source_fps": fps, "frames": int(len(qpos)), "fps": FPS,
        "annotation": open(a.annot, encoding="utf-8", errors="replace").read().strip().splitlines() if a.annot else None,
        "scale": scale, "anchor_xy_m": anchor.tolist(), "human_heights_m": heights, "robot_height_m": h_robot,
        "robot_height_definition": "HoloRetarget's G1 (g1_mocap_29dof.xml) visual meshes, z extent at zero joint angles",
        "pair_xy_max_error_m": xy_err, "ground_offset_z_m": ground,
        "retarget": {"tool": "HoloRetarget (HoloMotion)", "newton_iterations": a.iterations},
        "feet_on_floor": {"model": os.path.basename(a.robot_mjcf), "lift_m": lift, "lowest_foot_point_after": feet,
                          "rule": "each robot raised by minus the median over frames of its lowest foot-sphere point"},
        "dof_names": list(UNITREE_G1_29DOF_NAMES), "seconds": round(time.time() - t0, 1),
    }  # fmt: skip
    np.savez_compressed(a.out, root_p=qpos[..., :3], root_quat=qpos[..., 3:7], dof=qpos[..., 7:], dof_names=np.array(UNITREE_G1_29DOF_NAMES),
                        fps=np.array(FPS), record=np.array(json.dumps(record)))  # fmt: skip
    json.dump(record, open(a.out + ".json", "w"), indent=1)
    print(json.dumps({k: v for k, v in record.items() if k not in ("annotation", "dof_names")}, indent=1))


if __name__ == "__main__":
    main()
