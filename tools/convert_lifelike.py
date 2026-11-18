"""Retarget Tencent's Labrador motion capture to the Unitree Go2 (the quadruped tracking task's reference clips).

The motion capture is the 33 BVH files of "Lifelike agility and play in quadrupedal robots using reinforcement learning
and generative pre-trained models" (Han et al., Nature Machine Intelligence 2024;
https://github.com/Tencent-RoboticsX/lifelike-agility-and-play, ``data/raw_mocap_data``; MIT, its README: for research
purposes only), 120 fps, fetched by ``tools/download_lifelike.sh`` and not redistributed. The Go2 is MuJoCo Menagerie's
``unitree_go2/go2.xml`` (BSD-3-Clause).

The retargeting is supermimic-quadruped's ``quadmotion`` (https://github.com/Houserz/supermimic-quadruped, MIT), from the
user's own checkout (``--quadmotion``, its ``src`` directory): the dog's shoulder and hip joints give the trunk's frame,
each paw's offset from its shoulder or hip, scaled by the robot's standing height over the dog's, gives a foot target
below the robot's own thigh joint, planted paws are held in place on the floor, time is stretched by the square root of
the scale (so that flight phases keep gravity), and the legs are solved in closed form. Frames that fail a check (an
unreachable foot, a joint at its limit, the trunk or a knee in the floor, a capture glitch, ...) are cut out; each run of
valid frames of at least ``--min_seconds`` becomes a clip, its start at the origin facing +x.

``quadmotion`` reads the leg's geometry from the model and wants a foot site straight below the knee and a named trunk box,
which Menagerie's Go2 lacks: the conversion compiles a copy of the model with them added (``<out>/retarget_model``). The
Go2's foot sphere sits 2 mm behind that line, so the simulated foot is 2 mm behind the retargeted one.

Needs numpy, scipy, pyyaml and mujoco:
    python tools/convert_lifelike.py --quadmotion <supermimic-quadruped>/src
Writes, to ``--out`` (default data/lifelike_go2): ``<clip>.npz`` per clip (``qpos`` (T, 19): the trunk's position and
quaternion (w, x, y, z), then the 12 joints in the model's order; ``contacts`` (T, 4): planted feet, FL FR RL RR; ``fps``;
``joint_names``) and ``manifest.json`` (the calibration, and per source file the frames kept and what cut the others).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FOOT_GEOMS = ("FL", "FR", "RL", "RR")
STANCE = (0.9, -1.8)  # the thigh and calf angles of the model's "home" keyframe: the standing height dogs are scaled to


def retarget_model(go2_xml: str, out_dir: str) -> str:
    """A copy of the Go2 model with a foot site straight below each knee and the trunk's box named, and its description for
    ``quadmotion.robot.load_robot``. Returns the description's path."""
    import mujoco
    import yaml

    spec = mujoco.MjSpec.from_file(go2_xml)
    spec.meshdir = os.path.join(os.path.dirname(os.path.abspath(go2_xml)), spec.meshdir)
    boxes = [g for g in spec.body("base").geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX]
    if len(boxes) != 1:
        raise ValueError(f"{go2_xml}: expected one box on the trunk, found {len(boxes)}")
    boxes[0].name = "trunk_collision"
    for leg in FOOT_GEOMS:
        foot = spec.geom(leg)
        spec.body(f"{leg}_calf").add_site(name=f"{leg}_foot", pos=[0.0, 0.0, foot.pos[2]])
    spec.compile()
    os.makedirs(out_dir, exist_ok=True)
    xml = os.path.join(out_dir, "go2_retarget.xml")
    with open(xml, "w") as f:
        f.write(spec.to_xml())
    cfg = {
        "xml": os.path.abspath(xml),
        "root_body": "base",
        "legs": {leg: [f"{leg}_hip_joint", f"{leg}_thigh_joint", f"{leg}_calf_joint", f"{leg}_foot"] for leg in FOOT_GEOMS},
        "geoms": {"foot": "FL", "trunk": "trunk_collision"},
        "knee_sign": -1,
        "stance": list(STANCE),
    }
    path = os.path.join(out_dir, "go2.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist(), strict=True))


def at_origin(root_pos: np.ndarray, root_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The trajectory moved and turned about the vertical so that it starts at the origin facing +x."""
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_quat(root_quat, scalar_first=True)
    forward = rot[0].apply([1.0, 0.0, 0.0])
    undo = Rotation.from_euler("z", -float(np.arctan2(forward[1], forward[0])))
    pos = root_pos - [root_pos[0, 0], root_pos[0, 1], 0.0]
    return undo.apply(pos), (undo * rot).as_quat(scalar_first=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quadmotion", required=True, help="supermimic-quadruped's src directory (holds the quadmotion package)")
    ap.add_argument("--bvh_dir", default=os.path.join(REPO, "data/lifelike/raw_mocap_data"))
    ap.add_argument("--go2_xml", default=os.path.join(REPO, "data/menagerie/unitree_go2/go2.xml"))
    ap.add_argument("--out", default=os.path.join(REPO, "data/lifelike_go2"))
    ap.add_argument("--min_seconds", type=float, default=2.0, help="runs of valid frames shorter than this are dropped")
    ap.add_argument("--keep_static", type=float, default=1.0, help="seconds kept of a clip's leading and trailing stand-still")
    args = ap.parse_args()
    if not os.path.isdir(os.path.join(args.quadmotion, "quadmotion")):
        ap.error(f"--quadmotion: no quadmotion package in {args.quadmotion}")
    sys.path.insert(0, os.path.abspath(args.quadmotion))
    from quadmotion import dog
    from quadmotion.clip import FPS, froude_time_factor, trim_bounds
    from quadmotion.robot import LEGS, load_robot

    paths = sorted(glob.glob(os.path.join(args.bvh_dir, "*.bvh")))
    if not paths:
        ap.error(f"no BVH files in {args.bvh_dir} (tools/download_lifelike.sh)")
    robot = load_robot(retarget_model(args.go2_xml, os.path.join(args.out, "retarget_model")))
    skel = dog.TENCENT
    keypoints = [dog.load_keypoints(p, skel) for p in paths]
    cal = dog.calibrate(keypoints, skel, robot.nominal_root_height())
    checks = dog.checks_for(robot)
    manifest = {
        "source": "Tencent-RoboticsX/lifelike-agility-and-play, data/raw_mocap_data",
        "robot": os.path.abspath(args.go2_xml),
        "fps": FPS,
        "joint_names": list(robot.joint_names),
        "feet": list(LEGS),
        "calibration": {"scale": cal.scale, "time_factor": froude_time_factor(cal.scale), "dog_height_m": cal.dog_height,
                        "robot_height_m": cal.robot_height, "pitch_bias_deg": float(np.degrees(cal.pitch_bias))},  # fmt: skip
        "checks": {k: float(v) for k, v in vars(checks).items()},
        "files": {},
        "clips": {},
    }
    for old in glob.glob(os.path.join(args.out, "*.npz")):
        os.remove(old)
    for path, k in zip(paths, keypoints, strict=True):
        r = dog.retarget(k, skel, cal, robot, checks=checks)
        valid = r.valid
        record = {"source_frames": k.src_frames, "source_fps": k.src_fps, "frames": len(valid), "valid_frames": int(valid.sum()),
                  "cut_by": {n: int(m.sum()) for n, m in r.fail.items() if m.any()}, "clips": []}  # fmt: skip
        for a, b in runs(valid):
            if (b - a) / FPS < args.min_seconds:
                continue
            q = r.joint_pos[a:b]
            qd = np.zeros_like(q)
            qd[:-1] = np.diff(q, axis=0) * FPS
            lead, end = trim_bounds(qd, FPS, args.keep_static)
            a, b = a + lead, a + end
            if (b - a) / FPS < args.min_seconds:
                continue
            pos, quat = at_origin(r.root_pos[a:b], r.root_quat[a:b])
            qpos = np.concatenate((pos, quat, r.joint_pos[a:b]), -1).astype(np.float32)
            name = f"{k.name}_{len(record['clips'])}"  # the file's kept runs, in order
            np.savez(os.path.join(args.out, name + ".npz"), qpos=qpos, contacts=r.contacts[a:b], fps=np.float32(FPS),
                     joint_names=np.array(robot.joint_names))  # fmt: skip
            speed = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=-1) * FPS
            manifest["clips"][name] = {
                "file": os.path.basename(path), "frames": [a, b], "seconds": (b - a) / FPS,
                "root_height_m": [float(pos[:, 2].min()), float(pos[:, 2].max())],
                "speed_mps": [float(speed.mean()), float(speed.max())],
                "joint_speed_max_radps": float(np.abs(np.diff(qpos[:, 7:], axis=0)).max() * FPS),
                "ik_residual_max_mm": float(r.ik_residual[a:b].max() * 1000),
                "flight_frames": int((~r.contacts[a:b].any(1)).sum()),
            }  # fmt: skip
            record["clips"].append(name)
        manifest["files"][os.path.basename(path)] = record
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    seconds = sum(c["seconds"] for c in manifest["clips"].values())
    total = sum(f["frames"] for f in manifest["files"].values()) / FPS
    print(f"{len(paths)} files, {total:.0f} s at the robot's time -> {len(manifest['clips'])} clips, {seconds:.0f} s, in {args.out}")
    print(f"scale {cal.scale:.3f} (dog {cal.dog_height:.3f} m, robot {cal.robot_height:.3f} m), time x{froude_time_factor(cal.scale):.3f}")


if __name__ == "__main__":
    main()
