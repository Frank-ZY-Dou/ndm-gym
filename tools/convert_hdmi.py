"""Convert the motion clips of HDMI's 13 human-object interaction tasks (LeCAR-Lab/HDMI @ 32282f6, placed in data/hdmi/ by
tools/download_hdmi.sh) to ndm_gym's clip format (ndm_gym/motion/sources/hoi.py): data/hdmi/clips/<task>.npz and
<task>.json, one clip per task, named after HDMI's task config cfg/task/G1/hdmi/<task>.yaml.

A clip holds the reference HDMI's RobotObjectTracking command reads from the same file. References are to HDMI's source,
as in docs/hdmi_task_spec.md (MOT = active_adaptation/utils/motion.py, CMD = active_adaptation/envs/mdp/commands/hdmi/
command.py, REW = .../hdmi/rewards.py, G1 = active_adaptation/assets/g1.py, Y = cfg/task/base/hdmi-base.yaml,
T:<task> = cfg/task/G1/hdmi/<task>.yaml):

* frames: all 13 clips are at 50 fps, HDMI's control rate, at which HDMI uses every frame as stored (MOT:103-104); nothing
  is resampled. Values are HDMI's float64 cast to float32, as HDMI holds them (MOT:271-296).
* root: the pose of the pelvis body (root_body_name, CMD:75, 162-163).
* joints: the clip's joints by name, in ndm_gym's G1 order (ndm_gym.robots.g1.G1_JOINT_NAMES).
* object: the pose of the body named after the object asset (CMD:383, 550-551); for the door and the folding chair also
  the hinge angle (object_joint_name, CMD:379, 558) as object_joint_pos.
* extra objects: the poses of the bodies of extra_object_names (CMD:369-370, 477-490), as extra_pos / extra_quat_wxyz.
* contact: object_contact cast to bool as HDMI casts it (CMD:444-446; push_box stores 0/1 floats), one row per frame like
  the poses, with one column shared by all contact end effectors or one per end effector (REW:407). HDMI reads the flags
  one frame ahead of the pose references: its command update at counter t loads the pose references of frames t + F
  (F = 1, 2, 8, 16, 32), sets t to t + 1, and then loads the flags of rows t + F with the new t (CMD:247, 290, 565-568),
  all clamped to the clip's last frame. So the flags that go with the pose reference of frame f are row min(f + 1, T - 1);
  the sidecar records that lead as contact_lead_frames = 1.
* keypoints: HDMI's 16 tracking keypoints (Y:52-61, in the simulator's body order, CMD:61-62) with the link poses and
  velocities the clip stores (CMD:251-254), and the joint and hinge velocities (CMD:164-165, 185, 256, 501): HDMI's reference
  as it reads it, kept to check ndm_gym's forward kinematics (--check). HDMI's reference velocities are not finite
  differences of its poses; ndm_gym's MotionLib differentiates the poses instead.

Clips without wrist joints: open_chair_and_sit_amass (open_foldchair-sit), roll_ball_amass (roll_ball-hand) and push_box
lack left/right_wrist_pitch_joint and left/right_wrist_yaw_joint. HDMI sets a simulator joint that a clip lacks to 0, in
position and velocity (MOT:247-267), and for these three tasks 0 is also the joints' default position (G1:66-79; the -0.4 /
+0.4 wrist yaw default of e.g. T:move_suitcase:27-30 is not set in their configs). HDMI's wrist keypoints are the clip's
stored left/right_wrist_yaw_link poses (CMD:62, 251-254), which are the poses of wrist pitch = yaw = 0: in every frame the
stored wrist_pitch_link and wrist_yaw_link have the identity rotation relative to their parent links, to float precision.
The converter writes the same zeros, lists those joints in the sidecar (absent_joints, absent_joints_note), and --check
confirms that ndm_gym's forward kinematics of the zero-filled joints reproduces the stored wrist keypoints.

The task table (TASKS) is docs/hdmi_task_spec.md §8.1. Conversion checks it against HDMI's task configs, the clip lengths
and contact segments against the table, and every written clip against its source arrays; it fails on any difference.

Usage, from the repository root, in ndm_gym's environment:
  tools/download_hdmi.sh
  python tools/convert_hdmi.py           # data/hdmi -> data/hdmi/clips
  python tools/convert_hdmi.py --check   # ndm_gym's G1 FK of the converted clips vs HDMI's keypoints, per clip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ndm_gym.robots.g1 import G1_JOINT_NAMES  # noqa: E402

HDMI_DIR = os.path.join(REPO, "data", "hdmi")
HDMI_REPO = "https://github.com/LeCAR-Lab/HDMI"
COMMIT = "32282f6dcf26cae70b814d585ceb12cc38aa1b60"
FPS = 50.0
CONTACT_LEAD = 1  # flags of row f + 1 go with the pose reference of frame f (CMD:290, 565-568)
FK_TOL = 1e-4  # m and rad: ndm_gym's FK (float32) vs HDMI's stored keypoints

ROBOT_SPHERE = "g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule"
ROBOT_BOX = "g1_29dof_rubberhand-feet_box-eef_box-body_capsule"
ROBOT_NOHAND = "g1_29dof_nohand-feet_sphere-eef_L-body_capsule"
WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")  # contact_eef_body_name default (Y:109)

# Y:52-61 matched in the simulator's body order (CMD:61, record_motion/x65r1823/meta.json)
KEYPOINTS = (
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
    "left_knee_link", "right_knee_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_ankle_roll_link",
    "right_ankle_roll_link", "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
)  # fmt: skip

# USD of each object asset (active_adaptation/assets/__init__.py:13-33 and the usd_path of objects.py), under data/hdmi
OBJECT_USD = {
    name: f"active_adaptation/assets/objects/{path}.usd"
    for name, path in (
        ("box", "box/box"), ("suitcase", "suitcase/suitcase"), ("foam", "foam/foam"), ("stool", "stool/stool"),
        ("stool_support", "stool/stool_support"), ("bread_box", "bread_box/bread_box"), ("support0", "support/support0"),
        ("support1", "support/support1"), ("wood_board", "wood_board/wood_board"), ("platform0", "platform/platform0"),
        ("platform1", "platform/platform1"), ("wall0", "platform/wall0"), ("ball", "ball/ball"), ("stair", "stair/stair"),
        ("door", "door/door"), ("foldchair", "foldchair/foldchair"),
    )
}  # fmt: skip


class Task(NamedTuple):
    hdmi_name: str  # name
    data_path: str  # command.data_path: a regex, fullmatched against clip directories (MOT:206-217)
    frames: int  # clip length at 50 fps
    robot: str  # robot.robot_type
    object_name: str  # command.object_asset_name: the object's motion body (CMD:383)
    object_body: str  # command.object_body_name: the asset's body that carries the contact targets
    object_joint: str | None  # command.object_joint_name: the hinge, for an articulated object
    extras: tuple[str, ...]  # command.extra_object_names
    eefs: tuple[str, ...]  # command.contact_eef_body_name
    segments: tuple[tuple[tuple[int, int], ...], ...]  # per contact column, the [start, end) frame ranges of its flags


TASKS = {
    "carry_and_place_bread_box": Task("G1CarryAndPlaceBreadBox", "data/motion/data_for_sim/carry_and_place_bread_box-0829", 840, ROBOT_SPHERE,
                                      "bread_box", "bread_box", None, ("support0", "support1"), WRISTS, (((260, 550),),)),
    "carry_box_over_shoulder": Task("G1LiftBoxOverShoulder", "data/motion/data_for_sim/lift_box_over_shoulder", 590, ROBOT_SPHERE,
                                    "suitcase", "suitcase", None, (), WRISTS, (((180, 460),),)),
    "move_foam": Task("G1TrackFoam", "data/motion/data_for_sim/move_foam-climb-0825-2-0_580", 680, ROBOT_BOX,
                      "foam", "foam", None, ("stool_support",), WRISTS, (((240, 480),),)),
    "move_stool-climb": Task("G1TrackStool", "data/motion/data_for_sim/move_stool-climb-0814", 833, ROBOT_BOX,
                             "stool", "stool", None, (), WRISTS, (((240, 530),),)),
    "move_suitcase": Task("G1TrackSuitcase", "data/motion/g1/omomo/sub1_suitcase_011", 472, ROBOT_SPHERE,
                          "suitcase", "suitcase", None, (), WRISTS, (((170, 340),),)),
    "open_door-feet": Task("G1OpenDoorFeet", "data/motion/data_for_sim/open_door-feet-0828", 928, ROBOT_BOX,
                           "door", "door_panel", "door_joint", (), ("left_ankle_roll_link",), (((470, 540),),)),
    "open_foldchair-sit": Task("G1OpenFoldchairSit", "data/motion/g1/multi_motion_0726/open_chair_and_sit_amass", 585, ROBOT_BOX,
                               "foldchair", "seat_link", "foldchair_joint", (), WRISTS, (((120, 255),),)),
    "push_box": Task("G1PushBox", "data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height", 792, ROBOT_NOHAND,
                     "box", "box", None, (), WRISTS, (((200, 542),),)),
    "push_door-hand": Task("G1PushDoorHand", "data/motion/data_for_sim/push_door-hand-0828", 573, ROBOT_SPHERE,
                           "door", "door_panel", "door_joint", (), ("right_wrist_yaw_link",), (((230, 300),),)),
    "roll_ball-hand": Task("G1RollBall", "data/motion/g1/multi_motion_0726/roll_ball_amass", 602, ROBOT_SPHERE,
                           "ball", "ball", None, (), WRISTS, (((0, 602),),)),
    "topple_wood_board_and_cross": Task("G1ToppleWoodBoardAndCross", "data/motion/data_for_sim/topple_wood_board_and_cross", 582, ROBOT_SPHERE,
                                        "wood_board", "wood_board", None, ("platform0", "platform1", "wall0"), WRISTS, (((150, 270),),)),
    "traverse_door": Task("G1TraverseDoor", "data/motion/data_for_sim/traverse_door", 1463, ROBOT_BOX,
                          "door", "door_panel", "door_joint", (), ("right_wrist_yaw_link", "left_ankle_roll_link"), (((270, 370),), ((870, 920),))),
    "truman": Task("G1Truman", "data/motion/data_for_sim/combo-2", 3352, ROBOT_BOX,
                   "stair", "stair", None, (), WRISTS, (((850, 1130),), ((800, 900), (1100, 1130)))),
}  # fmt: skip

CMD_PY = "active_adaptation/envs/mdp/commands/hdmi/command.py"
CONTACT_NOTE = (
    f"contact[i] is HDMI's object_contact row i cast to bool ({CMD_PY}:444-446), aligned with the pose of frame i; one "
    "column is shared by all contact_eef_bodies, else there is one column per end effector, in their order "
    "(active_adaptation/envs/mdp/commands/hdmi/rewards.py:407). HDMI reads the flags one frame ahead of the pose references "
    f"({CMD_PY}:247, 290, 565-568): the flags that go with the pose reference of frame f are row "
    "min(f + contact_lead_frames, frames - 1)."
)
ABSENT_NOTE = (
    "The clip lacks these G1 joints. HDMI sets a simulator joint that a clip lacks to 0, in position and velocity "
    "(active_adaptation/utils/motion.py:247-267), which is also this task's default position of these joints "
    "(active_adaptation/assets/g1.py:66-79, no override in the task config); joint_pos and joint_vel hold those zeros. "
    f"HDMI's wrist keypoints are the clip's stored wrist_yaw_link poses ({CMD_PY}:62, 251-254), and those are the poses "
    "of wrist pitch = yaw = 0: wrist_pitch_link and wrist_yaw_link have the identity rotation relative to their parent "
    "links in every frame."
)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def match_clips(root: str | Path, pattern: str) -> list[Path]:
    """Clip directories whose path relative to ``root`` fullmatches the data_path regex, as HDMI selects them (MOT:206-217)."""
    root, rx = Path(root), re.compile(pattern)
    return sorted(p.parent for p in root.rglob("motion.npz") if rx.fullmatch(p.parent.relative_to(root).as_posix()))


def remap_joints(x: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Columns of ``x`` (T, len(names)) in G1_JOINT_NAMES order, G1 joints missing from ``names`` set to 0 as HDMI sets them
    (MOT:247-267); also the missing joints' names."""
    out = np.zeros((x.shape[0], len(G1_JOINT_NAMES)), dtype=np.float64)
    for i, n in enumerate(G1_JOINT_NAMES):
        if n in names:
            out[:, i] = x[:, names.index(n)]
    return out, [n for n in G1_JOINT_NAMES if n not in names]


def contact_segments(flags: np.ndarray) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Per column of boolean flags (T, C), the [start, end) frame ranges where it is set."""
    out = []
    for col in np.asarray(flags, dtype=bool).T:
        d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
        out.append(tuple(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist())))
    return tuple(out)


def convert(name: str, task: Task, motion: dict, meta: dict) -> tuple[dict[str, np.ndarray], dict]:
    """One task's clip (HDMI's motion.npz arrays and meta.json) as ndm_gym clip arrays and the sidecar's clip fields; raises
    on anything that does not match ``task`` or HDMI's format."""
    bodies, joints = list(meta["body_names"]), list(meta["joint_names"])
    if float(meta["fps"]) != FPS:
        raise ValueError(f"{name}: {meta['fps']} fps; HDMI uses object clips only at {FPS:g} fps (MOT:103-108)")
    T = motion["joint_pos"].shape[0]
    for k, last in (("body_pos_w", 3), ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
        if motion[k].shape != (T, len(bodies), last):
            raise ValueError(f"{name}: {k} has shape {motion[k].shape}, expected ({T}, {len(bodies)}, {last})")
    for k in ("joint_pos", "joint_vel"):
        if motion[k].shape != (T, len(joints)):
            raise ValueError(f"{name}: {k} has shape {motion[k].shape}, expected ({T}, {len(joints)})")
    if T != task.frames:
        raise ValueError(f"{name}: {T} frames, the table says {task.frames}")
    missing = sorted({"pelvis", task.object_name, *task.extras, *KEYPOINTS} - set(bodies))
    if missing:
        raise ValueError(f"{name}: no motion body {missing}")
    others = [j for j in joints if j not in G1_JOINT_NAMES]
    if others != ([task.object_joint] if task.object_joint else []):
        raise ValueError(f"{name}: joints {others} besides the G1's, expected the hinge {task.object_joint}")
    f32 = lambda x: np.asarray(x, dtype=np.float32)  # noqa: E731
    pos, quat = motion["body_pos_w"], motion["body_quat_w"]
    b = bodies.index
    q, absent = remap_joints(motion["joint_pos"], joints)
    qd, _ = remap_joints(motion["joint_vel"], joints)
    kp = [b(k) for k in KEYPOINTS]
    arrays = {
        "root_pos": f32(pos[:, b("pelvis")]), "root_quat_wxyz": f32(quat[:, b("pelvis")]), "joint_pos": f32(q),
        "object_pos": f32(pos[:, b(task.object_name)]), "object_quat_wxyz": f32(quat[:, b(task.object_name)]),
    }  # fmt: skip
    if task.object_joint:
        arrays["object_joint_pos"] = f32(motion["joint_pos"][:, [joints.index(task.object_joint)]])
    contact = np.asarray(motion["object_contact"])
    if contact.ndim != 2 or contact.shape[0] != T or contact.shape[1] not in (1, len(task.eefs)):
        raise ValueError(f"{name}: object_contact has shape {contact.shape}, expected ({T}, 1 or {len(task.eefs)})")
    arrays["contact"] = contact != 0  # torch.tensor(..., dtype=torch.bool): nonzero is set (CMD:446)
    segments = contact_segments(arrays["contact"])
    if segments != task.segments:
        raise ValueError(f"{name}: contact segments {segments}, the table says {task.segments}")
    if task.extras:
        arrays["extra_pos"] = f32(pos[:, [b(e) for e in task.extras]])
        arrays["extra_quat_wxyz"] = f32(quat[:, [b(e) for e in task.extras]])
    arrays.update(keypoint_pos=f32(pos[:, kp]), keypoint_quat_wxyz=f32(quat[:, kp]), keypoint_lin_vel=f32(motion["body_lin_vel_w"][:, kp]),
                  keypoint_ang_vel=f32(motion["body_ang_vel_w"][:, kp]), joint_vel=f32(qd))  # fmt: skip
    if task.object_joint:
        arrays["object_joint_vel"] = f32(motion["joint_vel"][:, [joints.index(task.object_joint)]])
    side = {
        "source": "hdmi", "source_repo": HDMI_REPO, "source_commit": COMMIT, "task": name, "hdmi_task_name": task.hdmi_name,
        "hdmi_task_config": f"cfg/task/G1/hdmi/{name}.yaml", "data_path": task.data_path, "fps": FPS, "frames": T,
        "robot_type": task.robot, "robot_usd": f"active_adaptation/assets/g1/{task.robot}.usd",
        "object_name": task.object_name, "object_body_name": task.object_body, "object_joint_name": task.object_joint,
        "object_usd": OBJECT_USD[task.object_name], "extra_objects": list(task.extras),
        "extra_object_usd": {e: OBJECT_USD[e] for e in task.extras},
        "contact_eef_bodies": list(task.eefs), "contact_columns": int(contact.shape[1]),
        "contact_segments": [[list(s) for s in col] for col in segments], "contact_lead_frames": CONTACT_LEAD,
        "contact_source_dtype": str(contact.dtype), "contact_note": CONTACT_NOTE,
        "root_body": "pelvis", "quat_convention": "wxyz", "joint_names": list(G1_JOINT_NAMES), "source_joint_names": joints,
        "absent_joints": absent, "absent_joints_note": ABSENT_NOTE if absent else None, "keypoint_names": list(KEYPOINTS),
        "paths_note": "paths are relative to data/hdmi, HDMI's repository layout",
    }  # fmt: skip
    return arrays, side


def write_clip(out_dir: str | Path, name: str, arrays: dict[str, np.ndarray], side: dict) -> None:
    out = {**arrays, "fps": np.asarray(FPS, dtype=np.float32)}
    np.savez(os.path.join(out_dir, f"{name}.npz"), **out)
    side = {**side, "arrays": {k: [list(v.shape), str(v.dtype)] for k, v in out.items()}}
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        f.write(json.dumps(side, indent=1) + "\n")


def check_configs(hdmi_dir: str | Path, robot_links: list[str]) -> None:
    """TASKS and KEYPOINTS against HDMI's task configs (cfg/task/G1/hdmi/<task>.yaml over cfg/task/base/hdmi-base.yaml)."""
    import yaml

    cfg_dir = Path(hdmi_dir) / "cfg" / "task"
    base = yaml.safe_load((cfg_dir / "base" / "hdmi-base.yaml").read_text())["command"]
    pats = [re.compile(p) for p in base["tracking_keypoint_names"]]
    if {n for n in robot_links if any(p.fullmatch(n) for p in pats)} != set(KEYPOINTS):
        raise ValueError(f"KEYPOINTS differ from Y's tracking_keypoint_names {base['tracking_keypoint_names']}")
    errors = []
    for name, t in TASKS.items():
        cfg = yaml.safe_load((cfg_dir / "G1" / "hdmi" / f"{name}.yaml").read_text())
        cmd = cfg["command"]
        got = Task(cfg["name"], cmd["data_path"], t.frames, cfg["robot"]["robot_type"], cmd["object_asset_name"], cmd["object_body_name"],
                   cmd.get("object_joint_name"), tuple(cmd.get("extra_object_names", base["extra_object_names"])),
                   tuple(cmd.get("contact_eef_body_name", base["contact_eef_body_name"])), t.segments)  # fmt: skip
        errors += [f"{name}.{k}: table {a!r}, config {g!r}" for k, a, g in zip(Task._fields, t, got) if a != g]
    if errors:
        raise ValueError("TASKS differ from HDMI's task configs:\n  " + "\n  ".join(errors))


def convert_all(hdmi_dir: str | Path, out_dir: str | Path) -> None:
    import torch

    from ndm_gym.motion.sources import hoi

    hdmi_dir = Path(hdmi_dir)
    with open(hdmi_dir / "COMMIT") as f:
        if f.read().strip() != COMMIT:
            raise ValueError(f"{hdmi_dir} is not HDMI @ {COMMIT}; run tools/download_hdmi.sh")
    sources = {}
    for name, task in TASKS.items():
        dirs = match_clips(hdmi_dir, task.data_path)
        if len(dirs) != 1:  # RobotObjectTracking takes exactly one clip (CMD:443)
            raise ValueError(f"{name}: data_path {task.data_path} matches {len(dirs)} clips in {hdmi_dir}")
        sources[name] = dirs[0]
    with open(sources["truman"] / "meta.json") as f:
        check_configs(hdmi_dir, [b for b in json.load(f)["body_names"] if b not in OBJECT_USD])
    os.makedirs(out_dir, exist_ok=True)
    for name, task in TASKS.items():
        src = sources[name]
        with open(src / "meta.json") as f:
            meta = json.load(f)
        with np.load(src / "motion.npz", allow_pickle=False) as z:
            motion = {k: z[k] for k in z.files}
        for usd in [f"active_adaptation/assets/g1/{task.robot}.usd"] + [OBJECT_USD[o] for o in (task.object_name, *task.extras)]:
            if not (hdmi_dir / usd).is_file():
                raise FileNotFoundError(f"{name}: {hdmi_dir / usd} is missing; run tools/download_hdmi.sh")
        arrays, side = convert(name, task, motion, meta)
        rel = src.relative_to(hdmi_dir).as_posix()
        side["source_files"] = {f"{rel}/{f}": sha256(src / f) for f in ("motion.npz", "meta.json")}
        write_clip(out_dir, name, arrays, side)
        clip = hoi.load_clip(os.path.join(out_dir, f"{name}.npz"))  # the written clip is the source, value for value
        for got, key in ((clip.root_p, "root_pos"), (clip.root_quat, "root_quat_wxyz"), (clip.joints, "joint_pos"), (clip.obj_p, "object_pos"),
                         (clip.obj_quat, "object_quat_wxyz"), (clip.obj_joint, "object_joint_pos"), (clip.contact, "contact"),
                         (clip.extra_p, "extra_pos"), (clip.extra_quat, "extra_quat_wxyz")):  # fmt: skip
            if (got is None) != (key not in arrays) or (got is not None and not torch.equal(got, torch.from_numpy(arrays[key].astype(np.float32)))):
                raise RuntimeError(f"{name}: {key} does not read back as converted")
        print(f"{name:28s} {rel}  {task.frames} frames, joints absent: {side['absent_joints'] or '-'}")
    stray = sorted(({p.stem for p in Path(out_dir).glob("*.npz")} | {p.stem for p in Path(out_dir).glob("*.json")}) - set(TASKS))
    if stray:
        raise RuntimeError(f"{out_dir} holds clips that are not HDMI tasks: {stray}")
    print(f"{len(TASKS)} clips in {out_dir}")


def check(clips_dir: str | Path) -> list[dict]:
    """Per task: the converted clip loaded into ndm_gym's MotionLib with HDMI's keypoints as the tracked bodies. Its 50 Hz
    arrays must be the clip's own (no frame dropped or interpolated), the G1 forward kinematics of root and joints must
    reproduce HDMI's stored keypoint positions and rotations within FK_TOL, and the contact segments must be the table's.
    Returns the maximum errors; raises on any failure."""
    import torch

    from ndm_gym.core.rotations import quat_to_mat, so3_log
    from ndm_gym.motion.library import MotionLib
    from ndm_gym.motion.sources import hoi
    from ndm_gym.robots import g1

    kin = g1.kinematics("cpu", tracked=list(KEYPOINTS))
    wrists = [KEYPOINTS.index(k) for k in WRISTS]
    rows, errors = [], []
    for name, task in TASKS.items():
        path = os.path.join(clips_dir, f"{name}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} is missing; run tools/download_hdmi.sh, then python tools/convert_hdmi.py")
        clip = hoi.load_clip(path)
        m = MotionLib([clip], kin, "cpu")
        with np.load(path) as z:
            kp, kq = torch.from_numpy(z["keypoint_pos"]), quat_to_mat(torch.from_numpy(z["keypoint_quat_wxyz"]))
        pairs = [("root_pos", m.root_p, clip.root_p), ("joint_pos", m.q, clip.joints), ("object_pos", m.obj_p, clip.obj_p),
                 ("contact", m.contact, clip.contact)]  # fmt: skip
        pairs += [("object_joint_pos", m.obj_joint, clip.obj_joint)] if clip.obj_joint is not None else []
        pairs += [("extra_pos", m.extra_p, clip.extra_p)] if clip.extra_p is not None else []
        differ = [k for k, a, b in pairs if not torch.equal(a, b)]
        dp = (m.body_p - kp).norm(dim=-1)
        dr = so3_log(m.body_R.transpose(-1, -2) @ kq).norm(dim=-1)
        segs = contact_segments(m.contact.numpy() > 0.5)
        row = {"task": name, "frames": m.num_frames, "absent_joints": len(clip.meta["absent_joints"]), "fk_pos_max_m": float(dp.max()),
               "fk_rot_max_rad": float(dr.max()), "fk_wrist_pos_max_m": float(dp[:, wrists].max()), "contact_segments": segs}  # fmt: skip
        rows.append(row)
        if m.num_frames != task.frames or differ:
            errors.append(f"{name}: {m.num_frames} frames (expected {task.frames}); differing from the clip: {differ}")
        if row["fk_pos_max_m"] > FK_TOL or row["fk_rot_max_rad"] > FK_TOL:
            errors.append(f"{name}: FK error {row['fk_pos_max_m']:.2e} m, {row['fk_rot_max_rad']:.2e} rad (tolerance {FK_TOL:g})")
        if segs != task.segments:
            errors.append(f"{name}: contact segments {segs}, the table says {task.segments}")
    if errors:
        raise AssertionError("\n".join(errors))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hdmi-dir", default=HDMI_DIR, help="HDMI files as tools/download_hdmi.sh places them")
    ap.add_argument("--out-dir", default=None, help="converted clips (default: <hdmi-dir>/clips)")
    ap.add_argument("--check", action="store_true", help="check the converted clips with ndm_gym's G1 forward kinematics")
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(a.hdmi_dir, "clips")
    if not a.check:
        convert_all(a.hdmi_dir, out_dir)
        return
    print(f"{'task':28s} {'frames':>6s} {'absent':>6s} {'FK pos max (m)':>14s} {'wrist pos max':>13s} {'FK rot max (rad)':>16s}  contact segments")
    for r in check(out_dir):
        print(f"{r['task']:28s} {r['frames']:6d} {r['absent_joints']:6d} {r['fk_pos_max_m']:14.2e} {r['fk_wrist_pos_max_m']:13.2e} "
              f"{r['fk_rot_max_rad']:16.2e}  {r['contact_segments']}")  # fmt: skip


if __name__ == "__main__":
    main()
