"""ACT's simulated ALOHA scenes (github.com/tonyzhaozh/act, MIT; Zhao et al., "Learning Fine-Grained Bimanual Manipulation
with Low-Cost Hardware", RSS 2023) for demonstration tracking: two ViperX 300 arms, six joints and two finger slides each, on
a table with a cube (Transfer Cube) or a peg and a socket (Insertion). The scenes are ACT's MJCF files as they are, under a
local copy of its assets (tools/make_aloha_demos.py): their 16 position actuators with ACT's gains and force ranges, their
joints' friction and their contact parameters. ACT controls a gripper by one value, its two fingers mirrored; here too one
action drives both (``mirror``).

ACT scores a task by the geoms in contact (``get_reward`` of its TransferCubeTask and InsertionTask: stages 0 to 4, an
episode a success if its highest reward is 4). ``CONTACTS`` names the pairs it reads, and the evaluation reads the
simulator's contacts for them (sim.rigid.robomimic_sim, tasks.robomimic_tracking.staged_reward). Training's log has the last
stage as a function of the state instead, a proxy set on the benchmark's demonstrations (every frame with ACT's reward of
4 passes; 99.4 % and 98.8 % of the frames agree):

- Transfer Cube, "the cube in the left gripper and off the table": the cube's centre above ``TRANSFER_Z`` (it rests at
  0.02 m), within ``TRANSFER_REACH`` of the left gripper's grasp point (the point ``GRASP_REACH`` along the gripper from
  its link, where ACT welds its end-effector target), and the gripper's opening below ``TRANSFER_OPENING`` (0.085 m on
  the cube, 0.11 m open).
- Insertion, "the peg touching the pin": the peg's centre in the socket's frame within ``INSERTION_X`` along the socket's
  axis (the peg's end at the pin's: 0.04 + 0.06 m) and within ``INSERTION_LATERAL`` across it.
"""

from __future__ import annotations

import os

import mujoco

from .robomimic_scenes import ARM_SCALE, FINGER_SCALE, RobomimicScene

CONTROL_DT = 0.02  # ACT's DT (50 Hz)
HOLD = 25  # frames of the last state appended to a demonstration (0.5 s)
TASKS = {
    "aloha_transfer_cube": {"scene": "bimanual_viperx_transfer_cube.xml", "data": "transfer_cube", "objects": ("red_box_joint",),
                            "success": "aloha_transfer"},
    "aloha_insertion": {"scene": "bimanual_viperx_insertion.xml", "data": "insertion", "objects": ("red_peg_joint", "blue_socket_joint"),
                        "success": "aloha_insertion"},
}  # fmt: skip
ARMS = ("vx300s_left", "vx300s_right")
# ACT's reward reads these geoms' contacts: of the left arm its left finger, of the right arm its right finger
LEFT_FINGER, RIGHT_FINGER, SOCKET = "vx300s_left/10_left_gripper_finger", "vx300s_right/10_right_gripper_finger", ("socket-1", "socket-2", "socket-3", "socket-4")
CONTACTS = {  # per task, the groups of geom pairs in the order tasks.robomimic_tracking.staged_reward takes them
    "aloha_transfer_cube": ((("red_box", LEFT_FINGER),), (("red_box", RIGHT_FINGER),), (("red_box", "table"),)),
    "aloha_insertion": (tuple((s, LEFT_FINGER) for s in SOCKET), (("red_peg", RIGHT_FINGER),), (("red_peg", "table"),),
                        tuple((s, "table") for s in SOCKET), tuple(("red_peg", s) for s in SOCKET), (("red_peg", "pin"),)),
}  # fmt: skip
MAX_REWARD = 4
KEYPOINTS = ("gripper_link", "left_finger_link", "right_finger_link")  # per arm: its end effector's link and its fingers'
GRASP_REACH, FINGER_REACH = 0.1347, 0.0687  # from a gripper's link along it: its grasp point, its fingers' links (m)
GRASP_RATIO = GRASP_REACH / FINGER_REACH
TRANSFER_Z, TRANSFER_REACH, TRANSFER_OPENING = 0.03, 0.03, 0.09
INSERTION_X, INSERTION_LATERAL = 0.102, 0.014


def load_scene(assets_root: str, task: str) -> mujoco.MjModel:
    """ACT's joint-controlled scene of ``task`` from ``assets_root`` (tools/make_aloha_demos.py's ``act_assets``)."""
    if task not in TASKS:
        raise ValueError(f"task {task!r}, one of {sorted(TASKS)}")
    path = os.path.join(assets_root, TASKS[task]["scene"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: ACT's scenes are not there (python tools/make_aloha_demos.py)")
    return mujoco.MjModel.from_xml_path(path)


def aloha_scene(m: mujoco.MjModel, task: str) -> RobomimicScene:
    """What the tracking task reads of a compiled ALOHA scene: the left arm first, each gripper's fingers one mirrored pair."""
    if task not in TASKS:
        raise ValueError(f"task {task!r}, one of {sorted(TASKS)}")
    name = lambda kind, i: mujoco.mj_id2name(m, kind, i)  # noqa: E731
    act_q, act_d, lo, hi, scale, arm, joints = [], [], [], [], [], [], []
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError(f"actuator {a}: not a joint transmission")
        j = int(m.actuator_trnid[a, 0])
        joint = name(mujoco.mjtObj.mjOBJ_JOINT, j)
        if not joint.startswith(ARMS) or not m.jnt_limited[j]:
            raise ValueError(f"actuator {a} drives {joint}: not a limited joint of {ARMS}")
        is_arm = not joint.endswith("_finger")
        act_q.append(int(m.jnt_qposadr[j])), act_d.append(int(m.jnt_dofadr[j])), arm.append(is_arm), joints.append(joint)
        lo.append(float(m.jnt_range[j, 0])), hi.append(float(m.jnt_range[j, 1])), scale.append(ARM_SCALE if is_arm else FINGER_SCALE)
    mirror = []
    for side in ARMS:  # the two slides of a gripper open in opposite directions
        pair = (f"{side}/left_finger", f"{side}/right_finger")
        if any(p not in joints for p in pair):
            raise ValueError(f"the scene has no actuators on {pair}")
        mirror.append((joints.index(pair[0]), joints.index(pair[1])))
    if sum(arm) != 6 * len(ARMS) or joints.index(f"{ARMS[0]}/waist") > joints.index(f"{ARMS[1]}/waist"):
        raise ValueError("the scene's actuators are not ALOHA's: six joints and two fingers per arm, the left arm first")
    bodies = []
    for side in ARMS:
        for k in KEYPOINTS:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}/{k}")
            if b < 0:
                raise ValueError(f"the scene has no body {side}/{k}")
            bodies.append(int(b))
    free = {name(mujoco.mjtObj.mjOBJ_JOINT, j): (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])) for j in range(m.njnt)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}  # fmt: skip
    missing = [o for o in TASKS[task]["objects"] if o not in free]
    if missing:
        raise ValueError(f"the scene has no free joint {missing} of task {task!r}")
    objects = tuple(free[o] for o in TASKS[task]["objects"])
    geom = lambda g: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)  # noqa: E731
    absent = sorted({g for group in CONTACTS[task] for pair in group for g in pair if geom(g) < 0})
    if absent:
        raise ValueError(f"the scene has no geoms {absent} of ACT's reward")
    contacts = tuple(tuple((geom(a), geom(b)) for a, b in group) for group in CONTACTS[task])
    return RobomimicScene(task=task, nq=m.nq, nv=m.nv, nu=m.nu, act_q=tuple(act_q), act_d=tuple(act_d), act_lo=tuple(lo), act_hi=tuple(hi),
                          act_scale=tuple(scale), arm=tuple(arm), keypoint_bodies=tuple(bodies), objects=objects,
                          others=tuple(v for v in free.values() if v not in objects), success_kind=TASKS[task]["success"],
                          control_dt=CONTROL_DT, mirror=tuple(mirror), contacts=contacts)  # fmt: skip

