"""robomimic's scenes (robosuite 1.5.0) with the Franka Panda arms as joint position servos, for demonstration tracking.

A demonstration's scene is its ``model_file``, the MJCF robosuite wrote when it was recorded, with the meshes and textures of
robosuite's package resolved under a local copy of its assets (tools/download_robomimic.py). robosuite drives the arm's
torque actuators (``robotN_torq_jK``, gain 1, no bias) by operational-space control with gravity compensation; here each
becomes a position servo, force = kp (target - q) - kd qdot, clipped to the torque actuator's own range (80 N m on joints 1-5,
12 N m on 6 and 7), and the robots' bodies are compensated for gravity (``gravcomp``), as robosuite's controller compensates
it. The grippers keep their position servos on the finger slides. robosuite itself is not imported.

``RobomimicScene`` names what the task reads of a compiled scene: the actuated joints in the actuators' order (each arm's
seven, then its gripper's two fingers), the tracked keypoints (each gripper's end effector and fingers), the free joints of
the objects the task tracks and of the others, and the task's success test (robosuite's ``_check_success``, as a function of
the objects' positions).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import numpy as np

# each arm joint's servo gain (N m / rad) and damping ratio, joints 1-7 (the forearm and the wrist lighter, as their torques)
ARM_KP = (600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0)
ARM_DAMPING = 1.0
# the largest offset of a position target from the reference's: the arm's joints (rad), the gripper's fingers (m)
ARM_SCALE, FINGER_SCALE = 0.25, 0.02
KEYPOINT_BODIES = ("eef", "leftfinger", "rightfinger")  # per gripper: gripper<k>_right_<name>

# per task: the tracked objects' free joints (Transport's lid too: the arms open the start bin and set its lid down before the
# hand-over; untracked, a lid left in the gripper keeps the arm from the hammer), and robosuite's success test's constants
TASKS = {
    "lift": {"objects": ("cube_joint0",), "success": "lift"},
    "square": {"objects": ("SquareNut_joint0",), "success": "square"},
    "transport": {"objects": ("payload_joint0", "trash_joint0", "transport_start_bin_lid_joint0"), "success": "transport",
                  "bins": ("transport_target_bin_joint0", "transport_trash_bin_joint0")},
}
LIFT_TABLE_TOP, LIFT_HEIGHT = 0.8, 0.04  # robosuite Lift: the cube above the table's top by 4 cm
SQUARE_TABLE_TOP, SQUARE_PEG, SQUARE_XY, SQUARE_Z = 0.82, "peg1", 0.03, 0.05  # NutAssembly: the square nut on peg1
# TwoArmTransport: the payload in the target bin and the trash in the trash bin. robosuite tests each object's contact with
# its bin's base; here, as a function of the state, the object's origin inside its bin (in the bin's frame: within the base's
# half width, above the base's top and below the walls' top: the bins of TransportGroup, 0.3 x 0.3 x 0.15 m)
BIN_HALF_XY, BIN_FLOOR, BIN_TOP = 0.145, -0.065, 0.075


def resolve_assets(xml: str, assets_root: str) -> str:
    """``xml`` with the file paths into robosuite's package (``.../robosuite/models/assets/...``) under ``assets_root``."""
    return re.sub(r'file="[^"]*?robosuite/models/assets/', f'file="{assets_root}/', xml)


def build_spec(xml: str, assets_root: str, kp=ARM_KP, damping: float = ARM_DAMPING, gravcomp: bool = True) -> mujoco.MjSpec:
    """The scene's spec with the arms' torque actuators as position servos within their torque ranges and the robots' bodies
    compensated for gravity. The servo's damping is ``2 damping sqrt(kp)`` (a unit inertia's critical damping at 1)."""
    spec = mujoco.MjSpec.from_string(resolve_assets(xml, assets_root))
    n = 0
    for a in spec.actuators:
        m = re.fullmatch(r"robot\d+_torq_j(\d)", a.name)
        if not m:
            continue
        k = float(kp[int(m.group(1)) - 1])
        lo, hi = a.ctrlrange
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = k
        a.biasprm[0:3] = [0.0, -k, -2.0 * damping * np.sqrt(k)]
        a.forcelimited, a.forcerange = mujoco.mjtLimited.mjLIMITED_TRUE, [lo, hi]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        n += 1
    if n == 0 or n % 7:
        raise ValueError(f"the scene has {n} arm torque actuators (robosuite's robotN_torq_jK, 7 per Panda)")
    if gravcomp:
        for b in spec.bodies:
            if re.match(r"(robot|gripper)\d+_", b.name):
                b.gravcomp = 1.0
    return spec


def load_scene(xml: str, assets_root: str, **kw) -> mujoco.MjModel:
    return build_spec(xml, assets_root, **kw).compile()


@dataclass(frozen=True)
class RobomimicScene:
    """What the task reads of a compiled scene (``from_model``). Indices address qpos (``*_q``) and qvel (``*_d``)."""

    task: str
    nq: int
    nv: int
    nu: int
    act_q: tuple[int, ...]  # each actuator's joint, qpos address
    act_d: tuple[int, ...]
    act_lo: tuple[float, ...]  # its joint's range (the targets' clip)
    act_hi: tuple[float, ...]
    act_scale: tuple[float, ...]  # the largest offset of its target from the reference's
    arm: tuple[bool, ...]  # an arm joint (else a gripper finger)
    keypoint_bodies: tuple[int, ...]  # body ids, per gripper its end effector and fingers
    objects: tuple[tuple[int, int], ...]  # the tracked objects' free joints (qpos, qvel addresses)
    others: tuple[tuple[int, int], ...]  # the other free joints
    success_kind: str
    success_xy: tuple[float, float] = (0.0, 0.0)  # Square: its peg's position
    bins: tuple[tuple[int, int], ...] = ()  # Transport: the bins of the first tracked objects (the payload's, the trash's)

    @staticmethod
    def from_model(m: mujoco.MjModel, task: str) -> RobomimicScene:
        if task not in TASKS:
            raise ValueError(f"task {task!r}, one of {sorted(TASKS)}")
        name = lambda kind, i: mujoco.mj_id2name(m, kind, i)  # noqa: E731
        act_q, act_d, lo, hi, scale, arm = [], [], [], [], [], []
        for a in range(m.nu):
            if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(f"actuator {name(mujoco.mjtObj.mjOBJ_ACTUATOR, a)}: not a joint transmission")
            j = int(m.actuator_trnid[a, 0])
            is_arm = bool(re.fullmatch(r"robot\d+_torq_j\d", name(mujoco.mjtObj.mjOBJ_ACTUATOR, a)))
            act_q.append(int(m.jnt_qposadr[j])), act_d.append(int(m.jnt_dofadr[j])), arm.append(is_arm)
            r = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
            lo.append(float(r[0])), hi.append(float(r[1])), scale.append(ARM_SCALE if is_arm else FINGER_SCALE)
        grippers = sorted({int(g) for g in re.findall(r"gripper(\d+)_right_eef", " ".join(name(mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)))})
        bodies = []
        for g in grippers:
            for k in KEYPOINT_BODIES:
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"gripper{g}_right_{k}")
                if b < 0:
                    raise ValueError(f"the scene has no body gripper{g}_right_{k}")
                bodies.append(int(b))
        free = {name(mujoco.mjtObj.mjOBJ_JOINT, j): (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])) for j in range(m.njnt)
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}  # fmt: skip
        missing = [o for o in TASKS[task]["objects"] if o not in free]
        if missing:
            raise ValueError(f"the scene has no free joint {missing} of task {task!r}")
        objects = tuple(free[o] for o in TASKS[task]["objects"])
        xy, bins = (0.0, 0.0), ()
        if "bins" in TASKS[task]:
            missing = [o for o in TASKS[task]["bins"] if o not in free]
            if missing:
                raise ValueError(f"the scene has no free joint {missing} of task {task!r}")
            bins = tuple(free[o] for o in TASKS[task]["bins"])
        if TASKS[task]["success"] == "square":
            peg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, SQUARE_PEG)
            xy = tuple(float(x) for x in m.body_pos[peg, :2])
        return RobomimicScene(task=task, nq=m.nq, nv=m.nv, nu=m.nu, act_q=tuple(act_q), act_d=tuple(act_d), act_lo=tuple(lo), act_hi=tuple(hi),
                              act_scale=tuple(scale), arm=tuple(arm), keypoint_bodies=tuple(bodies), objects=objects,
                              others=tuple(v for k, v in free.items() if v not in objects), success_kind=TASKS[task]["success"],
                              success_xy=xy, bins=bins)  # fmt: skip

    @property
    def num_arms(self) -> int:
        return sum(self.arm) // 7


def contact_capacity(m: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, every: int = 5) -> tuple[int, int]:
    """MuJoCo-Warp's per-world contact and constraint capacities (``nconmax``, ``njmax``) for a reference (T, nq), (T, nv): twice
    the most contacts and constraint rows MuJoCo finds at its frames, at least 64 and 512."""
    d = mujoco.MjData(m)
    ncon = nefc = 0
    for t in range(0, len(qpos), every):
        d.qpos[:], d.qvel[:] = qpos[t], qvel[t]
        mujoco.mj_forward(m, d)
        ncon, nefc = max(ncon, d.ncon), max(nefc, d.nefc)
    return max(64, 2 * ncon), max(512, 2 * nefc)
