"""robomimic's scenes (robosuite 1.5.0) with the Franka Panda arms as joint position servos, for demonstration tracking.

A demonstration's scene is its ``model_file``, the MJCF robosuite wrote when it was recorded, with the meshes and textures of
robosuite's package resolved under a local copy of its assets (tools/download_robomimic.py). robosuite drives the arm's
torque actuators (``robotN_torq_jK``, gain 1, no bias) by operational-space control with gravity compensation; here each
becomes a position servo, force = kp (target - q) - kd qdot, clipped to the torque actuator's own range (80 N m on joints 1-5,
12 N m on 6 and 7), and the robots' bodies are compensated for gravity (``gravcomp``), as robosuite's controller compensates
it. The grippers keep their position servos on the finger slides. robosuite itself is not imported.

``RobomimicScene`` names what the task reads of a compiled scene: the actuated joints in the actuators' order (each arm's
seven, then its gripper's two fingers), the tracked keypoints (each gripper's end effector and fingers), the free joints of
the objects the task tracks and of the others, and what robosuite's success test of the task (``_check_success`` at the
pinned commit) reads of the simulator: bodies' and grip sites' positions, and contacts between geoms.
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
# hand-over; untracked, a lid left in the gripper keeps the arm from the hammer) and the success test's kind
TASKS = {
    "lift": {"objects": ("cube_joint0",), "success": "lift"},
    "square": {"objects": ("SquareNut_joint0",), "success": "square"},
    "transport": {"objects": ("payload_joint0", "trash_joint0", "transport_start_bin_lid_joint0"), "success": "transport"},
}
# robosuite's success tests (_check_success at the pinned commit) and what they read. Lift: the cube's body above the table's
# top (its table_offset, 0.8 m) by more than 4 cm.
LIFT_BODY, LIFT_TABLE_TOP, LIFT_HEIGHT = "cube_joint0", 0.8, 0.04
# NutAssemblySquare (single_object_mode 2): a nut on its own peg (on_peg: its body within 3 cm of the peg's in x and y, below
# the table's top, 0.82 m, plus 5 cm) and released (1 - tanh(10 d) < 0.6, d its body's distance to the nearer grip site);
# success if one is. The round nut stays in the scene, cleared off the table.
SQUARE_NUTS = (("SquareNut_joint0", "peg1"), ("RoundNut_joint0", "peg2"))
SQUARE_TABLE_TOP, SQUARE_XY, SQUARE_Z, SQUARE_REACH = 0.82, 0.03, 0.05, 0.6
# TwoArmTransport: the payload touches the target bin's base and the trash the trash bin's (TransportGroup's
# payload_in_target_bin and trash_in_trash_bin: sim_utils.check_contact between an object's contact geoms, those of group 0,
# and the base geom, over MuJoCo's contact list)
TRANSPORT_CONTACTS = (("payload_joint0", "transport_target_bin_base"), ("trash_joint0", "transport_trash_bin_base"))


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
    ngeom: int = 0
    grip_sites: tuple[int, ...] = ()  # each gripper's grip site (robosuite's eef_site_id), site ids
    success_bodies: tuple[int, ...] = ()  # Lift: the cube's body; Square: each nut's body
    pegs: tuple[int, ...] = ()  # Square: each nut's peg, body ids
    contact_pairs: tuple[tuple[tuple[int, ...], int], ...] = ()  # Transport: per test, an object's contact geoms and its bin's base geom

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
            if not m.jnt_limited[j]:
                raise ValueError(f"actuator {name(mujoco.mjtObj.mjOBJ_ACTUATOR, a)}: its joint has no range (the targets' clip)")
            lo.append(float(m.jnt_range[j, 0])), hi.append(float(m.jnt_range[j, 1])), scale.append(ARM_SCALE if is_arm else FINGER_SCALE)
        body_names = " ".join(name(mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody))
        grippers = sorted({int(g) for g in re.findall(r"gripper(\d+)_right_eef", body_names)})
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
        kind, success = TASKS[task]["success"], {}
        if kind == "lift":
            success["success_bodies"] = (_free_body(m, LIFT_BODY),)
        elif kind == "square":
            success["success_bodies"] = tuple(_free_body(m, nut) for nut, _ in SQUARE_NUTS)
            success["pegs"] = tuple(_id(m, mujoco.mjtObj.mjOBJ_BODY, peg) for _, peg in SQUARE_NUTS)
        else:
            success["contact_pairs"] = tuple((_contact_geoms(m, _free_body(m, o)), _id(m, mujoco.mjtObj.mjOBJ_GEOM, base))
                                             for o, base in TRANSPORT_CONTACTS)  # fmt: skip
        sites = tuple(_id(m, mujoco.mjtObj.mjOBJ_SITE, f"gripper{g}_right_grip_site") for g in grippers)
        return RobomimicScene(task=task, nq=m.nq, nv=m.nv, nu=m.nu, act_q=tuple(act_q), act_d=tuple(act_d), act_lo=tuple(lo), act_hi=tuple(hi),
                              act_scale=tuple(scale), arm=tuple(arm), keypoint_bodies=tuple(bodies), objects=objects,
                              others=tuple(v for k, v in free.items() if v not in objects), success_kind=kind, ngeom=m.ngeom,
                              grip_sites=sites, **success)  # fmt: skip

    @property
    def num_arms(self) -> int:
        return sum(self.arm) // 7


def _id(m: mujoco.MjModel, kind, name: str) -> int:
    i = mujoco.mj_name2id(m, kind, name)
    if i < 0:
        raise ValueError(f"the scene has no {kind.name[6:].lower()} {name!r}")
    return int(i)


def _free_body(m: mujoco.MjModel, joint: str) -> int:
    """The body of a free joint: an object's root body (robosuite's ``root_body``, whose position its tests read)."""
    j = _id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"the joint {joint!r} is not a free joint")
    return int(m.jnt_bodyid[j])


def _contact_geoms(m: mujoco.MjModel, root: int) -> tuple[int, ...]:
    """robosuite's ``contact_geoms`` of the object whose root body is ``root``: the geoms of group 0 in its bodies."""
    geoms = tuple(g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root and m.geom_group[g] == 0)
    if not geoms:
        raise ValueError(f"the body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root)!r} has no geom of group 0")
    return geoms


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
