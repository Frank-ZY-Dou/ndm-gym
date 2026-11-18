"""Unitree Go2 (MuJoCo Menagerie's ``unitree_go2``) with joint position servos, for quadruped motion tracking.

The trunk is a free body and each leg a chain of three hinges (hip abduction, thigh, calf), so the robot's state is
MuJoCo's: ``qpos`` (19: the trunk's position and quaternion w, x, y, z, then the 12 joints in ``JOINT_NAMES``' order) and
``qvel`` (18: the trunk's world linear velocity and body-frame angular velocity, then the joints'). Menagerie's motors
take torques; ``load_scene`` turns them into position servos about the policy's targets (gains ``KP``, ``KD``) with the
motors' torque limits, sets the joints' damping to ``JOINT_DAMPING`` (Unitree's own model's, in place of Menagerie's 2),
and zeroes the geoms' margins, which MuJoCo-Warp refuses with multi-CCD. ``Go2Kinematics`` is the
differentiable forward kinematics of the tracked keypoints (the trunk and the four feet), built from the compiled model,
shared by the reference, the simulator's states and the world model's predictions.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
import torch

from ..core.kinematics import mat_mat, mat_vec
from ..core.rotations import axis_angle_to_mat, quat_to_mat

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_PATH = os.path.join(REPO, "data", "menagerie", "unitree_go2", "scene.xml")
LEGS = ("FL", "FR", "RL", "RR")
JOINT_NAMES = [f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")]
NUM_J = len(JOINT_NAMES)
NQ, NV = 7 + NUM_J, 6 + NUM_J  # qpos and qvel: the trunk's free joint, then the joints
TRUNK = "base"
FEET = list(LEGS)  # the foot spheres' geoms
NUM_KP = 1 + len(FEET)  # tracked keypoints: the trunk's origin, then the feet's centres

SIM_DT = 0.002  # the physics step
KP, KD = 40.0, 1.0  # the servos' stiffness (N m / rad) and damping (N m s / rad), on top of the joints' damping
# The joints' damping (N m s / rad). Menagerie's model has 2, where Unitree's own MuJoCo model of the Go2 (unitree_mujoco,
# unitree_robots/go2/go2.xml), otherwise equal, has 0.1: at 2, the reference's joint speeds (up to 16 rad/s on the runs) take
# more than the hip and thigh motors' 23.7 N m to overcome, and the robot falls behind the fast clips.
JOINT_DAMPING = 0.1
# the largest offset of a position target from the reference pose (rad): hip abduction, thigh, calf, per leg
ACTION_SCALE = [0.3, 0.5, 0.5] * len(LEGS)


def load_scene(path: str = SCENE_PATH, kp: float = KP, kd: float = KD, timestep: float = SIM_DT,
               joint_damping: float | None = JOINT_DAMPING) -> mujoco.MjModel:
    """Menagerie's Go2 scene (the robot and the floor) with each motor turned into a position servo of gains ``kp`` and
    ``kd``: its control a joint position within the joint's range, its force limited to the motor's torque range. The
    geoms' margins are zeroed (MuJoCo-Warp does not take a non-zero margin with multi-CCD)."""
    spec = mujoco.MjSpec.from_file(path)
    for a in spec.actuators:
        torque = np.array(a.ctrlrange, dtype=np.float64)
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[:3] = [kp, 0.0, 0.0]
        a.biasprm[:3] = [0.0, -kp, -kd]
        a.forcerange, a.forcelimited = torque, mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange, a.ctrllimited = np.array(spec.joint(a.target).range, dtype=np.float64), mujoco.mjtLimited.mjLIMITED_TRUE
    model = spec.compile()
    model.geom_margin[:] = 0.0
    model.opt.timestep = timestep
    if joint_damping is not None:  # None: the model's own (Menagerie: 2)
        model.dof_damping[6:] = joint_damping
    if [model.joint(int(model.actuator_trnid[a, 0])).name for a in range(model.nu)] != JOINT_NAMES or model.nq != NQ or model.nv != NV:
        raise ValueError(f"{path}: expected the Go2's free trunk and its 12 actuated joints in the order {JOINT_NAMES}")
    return model


class Go2Kinematics:
    """Keypoint positions and the trunk's rotation from ``qpos``, for a compiled model ``model``."""

    def __init__(self, model: mujoco.MjModel, device: str):
        self.num_joints, self.num_keypoints = NUM_J, NUM_KP  # the sizes hand_fast.Collector reads
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TRUNK)
        geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FEET]
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        if min([trunk] + geom_ids + joint_ids) < 0 or model.jnt_type[model.body_jntadr[trunk]] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("the model lacks the Go2's free trunk, a joint or a foot")
        column = {j: i for i, j in enumerate(joint_ids)}
        chain = set()  # the bodies between the feet and the trunk
        for g in geom_ids:
            b = int(model.geom_bodyid[g])
            while b != trunk:
                if b == 0:
                    raise ValueError("a foot does not hang from the trunk")
                chain.add(b)
                b = int(model.body_parentid[b])
        local, self.steps = {trunk: -1}, []
        for b in sorted(chain):  # MuJoCo orders parents before children
            j = int(model.body_jntadr[b])
            if model.body_jntnum[b] != 1 or j not in column or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"body {model.body(b).name}: one of the Go2's hinges expected")
            self.steps.append({"parent": local[int(model.body_parentid[b])], "column": 7 + column[j],
                               "axis": torch.tensor(model.jnt_axis[j], dtype=torch.float32, device=device),
                               "anchor": torch.tensor(model.jnt_pos[j], dtype=torch.float32, device=device)})  # fmt: skip
            local[b] = len(self.steps) - 1
        order = sorted(chain)
        self.body_pos = torch.tensor(model.body_pos[order], dtype=torch.float32, device=device)
        self.body_rot = quat_to_mat(torch.tensor(model.body_quat[order], dtype=torch.float32, device=device))
        self.foot_body = [local[int(model.geom_bodyid[g])] for g in geom_ids]
        self.foot_pos = torch.tensor(model.geom_pos[geom_ids], dtype=torch.float32, device=device)
        lo, hi = model.jnt_range[joint_ids].T
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``qpos`` (..., NQ) -> keypoint positions (..., NUM_KP, 3), the trunk's first, and the trunk's rotation (..., 3, 3)."""
        p0, R0 = qpos[..., :3], quat_to_mat(qpos[..., 3:7] / qpos[..., 3:7].norm(dim=-1, keepdim=True))
        pos, rot = [], []
        for i, s in enumerate(self.steps):
            pp, Rp = (p0, R0) if s["parent"] < 0 else (pos[s["parent"]], rot[s["parent"]])
            p, R = pp + mat_vec(Rp, self.body_pos[i]), mat_mat(Rp, self.body_rot[i])
            Rj = axis_angle_to_mat(s["axis"], qpos[..., s["column"]])
            p = p + mat_vec(R, s["anchor"] - mat_vec(Rj, s["anchor"]))
            pos.append(p), rot.append(mat_mat(R, Rj))
        feet = [pos[b] + mat_vec(rot[b], self.foot_pos[k]) for k, b in enumerate(self.foot_body)]
        return torch.stack([p0] + feet, -2), R0
