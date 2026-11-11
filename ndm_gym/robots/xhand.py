"""XHand (right, 12 finger joints) on a position-controlled floating palm, as in SPIDER's hand-object scenes.

The palm moves on three slides and three hinges, so the hand's state is its 18 joint positions and velocities, in the
order of the scene's actuators. ``HandKinematics`` is the differentiable forward kinematics of the tracked keypoints
(the palm and the five fingertips), built from the compiled scene, shared by the reference, the simulator's states and
the world model's predictions.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from ..core.kinematics import mat_mat, mat_vec
from ..core.rotations import axis_angle_to_mat, quat_to_mat

PALM_JOINTS = [
    "R_forearm_tx_link_joint", "R_forearm_ty_link_joint", "R_forearm_tz_link_joint",
    "R_forearm_roll_link_joint", "R_forearm_pitch_link_joint", "R_forearm_yaw_link_joint",
]  # fmt: skip
FINGER_JOINTS = [
    "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
    "right_hand_mid_joint1", "right_hand_mid_joint2",
    "right_hand_ring_joint1", "right_hand_ring_joint2",
    "right_hand_pinky_joint1", "right_hand_pinky_joint2",
]  # fmt: skip
JOINT_NAMES = PALM_JOINTS + FINGER_JOINTS
NUM_J = len(JOINT_NAMES)
NUM_PALM = len(PALM_JOINTS)

# tracked sites: the palm, then the fingertips
KEYPOINTS = ["right_palm", "right_thumb_tip", "right_index_tip", "right_middle_tip", "right_ring_tip", "right_pinky_tip"]
NUM_KP = len(KEYPOINTS)
OBJECT_BODY = "right_object"

# the largest offset of a position target from the reference pose: palm translation (m), palm rotation and fingers (rad)
ACTION_SCALE = [0.03] * 3 + [0.3] * 3 + [0.4] * len(FINGER_JOINTS)


class HandKinematics:
    """Keypoint positions and the palm's rotation from the joint positions, for a compiled scene ``model``."""

    def __init__(self, model: mujoco.MjModel, device: str):
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in KEYPOINTS]
        if min(joint_ids + site_ids) < 0:
            raise ValueError("the scene lacks an XHand joint or keypoint site")
        column = {j: i for i, j in enumerate(joint_ids)}
        chain = set()  # the bodies between the keypoints and the world
        for s in site_ids:
            b = model.site_bodyid[s]
            while b != 0:
                chain.add(int(b))
                b = model.body_parentid[b]
        local, self.steps = {0: -1}, []
        for b in sorted(chain):  # MuJoCo orders parents before children
            if model.body_jntnum[b] > 1:
                raise ValueError(f"body {model.body(b).name} has more than one joint")
            step = {"parent": local[int(model.body_parentid[b])], "kind": None}
            if model.body_jntnum[b] == 1:
                j = int(model.body_jntadr[b])
                kind = {int(mujoco.mjtJoint.mjJNT_SLIDE): "slide", int(mujoco.mjtJoint.mjJNT_HINGE): "hinge"}.get(int(model.jnt_type[j]))
                if j not in column or kind is None:
                    raise ValueError(f"joint {model.joint(j).name} is not one of the hand's slides and hinges")
                step.update(kind=kind, column=column[j],
                            axis=torch.tensor(model.jnt_axis[j], dtype=torch.float32, device=device),
                            anchor=torch.tensor(model.jnt_pos[j], dtype=torch.float32, device=device))  # fmt: skip
            local[b] = len(self.steps)
            self.steps.append(step)
        order = sorted(chain)
        self.body_pos = torch.tensor(model.body_pos[order], dtype=torch.float32, device=device)
        self.body_rot = quat_to_mat(torch.tensor(model.body_quat[order], dtype=torch.float32, device=device))
        self.site_body = [local[int(model.site_bodyid[s])] for s in site_ids]
        self.site_pos = torch.tensor(model.site_pos[site_ids], dtype=torch.float32, device=device)
        lo, hi = model.jnt_range[joint_ids].T
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint positions (..., NUM_J) -> keypoint positions (..., NUM_KP, 3) and the palm's rotation (..., 3, 3)."""
        pos, rot = [], []
        for i, s in enumerate(self.steps):
            p0, R0 = self.body_pos[i], self.body_rot[i]
            if s["parent"] < 0:
                p, R = p0.expand(q.shape[:-1] + (3,)), R0.expand(q.shape[:-1] + (3, 3))
            else:
                p, R = pos[s["parent"]] + mat_vec(rot[s["parent"]], p0), mat_mat(rot[s["parent"]], R0)
            if s["kind"] == "slide":
                p = p + mat_vec(R, s["axis"]) * q[..., s["column"], None]
            elif s["kind"] == "hinge":
                Rj = axis_angle_to_mat(s["axis"], q[..., s["column"]])
                p = p + mat_vec(R, s["anchor"] - mat_vec(Rj, s["anchor"]))
                R = mat_mat(R, Rj)
            pos.append(p), rot.append(R)
        keypoints = torch.stack([pos[b] + mat_vec(rot[b], self.site_pos[k]) for k, b in enumerate(self.site_body)], -2)
        return keypoints, rot[self.site_body[0]]


def scene_indices(model: mujoco.MjModel) -> dict[str, np.ndarray | int]:
    """Where the hand's joints, their actuators and the object's free joint sit in the scene's qpos, qvel and ctrl."""
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    actuator_of = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    if min(joint_ids) < 0 or body < 0 or any(j not in actuator_of for j in joint_ids):
        raise ValueError("the scene lacks an actuated XHand joint or the object")
    free = int(model.body_jntadr[body])
    if int(model.jnt_type[free]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("the object needs a free joint")
    return {
        "qpos": model.jnt_qposadr[joint_ids].astype(np.int64),
        "qvel": model.jnt_dofadr[joint_ids].astype(np.int64),
        "ctrl": np.array([actuator_of[j] for j in joint_ids], dtype=np.int64),
        "object_qpos": int(model.jnt_qposadr[free]),
        "object_qvel": int(model.jnt_dofadr[free]),
    }
