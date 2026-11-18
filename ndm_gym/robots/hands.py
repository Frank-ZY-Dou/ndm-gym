"""Dexterous hands on a position-controlled floating palm, as in SPIDER's hand-object scenes: the XHand (12 finger
joints), the Allegro hand (16, four fingers) and the Inspire hand (12).

A hand is read from its compiled scene. Its joints are the actuated ones in the order of the scene's actuators: the
palm's three slides and three hinges, then the fingers' hinges. Its state is their positions and velocities.
``HandKinematics`` is the differentiable forward kinematics of the tracked keypoints (the palm and the fingertips the
hand has), shared by the reference, the simulator's states and the world model's predictions.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from ..core.kinematics import mat_mat, mat_vec
from ..core.rotations import axis_angle_to_mat, quat_to_mat

NUM_PALM = 6  # the palm's joints: three slides, then three hinges
PALM_SITE = "right_palm"
FINGERTIP_SITES = ("right_thumb_tip", "right_index_tip", "right_middle_tip", "right_ring_tip", "right_pinky_tip")
OBJECT_BODY = "right_object"

# the largest offset of a position target from the reference pose: palm translation (m), palm rotation and fingers (rad)
PALM_TRANSLATION_SCALE, PALM_ROTATION_SCALE, FINGER_SCALE = 0.03, 0.3, 0.4


def hand_joints(model: mujoco.MjModel) -> list[int]:
    """The ids of the hand's joints in the order of the scene's actuators: the palm's slides and hinges, the fingers'."""
    joints = [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]
    kinds = [int(model.jnt_type[j]) for j in joints]
    slide, hinge = int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)
    if len(set(joints)) != len(joints) or kinds[:NUM_PALM] != [slide] * 3 + [hinge] * 3 or set(kinds[NUM_PALM:]) != {hinge}:
        raise ValueError("the scene's actuators are not a palm's three slides and three hinges followed by finger hinges")
    return joints


def hand_keypoints(model: mujoco.MjModel) -> list[int]:
    """The ids of the tracked sites: the palm, then the fingertips the hand has (thumb to pinky)."""
    names = [PALM_SITE] + [n for n in FINGERTIP_SITES if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) >= 0]
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in names]
    if sites[0] < 0 or len(sites) < 3:
        raise ValueError("the scene lacks the palm's site or fingertip sites")
    return sites


class HandKinematics:
    """Keypoint positions and the palm's rotation from the joint positions, for a compiled scene ``model``."""

    def __init__(self, model: mujoco.MjModel, device: str):
        joint_ids, site_ids = hand_joints(model), hand_keypoints(model)
        self.num_joints, self.num_keypoints = len(joint_ids), len(site_ids)
        column = {j: i for i, j in enumerate(joint_ids)}
        chain = set()  # the bodies between the keypoints and the world
        for s in site_ids:
            b = model.site_bodyid[s]
            while b != 0:
                chain.add(int(b))
                b = model.body_parentid[b]
        order = sorted(chain)  # MuJoCo orders parents before children
        local = {0: -1, **{b: i for i, b in enumerate(order)}}
        tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.steps = []
        for b in order:
            joints = []
            for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + int(model.body_jntnum[b])):
                if j not in column:
                    raise ValueError(f"joint {model.joint(j).name} moves a keypoint and has no actuator")
                kind = "slide" if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE) else "hinge"
                joints.append({"kind": kind, "column": column[j], "axis": tensor(model.jnt_axis[j]), "anchor": tensor(model.jnt_pos[j])})
            self.steps.append({"parent": local[int(model.body_parentid[b])], "joints": joints})
        self.body_pos = tensor(model.body_pos[order])
        self.body_rot = quat_to_mat(tensor(model.body_quat[order]))
        self.site_body = [local[int(model.site_bodyid[s])] for s in site_ids]
        palm_chain, b = [], self.site_body[0]  # the bodies from the world to the palm's site, parents first (``palm``)
        while b >= 0:
            palm_chain.append(b)
            b = self.steps[b]["parent"]
        self.palm_chain = palm_chain[::-1]
        self.site_pos = tensor(model.site_pos[site_ids])
        lo, hi = model.jnt_range[joint_ids].T
        self.q_lo, self.q_hi = tensor(lo), tensor(hi)
        fingers = self.num_joints - NUM_PALM
        self.action_scale = tensor([PALM_TRANSLATION_SCALE] * 3 + [PALM_ROTATION_SCALE] * 3 + [FINGER_SCALE] * fingers)

    def _bodies(self, q: torch.Tensor, bodies) -> tuple[dict, dict]:
        """The positions and rotations of the chain's ``bodies`` (parents before children). A body's joints act in their
        order, as in MuJoCo."""
        pos, rot = {}, {}
        for i in bodies:
            s, p0, R0 = self.steps[i], self.body_pos[i], self.body_rot[i]
            if s["parent"] < 0:
                p, R = p0.expand(q.shape[:-1] + (3,)), R0.expand(q.shape[:-1] + (3, 3))
            else:
                p, R = pos[s["parent"]] + mat_vec(rot[s["parent"]], p0), mat_mat(rot[s["parent"]], R0)
            for joint in s["joints"]:
                if joint["kind"] == "slide":
                    p = p + mat_vec(R, joint["axis"]) * q[..., joint["column"], None]
                else:
                    Rj = axis_angle_to_mat(joint["axis"], q[..., joint["column"]])
                    p = p + mat_vec(R, joint["anchor"] - mat_vec(Rj, joint["anchor"]))
                    R = mat_mat(R, Rj)
            pos[i], rot[i] = p, R
        return pos, rot

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint positions (..., num_joints) -> keypoint positions (..., num_keypoints, 3) and the palm's rotation
        (..., 3, 3)."""
        pos, rot = self._bodies(q, range(len(self.steps)))
        keypoints = torch.stack([pos[b] + mat_vec(rot[b], self.site_pos[k]) for k, b in enumerate(self.site_body)], -2)
        return keypoints, rot[self.site_body[0]]

    def palm(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward``'s palm alone: its keypoint (..., 1, 3) and rotation (..., 3, 3), from the bodies of the palm's chain
        (the same operations, so the same values); what the world model reads (train_hand --wm_palm_fk)."""
        pos, rot = self._bodies(q, self.palm_chain)
        b = self.site_body[0]
        return (pos[b] + mat_vec(rot[b], self.site_pos[0])).unsqueeze(-2), rot[b]


def scene_indices(model: mujoco.MjModel) -> dict[str, np.ndarray | int]:
    """Where the hand's joints, their actuators and the object's free joint sit in the scene's qpos, qvel and ctrl."""
    joint_ids = hand_joints(model)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    if body < 0 or int(model.body_jntnum[body]) != 1 or int(model.jnt_type[model.body_jntadr[body]]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("the scene lacks the object or its free joint")
    free = int(model.body_jntadr[body])
    return {
        "qpos": model.jnt_qposadr[joint_ids].astype(np.int64),
        "qvel": model.jnt_dofadr[joint_ids].astype(np.int64),
        "ctrl": np.arange(model.nu, dtype=np.int64),
        "object_qpos": int(model.jnt_qposadr[free]),
        "object_qvel": int(model.jnt_dofadr[free]),
    }
