"""Dexterous hands on position-controlled floating palms, as in SPIDER's hand-object scenes: the XHand (12 finger
joints), the Allegro hand (16, four fingers) and the Inspire hand (12); one hand, or a right and a left one.

The hands are read from the compiled scene. Their joints are the actuated ones in the order of the scene's actuators:
per hand the palm's three slides and three hinges, then the fingers' hinges. Their state is those joints' positions and
velocities. ``HandKinematics`` is the differentiable forward kinematics of the tracked keypoints (per hand the palm and
the fingertips it has), shared by the reference, the simulator's states and the world model's predictions.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from ..core.kinematics import mat_mat, mat_vec
from ..core.rotations import axis_angle_to_mat, quat_to_mat

NUM_PALM = 6  # a palm's joints: three slides, then three hinges
SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
OBJECT_BODY = "right_object"

# the largest offset of a position target from the reference pose: palm translation (m), palm rotation and fingers (rad)
PALM_TRANSLATION_SCALE, PALM_ROTATION_SCALE, FINGER_SCALE = 0.03, 0.3, 0.4


def hand_joints(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    """The ids of the hands' joints in the order of the scene's actuators, and where each hand starts in them: a hand is
    a palm's three slides and three hinges followed by its fingers' hinges."""
    joints = [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]
    kinds = [int(model.jnt_type[j]) for j in joints]
    slide, hinge = int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)
    starts = [i for i in range(len(kinds)) if kinds[i : i + 3] == [slide] * 3]
    expected = []
    for start, stop in zip(starts, starts[1:] + [len(kinds)], strict=True):
        expected += [slide] * 3 + [hinge] * (stop - start - 3)
        if stop - start <= NUM_PALM:
            expected = None
            break
    if len(set(joints)) != len(joints) or not starts or kinds != expected:
        raise ValueError("the scene's actuators are not, per hand, a palm's three slides and three hinges followed by finger hinges")
    return joints, starts


def hand_keypoints(model: mujoco.MjModel) -> list[list[int]]:
    """Per hand of the scene (right, then left), the ids of its tracked sites: the palm, then the fingertips it has
    (thumb to pinky)."""
    site = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)  # noqa: E731
    hands = []
    for side in SIDES:
        if site(f"{side}_palm") >= 0:
            tips = [site(f"{side}_{finger}_tip") for finger in FINGERS]
            hands.append([site(f"{side}_palm")] + [t for t in tips if t >= 0])
    if not hands or min(len(h) for h in hands) < 3:
        raise ValueError("the scene lacks a palm's site or fingertip sites")
    return hands


class HandKinematics:
    """Keypoint positions and the palms' rotations from the joint positions, for a compiled scene ``model``. The hands
    are in the order of their joints; a hand's keypoints are consecutive, its palm first."""

    def __init__(self, model: mujoco.MjModel, device: str):
        joint_ids, starts = hand_joints(model)
        self.num_joints, self.num_hands = len(joint_ids), len(starts)
        column = {j: i for i, j in enumerate(joint_ids)}

        def chain_of(site_id: int) -> list[int]:  # the bodies between a site and the world
            bodies, b = [], int(model.site_bodyid[site_id])
            while b != 0:
                bodies.append(b)
                b = int(model.body_parentid[b])
            return bodies

        def first_joint(sites: list[int]) -> int:  # the column of the palm's first joint: orders the hands as their joints
            return min(
                column[j] for b in chain_of(sites[0]) for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + int(model.body_jntnum[b]))
            )

        hands = sorted(hand_keypoints(model), key=first_joint)
        if len(hands) != self.num_hands:
            raise ValueError(f"the scene has {len(hands)} palms' sites for {self.num_hands} hands' actuators")
        site_ids = [s for sites in hands for s in sites]
        self.num_keypoints = len(site_ids)
        self.palm_keypoints = [sum(len(h) for h in hands[:i]) for i in range(self.num_hands)]
        self.hand_keypoints = [(start, start + len(sites)) for start, sites in zip(self.palm_keypoints, hands, strict=True)]
        self.fingertip_keypoints = [k for k in range(self.num_keypoints) if k not in self.palm_keypoints]
        order = sorted({b for s in site_ids for b in chain_of(s)})  # MuJoCo orders parents before children
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
        scale, fingers, translations = [], [], []
        for start, stop in zip(starts, starts[1:] + [self.num_joints], strict=True):
            scale += [PALM_TRANSLATION_SCALE] * 3 + [PALM_ROTATION_SCALE] * 3 + [FINGER_SCALE] * (stop - start - NUM_PALM)
            fingers += range(start + NUM_PALM, stop)
            translations += range(start, start + 3)
        self.action_scale = tensor(scale)
        self.finger_joints = torch.tensor(fingers, device=device)  # the columns of the fingers' joints
        self.palm_translations = translations  # the columns of the palms' slides

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint positions (..., num_joints) -> keypoint positions (..., num_keypoints, 3) and the palms' rotations
        (..., num_hands, 3, 3). A body's joints act in their order, as in MuJoCo."""
        pos, rot = [], []
        for s, p0, R0 in zip(self.steps, self.body_pos, self.body_rot, strict=True):
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
            pos.append(p), rot.append(R)
        keypoints = torch.stack([pos[b] + mat_vec(rot[b], self.site_pos[k]) for k, b in enumerate(self.site_body)], -2)
        return keypoints, torch.stack([rot[self.site_body[k]] for k in self.palm_keypoints], -3)

    def fingertips_about_palms(self, keypoints: torch.Tensor, palm_R: torch.Tensor) -> torch.Tensor:
        """Every fingertip about its hand's palm, in that palm's frame: (..., num_keypoints - num_hands, 3)."""
        parts = [
            (keypoints[..., start + 1 : stop, :] - keypoints[..., start : start + 1, :]) @ palm_R[..., hand, :, :]
            for hand, (start, stop) in enumerate(self.hand_keypoints)
        ]  # rows x^T R = (R^T x)^T
        return torch.cat(parts, -2)

    def palm(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward``'s palm alone: its keypoint (..., 1, 3) and rotation (..., 3, 3), from the bodies of the palm's chain
        (the same operations, so the same values); what the world model reads (train_hand --wm_palm_fk)."""
        pos, rot = self._bodies(q, self.palm_chain)
        b = self.site_body[0]
        return (pos[b] + mat_vec(rot[b], self.site_pos[0])).unsqueeze(-2), rot[b]


def scene_indices(model: mujoco.MjModel) -> dict[str, np.ndarray | int]:
    """Where the hands' joints, their actuators and the object's free joint sit in the scene's qpos, qvel and ctrl."""
    joint_ids, _ = hand_joints(model)
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
