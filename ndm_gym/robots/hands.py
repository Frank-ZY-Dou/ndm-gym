"""Dexterous hands on position-controlled floating palms, as in SPIDER's hand-object scenes: the XHand (12 finger
joints), the Allegro hand (16, four fingers) and the Inspire hand (12); one hand, or a right and a left one. A hand can
also be mounted on a robot's arm (a Unitree G1's), whose hinges then move its palm.

The hands are read from the compiled scene. Their joints are the actuated ones in the order of the scene's actuators:
per hand the joints that move its palm (a floating palm's three slides and three hinges, or an arm's hinges), then the
fingers' hinges. Their state is those joints' positions and velocities. ``HandKinematics`` is the differentiable forward
kinematics of the tracked keypoints (per hand the palm and the fingertips it has), shared by the reference, the
simulator's states and the world model's predictions.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from ..core.kinematics import mat_mat, mat_vec
from ..core.rotations import axis_angle_to_mat, quat_to_mat

SIDES = ("right", "left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
OBJECT_BODY = "right_object"
NUM_PALM = 6  # a floating palm's joints, three slides then three hinges, first in its hand's (the single-hand expressions' slices)

# the largest offset of a position target from the reference pose: palm translation (m), palm rotation and fingers (rad);
# an arm's hinges move the palm as a floating palm's hinges do, and take the palm rotation's
PALM_TRANSLATION_SCALE, PALM_ROTATION_SCALE, FINGER_SCALE = 0.03, 0.3, 0.4


def body_joints(model: mujoco.MjModel, body: int) -> range:
    return range(int(model.body_jntadr[body]), int(model.body_jntadr[body]) + int(model.body_jntnum[body]))


def ancestors(model: mujoco.MjModel, body: int) -> list[int]:
    """``body`` and the bodies between it and the world."""
    bodies = []
    while body != 0:
        bodies.append(body)
        body = int(model.body_parentid[body])
    return bodies


def hand_joints(model: mujoco.MjModel) -> tuple[list[int], list[int], set[int]]:
    """The ids of the hands' joints in the order of the scene's actuators, where each hand starts in them, and the ids of
    the joints that move a palm. Per palm's site (``<side>_palm``), a hand is the joints that move it (a floating palm's
    slides and hinges, or the hinges of the arm it is on), then the joints of the bodies it carries (the fingers'); each
    is actuated once, a hand's joints follow each other, its palm's first."""
    joints = [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]
    column = {j: i for i, j in enumerate(joints)}
    by_joint = all(int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) for a in range(model.nu))
    hands = []
    for side in SIDES:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm")
        if site >= 0:
            palm_body = int(model.site_bodyid[site])
            palm = [j for b in ancestors(model, palm_body) for j in body_joints(model, b)]
            fingers = [j for b in range(model.nbody) if b != palm_body and palm_body in ancestors(model, b) for j in body_joints(model, b)]
            hands.append((palm, fingers))
    hands.sort(key=lambda hand: min((column.get(j, len(joints)) for j in hand[0]), default=len(joints)))
    starts, expected = [], []
    for palm, fingers in hands:
        starts.append(len(expected))
        expected += sorted(palm, key=lambda j: column.get(j, -1)) + sorted(fingers, key=lambda j: column.get(j, -1))
    shared = len({j for palm, _ in hands for j in palm}) < sum(len(palm) for palm, _ in hands)
    if not by_joint or not hands or shared or any(not palm for palm, _ in hands) or joints != expected:
        raise ValueError("the scene's actuators are not, per palm's site, the joints that move the palm followed by those of the "
                         "fingers it carries, each actuated once and moving one palm")  # fmt: skip
    return joints, starts, {j for palm, _ in hands for j in palm}


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
        joint_ids, starts, palm_joints = hand_joints(model)
        self.num_joints, self.num_hands = len(joint_ids), len(starts)
        column = {j: i for i, j in enumerate(joint_ids)}

        def chain_of(site_id: int) -> list[int]:  # the bodies between a site and the world
            return ancestors(model, int(model.site_bodyid[site_id]))

        def first_joint(sites: list[int]) -> int:  # the column of the palm's first joint: orders the hands as their joints
            return min(column[j] for b in chain_of(sites[0]) for j in body_joints(model, b))

        hands = sorted(hand_keypoints(model), key=first_joint)
        if len(hands) != self.num_hands:
            raise ValueError(f"the scene has {len(hands)} palms' sites for {self.num_hands} hands' actuators")
        site_ids = [s for sites in hands for s in sites]
        self.num_keypoints = len(site_ids)
        palms = [sum(len(h) for h in hands[:i]) for i in range(self.num_hands)]
        self.hand_keypoints = [(start, start + len(sites)) for start, sites in zip(palms, hands, strict=True)]
        self.palm_keypoints = torch.tensor(palms, device=device)
        self.fingertip_keypoints = torch.tensor([k for k in range(self.num_keypoints) if k not in palms], device=device)
        order = sorted({b for s in site_ids for b in chain_of(s)})  # MuJoCo orders parents before children
        local = {0: -1, **{b: i for i, b in enumerate(order)}}
        tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.steps = []
        for b in order:
            joints = []
            for j in body_joints(model, b):
                if j not in column:
                    raise ValueError(f"joint {model.joint(j).name} moves a keypoint and has no actuator")
                kind = "slide" if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE) else "hinge"
                joints.append({"kind": kind, "column": column[j], "axis": tensor(model.jnt_axis[j]), "anchor": tensor(model.jnt_pos[j])})
            self.steps.append({"parent": local[int(model.body_parentid[b])], "joints": joints})
        self.body_pos = tensor(model.body_pos[order])
        self.body_rot = quat_to_mat(tensor(model.body_quat[order]))
        self.site_body = [local[int(model.site_bodyid[s])] for s in site_ids]
        palm_chains = set()  # the bodies between the world and the palms' sites (``palm``)
        for start, _ in self.hand_keypoints:
            b = self.site_body[start]
            while b >= 0:
                palm_chains.add(b)
                b = self.steps[b]["parent"]
        self.palm_chains = sorted(palm_chains)  # parents before children, as ``steps``
        self.site_pos = tensor(model.site_pos[site_ids])
        lo, hi = model.jnt_range[joint_ids].T
        self.q_lo, self.q_hi = tensor(lo), tensor(hi)
        slides = {j for j in joint_ids if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE)}
        self.action_scale = tensor([FINGER_SCALE if j not in palm_joints else PALM_TRANSLATION_SCALE if j in slides else PALM_ROTATION_SCALE
                                    for j in joint_ids])  # fmt: skip
        fingers = [c for c, j in enumerate(joint_ids) if j not in palm_joints]
        self.finger_joints = torch.tensor(fingers, device=device)  # the columns of the fingers' joints
        self.palm_translations = [c for c, j in enumerate(joint_ids) if j in palm_joints and j in slides]  # the columns of the palms' slides

    def _bodies(self, q: torch.Tensor, bodies) -> tuple[dict, dict]:
        """The positions and rotations of ``bodies`` (parents before children). A body's joints act in their order, as in
        MuJoCo."""
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
        """Joint positions (..., num_joints) -> keypoint positions (..., num_keypoints, 3) and the palms' rotations
        (..., num_hands, 3, 3); with one hand the palm's rotation (..., 3, 3)."""
        pos, rot = self._bodies(q, range(len(self.steps)))
        keypoints = torch.stack([pos[b] + mat_vec(rot[b], self.site_pos[k]) for k, b in enumerate(self.site_body)], -2)
        if self.num_hands == 1:
            return keypoints, rot[self.site_body[0]]
        return keypoints, torch.stack([rot[self.site_body[start]] for start, _ in self.hand_keypoints], -3)

    def palm(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward``'s palms alone: their keypoints (..., num_hands, 3) and rotations (as ``forward``'s), from the
        bodies of the palms' chains (the same operations, so the same values); what the world model reads (train_hand
        --wm_palm_fk)."""
        pos, rot = self._bodies(q, self.palm_chains)
        if self.num_hands == 1:
            b = self.site_body[0]
            return (pos[b] + mat_vec(rot[b], self.site_pos[0])).unsqueeze(-2), rot[b]
        bodies = [(start, self.site_body[start]) for start, _ in self.hand_keypoints]
        return torch.stack([pos[b] + mat_vec(rot[b], self.site_pos[k]) for k, b in bodies], -2), torch.stack([rot[b] for _, b in bodies], -3)

    def fingertips_about_palms(self, keypoints: torch.Tensor, palm_R: torch.Tensor) -> torch.Tensor:
        """Every fingertip about its hand's palm, in that palm's frame: (..., num_keypoints - num_hands, 3)."""
        if self.num_hands == 1:
            return (keypoints[..., 1:, :] - keypoints[..., :1, :]) @ palm_R  # rows x^T R = (R^T x)^T
        parts = [
            (keypoints[..., start + 1 : stop, :] - keypoints[..., start : start + 1, :]) @ palm_R[..., hand, :, :]
            for hand, (start, stop) in enumerate(self.hand_keypoints)
        ]  # rows x^T R = (R^T x)^T
        return torch.cat(parts, -2)


def scene_indices(model: mujoco.MjModel) -> dict[str, np.ndarray | int]:
    """Where the hands' joints, their actuators and the object's free joint sit in the scene's qpos, qvel and ctrl."""
    joint_ids, _, _ = hand_joints(model)
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
