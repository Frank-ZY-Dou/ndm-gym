"""Hand-object motion tracking: one or two dexterous hands follow a kinematic reference of the hands and the object
they handle.

State tuple ``S = (q, qd, p, R, v, w)``: the hands' joint positions and velocities (ndm_gym.robots.hands), and the
object's position, rotation matrix and world linear and angular velocity. Observation, goal, reward and termination are
analytic functions of the state, its forward kinematics ``fk`` (HandKinematics.forward of the joint positions, computed
once per state) and the reference frame, shared by the simulator's transitions and the world model's imagined rollouts.
A step taken at frame ``f`` tracks frame ``f + 1``; the reference ends in ``HOLD`` frames of its last pose, and imagined
rollouts stay on the last frame once they reach it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..core.rotations import mat_to_6d, quat_to_mat, rot_angle_sq
from ..motion.library import _angular_velocity, _gradient
from ..robots.hands import NUM_PALM, HandKinematics
from ..state import CONTROL_DT, OBJ_DIM, flatten_object, unflatten_object

HOLD = 25  # frames of the last pose appended to a clip (0.5 s)


def flatten_state(S) -> torch.Tensor:
    return torch.cat((S[0], S[1], flatten_object(S[2:])), -1)


def unflatten_state(x: torch.Tensor):
    num_joints = (x.shape[-1] - OBJ_DIM) // 2
    return x[..., :num_joints], x[..., num_joints : 2 * num_joints], *unflatten_object(x[..., 2 * num_joints :])


class HandReference(NamedTuple):
    """A clip at the control rate, (T, ...) per field: the state's components, the keypoints and the palms' rotations."""

    q: torch.Tensor
    qd: torch.Tensor
    p: torch.Tensor
    R: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    keypoints: torch.Tensor
    palm_R: torch.Tensor

    @property
    def frames(self) -> int:
        return self.q.shape[0]

    def state(self, f: torch.Tensor):
        return self.q[f], self.qd[f], self.p[f], self.R[f], self.v[f], self.w[f]

    def next_frame(self, f: torch.Tensor) -> torch.Tensor:
        return (f + 1).clamp(max=self.frames - 1)


def build_reference(clip: dict[str, torch.Tensor], kin: HandKinematics, device: str) -> HandReference:
    """The reference of a clip (motion.sources.spider.load_reference, sampled every CONTROL_DT), its last pose held for
    ``HOLD`` frames; velocities by central differences."""
    hold = lambda x: torch.cat((x, x[-1:].expand((HOLD,) + x.shape[1:])), 0)  # noqa: E731
    q, p, R = hold(clip["q"]), hold(clip["p"]), quat_to_mat(hold(clip["quat"]))
    keypoints, palm_R = kin.forward(q.to(device))
    fields = (q, _gradient(q, CONTROL_DT), p, R, _gradient(p, CONTROL_DT), _angular_velocity(R))
    return HandReference(*(x.to(device) for x in fields), keypoints, palm_R)


@dataclass
class RewardCfg:
    keypoint_std: float = 0.04
    finger_std: float = 0.4
    palm_ori_std: float = 0.4
    object_pos_std: float = 0.04
    object_ori_std: float = 0.5
    w_keypoint: float = 1.0
    w_finger: float = 0.5
    w_palm_ori: float = 0.5
    w_object_pos: float = 2.0
    w_object_ori: float = 1.0
    w_action_rate: float = -0.05
    # the target-rate penalty (train_hand --target_rate_w; 0: off), a cost per step on the policy's part of the targets' change, each
    # joint over its kernel's width, squared, mean over the joints: in the reward with a minus sign, in SuperTrack's loss as it is
    # (``HandTask.smooth_cost``)
    w_target_rate: float = 0.0
    term_object: float = 0.15  # m, the object's distance from its reference
    term_palm: float = 0.15  # m, a palm's

    def scaled(self, k: float) -> RewardCfg:
        """The tracking kernels' widths scaled by ``k`` (training's --reward_std_scale)."""
        widths = ("keypoint_std", "finger_std", "palm_ori_std", "object_pos_std", "object_ori_std")
        return RewardCfg(**{**self.__dict__, **{n: getattr(self, n) * k for n in widths}})


class HandTask:
    def __init__(self, kin: HandKinematics, ref: HandReference, cfg: RewardCfg = RewardCfg()):
        self.kin, self.ref, self.cfg = kin, ref, cfg
        self.scale = kin.action_scale
        joints, fingertips = kin.num_joints, kin.num_keypoints - kin.num_hands
        self.state_dim = 2 * joints + OBJ_DIM
        self.obs_dim = joints + joints + 3 * fingertips + kin.num_hands * (6 + 3 + 6 + 3 + 3) + joints
        self.goal_frame_dim = joints + 3 + 6
        self.max_step_reward = (cfg.w_keypoint + cfg.w_finger + cfg.w_palm_ori + cfg.w_object_pos + cfg.w_object_ori) * CONTROL_DT
        # smooth_cost's constant: each joint's target change per unit of action over its kernel's width (each palm's translation over
        # keypoint_std, its rotation over palm_ori_std, the fingers over finger_std)
        width = torch.full((joints,), cfg.palm_ori_std, device=self.scale.device)
        width[kin.palm_translations] = cfg.keypoint_std
        width[kin.finger_joints] = cfg.finger_std
        self.rate_k = self.scale / width

    def setpoint(self, a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Position targets for policy output ``a`` in [-1, 1] at frame ``f``: offsets about the next frame's pose."""
        return torch.clamp(self.ref.q[self.ref.next_frame(f)] + self.scale * a, self.kin.q_lo, self.kin.q_hi)

    def obs(self, S, fk, a_prev: torch.Tensor) -> torch.Tensor:
        """The hands' joints, and per hand its fingertips and the object about its palm in the palm's frame."""
        q, qd, p, R, v, w = S
        keypoints, palm_R = fk
        if self.kin.num_hands == 1:
            Rt = palm_R.transpose(-1, -2)
            local = lambda x: (Rt @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731
            tips = (keypoints[..., 1:, :] - keypoints[..., :1, :]) @ palm_R  # rows x^T R = (R^T x)^T
            return torch.cat((q, 0.1 * qd, tips.flatten(-2), mat_to_6d(palm_R), local(p - keypoints[..., 0, :]), mat_to_6d(Rt @ R), local(v),
                              0.1 * local(w), a_prev), -1)  # fmt: skip
        tips = self.kin.fingertips_about_palms(keypoints, palm_R)
        parts = [q, 0.1 * qd]
        for hand, (start, stop) in enumerate(self.kin.hand_keypoints):
            Rh = palm_R[..., hand, :, :]
            Rt = Rh.transpose(-1, -2)
            local = lambda x, Rt=Rt: (Rt @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731
            parts += [tips[..., start - hand : stop - hand - 1, :].flatten(-2), mat_to_6d(Rh), local(p - keypoints[..., start, :]),
                      mat_to_6d(Rt @ R), local(v), 0.1 * local(w)]  # fmt: skip
        return torch.cat((*parts, a_prev), -1)

    def goal(self, S, f: torch.Tensor, goal_frames: int) -> torch.Tensor:
        """The next ``goal_frames`` reference frames about the state: joint offsets, the object's offset and relative rotation."""
        q, _, p, R, _, _ = S
        Rt = R.transpose(-1, -2)
        parts = []
        for k in range(1, goal_frames + 1):
            g = (f + k).clamp(max=self.ref.frames - 1)
            parts += [self.ref.q[g] - q, self.ref.p[g] - p, mat_to_6d(Rt @ self.ref.R[g])]
        return torch.cat(parts, -1)

    def errors(self, S, fk, f: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tracking errors against frame ``f``: keypoints (..., num_keypoints) and object position (m), finger joints (rad, RMS),
        and the squared rotation angles of the palms (their mean) and the object."""
        q, _, p, R, _, _ = S
        keypoints, palm_R = fk
        one = self.kin.num_hands == 1
        fingers = (q[..., NUM_PALM:] - self.ref.q[f][..., NUM_PALM:]) if one else (q - self.ref.q[f])[..., self.kin.finger_joints]
        palm_ori_sq = rot_angle_sq(palm_R, self.ref.palm_R[f])
        return {
            "keypoint": torch.linalg.vector_norm(keypoints - self.ref.keypoints[f], dim=-1),
            "finger": fingers.pow(2).mean(-1).sqrt(),
            "palm_ori_sq": palm_ori_sq if one else palm_ori_sq.mean(-1),
            "object_pos": torch.linalg.vector_norm(p - self.ref.p[f], dim=-1),
            "object_ori_sq": rot_angle_sq(R, self.ref.R[f]),
        }

    def reward(self, S, fk, f: torch.Tensor, a: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """The tracking reward of reaching ``S`` at frame ``f`` with action ``a`` after ``a_prev``."""
        c, e = self.cfg, self.errors(S, fk, f)
        r = (
            c.w_keypoint * torch.exp(-e["keypoint"].pow(2).mean(-1) / c.keypoint_std**2)
            + c.w_finger * torch.exp(-e["finger"].pow(2) / c.finger_std**2)
            + c.w_palm_ori * torch.exp(-e["palm_ori_sq"] / c.palm_ori_std**2)
            + c.w_object_pos * torch.exp(-e["object_pos"].pow(2) / c.object_pos_std**2)
            + c.w_object_ori * torch.exp(-e["object_ori_sq"] / c.object_ori_std**2)
            + c.w_action_rate * (a - a_prev).pow(2).mean(-1)
        )
        cost = self.smooth_cost(S, a, a_prev)
        if cost is not None:
            r = r - cost
        return r * CONTROL_DT

    def smooth_cost(self, S, a: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor | None:
        """The target-rate penalty of reaching ``S`` with action ``a`` after ``a_prev`` (per step, before CONTROL_DT), or None when it
        is off: ``w_target_rate`` times the mean over the joints of the policy's part of the targets' change over the joint's kernel
        width, squared (a change of the width costs the weight). A target that jumps back and forth from one step to the next costs
        its jumps; one that follows the reference costs nothing, since the reference's own motion is not the policy's part."""
        c = self.cfg
        if not c.w_target_rate:
            return None
        da = a - a_prev
        cost = torch.zeros_like(da[..., 0])
        return cost + c.w_target_rate * (self.rate_k * da).pow(2).mean(-1)

    def termination(self, S, fk, f: torch.Tensor) -> torch.Tensor:
        e = self.errors(S, fk, f)
        if self.kin.num_hands == 1:
            return (e["object_pos"] > self.cfg.term_object) | (e["keypoint"][..., 0] > self.cfg.term_palm)
        return (e["object_pos"] > self.cfg.term_object) | (e["keypoint"][..., self.kin.palm_keypoints] > self.cfg.term_palm).any(-1)

    def soft_termination(self, S, fk, f: torch.Tensor, width: float = 0.015) -> torch.Tensor:
        """The termination's probability with the thresholds as sigmoids, differentiable in the state."""
        e = self.errors(S, fk, f)
        if self.kin.num_hands == 1:
            alive = (1 - torch.sigmoid((e["object_pos"] - self.cfg.term_object) / width)) * (
                1 - torch.sigmoid((e["keypoint"][..., 0] - self.cfg.term_palm) / width))  # fmt: skip
            return 1 - alive
        palms = 1 - torch.sigmoid((e["keypoint"][..., self.kin.palm_keypoints] - self.cfg.term_palm) / width)
        return 1 - (1 - torch.sigmoid((e["object_pos"] - self.cfg.term_object) / width)) * palms.prod(-1)

    def critic_features(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, goal_frames: int) -> torch.Tensor:
        return torch.cat((self.obs(S, fk, a_prev), self.goal(S, f, goal_frames)), -1)
