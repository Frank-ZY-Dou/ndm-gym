"""Quadruped motion tracking: the Go2 follows retargeted dog motion capture (motion.sources.lifelike), one or many clips.

State tuple ``S = (qpos, qvel)``, MuJoCo's generalized coordinates of the Go2 (robots.go2): the trunk's position and
quaternion and the joints, the trunk's world linear and body-frame angular velocity and the joints'. Observation, goal,
reward and termination are analytic functions of the state, its forward kinematics ``fk`` (Go2Kinematics.forward of
qpos, computed once per state: the trunk and the feet, and the trunk's rotation) and the reference frame, shared by the
simulator's transitions and the world model's imagined rollouts.

The clips are concatenated into one reference; a frame index addresses a frame of one clip, and ``last`` holds the
index of its clip's last frame. A step taken at frame ``f`` tracks frame ``f + 1``; imagined rollouts and goals stay on a
clip's last frame once they reach it. The observation does not depend on the trunk's heading and horizontal position;
the goal reads the reference in the trunk's heading frame; the reward and the termination compare the trunk with the
reference in the world (the clip's own frame), so that drift costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..core.rotations import mat_to_6d, quat_to_mat, rot_angle_sq, yaw_mat
from ..motion.library import _angular_velocity, _gradient
from ..robots.go2 import ACTION_SCALE, NQ, NUM_J, NUM_KP, NV, Go2Kinematics
from ..state import CONTROL_DT

STATE_DIM = NQ + NV
NUM_FEET = NUM_KP - 1
OBS_DIM = 1 + 6 + 3 + 3 + NUM_J + NUM_J + 3 * NUM_FEET + NUM_J
GOAL_FRAME_DIM = 3 + 6 + NUM_J + 3 + 3


def flatten_state(S) -> torch.Tensor:
    return torch.cat(S, -1)


def unflatten_state(x: torch.Tensor):
    return x[..., :NQ], x[..., NQ:]


def heading_local(H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """World vectors ``x`` (..., 3) in the heading frame ``H`` (..., 3, 3)."""
    return (H.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)


def feet_local(fk) -> torch.Tensor:
    """The feet about the trunk in the trunk's heading frame (..., NUM_FEET, 3)."""
    keypoints, R = fk
    return (keypoints[..., 1:, :] - keypoints[..., :1, :]) @ yaw_mat(R)  # rows x^T H = (H^T x)^T


class QuadrupedReference(NamedTuple):
    """The clips at the control rate, concatenated, (F, ...) per field: the state, the keypoints and the trunk's rotation
    (Go2Kinematics.forward), the planted feet, each frame's clip and its clip's last frame; per clip its first and last
    frame (C,)."""

    qpos: torch.Tensor
    qvel: torch.Tensor
    keypoints: torch.Tensor
    R: torch.Tensor
    contacts: torch.Tensor
    clip: torch.Tensor
    last: torch.Tensor
    clip_first: torch.Tensor
    clip_last: torch.Tensor

    @property
    def frames(self) -> int:
        return self.qpos.shape[0]

    @property
    def num_clips(self) -> int:
        return self.clip_first.shape[0]

    def state(self, f: torch.Tensor):
        return self.qpos[f], self.qvel[f]

    def next_frame(self, f: torch.Tensor) -> torch.Tensor:
        return torch.minimum(f + 1, self.last[f])

    def ahead(self, f: torch.Tensor, k: int) -> torch.Tensor:
        """The frame ``k`` steps after ``f``, at most its clip's last."""
        return torch.minimum(f + k, self.last[f])


def build_reference(clips: list[dict], kin: Go2Kinematics, device: str) -> QuadrupedReference:
    """The reference of clips (motion.sources.lifelike.load_clip, sampled every CONTROL_DT); velocities by central
    differences within each clip, the trunk's angular velocity in its body frame as MuJoCo's free joint keeps it."""
    qpos, qvel, contacts, clip, first, last = [], [], [], [], [], []
    n = 0
    for c, x in enumerate(clips):
        q = x["qpos"].double()
        R = quat_to_mat(q[:, 3:7])
        w_body = (R.transpose(-1, -2) @ _angular_velocity(R).unsqueeze(-1)).squeeze(-1)
        qpos.append(q), qvel.append(torch.cat((_gradient(q[:, :3], CONTROL_DT), w_body, _gradient(q[:, 7:], CONTROL_DT)), -1))
        contacts.append(x["contacts"].bool()), clip.append(torch.full((q.shape[0],), c, dtype=torch.long))
        first.append(n), last.append(n + q.shape[0] - 1)
        n += q.shape[0]
    clip = torch.cat(clip)
    first, last = torch.tensor(first), torch.tensor(last)
    qpos = torch.cat(qpos).float().to(device)
    keypoints, R = kin.forward(qpos)
    return QuadrupedReference(qpos, torch.cat(qvel).float().to(device), keypoints, R, torch.cat(contacts).to(device), clip.to(device),
                              last[clip].to(device), first.to(device), last.to(device))  # fmt: skip


@dataclass
class RewardCfg:
    feet_std: float = 0.06  # m, the feet about the trunk in its heading frame, against the reference's
    joint_std: float = 0.3  # rad, RMS over the joints
    trunk_pos_std: float = 0.15  # m, in the world
    trunk_ori_std: float = 0.3  # rad
    lin_vel_std: float = 0.5  # m/s, the trunk's
    ang_vel_std: float = 1.5  # rad/s, the trunk's
    w_feet: float = 1.0
    w_joint: float = 0.5
    w_trunk_pos: float = 1.0
    w_trunk_ori: float = 1.0
    w_lin_vel: float = 0.5
    w_ang_vel: float = 0.25
    w_action_rate: float = -0.05
    term_height: float = 0.12  # m, the trunk's height from its reference's
    term_ori: float = 1.0  # rad, the trunk's rotation from its reference's
    term_pos: float = 0.5  # m, the trunk's distance from its reference

    def scaled(self, k: float) -> RewardCfg:
        """The tracking kernels' widths scaled by ``k`` (training's --reward_std_scale)."""
        widths = ("feet_std", "joint_std", "trunk_pos_std", "trunk_ori_std", "lin_vel_std", "ang_vel_std")
        return RewardCfg(**{**self.__dict__, **{n: getattr(self, n) * k for n in widths}})


class QuadrupedTask:
    # the sizes and the state's layout that the floating-base trainers read (algos.quadruped_algos, sim.rigid.quadruped_sim,
    # train_quadruped; tasks.humanoid_tracking has its own), and the errors of the training log (their columns, in mm)
    obs_dim, goal_frame_dim, state_dim, num_joints = OBS_DIM, GOAL_FRAME_DIM, STATE_DIM, NUM_J
    unflatten = staticmethod(unflatten_state)
    log_errors = (("trunk_mm", "trunk_pos"), ("feet_mm", "feet"))

    def __init__(self, kin: Go2Kinematics, ref: QuadrupedReference, cfg: RewardCfg = RewardCfg()):
        self.kin, self.ref, self.cfg = kin, ref, cfg
        self.scale = torch.tensor(ACTION_SCALE, device=ref.qpos.device)
        self.ref_feet = feet_local((ref.keypoints, ref.R))
        self.max_step_reward = (cfg.w_feet + cfg.w_joint + cfg.w_trunk_pos + cfg.w_trunk_ori + cfg.w_lin_vel + cfg.w_ang_vel) * CONTROL_DT

    def setpoint(self, a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Joint position targets for policy output ``a`` in [-1, 1] at frame ``f``: offsets about the next frame's pose."""
        return torch.clamp(self.ref.qpos[self.ref.next_frame(f), 7:] + self.scale * a, self.kin.q_lo, self.kin.q_hi)

    def obs(self, S, fk, a_prev: torch.Tensor) -> torch.Tensor:
        """The trunk's height, tilt (its rotation in its heading frame) and body-frame velocities, the joints, the feet
        about the trunk in the trunk's frame and the previous action."""
        qpos, qvel = S
        keypoints, R = fk
        Rt = R.transpose(-1, -2)
        feet = (keypoints[..., 1:, :] - keypoints[..., :1, :]) @ R  # rows x^T R = (R^T x)^T
        return torch.cat((qpos[..., 2:3], mat_to_6d(yaw_mat(R).transpose(-1, -2) @ R), (Rt @ qvel[..., :3].unsqueeze(-1)).squeeze(-1),
                          qvel[..., 3:6], qpos[..., 7:], 0.1 * qvel[..., 6:], feet.flatten(-2), a_prev), -1)  # fmt: skip

    def goal(self, S, fk, f: torch.Tensor, goal_frames: int) -> torch.Tensor:
        """The next ``goal_frames`` reference frames in the trunk's heading frame: the trunk's offset, rotation and
        velocities (the angular one in the reference trunk's frame), and the joints' offsets."""
        qpos, _ = S
        H = yaw_mat(fk[1])
        parts = []
        for k in range(1, goal_frames + 1):
            g = self.ref.ahead(f, k)
            parts += [heading_local(H, self.ref.qpos[g, :3] - qpos[..., :3]), mat_to_6d(H.transpose(-1, -2) @ self.ref.R[g]), self.ref.qpos[g, 7:] - qpos[..., 7:],
                      heading_local(H, self.ref.qvel[g, :3]), self.ref.qvel[g, 3:6]]  # fmt: skip
        return torch.cat(parts, -1)

    def errors(self, S, fk, f: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tracking errors against frame ``f``: the feet about the trunk (..., NUM_FEET) and in the world (m), the trunk's
        position and height (m), its squared rotation angle, the joints (rad, RMS) and the trunk's velocities."""
        qpos, qvel = S
        keypoints, R = fk
        ref = self.ref
        return {
            "feet": torch.linalg.vector_norm(feet_local(fk) - self.ref_feet[f], dim=-1),
            "feet_world": torch.linalg.vector_norm(keypoints[..., 1:, :] - ref.keypoints[f][..., 1:, :], dim=-1),
            "trunk_pos": torch.linalg.vector_norm(qpos[..., :3] - ref.qpos[f, :3], dim=-1),
            "trunk_height": (qpos[..., 2] - ref.qpos[f, 2]).abs(),
            "trunk_ori_sq": rot_angle_sq(R, ref.R[f]),
            "joint": (qpos[..., 7:] - ref.qpos[f, 7:]).pow(2).mean(-1).sqrt(),
            "lin_vel": torch.linalg.vector_norm(qvel[..., :3] - ref.qvel[f, :3], dim=-1),
            "ang_vel": torch.linalg.vector_norm(qvel[..., 3:6] - ref.qvel[f, 3:6], dim=-1),
        }

    def reward(self, S, fk, f: torch.Tensor, a: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """The tracking reward of reaching ``S`` at frame ``f`` with action ``a`` after ``a_prev``."""
        c, e = self.cfg, self.errors(S, fk, f)
        r = (
            c.w_feet * torch.exp(-e["feet"].pow(2).mean(-1) / c.feet_std**2)
            + c.w_joint * torch.exp(-e["joint"].pow(2) / c.joint_std**2)
            + c.w_trunk_pos * torch.exp(-e["trunk_pos"].pow(2) / c.trunk_pos_std**2)
            + c.w_trunk_ori * torch.exp(-e["trunk_ori_sq"] / c.trunk_ori_std**2)
            + c.w_lin_vel * torch.exp(-e["lin_vel"].pow(2) / c.lin_vel_std**2)
            + c.w_ang_vel * torch.exp(-e["ang_vel"].pow(2) / c.ang_vel_std**2)
            + c.w_action_rate * (a - a_prev).pow(2).mean(-1)
        )
        return r * CONTROL_DT

    def termination(self, S, fk, f: torch.Tensor) -> torch.Tensor:
        """The trunk too low or too high, turned too far or too far away from the reference (a fall ends there too)."""
        c, e = self.cfg, self.errors(S, fk, f)
        return (e["trunk_height"] > c.term_height) | (e["trunk_ori_sq"] > c.term_ori**2) | (e["trunk_pos"] > c.term_pos)

    def soft_termination(self, S, fk, f: torch.Tensor, widths: tuple[float, float, float] = (0.012, 0.08, 0.03)) -> torch.Tensor:
        """The termination's probability with the thresholds as sigmoids of ``widths`` (the height's, the rotation's and the
        position's), differentiable in the state."""
        c, e = self.cfg, self.errors(S, fk, f)
        ori = torch.sqrt(e["trunk_ori_sq"])
        alive = ((1 - torch.sigmoid((e["trunk_height"] - c.term_height) / widths[0])) * (1 - torch.sigmoid((ori - c.term_ori) / widths[1]))
                 * (1 - torch.sigmoid((e["trunk_pos"] - c.term_pos) / widths[2])))  # fmt: skip
        return 1 - alive

    def critic_features(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, goal_frames: int) -> torch.Tensor:
        return torch.cat((self.obs(S, fk, a_prev), self.goal(S, fk, f, goal_frames)), -1)
