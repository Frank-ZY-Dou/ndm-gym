"""Humanoid motion tracking on MuJoCo-Warp: a Unitree humanoid (robots.humanoid: the H1, or the G1 for reference) follows
LAFAN1 retargeted to it (motion.sources.lafan1), one clip or many with one policy.

The floating-base path of the quadruped task (tasks.quadruped_tracking: its simulator, world model, trainer and evaluation)
with a humanoid's task. State tuple ``S = (qpos, qvel)``, MuJoCo's generalized coordinates of the robot: the pelvis' position
and quaternion and the joints, the pelvis' world linear and body-frame angular velocity and the joints'. Observation, goal,
reward and termination are analytic functions of the state, its forward kinematics ``fk`` (HumanoidKinematics.forward of
qpos: every body's position and rotation, computed once per state) and the reference frame, shared by the simulator's
transitions and the world model's imagined rollouts.

The observation and the goal are the quadruped task's with the tracked bodies in the feet's place: the pelvis' height, tilt
and body-frame velocities, the joints, the tracked bodies about the pelvis in its frame and the previous action; per goal
frame the reference pelvis' offset, rotation and velocities in the pelvis' heading frame and the joints' offsets. The reward
and the termination are the G1 / LAFAN1 task's (tasks.tracking.MDP): Gaussian kernels of the tracked bodies' positions
(the reference re-anchored to the pelvis' horizontal position and heading, as BeyondMimic's), rotations, linear and angular
velocities, and of the pelvis' position and rotation in the world at half weight, with the action-rate and soft joint-limit
penalties; the episode terminates when the pelvis' height is 0.25 m off the reference's or its tilt (the z axis' height)
0.8 off, the G1's termination exactly. ``term_pos`` (off by default) adds a term beyond the G1's: the pelvis more than that
far from the reference's (m).

The position target of an action is the next reference pose plus the action times the robot's action scale, clamped to
the targets the scene's servos accept (robots.humanoid.load_scene's ``target_range``: the joints' ranges by default).

The clips are concatenated into one reference, a frame index addressing a frame of one clip, as the quadruped task's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..core.rotations import mat_to_6d, quat_slerp, quat_to_mat, rot_angle_sq, yaw_mat
from ..motion.library import _angular_velocity, _gradient
from ..motion.sources import Clip
from ..robots.humanoid import HumanoidKinematics
from ..state import CONTROL_DT


def heading_local(H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """World vectors ``x`` (..., 3) in the heading frame ``H`` (..., 3, 3)."""
    return (H.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)


def resample(clip: Clip, dt: float = CONTROL_DT) -> torch.Tensor:
    """A clip's qpos (T, 7 + J) every ``dt`` seconds, as the library's MotionLib resamples (frames ``floor(duration / dt) + 1``,
    the position and the joints linearly interpolated, the quaternion spherically), the quaternion's sign made continuous."""
    q = clip.root_quat.double() / clip.root_quat.double().norm(dim=-1, keepdim=True)
    flip = torch.cumprod(torch.where((q[1:] * q[:-1]).sum(-1) < 0, -1.0, 1.0), 0)
    q[1:] *= flip.unsqueeze(-1)
    q = q * torch.where(q[:1, :1] < 0, -1.0, 1.0)
    x = torch.cat((clip.root_p.double(), q, clip.joints.double()), -1)
    T = x.shape[0]
    n = int((T - 1) / clip.fps / dt + 1e-9) + 1
    t = torch.arange(n, dtype=torch.float64) * dt * clip.fps
    i0 = t.floor().long().clamp(max=T - 2)
    a = (t - i0).clamp(0, 1)
    out = x[i0] + (x[i0 + 1] - x[i0]) * a[:, None]
    out[:, 3:7] = quat_slerp(x[i0, 3:7], x[i0 + 1, 3:7], a)
    return out


class HumanoidReference(NamedTuple):
    """The clips at the control rate, concatenated, (F, ...) per field: the state, the tracked bodies' positions, rotations
    and velocities (HumanoidKinematics), each frame's clip and its clip's last frame; per clip its first and last frame (C,)."""

    qpos: torch.Tensor
    qvel: torch.Tensor
    body_p: torch.Tensor
    body_R: torch.Tensor
    body_v: torch.Tensor
    body_w: torch.Tensor
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

    @property
    def R(self) -> torch.Tensor:
        """The pelvis' rotation (F, 3, 3)."""
        return self.body_R[:, 0]

    def next_frame(self, f: torch.Tensor) -> torch.Tensor:
        return torch.minimum(f + 1, self.last[f])

    def ahead(self, f: torch.Tensor, k: int) -> torch.Tensor:
        """The frame ``k`` steps after ``f``, at most its clip's last."""
        return torch.minimum(f + k, self.last[f])


REF_GROUNDS = ("none", "clip", "stance")
# ground_reference's settings: a supporting foot's ankle slower than these vertically and horizontally (m/s) with its lowest
# point within SUPPORT_BAND of the lower foot's; the clip's offset puts this percentile of its supporting feet's lowest
# points on the floor; the stance correction is smoothed by a Gaussian of this width (s) and at most this large (m)
SUPPORT_VZ, SUPPORT_VXY, SUPPORT_BAND = 0.2, 0.3, 0.03
GROUND_PERCENTILE = 25.0
STANCE_SIGMA, STANCE_MAX = 0.1, 0.05


def supporting_feet(low: torch.Tensor, ankle: torch.Tensor, dt: float = CONTROL_DT) -> torch.Tensor:
    """Per frame and foot (T, feet), whether the foot supports, from its lowest collision point's height ``low`` (T, feet)
    and its body's origin ``ankle`` (T, feet, 3), independently of the clip's height: the origin's vertical speed below
    ``SUPPORT_VZ``, its horizontal speed below ``SUPPORT_VXY`` and its lowest point within ``SUPPORT_BAND`` of the lowest foot's."""
    v = _gradient(ankle.reshape(ankle.shape[0], -1), dt).reshape(ankle.shape)
    return (v[..., 2].abs() < SUPPORT_VZ) & (v[..., :2].norm(dim=-1) < SUPPORT_VXY) & (low <= low.min(-1, keepdim=True).values + SUPPORT_BAND)


def _gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """``x`` (T,) convolved with a normalized Gaussian of ``sigma`` frames, the ends padded with zeros (so that a correction
    fades out at the clip's ends only where it is not supported there)."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    return torch.nn.functional.conv1d(x.view(1, 1, -1), k.view(1, 1, -1), padding=r).view(-1)


def ground_reference(q: torch.Tensor, kin: HumanoidKinematics, ground: str) -> torch.Tensor:
    """One clip's qpos (T, nq) at the control rate with its pelvis' height corrected against the floor; the horizontal
    trajectory, the rotation and the joints unchanged (build_reference then takes the velocities and the bodies from the
    corrected positions). The feet are the robot's collision geometry (HumanoidKinematics.feet_lowest: the full convex hull of
    the H1's ankle mesh), the supporting feet ``supporting_feet``'s.

    - "clip": one vertical offset per clip, the ``GROUND_PERCENTILE``-th percentile of the supporting feet's lowest points on
      the floor. Other projects correct each clip once as well: H2O puts the lowest link origin of a clip 0.08 m above the
      floor, ProtoMotions the lowest body at 0 (ideas only; measured on the H1's LAFAN1 retargets, both leave most of the
      stance frames floating more than 2 cm, logs/p3/measure_ground.py, while the 25th percentile of the stance feet takes
      the stance frames more than 1 cm into the floor from 28 % to 5 % at 10 % floating more than 2 cm).
    - "stance": "clip"'s offset and then, where a foot supports, the pelvis moved by the supporting feet's lowest point's
      height (the lowest of them) so that it touches the floor, both ways, at most ``STANCE_MAX``; zero where no foot
      supports, the whole correction smoothed by a Gaussian of ``STANCE_SIGMA`` (so that it fades in and out of stance)."""
    if ground == "none":
        return q
    if ground not in REF_GROUNDS:
        raise ValueError(f"ground {ground!r}: one of {REF_GROUNDS}")
    if not kin.spec.feet:
        raise ValueError(f"ground {ground!r}: {kin.spec.name} has no feet (HumanoidSpec.feet)")
    with torch.no_grad():
        fk = kin.forward(q.to(kin.device, torch.float32))
        low, ankle = kin.feet_lowest(fk).double().cpu(), fk[0][:, kin.feet].double().cpu()
    sup = supporting_feet(low, ankle)
    if not sup.any():
        raise ValueError("ground_reference: no supporting foot in the clip")
    dz = -torch.quantile(low[sup], GROUND_PERCENTILE / 100)
    q = q.clone()
    q[:, 2] += dz
    if ground == "stance":
        low = low + dz
        lowest = torch.where(sup, low, torch.inf).min(-1).values
        c = torch.where(sup.any(-1), (-lowest).clamp(-STANCE_MAX, STANCE_MAX), torch.zeros_like(lowest))
        q[:, 2] += _gaussian_smooth(c, STANCE_SIGMA / CONTROL_DT)
    return q


def build_reference(clips: list[Clip], kin: HumanoidKinematics, device: str, ground: str = "none") -> HumanoidReference:
    """The reference of LAFAN1 clips (motion.sources.lafan1) resampled to the control rate, each clip's heights corrected by
    ``ground_reference(..., ground)`` (``REF_GROUNDS``; "none": as retargeted, the default); velocities by central
    differences within each clip from the (corrected) positions, the pelvis' angular velocity in its body frame as MuJoCo's
    free joint keeps it, the bodies' from the state (HumanoidKinematics.velocities), as the simulator's and the world
    model's states give them."""
    if ground not in REF_GROUNDS:
        raise ValueError(f"ground {ground!r}: one of {REF_GROUNDS}")
    qpos, qvel, clip, first, last = [], [], [], [], []
    n = 0
    for c, x in enumerate(clips):
        q = resample(x)
        if ground != "none":
            q = ground_reference(q, kin, ground)
        R = quat_to_mat(q[:, 3:7])
        w_body = (R.transpose(-1, -2) @ _angular_velocity(R).unsqueeze(-1)).squeeze(-1)
        qpos.append(q), qvel.append(torch.cat((_gradient(q[:, :3], CONTROL_DT), w_body, _gradient(q[:, 7:], CONTROL_DT)), -1))
        clip.append(torch.full((q.shape[0],), c, dtype=torch.long))
        first.append(n), last.append(n + q.shape[0] - 1)
        n += q.shape[0]
    clip, first, last = torch.cat(clip), torch.tensor(first), torch.tensor(last)
    qpos, qvel = torch.cat(qpos).float().to(device), torch.cat(qvel).float().to(device)
    fk = kin.forward(qpos)
    V, W = kin.velocities(fk, qvel)
    t = kin.tracked
    return HumanoidReference(qpos, qvel, fk[0][:, t], fk[1][:, t], V[:, t], W[:, t], clip.to(device), last[clip].to(device), first.to(device),
                             last.to(device))  # fmt: skip


@dataclass
class RewardCfg:
    """The G1 / LAFAN1 task's reward and termination (tasks.tracking.RewardCfg); ``term_pos``: the pelvis' distance from the
    reference as a further termination (inf: none, the G1's)."""

    body_pos_std: float = 0.3  # m
    body_ori_std: float = 0.4  # rad
    body_linvel_std: float = 1.0  # m/s
    body_angvel_std: float = 3.14  # rad/s
    root_pos_std: float = 0.3  # m, in the world
    root_ori_std: float = 0.4  # rad
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_linvel: float = 1.0
    w_body_angvel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25  # m, the pelvis' height from its reference's
    term_tilt: float = 0.8  # the pelvis' z axis' height from its reference's
    term_pos: float = float("inf")  # m, the pelvis' distance from its reference (inf: no such term)

    def scaled(self, k: float) -> RewardCfg:
        """The tracking kernels' widths scaled by ``k`` (training's --reward_std_scale)."""
        widths = ("body_pos_std", "body_ori_std", "body_linvel_std", "body_angvel_std", "root_pos_std", "root_ori_std")
        return RewardCfg(**{**self.__dict__, **{n: getattr(self, n) * k for n in widths}})


class HumanoidTask:
    log_errors = (("root_mm", "root_pos"), ("mpjpe_l_mm", "body_local"))  # the training log's errors (train_quadruped)

    def __init__(self, kin: HumanoidKinematics, ref: HumanoidReference, cfg: RewardCfg = RewardCfg(), goal_anchor: str = "robot",
                 score_ref: HumanoidReference | None = None):
        """``ref``: the reference the policy follows (its goal, the feed-forward pose of its position targets, the
        simulator's starts); ``score_ref``: the one the reward, the errors and the termination measure against, ``ref``
        by default (another: the evaluation's, the same clips and frames with another height correction, ground_reference)."""
        if goal_anchor not in ("robot", "reference"):
            raise ValueError(f"goal_anchor {goal_anchor!r}: 'robot' or 'reference'")
        if score_ref is not None and (score_ref.frames != ref.frames or not torch.equal(score_ref.clip_first, ref.clip_first)):
            raise ValueError("score_ref: the reference's clips and frames expected")
        self.kin, self.ref, self.cfg, self.goal_anchor = kin, ref, cfg, goal_anchor
        self.score_ref = ref if score_ref is None else score_ref
        J, B = kin.num_joints, len(kin.tracked)
        self.nq, self.num_joints = 7 + J, J
        self.state_dim = 2 * (7 + J) - 1
        self.obs_dim = 1 + 6 + 3 + 3 + J + J + 3 * (B - 1) + J
        self.goal_frame_dim = 3 + 6 + J + 3 + 3
        self.scale = torch.tensor(kin.spec.action_scale, device=ref.qpos.device)
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        self.max_step_reward = (cfg.w_body_pos + cfg.w_body_ori + cfg.w_body_linvel + cfg.w_body_angvel + cfg.w_root_pos + cfg.w_root_ori) * CONTROL_DT

    def unflatten(self, x: torch.Tensor):
        return x[..., : self.nq], x[..., self.nq :]

    def tracked(self, fk):
        """The tracked bodies' positions (..., B, 3) and rotations (..., B, 3, 3) of every body's ``fk``."""
        return fk[0][..., self.kin.tracked, :], fk[1][..., self.kin.tracked, :, :]

    def setpoint(self, a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Joint position targets for policy output ``a`` at frame ``f``: offsets about the next frame's pose, clamped to the
        targets the servos accept (HumanoidKinematics.u_lo / u_hi, the scene's ctrlrange: the joints' ranges by default,
        robots.humanoid.TARGET_RANGES), so that the simulator, the world model's imagined steps and the evaluation all
        execute the same target."""
        return torch.clamp(self.ref.qpos[self.ref.next_frame(f), 7:] + self.scale * a, self.kin.u_lo, self.kin.u_hi)

    def obs(self, S, fk, a_prev: torch.Tensor) -> torch.Tensor:
        """The pelvis' height, tilt (its rotation in its heading frame) and body-frame velocities, the joints, the tracked
        bodies about the pelvis in its frame and the previous action."""
        qpos, qvel = S
        bp, _ = self.tracked(fk)
        R = fk[1][..., 0, :, :]
        Rt = R.transpose(-1, -2)
        rel = (bp[..., 1:, :] - bp[..., :1, :]) @ R  # rows x^T R = (R^T x)^T
        return torch.cat((qpos[..., 2:3], mat_to_6d(yaw_mat(R).transpose(-1, -2) @ R), (Rt @ qvel[..., :3].unsqueeze(-1)).squeeze(-1),
                          qvel[..., 3:6], qpos[..., 7:], 0.1 * qvel[..., 6:], rel.flatten(-2), a_prev), -1)  # fmt: skip

    def goal(self, S, fk, f: torch.Tensor, goal_frames: int) -> torch.Tensor:
        """The next ``goal_frames`` reference frames in the pelvis' heading frame: the pelvis' offset, rotation and
        velocities (the angular one in the reference pelvis' frame), and the joints' offsets. The offset is from the pelvis
        (``goal_anchor="robot"``), or, with ``goal_anchor="reference"``, horizontally from the reference pelvis at ``f``
        (vertically still from the pelvis): no feedback of the horizontal drift, as the G1's goal (motion.library.MotionLib.goal)
        has none."""
        qpos, _ = S
        H = yaw_mat(fk[1][..., 0, :, :])
        origin = qpos[..., :3]
        if self.goal_anchor == "reference":
            origin = torch.cat((self.ref.qpos[f, :2], qpos[..., 2:3]), -1)
        parts = []
        for k in range(1, goal_frames + 1):
            g = self.ref.ahead(f, k)
            parts += [heading_local(H, self.ref.qpos[g, :3] - origin), mat_to_6d(H.transpose(-1, -2) @ self.ref.R[g]), self.ref.qpos[g, 7:] - qpos[..., 7:],
                      heading_local(H, self.ref.qvel[g, :3]), self.ref.qvel[g, 3:6]]  # fmt: skip
        return torch.cat(parts, -1)

    def anchored(self, S, fk, f: torch.Tensor):
        """The reference's tracked bodies at frame ``f`` re-anchored to the pelvis' horizontal position and heading (the G1
        task's, BeyondMimic's): positions (..., B, 3), rotations (..., B, 3, 3) and the heading change (..., 3, 3)."""
        qpos, R, ref = S[0], fk[1][..., 0, :, :], self.score_ref
        rp, rR = ref.qpos[f, :3], ref.R[f]
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((qpos[..., :2], rp[..., 2:3]), -1)
        bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (ref.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        return bp, dR.unsqueeze(-3) @ ref.body_R[f], dR

    def errors(self, S, fk, f: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tracking errors against frame ``f``: per tracked body its distance in the world and about the pelvis (the G1
        evaluation's MPJPE, global and local; m), the pelvis' position, height (m) and squared rotation angle, its tilt
        error (the termination's), the joints (rad, mean absolute) and the pelvis' velocities."""
        qpos, qvel = S
        bp, _ = self.tracked(fk)
        ref = self.score_ref
        rp = ref.qpos[f, :3]
        R = fk[1][..., 0, :, :]
        return {
            "body_world": torch.linalg.vector_norm(bp - ref.body_p[f], dim=-1),
            "body_local": torch.linalg.vector_norm((bp - qpos[..., None, :3]) - (ref.body_p[f] - rp.unsqueeze(-2)), dim=-1),
            "root_pos": torch.linalg.vector_norm(qpos[..., :3] - rp, dim=-1),
            "root_height": (qpos[..., 2] - rp[..., 2]).abs(),
            "root_ori_sq": rot_angle_sq(R, ref.R[f]),
            "root_tilt": (R[..., 2, 2] - ref.R[f][..., 2, 2]).abs(),
            "joint": (qpos[..., 7:] - ref.qpos[f, 7:]).abs().mean(-1),
            "lin_vel": torch.linalg.vector_norm(qvel[..., :3] - ref.qvel[f, :3], dim=-1),
            "ang_vel": torch.linalg.vector_norm(qvel[..., 3:6] - ref.qvel[f, 3:6], dim=-1),
        }

    @staticmethod
    def step_metrics(errs: dict) -> dict:
        """The reported errors of ``errors`` (any leading shape; evaluate_humanoid): MPJPE in the world and about the pelvis
        (mm), the pelvis' position (mm) and rotation (rad), the joints (rad and degrees, mean absolute)."""
        return {"mpjpe_g_mm": 1000 * errs["body_world"].mean(-1), "mpjpe_l_mm": 1000 * errs["body_local"].mean(-1), "root_mm": 1000 * errs["root_pos"],
                "root_ori_rad": errs["root_ori_sq"].sqrt(), "joint_rad": errs["joint"], "joint_deg": torch.rad2deg(errs["joint"])}  # fmt: skip

    def reward_terms(self, S, fk, f: torch.Tensor, vel=None) -> dict[str, torch.Tensor]:
        """The G1 task's squared errors (tasks.tracking.MDP.reward): the tracked bodies' positions about the re-anchored
        reference, rotations, linear and angular velocities (means over the bodies), the pelvis' position and rotation.
        ``vel``: every body's velocities (HumanoidKinematics.velocities), if computed already."""
        qpos, qvel = S
        ref = self.score_ref
        bp, bR = self.tracked(fk)
        V, W = self.kin.velocities(fk, qvel) if vel is None else vel
        t = self.kin.tracked
        ref_bp, ref_bR, _ = self.anchored(S, fk, f)
        return {
            "body_pos": (ref_bp - bp).pow(2).sum(-1).mean(-1),
            "body_ori": rot_angle_sq(ref_bR, bR).mean(-1),
            "body_linvel": (ref.body_v[f] - V[..., t, :]).pow(2).sum(-1).mean(-1),
            "body_angvel": (ref.body_w[f] - W[..., t, :]).pow(2).sum(-1).mean(-1),
            "root_pos": (ref.qpos[f, :3] - qpos[..., :3]).pow(2).sum(-1),
            "root_ori": rot_angle_sq(ref.R[f], fk[1][..., 0, :, :]),
        }

    def reward(self, S, fk, f: torch.Tensor, a: torch.Tensor, a_prev: torch.Tensor, vel=None) -> torch.Tensor:
        """The tracking reward of reaching ``S`` at frame ``f`` with action ``a`` after ``a_prev`` (already times dt)."""
        c, e, q = self.cfg, self.reward_terms(S, fk, f, vel), S[0][..., 7:]
        r = (
            c.w_body_pos * torch.exp(-e["body_pos"] / c.body_pos_std**2)
            + c.w_body_ori * torch.exp(-e["body_ori"] / c.body_ori_std**2)
            + c.w_body_linvel * torch.exp(-e["body_linvel"] / c.body_linvel_std**2)
            + c.w_body_angvel * torch.exp(-e["body_angvel"] / c.body_angvel_std**2)
            + c.w_root_pos * torch.exp(-e["root_pos"] / c.root_pos_std**2)
            + c.w_root_ori * torch.exp(-e["root_ori"] / c.root_ori_std**2)
            + c.w_action_rate * (a - a_prev).pow(2).sum(-1)
            + c.w_joint_limit * (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        )
        return r * CONTROL_DT

    def _term_errors(self, S, fk, f: torch.Tensor):
        qpos = S[0]
        rp, R = self.score_ref.qpos[f, :3], fk[1][..., 0, :, :]
        return (qpos[..., 2] - rp[..., 2]).abs(), (R[..., 2, 2] - self.score_ref.R[f][..., 2, 2]).abs(), torch.linalg.vector_norm(qpos[..., :3] - rp, dim=-1)

    def termination(self, S, fk, f: torch.Tensor) -> torch.Tensor:
        """The pelvis too high or low or tilted too far from the reference's (a fall ends there too), or, with ``term_pos``,
        too far away."""
        c = self.cfg
        h, tilt, pos = self._term_errors(S, fk, f)
        out = (h > c.term_height) | (tilt > c.term_tilt)
        return out if c.term_pos == float("inf") else out | (pos > c.term_pos)

    def soft_termination(self, S, fk, f: torch.Tensor, widths: tuple[float, float, float] = (0.02, 0.04, 0.05)) -> torch.Tensor:
        """The termination's probability with the thresholds as sigmoids of ``widths`` (the height's and the tilt's are the
        G1 task's, tasks.tracking.MDP.soft_termination), differentiable in the state."""
        c = self.cfg
        h, tilt, pos = self._term_errors(S, fk, f)
        alive = (1 - torch.sigmoid((h - c.term_height) / widths[0])) * (1 - torch.sigmoid((tilt - c.term_tilt) / widths[1]))
        if c.term_pos != float("inf"):
            alive = alive * (1 - torch.sigmoid((pos - c.term_pos) / widths[2]))
        return 1 - alive

    def critic_features(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, goal_frames: int) -> torch.Tensor:
        return torch.cat((self.obs(S, fk, a_prev), self.goal(S, fk, f, goal_frames)), -1)
