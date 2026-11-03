"""HDMI's human-object interaction task (LeCAR-Lab/HDMI @ 32282f6): observation groups, reward groups and terminations
as functions of explicit tensors, shared by the simulator (labels transitions) and the world models (imagined rollouts).
The per-task data is in :mod:`ndm_gym.tasks.hdmi_tasks`.

Every formula follows docs/hdmi_task_spec.md (§ below), which cites HDMI's source lines. Conventions:

* Orientations are rotation matrices (v_world = R v_body), as elsewhere in ndm_gym. HDMI's quaternion formulas are
  evaluated in matrix form: quat_apply_inverse(q, v) = R^T v, q_a* (x) q_b = R_a^T R_b, yaw_quat(q) = the heading
  rotation of R (yaw = atan2(R10, R00)), |axis_angle_from_quat(q_a* (x) q_b)| = the rotation angle of R_a^T R_b.
* Joints follow HDMI's simulator order (JOINT_NAMES, §3.3); its first 23 are the action and tracking joints. Keypoints
  are HDMI's 16 tracking bodies in simulator order (KEYPOINT_NAMES, §3.4); keypoint 0, the pelvis, is the root.
* Robot, object and reference positions share one frame (HDMI shifts the reference by the env origin, CMD:171).
* Timing (§0): step n runs the physics, HDMI's update hooks, the rewards, the command update CU(n), the observations
  and the terminations. Rewards and terminations compare the state after the step (time n) with the reference of the
  previous command update CU(n-1), whose index 0 is frame f_{n-1} = s + n - 1, while several terms read CU(n-1)'s
  snapshots of the state at time n-1; contact flags lead the poses by one frame. HDMIInput therefore holds both
  states and both reference windows. After a reset HDMI returns all-zero observations and, at n = 1, still holds the
  previous episode's command buffers (its PPO drops those transitions, §0): what to pass then is the caller's choice.
* Not here: truncation (clip end t >= L or the episode length, §0), HDMI's running observation normalizer (§4), the
  ``ref_joint_pos_`` group (the teacher's residual base, §4.5) and the always-zero debug reward group (§5.6).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import NamedTuple

import torch

from ..core.rotations import yaw_mat
from ..state import CONTROL_DT
from .hdmi_tasks import HDMITaskCfg, Term, Termination

JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)  # Isaac's order of HDMI's G1 (REC:3-31)  # fmt: skip
NUM_ACTIONS = 23  # every joint but the six wrist joints, i.e. the first 23; also the tracking joints (Y:27-43, 63-70)
TRACKING_JOINT_NAMES = JOINT_NAMES[:NUM_ACTIONS]
KEYPOINT_NAMES = (
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link",
    "right_knee_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
)  # HDMI's tracking keypoints in simulator order (Y:52-61, CMD:61-63)  # fmt: skip
FEET = (10, 11)  # left/right ankle_roll_link: the feet of the feet terms and body_pos_b / body_vel_b (Y:151-152, 228-230)
FEET_SLICE = slice(FEET[0], FEET[-1] + 1)  # the same, as a slice: indexing with a tuple copies it to the device every call
assert FEET == tuple(range(FEET[0], FEET[-1] + 1))
HEIGHT_BODIES = (0, 5, 10, 11)  # body_height's pelvis, torso and ankles, in simulator order (Y:153)
FUTURE_STEPS = (1, 2, 8, 16, 32)  # Y:93
PHYSICS_DT = 0.005  # Y:18
DECIMATION = 4  # physics substeps per control step (ENV:139)
EFFORT_LIMITS = (88, 88, 88, 139, 139, 50, 88, 88, 50, 139, 139, 25, 25, 50, 50, 25, 25, 50, 50, 25, 25, 25, 25, 25, 25, 5, 5, 5, 5)  # §3.3
CONTACT_FORCE_THRESHOLD = 1.0  # N, a body is in contact when its net contact force exceeds it [IL] (§3.4)
GRAVITY = 9.81  # impact_force_l2 normalizes by the robot's weight (RF:87-88)
GROUPS = ("tracking", "object_tracking", "loco", "feet")

# observation histories (Y:125-140) and noise: Gaussian clipped at 3 std (OC:11-12) unless noted
POLICY_JOINT_HISTORY = (0, 1, 2, 3, 4, 8)
PRIV_HISTORY = 9  # history steps 0..8
NOISE_ANG_VEL, NOISE_GRAVITY, NOISE_JOINT_POS = 0.05, 0.05, 0.015
NOISE_OBJECT = 0.01  # object_xy_b, object_heading_b, ref_contact_pos_b (the latter adds a second draw clipped at 1 std)
EPISODIC_CONTACT_NOISE = 0.02  # ref_contact_pos_b: uniform per env, end effector and axis, drawn at reset (OBS:281-283)


class HDMIState(NamedTuple):
    """Robot and object at one instant, read from the simulator or predicted by a world model; world frame."""

    body_pos: torch.Tensor  # (N, 16, 3) keypoint link-frame positions
    body_rot: torch.Tensor  # (N, 16, 3, 3) keypoint link-frame orientations
    body_lin_vel: torch.Tensor  # (N, 16, 3) keypoint centre-of-mass linear velocities (Isaac's body_com_lin_vel_w)
    body_ang_vel: torch.Tensor  # (N, 16, 3) keypoint angular velocities
    joint_pos: torch.Tensor  # (N, 29)
    joint_vel: torch.Tensor  # (N, 29)
    obj_pos: torch.Tensor  # (N, 3) the tracked object's root link (a door's frame, the chair's base, §9)
    obj_rot: torch.Tensor  # (N, 3, 3)
    eef_pos: torch.Tensor  # (N, E, 3) end-effector points e_e (HDMITask.contact_points)
    target_pos: torch.Tensor  # (N, E, 3) contact target points c_e on the object's contact body (HDMITask.contact_points)
    eef_force: torch.Tensor  # (N, E, 3) contact force on each end effector from the object's contact body, in that body's frame (§5.3)
    obj_joint_pos: torch.Tensor | None = None  # (N,) hinge angle of an articulated object
    obj_joint_vel: torch.Tensor | None = None  # (N,)

    @property
    def root_pos(self) -> torch.Tensor:
        return self.body_pos[:, 0]

    @property
    def root_rot(self) -> torch.Tensor:
        return self.body_rot[:, 0]


class HDMIRef(NamedTuple):
    """The reference one command update loads (reference_window): index k of the second axis is frame t + F[k] for
    poses and joints and frame t + 1 + F[k] for the contact flags, t being the frame counter before the update."""

    body_pos: torch.Tensor  # (N, 5, 16, 3)
    body_rot: torch.Tensor  # (N, 5, 16, 3, 3)
    body_lin_vel: torch.Tensor  # (N, 5, 16, 3)
    body_ang_vel: torch.Tensor  # (N, 5, 16, 3)
    joint_pos: torch.Tensor  # (N, 5, 23) tracking joints
    joint_vel: torch.Tensor  # (N, 5, 23)
    obj_pos: torch.Tensor  # (N, 5, 3)
    obj_rot: torch.Tensor  # (N, 5, 3, 3)
    contact: torch.Tensor  # (N, 5, C) bool: the end effectors should touch the object (§1.1); C = 1 is shared by all
    phase: torch.Tensor  # (N,) (t + 1) / L, the motion phase after the update
    obj_joint_pos: torch.Tensor | None = None  # (N, 5) hinge


class HDMIInput(NamedTuple):
    """Everything step n needs (§0). From ``prev`` (CU(n-1)'s snapshots) the terms read the root pose, the keypoints
    (priv observations), the object's pose and hinge angle (rewards), the contact points and forces; from ``now`` the
    keypoints and joints (rewards, terminations, observations), the object (terminations, observations) and its hinge
    (observations). ``now``'s contact points and forces become the next step's ``prev``."""

    now: HDMIState  # after this step's physics (time n): HDMI's direct simulator reads
    prev: HDMIState  # after the previous step (time n-1): the snapshots HDMI's command took in CU(n-1)
    ref: HDMIRef  # CU(n-1)'s reference: rewards, terminations and the observation terms HDMI computes in update()
    ref_next: HDMIRef  # CU(n)'s reference: future joints, contact flags and phase, which HDMI observes after CU(n)
    joint_pos_sub: torch.Tensor  # (N, 2, 29) joint positions after physics substeps 2 and 3 (HDMI's substep % 2 slots, OC:101-102)
    joint_vel_sub: torch.Tensor  # (N, 2, 29) joint velocities after physics substeps 2 and 3 (RC:131-132)
    joint_offset: torch.Tensor  # (N, 29) random joint offsets (§7): added to the PD targets, subtracted from observed positions
    applied_action: torch.Tensor  # (N, 23) the filtered action after the last substep (§3.1)
    applied_torque: torch.Tensor  # (N, 29) the actuators' torque estimate after the last substep [IL]
    action_buf: torch.Tensor  # (N, 23, 3) raw policy actions a_n, a_{n-1}, a_{n-2} on the last axis, zero before the episode (ACT:98-102)
    feet_force: torch.Tensor  # (N, 4, 2, 3) net contact force on the left/right ankle_roll links after each physics substep, world frame
    obj_joint_torque: torch.Tensor | None = None  # (N,) hinge torque applied in the last substep (OBJ:33-40)


class HDMIClip(NamedTuple):
    """One clip's reference at 50 fps in HDMI's orders (§1), T frames."""

    body_pos: torch.Tensor  # (T, 16, 3)
    body_rot: torch.Tensor  # (T, 16, 3, 3)
    body_lin_vel: torch.Tensor  # (T, 16, 3)
    body_ang_vel: torch.Tensor  # (T, 16, 3)
    joint_pos: torch.Tensor  # (T, 23)
    joint_vel: torch.Tensor  # (T, 23)
    obj_pos: torch.Tensor  # (T, 3)
    obj_rot: torch.Tensor  # (T, 3, 3)
    contact: torch.Tensor  # (T, C) bool, the clip's flags row by row, unshifted: reference_window applies HDMI's one-frame lead
    obj_joint_pos: torch.Tensor | None = None  # (T,)


class FeetContact(NamedTuple):
    """The feet's contact-sensor state the feet terms read [IL], kept by HDMITask.advance_feet."""

    first_contact: torch.Tensor  # (N, 2) bool: contact made within the last control step (compute_first_contact(dt))
    contact_time: torch.Tensor  # (N, 2) s, current contact time
    last_air_time: torch.Tensor  # (N, 2) s, duration of the last swing


class HDMIOutput(NamedTuple):
    """One step's observations, rewards and terminations (HDMITask.step)."""

    obs: dict[str, torch.Tensor]  # "command" (N, 356), "policy" (N, 249), "object" (N, 4 + 3E), "priv" (N, HDMITask.obs_dims["priv"])
    reward: torch.Tensor  # (N, 4) per group (GROUPS): the sum of its enabled terms, times dt; HDMI's PPO maximizes their sum (§5.1)
    terms: dict[str, torch.Tensor]  # "group/term": weight x value, (N,); disabled terms included, as HDMI logs them
    terminated: torch.Tensor  # (N,) bool
    term_flags: dict[str, torch.Tensor]  # per termination term, (N,) bool


def action_pipeline(action: torch.Tensor, action_buf: torch.Tensor, a_filt: torch.Tensor, delay: torch.Tensor,
                    alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """HDMI's action delay and filter over one control step (§3.1). ``action`` (N, 23) enters the buffer (N, 23, 3) of raw
    actions; at physics substep i the action delayed by ``delay`` (N,) substeps drives ã <- ã + alpha (a_used - ã),
    ``alpha`` (N, 1). Returns the filtered action after each substep (N, 4, 23) and the buffer and ã after the step.
    It does not depend on the physics, so imagination reproduces the simulator's PD targets exactly."""
    buf = torch.cat((action.unsqueeze(-1), action_buf[..., :-1]), -1)
    envs = torch.arange(action.shape[0], device=action.device)
    out = []
    for i in range(DECIMATION):
        idx = torch.div(delay - i + DECIMATION - 1, DECIMATION, rounding_mode="floor").clamp(0, buf.shape[-1] - 1)
        a_filt = a_filt + alpha * (buf[envs, :, idx] - a_filt)  # a_used = a_{k - idx}: a pure delay of d substeps
        out.append(a_filt)
    return torch.stack(out, 1), buf, a_filt


def reference_window(clip: HDMIClip, t: torch.Tensor, steps: torch.Tensor | None = None) -> HDMIRef:
    """The reference a command update with frame counter ``t`` (N,) loads before incrementing it (CMD:247, 290, 565-568):
    poses and joints at frames t + F, contact flags at t + 1 + F (one frame ahead, §0), frames past the clip's end held
    at its last frame (MOT:333-338), and the phase (t + 1) / L that ref_motion_phase reads after the update. ``steps``:
    FUTURE_STEPS as a tensor on t's device, which hot loops pass so that it is not copied to the device every call."""
    T = clip.body_pos.shape[0]
    steps = torch.tensor(FUTURE_STEPS, device=t.device) if steps is None else steps
    f = (t.unsqueeze(-1) + steps).clamp(max=T - 1)
    fc = (t.unsqueeze(-1) + 1 + steps).clamp(max=T - 1)
    return HDMIRef(clip.body_pos[f], clip.body_rot[f], clip.body_lin_vel[f], clip.body_ang_vel[f], clip.joint_pos[f], clip.joint_vel[f],
                   clip.obj_pos[f], clip.obj_rot[f], clip.contact[fc], (t + 1).to(clip.body_pos.dtype) / T,
                   None if clip.obj_joint_pos is None else clip.obj_joint_pos[f])  # fmt: skip


# ---------------- frames ----------------
# 3x3 products are written as elementwise products and sums (as core.kinematics.mat_vec): torch.compile fuses them with
# their neighbours, whereas a batched 3x3 ``@`` is a separate cuBLAS call, which also runs in TF32 for large batches
# when TF32 matmuls are allowed (as in training)
def _inv(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R^T v for (..., 3, 3) and (..., 3) with broadcast batch dims (HDMI's quat_apply_inverse)."""
    return (R * v.unsqueeze(-1)).sum(-2)


def _mul(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return (R * v.unsqueeze(-2)).sum(-1)


def relative_rot(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Ra^T Rb for (..., 3, 3) with broadcast batch dims (HDMI's q_a* (x) q_b)."""
    return (Ra.unsqueeze(-1) * Rb.unsqueeze(-2)).sum(-3)


def _flat(p: torch.Tensor) -> torch.Tensor:
    return torch.cat((p[..., :2], torch.zeros_like(p[..., 2:])), -1)


def rot_angle(R: torch.Tensor) -> torch.Tensor:
    """Rotation angle in [0, pi] of R, i.e. |axis_angle_from_quat(q)|; atan2 keeps it accurate near 0 and pi."""
    s = torch.stack((R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), -1).norm(dim=-1)
    return torch.atan2(s, R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1)


def _yaw_angle(R: torch.Tensor) -> torch.Tensor:
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])  # OBS:260-263


def _wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    w = (a + torch.pi) % (2 * torch.pi)  # Isaac Lab's wrap_to_pi
    return torch.where((w == 0) & (a > 0), torch.pi, w - torch.pi)


def heading_pos(pos: torch.Tensor, root_pos: torch.Tensor, root_rot: torch.Tensor) -> torch.Tensor:
    """Positions (..., B, 3) in the root's heading frame, R(yaw q_root)^T (p - p_root) with the root's height set to 0
    (REW:62-89, OBS:95-115); the root's batch dims are those of ``pos`` without the body axis."""
    return _inv(yaw_mat(root_rot).unsqueeze(-3), pos - _flat(root_pos).unsqueeze(-2))


def heading_rot(rot: torch.Tensor, root_rot: torch.Tensor) -> torch.Tensor:
    """Orientations (..., B, 3, 3) relative to the root's heading, yaw(q_root)* (x) q (REW:186-208)."""
    return relative_rot(yaw_mat(root_rot).unsqueeze(-3), rot)


def contact_points(eef_pos, eef_rot, eef_offset, obj_body_pos, obj_body_rot, target_offset) -> tuple[torch.Tensor, torch.Tensor]:
    """End-effector points e_e = p_eef + R_eef d_e (N, E, 3) and contact targets c_e = p_ob + R_ob o_e (N, E, 3) on the
    object's contact body (CMD:571-577); ``target_offset`` (N, E, 3) already multiplied by the env's object scale
    (CMD:427-439), ``eef_offset`` (E, 3) or (N, E, 3)."""
    return eef_pos + _mul(eef_rot, eef_offset), obj_body_pos.unsqueeze(-2) + _mul(obj_body_rot.unsqueeze(-3), target_offset)


def object_frame_force(obj_body_rot: torch.Tensor, force_w: torch.Tensor) -> torch.Tensor:
    """End-effector contact forces (N, E, 3) from the world frame to the object contact body's frame (CMD:584)."""
    return _inv(obj_body_rot.unsqueeze(-3), force_w)


# ---------------- rewards (unweighted values, (N,)) ----------------
# tracking group (§5.2): exp(-mean error / sigma) of unsquared norms; robot keypoints and joints at time n against frame
# f_{n-1}; the local terms put the robot's keypoints in the heading frame of CU(n-1)'s root snapshot (time n-1)
def keypoint_pos_tracking_local(x: HDMIInput, idx, sigma: float) -> torch.Tensor:
    r = heading_pos(x.ref.body_pos[:, 0, idx], x.ref.body_pos[:, 0, 0], x.ref.body_rot[:, 0, 0])
    b = heading_pos(x.now.body_pos[:, idx], x.prev.root_pos, x.prev.root_rot)
    return torch.exp(-(r - b).norm(dim=-1).mean(-1) / sigma)


def keypoint_ori_tracking_local(x: HDMIInput, idx, sigma: float) -> torch.Tensor:
    r = heading_rot(x.ref.body_rot[:, 0, idx], x.ref.body_rot[:, 0, 0])
    b = heading_rot(x.now.body_rot[:, idx], x.prev.root_rot)
    return torch.exp(-rot_angle(relative_rot(r, b)).mean(-1) / sigma)


def keypoint_pos_tracking(x: HDMIInput, idx, sigma: float) -> torch.Tensor:
    return torch.exp(-(x.ref.body_pos[:, 0, idx] - x.now.body_pos[:, idx]).norm(dim=-1).mean(-1) / sigma)


def keypoint_ori_tracking(x: HDMIInput, idx, sigma: float) -> torch.Tensor:
    return torch.exp(-rot_angle(relative_rot(x.ref.body_rot[:, 0, idx], x.now.body_rot[:, idx])).mean(-1) / sigma)


def keypoint_lin_vel_tracking(x: HDMIInput, idx, sigma: float) -> torch.Tensor:
    return torch.exp(-(x.ref.body_lin_vel[:, 0, idx] - x.now.body_lin_vel[:, idx]).norm(dim=-1).mean(-1) / sigma)


def keypoint_ang_vel_tracking(x: HDMIInput, idx, sigma: float) -> torch.Tensor:
    return torch.exp(-(x.ref.body_ang_vel[:, 0, idx] - x.now.body_ang_vel[:, idx]).norm(dim=-1).mean(-1) / sigma)


def joint_pos_tracking(x: HDMIInput, idx, sigma: float) -> torch.Tensor:
    return torch.exp(-(x.ref.joint_pos[:, 0, idx] - x.now.joint_pos[:, idx]).abs().mean(-1) / sigma)


def joint_vel_tracking(x: HDMIInput, idx, sigma: float) -> torch.Tensor:
    return torch.exp(-(x.ref.joint_vel[:, 0, idx] - x.now.joint_vel[:, idx]).abs().mean(-1) / sigma)


# object-tracking group (§5.3): CU(n-1)'s snapshots (time n-1) against frame f_{n-1}, contact flags at frame f_n
def object_pos_tracking(x: HDMIInput, sigma: float) -> torch.Tensor:
    return torch.exp(-(x.ref.obj_pos[:, 0] - x.prev.obj_pos).norm(dim=-1) / sigma)


def object_ori_tracking(x: HDMIInput, sigma: float) -> torch.Tensor:
    return torch.exp(-rot_angle(relative_rot(x.ref.obj_rot[:, 0], x.prev.obj_rot)) / sigma)


def object_joint_pos_tracking(x: HDMIInput, sigma: float) -> torch.Tensor:
    return torch.exp(-(x.ref.obj_joint_pos[:, 0] - x.prev.obj_joint_pos).abs() / sigma)


def _contact_factor(x: HDMIInput, pos_sigma, pos_tolerance, frc_sigma, frc_thres, per_axis) -> torch.Tensor:
    """exp(-d_e / pos_sigma) exp(phi_e / frc_sigma) per end effector (N, E) (REW:390-405); phi_e is min(|F_e| - thres, 0)
    or, in HDMI's branch for a non-float threshold, the mean over the axes of min(|F_e,axis| - thres, 0)."""
    d = ((x.prev.eef_pos - x.prev.target_pos).norm(dim=-1) - pos_tolerance).clamp_min(0.0)
    F = x.prev.eef_force
    phi = (F.abs() - frc_thres).clamp_max(0.0).mean(-1) if per_axis else (F.norm(dim=-1) - frc_thres).clamp_max(0.0)
    return torch.exp(-d / pos_sigma) * torch.exp(phi / frc_sigma)


def eef_contact_exp(x: HDMIInput, pos_sigma, pos_tolerance, frc_sigma, frc_thres, per_axis, gain) -> torch.Tensor:
    """mean_e [g_e gain r_e + 1 - g_e]: exactly 1 outside the annotated contact (REW:399-408)."""
    r = _contact_factor(x, pos_sigma, pos_tolerance, frc_sigma, frc_thres, per_axis)
    g = x.ref.contact[:, 0].to(r.dtype)  # (N, C); one column broadcasts over the end effectors (REW:405-407)
    return (r * g * gain + 1 - g).mean(-1)


def eef_contact_exp_max(x: HDMIInput, pos_sigma, pos_tolerance, frc_sigma, frc_thres, per_axis) -> torch.Tensor:
    """max_e r_e while any end effector's flag is on, else 0 (REW:441-449; roll_ball)."""
    r = _contact_factor(x, pos_sigma, pos_tolerance, frc_sigma, frc_thres, per_axis)
    return r.amax(-1) * x.ref.contact[:, 0].any(-1).to(r.dtype)


CONTACT_REWARDS = ("eef_contact_exp", "eef_contact_exp_max")


def contact_reward(cfg) -> tuple[str, dict]:
    """The task's enabled contact reward (eef_contact_exp, or roll_ball's eef_contact_exp_max) and its factor's parameters."""
    found = [t for ts in cfg.rewards.values() for t in ts.values() if t.enabled and t.kind in CONTACT_REWARDS]
    if len(found) != 1:
        raise ValueError(f"{cfg.name}: {len(found)} enabled contact rewards, expected 1")
    return found[0].kind, {k: v for k, v in found[0].params.items() if k != "gain"}


def contact_error(s: HDMIState, flag: torch.Tensor, kind: str, pos_sigma, pos_tolerance, frc_sigma, frc_thres, per_axis) -> torch.Tensor:
    """HDMI's contact reward as an error (N,): -log of its factor per end effector, d_e / pos_sigma plus the force's
    shortfall below frc_thres over frc_sigma, from the state ``s``; the mean over the end effectors of the flagged ones'
    (eef_contact_exp; ``flag`` (N, C), one column broadcasts) or the best end effector's while any flag is on
    (eef_contact_exp_max)."""
    d = ((s.eef_pos - s.target_pos).norm(dim=-1) - pos_tolerance).clamp_min(0.0)
    F = s.eef_force
    short = (frc_thres - F.abs()).clamp_min(0.0).mean(-1) if per_axis else (frc_thres - F.norm(dim=-1)).clamp_min(0.0)
    c = d / pos_sigma + short / frc_sigma
    g = flag.to(c.dtype)
    return c.amin(-1) * g.amax(-1) if kind == "eef_contact_exp_max" else (c * g).mean(-1)


# loco group (§5.4)
def action_rate_l2(x: HDMIInput) -> torch.Tensor:
    return -(x.action_buf[..., 0] - x.action_buf[..., 1]).square().sum(-1)


def joint_vel_l2(x: HDMIInput, idx) -> torch.Tensor:
    return -x.joint_vel_sub[..., idx].mean(1).square().clamp_max(5.0).sum(-1)


def joint_pos_limits(x: HDMIInput, idx, soft_lo: torch.Tensor, soft_hi: torch.Tensor) -> torch.Tensor:
    q = x.now.joint_pos[:, idx]
    return -((soft_lo - q).clamp_min(0.0) + (q - soft_hi).clamp_min(0.0)).sum(-1)


def joint_torque_limits(x: HDMIInput, idx, soft_effort: torch.Tensor) -> torch.Tensor:
    tau = x.applied_torque[:, idx]
    return -((tau / soft_effort - 1.0).clamp_min(0.0) + (-tau / soft_effort - 1.0).clamp_min(0.0)).sum(-1)


def survival(x: HDMIInput) -> torch.Tensor:
    return torch.ones_like(x.now.body_pos[:, 0, 0])


def feet_stumble(x: HDMIInput) -> torch.Tensor:
    """-mean over the feet of 1[|F_xy| > 0.5 N] on the latest substep's net force (RF:71-80; truman's loco group)."""
    return -(x.feet_force[:, -1, :, :2].norm(dim=-1) > 0.5).to(x.feet_force.dtype).mean(-1)


# feet group (§5.5)
def impact_force_l2(x: HDMIInput, feet: FeetContact, robot_weight: float) -> torch.Tensor:
    """-sum_f min((F_f / W)^2 1[first contact], 10); F_f is the mean net force magnitude over the sensor's history, the
    last 3 physics substeps (RF:95-103, LOC:113-117)."""
    F = x.feet_force[:, -3:].norm(dim=-1).mean(1) / robot_weight
    return -(F.square() * feet.first_contact).clamp_max(10.0).sum(-1)


def feet_slip(x: HDMIInput, feet: FeetContact, tolerance: float) -> torch.Tensor:
    """-sum_f 1[contact time > 0.02 s] clamp(|v_f,xy| - tol, 0, 1), centre-of-mass velocities (RF:25-32)."""
    v = (x.now.body_lin_vel[:, FEET_SLICE, :2].norm(dim=-1) - tolerance).clamp(0.0, 1.0)
    return -((feet.contact_time > 0.02) * v).sum(-1)


def feet_air_time(feet: FeetContact, thres: float) -> torch.Tensor:
    """sum_f min(t_air - thres, 0) 1[first contact]: a penalty on touchdowns after short swings (RF:143-154)."""
    return ((feet.last_air_time - thres).clamp_max(0.0) * feet.first_contact).sum(-1)


# HDMI's reward classes (REW, the YAML's kind) with (body or joint) selections and a sigma
_KEYPOINT_REWARDS = {
    "keypoint_pos_tracking_local_product": keypoint_pos_tracking_local,
    "keypoint_ori_tracking_local_product": keypoint_ori_tracking_local,
    "keypoint_pos_tracking_product": keypoint_pos_tracking,
    "keypoint_ori_tracking_product": keypoint_ori_tracking,
    "keypoint_lin_vel_tracking_product": keypoint_lin_vel_tracking,
    "keypoint_ang_vel_tracking_product": keypoint_ang_vel_tracking,
}
_JOINT_REWARDS = {"joint_pos_tracking_product": joint_pos_tracking, "joint_vel_tracking_product": joint_vel_tracking}
_OBJECT_REWARDS = {
    "object_pos_tracking": object_pos_tracking,
    "object_ori_tracking": object_ori_tracking,
    "object_joint_pos_tracking": object_joint_pos_tracking,
}


def reward_selection(cfg, kind: str) -> list[int]:
    """The keypoints (a keypoint reward) or tracking joints (a joint reward) that the task's one enabled reward of HDMI's
    class ``kind`` reads, as HDMITask selects them."""
    found = [t for ts in cfg.rewards.values() for t in ts.values() if t.enabled and t.kind == kind]
    if len(found) != 1:
        raise ValueError(f"{cfg.name}: {len(found)} enabled {kind} rewards, expected 1")
    if kind in _KEYPOINT_REWARDS:
        return _match(found[0].params["body_names"], KEYPOINT_NAMES)
    if kind in _JOINT_REWARDS:
        return _match(found[0].params["joint_names"], TRACKING_JOINT_NAMES)
    raise ValueError(f"{kind!r} selects neither keypoints nor joints")


# the loco and feet classes that are penalties: a value <= 0 (feet_air_time's min(t_air - thres, 0) at touchdowns included)
PENALTIES = ("action_rate_l2", "joint_vel_l2", "joint_pos_limits", "joint_torque_limits", "feet_stumble", "impact_force_l2", "feet_slip",
             "feet_air_time")


def penalty_terms(cfg) -> list[str]:
    """The task's enabled loco and feet terms but survival, as HDMITask.rewards keys them ("group/name"): HDMI's penalties,
    each a class of PENALTIES with a positive weight, so every term is <= 0; anything else is an error."""
    out = []
    for g in ("loco", "feet"):
        for name, t in cfg.rewards[g].items():
            if not t.enabled or t.kind == "survival":
                continue
            if t.kind not in PENALTIES or t.weight <= 0:
                raise ValueError(f"{cfg.name}: {g}/{name} ({t.kind}, weight {t.weight}) is not a penalty <= 0")
            out.append(f"{g}/{name}")
    return out


def _reward_max(t) -> float:
    """The largest value of a tracking or object-group reward of HDMI's class ``t.kind`` (before its weight)."""
    if t.kind in _KEYPOINT_REWARDS or t.kind in _JOINT_REWARDS or t.kind in _OBJECT_REWARDS:
        return 1.0  # exp(-error / sigma), 1 at zero error
    if t.kind == "eef_contact_exp_max":
        return 1.0  # the best end effector's exp(-d / pos_sigma) exp(phi / frc_sigma), d >= 0, phi <= 0
    if t.kind == "eef_contact_exp":
        return max(t.params["gain"], 1.0)  # mean_e [g_e gain r_e + 1 - g_e], r_e <= 1: gain with every flag on, 1 with none
    raise ValueError(f"the largest value of HDMI's {t.kind!r} reward is not known")


def full_tracking_score(cfg) -> float:
    """HDMI's full per-step tracking score of the task in HDMITask.rewards' term units (weight x value, before dt): the sum
    of weight x largest value over its enabled tracking and object-group terms, survival left out."""
    terms = [t for g in ("tracking", "object_tracking") for t in cfg.rewards[g].values() if t.enabled and t.kind != "survival"]
    return sum(t.weight * _reward_max(t) for t in terms)


# ---------------- terminations (errors, (N,); §6) ----------------
# robot keypoints and the object at time n against frame f_{n-1}; the local terms use the current root (TRM:94-95)
def body_pos_error(x: HDMIInput, idx) -> torch.Tensor:
    return (x.ref.body_pos[:, 0, idx] - x.now.body_pos[:, idx]).norm(dim=-1).amax(-1)


def body_z_error(x: HDMIInput, idx) -> torch.Tensor:
    return (x.ref.body_pos[:, 0, idx, 2] - x.now.body_pos[:, idx, 2]).abs().amax(-1)


def body_ori_error(x: HDMIInput, idx) -> torch.Tensor:
    return rot_angle(relative_rot(x.ref.body_rot[:, 0, idx], x.now.body_rot[:, idx])).amax(-1)


def body_pos_error_local(x: HDMIInput, idx) -> torch.Tensor:
    r = heading_pos(x.ref.body_pos[:, 0, idx], x.ref.body_pos[:, 0, 0], x.ref.body_rot[:, 0, 0])
    return (r - heading_pos(x.now.body_pos[:, idx], x.now.root_pos, x.now.root_rot)).norm(dim=-1).amax(-1)


def body_ori_error_local(x: HDMIInput, idx) -> torch.Tensor:
    r = heading_rot(x.ref.body_rot[:, 0, idx], x.ref.body_rot[:, 0, 0])
    return rot_angle(relative_rot(r, heading_rot(x.now.body_rot[:, idx], x.now.root_rot))).amax(-1)


def object_pos_error(x: HDMIInput) -> torch.Tensor:
    return (x.ref.obj_pos[:, 0] - x.now.obj_pos).norm(dim=-1)


def object_ori_error(x: HDMIInput) -> torch.Tensor:
    return rot_angle(relative_rot(x.now.obj_rot, x.ref.obj_rot[:, 0]))


def lost_contact(x: HDMIInput, pos_thres: float, frc_thres: float) -> torch.Tensor:
    """2 if an end effector flagged for contact at f_n is farther than pos_thres from its target or pressed with less
    than frc_thres, else 0, from CU(n-1)'s snapshots (TRM:177-190)."""
    p = x.prev
    touching = ((p.eef_pos - p.target_pos).norm(dim=-1) < pos_thres) & (p.eef_force.norm(dim=-1) >= frc_thres)
    return 2 * (x.ref.contact[:, 0] & ~touching).any(-1).to(p.eef_pos.dtype)


_BODY_ERRORS = {"cum_body_pos_error": body_pos_error, "cum_body_ori_error": body_ori_error, "cum_body_pos_error_local": body_pos_error_local,
                "cum_body_ori_error_local": body_ori_error_local, "cum_body_z_error": body_z_error}  # fmt: skip


# ---------------- observations (§4) ----------------
def command_obs(x: HDMIInput) -> torch.Tensor:
    """ref_body_pos_future_local (CU(n-1)'s keypoints in its frame-f_{n-1} root heading frame), ref_joint_pos_future and
    ref_motion_phase (CU(n)), (N, 356)."""
    ref, nxt = x.ref, x.ref_next
    local = heading_pos(ref.body_pos, ref.body_pos[:, :1, 0], ref.body_rot[:, :1, 0])
    return torch.cat((local.flatten(1), nxt.joint_pos.flatten(1), nxt.phase.unsqueeze(-1)), -1)


def policy_obs(x: HDMIInput, ang_vel_hist: torch.Tensor, gravity_hist: torch.Tensor, joint_pos_hist: torch.Tensor,
               hist_idx: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """Root angular velocity and projected gravity (newest), joint positions minus the offsets at POLICY_JOINT_HISTORY
    (slot 0 newest; OC:117-121), previous actions flattened as j * 3 + k (OC:131-136), (N, 249)."""
    q = joint_pos_hist[:, list(POLICY_JOINT_HISTORY) if hist_idx is None else hist_idx] - x.joint_offset.unsqueeze(1)
    return torch.cat((ang_vel_hist[:, 0], gravity_hist[:, 0], q.flatten(1), x.action_buf.flatten(1)), -1)


def object_obs(x: HDMIInput, xy_noise=None, heading_noise=None, contact_noise=None) -> torch.Tensor:
    """object_xy_b, object_heading_b and ref_contact_pos_b, (N, 4 + 3E): the object at time n and the contact targets
    at n-1 in the heading frame of CU(n-1)'s root snapshot, whose height is kept (OBS:285-352, 371-384)."""
    now, prev = x.now, x.prev
    H = yaw_mat(prev.root_rot)
    xy = _inv(H, now.obj_pos - prev.root_pos)[:, :2]
    phi = _wrap_to_pi(_yaw_angle(now.obj_rot) - _yaw_angle(prev.root_rot))
    contact = _inv(H.unsqueeze(1), prev.target_pos - prev.root_pos.unsqueeze(1))
    if xy_noise is not None:
        xy = xy + xy_noise
    if heading_noise is not None:
        phi = phi + heading_noise
    if contact_noise is not None:
        contact = contact + contact_noise
    return torch.cat((xy, phi.cos().unsqueeze(-1), phi.sin().unsqueeze(-1), contact.flatten(1)), -1)


def priv_obs(x: HDMIInput, ang_vel_hist: torch.Tensor, gravity_hist: torch.Tensor, joint_pos_hist: torch.Tensor, hinged: bool,
             height_idx: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """The critic's and teacher's group in HDMI's order (Y:137-164, plus the hinge terms of doors and the chair),
    (N, 1703 + 5C + 3E (+3))."""
    now, prev, ref = x.now, x.prev, x.ref
    Rs, ps = prev.root_rot, prev.root_pos  # CU(n-1)'s root snapshot
    Rr = ref.body_rot[:, :1, 0]  # reference root at f_{n-1}, (N, 1, 3, 3)
    Hr, Hs = yaw_mat(Rr).unsqueeze(-3), yaw_mat(Rs).unsqueeze(-3)
    ref_local = heading_pos(ref.body_pos, ref.body_pos[:, :1, 0], Rr)
    rob_local = heading_pos(prev.body_pos, ps, Rs).unsqueeze(1)
    diff_rot = relative_rot(heading_rot(prev.body_rot, Rs).unsqueeze(1), heading_rot(ref.body_rot, Rr))
    diff_lin = _inv(Hr, ref.body_lin_vel) - _inv(Hs, prev.body_lin_vel).unsqueeze(1)
    diff_ang = _inv(Hr, ref.body_ang_vel) - _inv(Hs, prev.body_ang_vel).unsqueeze(1)
    parts = [
        ang_vel_hist.flatten(1), gravity_hist.flatten(1), (joint_pos_hist - x.joint_offset.unsqueeze(1)).flatten(1),
        _inv(Rs.unsqueeze(1), ref.body_pos[:, :, 0] - ps.unsqueeze(1)).flatten(1),  # ref_root_pos_future_b
        relative_rot(Rs.unsqueeze(1), ref.body_rot[:, :, 0])[..., :2, :].flatten(1),  # ref_root_ori_future_b: rows 0-1
        (ref_local - rob_local).flatten(1), diff_rot[..., :2, :].flatten(1), diff_lin.flatten(1), diff_ang.flatten(1),
        _inv(now.root_rot, now.body_lin_vel[:, 0]),  # root_linvel_b: the last substep's root velocity (EMA with gamma 0)
        heading_pos(now.body_pos[:, FEET_SLICE], now.root_pos, now.root_rot).flatten(1),  # body_pos_b
        _inv(now.root_rot.unsqueeze(1), now.body_lin_vel[:, FEET_SLICE]).flatten(1),  # body_vel_b: full root rotation
        now.body_pos[:, list(HEIGHT_BODIES) if height_idx is None else height_idx, 2],  # body_height above the ground plane z = 0
        x.applied_action, x.applied_torque,
        _inv(Rs, now.obj_pos - ps), relative_rot(Rs, now.obj_rot).flatten(1),  # object_pos_b, object_ori_b (row-major)
        _inv(now.obj_rot.unsqueeze(1), ref.obj_pos - now.obj_pos.unsqueeze(1)).flatten(1),  # diff_object_pos_future
        relative_rot(now.obj_rot.unsqueeze(1), ref.obj_rot).flatten(1),  # diff_object_ori_future
        x.ref_next.contact.flatten(1).to(now.body_pos.dtype),  # ref_object_contact_future, CU(n)
        _inv(Rs.unsqueeze(1), prev.target_pos - prev.eef_pos).flatten(1),  # diff_contact_pos_b
    ]  # fmt: skip
    if hinged:  # CU(n)'s hinge snapshot, i.e. time n, and the torque the hinge actuator applied (T:open_door-feet:42-46)
        parts += [now.obj_joint_pos.unsqueeze(-1), now.obj_joint_vel.unsqueeze(-1), x.obj_joint_torque.unsqueeze(-1)]
    return torch.cat(parts, -1)


# ---------------- per-env state ----------------
def _match(patterns, names) -> list[int]:
    """Indices of ``names`` matched in full by one of ``patterns``, in the order of ``names``; like Isaac Lab's
    resolve_matching_names, a pattern that matches nothing or a name that matches twice is an error."""
    patterns = (patterns,) if isinstance(patterns, str) else tuple(patterns)
    hits = [[p for p in patterns if re.fullmatch(p, n)] for n in names]
    if any(len(h) > 1 for h in hits) or not all(any(p in h for h in hits) for p in patterns):
        raise ValueError(f"patterns {patterns} must each match one or more of {names}, and no name twice")
    return [i for i, h in enumerate(hits) if h]


def _n3(t: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(t).clamp(-3.0, 3.0)


def _push(buf: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Roll a history by one and write the newest value to slot 0 (OC:35-36), without writing into ``buf``."""
    return torch.cat((v.unsqueeze(1), buf[:, :-1]), 1)


class HDMITask:
    """HDMI's heads for one task and the per-env state HDMI keeps across steps: observation histories, the episodic
    observation noise, the termination counters and the feet's air and contact timers.

    ``joint_limits`` (29, 2): the robot's hard joint position limits in JOINT_NAMES order, which HDMI reads from its
    USD (§5.4). ``robot_mass``: total link mass (kg) that normalizes the impact force; HDMI uses env 0's after the
    startup mass randomization (§5.5). ``noise``: HDMI's observation noise (§4). State updates are out of place, so the
    rewards, terminations' errors and observations are differentiable through imagined rollouts.
    """

    uses_object = True

    def __init__(self, cfg: HDMITaskCfg, num_envs: int, joint_limits: torch.Tensor, robot_mass: float, noise: bool = True,
                 device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):  # fmt: skip
        self.cfg, self.num_envs, self.noise, self.device, self.dtype = cfg, num_envs, noise, torch.device(device), dtype
        self.num_eefs, self.num_contact_cols, self.hinged = cfg.num_eefs, cfg.contact_cols, cfg.object_joint is not None
        self.joint_limits = torch.as_tensor(joint_limits, dtype=dtype, device=self.device)
        if self.joint_limits.shape != (len(JOINT_NAMES), 2):
            raise ValueError(f"joint_limits must be ({len(JOINT_NAMES)}, 2), got {tuple(self.joint_limits.shape)}")
        self.robot_weight = robot_mass * GRAVITY
        self.eef_keypoints = [KEYPOINT_NAMES.index(b) for b in cfg.contact_eef_bodies]  # every end effector of the 13 tasks is a keypoint
        self.eef_offsets = torch.tensor(cfg.contact_eef_offsets, dtype=dtype, device=self.device)
        self.target_offsets = torch.tensor(cfg.contact_target_offsets, dtype=dtype, device=self.device)
        if tuple(cfg.rewards) != GROUPS:
            raise ValueError(f"reward groups must be {GROUPS}, got {tuple(cfg.rewards)}")
        self.reward_terms = {g: [(name, t.weight, t.enabled, self._reward_fn(t)) for name, t in ts.items()] for g, ts in cfg.rewards.items()}
        self.term_names = list(cfg.terminations)
        self._term_fns = [self._termination_fn(name, t) for name, t in cfg.terminations.items()]
        self.term_thresholds = torch.tensor([t.threshold for t in cfg.terminations.values()], dtype=dtype, device=self.device)
        self.term_min_steps = torch.tensor([t.min_steps for t in cfg.terminations.values()], device=self.device)
        # index constants on the device, made once: indexing with a Python list or tuple copies it there every call
        self._hist_idx = torch.tensor(POLICY_JOINT_HISTORY, device=self.device)
        self._height_idx = torch.tensor(HEIGHT_BODIES, device=self.device)
        self.future_steps = torch.tensor(FUTURE_STEPS, device=self.device)
        self._clear()

    @property
    def obs_dims(self) -> dict[str, int]:
        K, J, A, F, E, C = len(KEYPOINT_NAMES), len(JOINT_NAMES), NUM_ACTIONS, len(FUTURE_STEPS), self.num_eefs, self.num_contact_cols
        return {
            "command": F * K * 3 + F * A + 1,
            "policy": 3 + 3 + len(POLICY_JOINT_HISTORY) * J + 3 * A,
            "object": 2 + 2 + 3 * E,
            "priv": PRIV_HISTORY * (3 + 3 + J) + F * (3 + 6) + F * K * (3 + 6 + 3 + 3) + 3 + 3 * len(FEET) + 3 * len(FEET) + len(HEIGHT_BODIES)
                    + A + J + 3 + 9 + F * (3 + 9) + F * C + 3 * E + (3 if self.hinged else 0),
        }  # fmt: skip

    # ---------------- construction ----------------
    def _ix(self, idx: list[int]) -> torch.Tensor:
        """An index list as a tensor on the task's device, made once (a list index is copied there at every call)."""
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def _reward_fn(self, t: Term) -> Callable[[HDMIInput, FeetContact], torch.Tensor]:
        p, k = dict(t.params), t.kind
        if k in _KEYPOINT_REWARDS:
            f, idx, sigma = _KEYPOINT_REWARDS[k], self._ix(_match(p.pop("body_names"), KEYPOINT_NAMES)), p.pop("sigma")
            fn = lambda x, feet: f(x, idx, sigma)  # noqa: E731
        elif k in _JOINT_REWARDS:  # the reference holds the tracking joints, which are the first 23 joints
            f, idx, sigma = _JOINT_REWARDS[k], self._ix(_match(p.pop("joint_names"), TRACKING_JOINT_NAMES)), p.pop("sigma")
            fn = lambda x, feet: f(x, idx, sigma)  # noqa: E731
        elif k in _OBJECT_REWARDS:
            if k == "object_joint_pos_tracking" and not self.hinged:
                raise ValueError(f"{self.cfg.name}: {k} needs a hinged object")
            f, sigma = _OBJECT_REWARDS[k], p.pop("sigma")
            fn = lambda x, feet: f(x, sigma)  # noqa: E731
        elif k in ("eef_contact_exp", "eef_contact_exp_max"):
            kw = {n: p.pop(n) for n in ("pos_sigma", "pos_tolerance", "frc_sigma", "frc_thres", "per_axis")}
            if k == "eef_contact_exp":
                kw["gain"] = p.pop("gain")
            f = eef_contact_exp if k == "eef_contact_exp" else eef_contact_exp_max
            fn = lambda x, feet: f(x, **kw)  # noqa: E731
        elif k == "action_rate_l2":
            fn = lambda x, feet: action_rate_l2(x)  # noqa: E731
        elif k == "joint_vel_l2":
            idx = self._ix(_match(p.pop("joint_names"), JOINT_NAMES))
            fn = lambda x, feet: joint_vel_l2(x, idx)  # noqa: E731
        elif k == "joint_pos_limits":  # soft limits from the hard ones as RC:69-74
            idx, factor = self._ix(_match(p.pop("joint_names"), JOINT_NAMES)), p.pop("soft_factor")
            lim = self.joint_limits[idx]
            mid, span = (lim[:, 0] + lim[:, 1]) / 2, lim[:, 1] - lim[:, 0]
            lo, hi = mid - 0.5 * span * factor, mid + 0.5 * span * factor
            fn = lambda x, feet: joint_pos_limits(x, idx, lo, hi)  # noqa: E731
        elif k == "joint_torque_limits":
            ids = _match(p.pop("joint_names"), JOINT_NAMES)
            idx = self._ix(ids)
            soft = torch.tensor([EFFORT_LIMITS[i] for i in ids], dtype=self.dtype, device=self.device) * p.pop("soft_factor")
            fn = lambda x, feet: joint_torque_limits(x, idx, soft)  # noqa: E731
        elif k == "survival":
            fn = lambda x, feet: survival(x)  # noqa: E731
        elif k in ("feet_stumble", "impact_force_l2", "feet_slip", "feet_air_time"):
            if _match(p.pop("body_names"), KEYPOINT_NAMES) != list(FEET):
                raise ValueError(f"{k}: the feet terms act on the two ankle_roll links, whose forces HDMIInput carries")
            if k == "feet_stumble":
                fn = lambda x, feet: feet_stumble(x)  # noqa: E731
            elif k == "impact_force_l2":
                fn = lambda x, feet: impact_force_l2(x, feet, self.robot_weight)  # noqa: E731
            elif k == "feet_slip":
                tol = p.pop("tolerance")
                fn = lambda x, feet: feet_slip(x, feet, tol)  # noqa: E731
            else:
                thres = p.pop("thres")
                if p.pop("soft_discount") != 1.0:
                    raise NotImplementedError("feet_air_time's soft_discount != 1 changes HDMI's discount (RF:152-153); every HDMI task uses 1")
                fn = lambda x, feet: feet_air_time(feet, thres)  # noqa: E731
        else:
            raise ValueError(f"unknown HDMI reward {k!r}")
        if p:
            raise ValueError(f"{k}: unsupported parameters {sorted(p)}")
        return fn

    def _termination_fn(self, name: str, t: Termination) -> Callable[[HDMIInput], torch.Tensor]:
        p = dict(t.params)
        if name in _BODY_ERRORS:
            f, idx = _BODY_ERRORS[name], self._ix(_match(p.pop("body_names"), KEYPOINT_NAMES))
            fn = lambda x: f(x, idx)  # noqa: E731
        elif name == "cum_object_pos_error":
            fn = object_pos_error
        elif name == "cum_object_ori_error":
            fn = object_ori_error
        elif name == "cum_lost_contact_steps":  # every HDMI task gives frc_thres as a float: the force-magnitude branch (TRM:182-183)
            pos_thres, frc_thres = p.pop("pos_thres"), p.pop("frc_thres")
            fn = lambda x: lost_contact(x, pos_thres, frc_thres)  # noqa: E731
        else:
            raise ValueError(f"unknown HDMI termination {name!r}")
        if p:
            raise ValueError(f"{name}: unsupported parameters {sorted(p)}")
        return fn

    def _clear(self):
        N, J, E, dev, dt = self.num_envs, len(JOINT_NAMES), self.num_eefs, self.device, self.dtype
        z = lambda *s: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        # one instance per observation term, as in HDMI: the policy's histories carry noise, the critic's do not
        self.ang_vel_policy, self.gravity_policy, self.joint_pos_policy = z(N, 1, 3), z(N, 1, 3), z(N, PRIV_HISTORY, J)
        self.ang_vel_priv, self.gravity_priv, self.joint_pos_priv = z(N, PRIV_HISTORY, 3), z(N, PRIV_HISTORY, 3), z(N, PRIV_HISTORY, J)
        self.contact_noise = z(N, E, 3)
        self.term_count = torch.zeros(N, len(self.term_names), dtype=torch.int32, device=dev)
        # the feet's contact sensor [IL]: substeps in the air and in contact so far, and the last swing's duration (s)
        self.air_count = torch.zeros(N, 2, dtype=torch.long, device=dev)
        self.contact_count = torch.zeros(N, 2, dtype=torch.long, device=dev)
        self.last_air_time = z(N, 2)

    # ---------------- per-env state ----------------
    def reset(self, env_ids: torch.Tensor, state: HDMIState):
        """Episode start for ``env_ids`` with ``state`` the post-reset state of all envs (§0 reset): the histories are
        filled with it (OC:24-29, 51-57, 104-108), the contact-target noise is redrawn, the counters and timers cleared."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        n, J = len(ids), len(JOINT_NAMES)
        R0 = state.root_rot[ids]
        ang, grav, q = _inv(R0, state.body_ang_vel[ids, 0]), -R0[:, 2, :], state.joint_pos[ids]
        ang_p = ang.unsqueeze(1)
        if self.noise:  # the policy's angular velocity gets noise at reset, projected gravity does not (OC:27-29, 57)
            ang_p = ang_p + NOISE_ANG_VEL * _n3(ang_p)
        put = lambda buf, v: buf.index_put((ids,), v)  # noqa: E731
        self.ang_vel_policy, self.gravity_policy = put(self.ang_vel_policy, ang_p), put(self.gravity_policy, grav.unsqueeze(1))
        self.joint_pos_policy = put(self.joint_pos_policy, q.unsqueeze(1).expand(n, PRIV_HISTORY, J))
        self.ang_vel_priv = put(self.ang_vel_priv, ang.unsqueeze(1).expand(n, PRIV_HISTORY, 3))
        self.gravity_priv = put(self.gravity_priv, grav.unsqueeze(1).expand(n, PRIV_HISTORY, 3))
        self.joint_pos_priv = put(self.joint_pos_priv, q.unsqueeze(1).expand(n, PRIV_HISTORY, J))
        if self.noise:
            eps = (2 * torch.rand(n, self.num_eefs, 3, dtype=self.dtype, device=self.device) - 1) * EPISODIC_CONTACT_NOISE
            self.contact_noise = put(self.contact_noise, eps)
        self.term_count = put(self.term_count, torch.zeros((), dtype=torch.int32, device=self.device))
        zero = torch.zeros((), dtype=torch.long, device=self.device)
        self.air_count, self.contact_count = put(self.air_count, zero), put(self.contact_count, zero)
        self.last_air_time = put(self.last_air_time, torch.zeros((), dtype=self.dtype, device=self.device))

    def refill(self, mask: torch.Tensor, state: HDMIState):
        """``reset``'s histories for the envs where ``mask`` (N,) holds, by selection (no host synchronization, as a compiled
        caller needs): filled with ``state``, the policy's angular velocity with its reset noise, the termination counters
        cleared. The contact-target noise and the feet's timers are kept (algos.hoi_algos' --st_ref_replay)."""
        R0 = state.root_rot
        ang, grav, q = _inv(R0, state.body_ang_vel[:, 0]), -R0[:, 2, :], state.joint_pos
        ang_p = ang.unsqueeze(1)
        if self.noise:
            ang_p = ang_p + NOISE_ANG_VEL * _n3(ang_p)
        m = mask.view(-1, 1, 1)
        fill = lambda buf, v: torch.where(m, v.expand_as(buf), buf)  # noqa: E731
        self.ang_vel_policy, self.gravity_policy = fill(self.ang_vel_policy, ang_p), fill(self.gravity_policy, grav.unsqueeze(1))
        self.joint_pos_policy = fill(self.joint_pos_policy, q.unsqueeze(1))
        self.ang_vel_priv, self.gravity_priv = fill(self.ang_vel_priv, ang.unsqueeze(1)), fill(self.gravity_priv, grav.unsqueeze(1))
        self.joint_pos_priv = fill(self.joint_pos_priv, q.unsqueeze(1))
        self.term_count = torch.where(mask.unsqueeze(-1), 0, self.term_count)

    def advance_feet(self, feet_force: torch.Tensor) -> FeetContact:
        """The feet's contact sensor over one control step's physics substeps (Isaac Lab's ContactSensor, §5.5): a foot is
        in contact when its net force exceeds 1 N; on touchdown the last air time becomes the air time so far plus the
        touchdown substep; counted in substeps, so the 0.02 s comparisons are exact."""
        air, cnt, last = self.air_count, self.contact_count, self.last_air_time
        for k in range(DECIMATION):
            c = feet_force[:, k].detach().norm(dim=-1) > CONTACT_FORCE_THRESHOLD
            last = torch.where((air > 0) & c, (air + 1).to(self.dtype) * PHYSICS_DT, last)
            air = torch.where(c, 0, air + 1)
            cnt = torch.where(c, cnt + 1, 0)
        self.air_count, self.contact_count, self.last_air_time = air, cnt, last
        # compute_first_contact(dt): 0 < contact time < dt + 1e-8, i.e. made within this step's DECIMATION substeps
        return FeetContact((cnt > 0) & (cnt <= DECIMATION), cnt.to(self.dtype) * PHYSICS_DT, last)

    def draw_noise(self, push: bool = True) -> dict[str, torch.Tensor] | None:
        """The standard normal draws of one ``step`` (or, without ``push``, of one ``observations``), in the order they
        are drawn: the pushed histories' angular velocity, joint positions and gravity, then the object group's
        position, heading and contact targets (two draws); None without noise. Passed to ``step`` or ``observations``
        they give the same values as drawing inside, which a compiled caller needs to keep eager's random stream."""
        if not self.noise:
            return None
        N, E, J, kw = self.num_envs, self.num_eefs, len(JOINT_NAMES), {"dtype": self.dtype, "device": self.device}
        hist = {"ang": (N, 3), "q": (N, J), "grav": (N, 3)} if push else {}
        shapes = hist | {"xy": (N, 2), "heading": (N,), "contact": (N, E, 3), "contact2": (N, E, 3)}
        return {k: torch.randn(*s, **kw) for k, s in shapes.items()}

    def _push_histories(self, x: HDMIInput, noise: dict[str, torch.Tensor] | None):
        """The observation terms' update(): newest root angular velocity (root frame), projected gravity and joint
        positions averaged over physics substeps 2 and 3 (OC:31-36, 59-65, 110-115)."""
        R0 = x.now.root_rot
        ang, grav, q = _inv(R0, x.now.body_ang_vel[:, 0]), -R0[:, 2, :], x.joint_pos_sub.mean(1)
        self.ang_vel_priv = _push(self.ang_vel_priv, ang)
        self.gravity_priv = _push(self.gravity_priv, grav)
        self.joint_pos_priv = _push(self.joint_pos_priv, q)
        if self.noise:
            n3 = lambda k: noise[k].clamp(-3.0, 3.0)  # noqa: E731
            ang, q = ang + NOISE_ANG_VEL * n3("ang"), q + NOISE_JOINT_POS * n3("q")
            grav = grav + NOISE_GRAVITY * n3("grav")
            grav = grav / grav.norm(dim=-1, keepdim=True)
        self.ang_vel_policy = _push(self.ang_vel_policy, ang)
        self.gravity_policy = _push(self.gravity_policy, grav)
        self.joint_pos_policy = _push(self.joint_pos_policy, q)

    # ---------------- heads ----------------
    def contact_points(self, body_pos, body_rot, obj_body_pos, obj_body_rot, scale) -> tuple[torch.Tensor, torch.Tensor]:
        """The task's end-effector points and contact targets (N, E, 3) from the keypoints, the object's contact-body pose
        and the env's per-axis object scale ``scale`` (N, 3), 1 in env 0 (§5.3, §7)."""
        k = self.eef_keypoints
        return contact_points(body_pos[:, k], body_rot[:, k], self.eef_offsets, obj_body_pos, obj_body_rot, self.target_offsets * scale.unsqueeze(1))

    def rewards(self, x: HDMIInput, feet: FeetContact) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """(N, 4) group rewards, each the sum of its enabled terms' weight x value times dt (ENV:406-416, 620-647), and
        every term's weight x value."""
        terms, groups = {}, []
        for g in GROUPS:
            enabled = []
            for name, w, on, fn in self.reward_terms[g]:
                terms[f"{g}/{name}"] = v = w * fn(x, feet)
                if on:
                    enabled.append(v)
            groups.append(torch.stack(enabled, -1).sum(-1) if enabled else torch.zeros_like(x.now.body_pos[:, 0, 0]))
        return torch.stack(groups, -1) * CONTROL_DT, terms

    def termination_errors(self, x: HDMIInput) -> torch.Tensor:
        """(N, terms) errors in the order of ``term_names``; a term exceeds when its error >= its threshold (TRM:24)."""
        return torch.stack([fn(x) for fn in self._term_fns], -1)

    def observations(self, x: HDMIInput, noise: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """The four groups from the current histories; noise as HDMI draws it (OBS:286-300, 342-349, 372-380), from
        ``noise`` (``draw_noise``) if given."""
        xy_noise = heading_noise = None
        contact_noise = self.contact_noise  # episodic, zero without noise
        if self.noise:
            n = self.draw_noise(push=False) if noise is None else noise
            xy_noise, heading_noise = NOISE_OBJECT * n["xy"].clamp(-3.0, 3.0), NOISE_OBJECT * n["heading"].clamp(-3.0, 3.0)
            step = n["contact"].clamp(-3.0, 3.0) + n["contact2"].clamp(-1.0, 1.0)  # OBS:287 and 298
            contact_noise = contact_noise + NOISE_OBJECT * step
        return {
            "command": command_obs(x),
            "policy": policy_obs(x, self.ang_vel_policy, self.gravity_policy, self.joint_pos_policy, self._hist_idx),
            "object": object_obs(x, xy_noise, heading_noise, contact_noise),
            "priv": priv_obs(x, self.ang_vel_priv, self.gravity_priv, self.joint_pos_priv, self.hinged, self._height_idx),
        }

    def step(self, x: HDMIInput, noise: dict[str, torch.Tensor] | None = None) -> HDMIOutput:
        """One control step in HDMI's order (§0): the feet's sensor ticks through the physics substeps, the update hooks
        push the observation histories and count the terminations, then rewards, observations and termination flags.
        The observation noise comes from ``noise`` (``draw_noise``) if given."""
        self._check(x)
        noise = self.draw_noise() if noise is None else noise
        feet = self.advance_feet(x.feet_force)
        self._push_histories(x, noise)
        exceeded = self.termination_errors(x) >= self.term_thresholds
        self.term_count = torch.where(exceeded, self.term_count + 1, 0)  # consecutive steps over the threshold (TRM:23-26)
        reward, terms = self.rewards(x, feet)
        obs = self.observations(x, noise)
        fired = self.term_count >= self.term_min_steps
        return HDMIOutput(obs, reward, terms, fired.any(-1), {n: fired[:, i] for i, n in enumerate(self.term_names)})

    def _check(self, x: HDMIInput):
        N, E, C = self.num_envs, self.num_eefs, self.num_contact_cols
        shapes = {"now.body_pos": (x.now.body_pos, (N, len(KEYPOINT_NAMES), 3)), "prev.eef_pos": (x.prev.eef_pos, (N, E, 3)),
                  "ref.contact": (x.ref.contact, (N, len(FUTURE_STEPS), C)), "ref_next.contact": (x.ref_next.contact, (N, len(FUTURE_STEPS), C)),
                  "joint_pos_sub": (x.joint_pos_sub, (N, 2, len(JOINT_NAMES))), "action_buf": (x.action_buf, (N, NUM_ACTIONS, 3)),
                  "feet_force": (x.feet_force, (N, DECIMATION, 2, 3))}  # fmt: skip
        for name, (t, shape) in shapes.items():
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} has shape {tuple(t.shape)}, expected {shape}")
        if x.ref.contact.dtype != torch.bool or x.ref_next.contact.dtype != torch.bool:
            raise ValueError("reference contact flags must be bool")
        if x.now.body_pos.dtype != self.dtype:
            raise ValueError(f"inputs are {x.now.body_pos.dtype}, the task {self.dtype}")
        hinge = (x.now.obj_joint_pos, x.now.obj_joint_vel, x.prev.obj_joint_pos, x.ref.obj_joint_pos, x.obj_joint_torque)
        if any((h is None) == self.hinged for h in hinge):
            raise ValueError(f"{self.cfg.name}: hinge angle, velocity, reference and torque must be given exactly for a hinged object")
