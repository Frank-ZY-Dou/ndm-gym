"""Motion tracking with MuscleVAE's muscle-actuated character (docs/musclevae_design.md §4): the state its world models
predict, the reference tables, SuperMimic's tracking reward on its 23 bodies, and MuscleVAE's termination rule with its
form for imagined rollouts.

State: per body (body 0 the pelvis, the root) the center of mass p and the rotation R, and the velocities MuscleVAE's
environment records, backward differences over the 50 ms control step (CFG env_recompute_velocity, MU:39-50): v = (p -
p_prev) / dt and the rotation rate w = log(R R_prev^T) / dt. Positions then follow from the velocities exactly, p' = p +
dt v' and R' = exp(dt w') R, the semi-implicit Euler step of SuperTrack's and MuscleVAE's world models. At an episode's
first state the velocities are the simulator's (ENV:296), the angular one stored as the rate whose observation
(``their_angular_velocity``) is the simulator's. With it the muscle state: MuscleVAE's fatigue observation, the body
parts' f0-weighted MA, MR and MF (15). Body states are (p, R, v, w) tuples of (..., 23, 3) and (..., 23, 3, 3) tensors,
stored flat as (..., 23, 13): p, the quaternion wxyz, v, w.

Reference (``build_reference``): each frame's bodies by our forward kinematics of the clip's joint positions, raised as a
reset raises them (MuscleSim's Reference.lift), so that a reset starts on the reference; velocities by backward
differences at 20 Hz of the clip's own (unraised) motion, a clip's first frame taking its second's, as the reset's. Not
their dataset's placement of the bodies, which the character's hinges cannot reach (the lower legs, feet and hands 1 to
11 cm from our forward kinematics on average, up to 17.5 cm). The termination's head height is their placement's, as
MuscleSim's (within 1.2 mm of ours).

Reward (SuperMimic's, tasks.tracking.MDP.reward, on all 23 bodies): the sum of exp(-e / std^2) of the mean squared body
position, geodesic rotation, linear and angular velocity errors, plus the G1 task's action-rate penalty, times dt; the
reference re-anchored to the character's root (its horizontal position and heading; the velocities rotated with it),
since MuscleVAE's reference switches to a random frame of a random clip. The widths are SuperMimic's
(tasks.tracking.RewardCfg) times --reward_std_scale. The action rate (``action_rate``) is the G1's in its action units:
the muscle targets over ACTION_UNIT, their squared changes summed and scaled from the 284 muscles to the G1's 29
actuators. Unlike the G1's, no world-space root terms (after a switch the reference's global pose is unrelated to the
character's, and the task scores no drift) and no joint-limit penalty (the simulator's Euler hinges enforce the limits;
docs/musclevae_design.md §4).

Termination (ENV:193-218 with CFG, as ndm_gym.sim.rigid.muscle_sim.done_flags): a step is off track when the head's height
differs from the reference's by more than 0.5 m; an off count rises by one per step off track and falls by one (not
below 0) per step on track; terminated once it reaches 20 after 26 steps, or when the observation exceeds 50 in
magnitude. In imagination (``soft_termination``) each comparison is MuscleVAE's 1[x > threshold] in the forward pass
with the gradient of sigmoid((x - threshold) / (0.1 threshold)) (a straight-through estimator), so the count is theirs
exactly, and the count's rule fires by sigmoid((count - 19.5) / 0.05), within 5e-5 of 0 or 1 at every whole count.

References as in ndm_gym.sim.rigid.muscle_sim: MU = MuscleVAECore/Utils/motion_utils.py, ENV = MuscleVAECore/Env/
muscle_env.py, CFG = Data/NNModel/config.yml of wangchek/MuscleVAE_ @ 54c001c, which has no license: nothing is copied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..core.kinematics import mat_mat, mat_vec
from ..core.rotations import mat_to_quat, quat_to_mat, rot_angle_sq, so3_log, yaw_mat
from ..motion.sources import musclevae as clip_source
from ..robots import musclevae_char as mc
from ..robots.muscles import NUM_PARTS
from .tracking import RewardCfg

DT = 1.0 / mc.CONTROL_HZ
NB = mc.NUM_BODIES
# G1 settings in action units (its normalized setpoints u) on the muscle targets a: MuscleVAE's collection noise on a
# (0.05, MV:72) over SuperMimic's default on u (0.1); a = ACTION_UNIT u (docs/musclevae_design.md §4)
ACTION_UNIT = 0.05 / 0.1
G1_ACTUATORS = 29
BODY_DIM = 13  # p, quaternion wxyz, v, w
FATIGUE_DIM = 3 * NUM_PARTS  # 15
# MuscleVAE's termination (CFG), as ndm_gym.sim.rigid.muscle_sim's (train_muscles checks that they agree)
ERR_THRESHOLD, ERR_LENGTH, MIN_LENGTH, OBS_LIMIT = 0.5, 20, 26, 50.0
SOFT_REL_TAU = 0.1  # the straight-through comparisons' sigmoid width, relative to the threshold
FIRE_TAU = 0.05  # the count rule's sigmoid width (in steps)


# ---- the state
def flatten_bodies(S) -> torch.Tensor:
    """(p, R, v, w) -> (..., 23, 13)."""
    p, R, v, w = S
    return torch.cat((p, mat_to_quat(R), v, w), -1)


def unflatten_bodies(x: torch.Tensor):
    """(..., 23, 13) -> (p, R, v, w)."""
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13]


def _sinc_half(x: torch.Tensor) -> torch.Tensor:
    """2 sin(x / 2) / x, smooth at 0 (1 - x^2 / 24 there)."""
    small = x < 1e-3
    xs = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1.0 - x * x / 24.0, 2.0 * torch.sin(xs / 2.0) / xs)


def their_angular_velocity(w: torch.Tensor) -> torch.Tensor:
    """The rotation rates w (..., 3) of the state as MuscleVAE's backward-difference angular velocity, the vector part of
    2 (q' q^* - 1) / dt (MU:40-50) for q' = exp(dt w) q: (2 sin(|w| dt / 2) / (|w| dt)) w."""
    n = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-20)
    return w * _sinc_half(n * DT)


def rate_from_their(w_obs: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`their_angular_velocity` (for data only: at |w_obs| dt = 2, a half turn, its slope is
    infinite): the rate whose MuscleVAE angular velocity is ``w_obs``, rotation angles clipped at pi."""
    n = torch.sqrt((w_obs * w_obs).sum(-1, keepdim=True) + 1e-20)
    s = (n * DT / 2.0).clamp(max=1.0)
    small = s < 1e-3
    ss = torch.where(small, torch.ones_like(s), s)
    return w_obs * torch.where(small, 1.0 + s * s / 6.0, torch.asin(ss) / ss)


def body_states(body_q: torch.Tensor, body_qd: torch.Tensor, before: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
    """The state (N, 23, 13) of MuscleSim's bodies: their poses ``body_q`` (N, 23, 7, quaternions xyzw) with backward
    differences from the poses ``before`` the last step, or where ``fresh`` (N,) (reset since that step) the simulator's
    velocities ``body_qd`` (N, 23, 6)."""
    wxyz = lambda x: torch.cat((x[..., 6:7], x[..., 3:6]), -1)  # noqa: E731  (slices: no index tensor, which a CUDA graph capture refuses)
    pos, q = body_q[..., :3], wxyz(body_q)
    R, R0 = quat_to_mat(q), quat_to_mat(wxyz(before))
    v = (pos - before[..., :3]) / DT
    w = so3_log(mat_mat(R, R0.transpose(-1, -2))) / DT
    f = fresh[:, None, None]
    return torch.cat((pos, q, torch.where(f, body_qd[..., :3], v), torch.where(f, rate_from_their(body_qd[..., 3:]), w)), -1)


def head_height_error(S, head_z: torch.Tensor) -> torch.Tensor:
    """|head height - the reference's| (...)."""
    return (S[0][..., mc.HEAD, 2] - head_z).abs()


def observation_max(S) -> torch.Tensor:
    """The largest magnitude in MuscleVAE's termination check of the observation (MU:204-234, as MuscleSim's done_flags):
    positions relative to the root, velocities and angular velocities (theirs) in the root's frame, and the heights."""
    p, R, v, w = S
    Rt = R[..., :1, :, :].transpose(-1, -2)
    local = torch.cat([mat_vec(Rt, x) for x in (p - p[..., :1, :], v, their_angular_velocity(w))], -1)
    return torch.maximum(local.abs().flatten(-2).amax(-1), p[..., 2].abs().amax(-1))


def straight_through(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """1[x > threshold] with the gradient of sigmoid((x - threshold) / (SOFT_REL_TAU threshold))."""
    soft = torch.sigmoid((x - threshold) / (SOFT_REL_TAU * threshold))
    return (x > threshold).to(soft.dtype) + (soft - soft.detach())


def off_count_step(off: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """MuscleVAE's off-track count after a step: +1 off track, else -1 and at least 0 (``off`` 0 or 1, or soft)."""
    return off * (count + 1.0) + (1.0 - off) * (count - 1.0).clamp(min=0.0)


def soft_termination(S, head_z: torch.Tensor, count: torch.Tensor, ep_step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MuscleVAE's termination of an imagined step (module docstring) from the state after it, the reference's head
    heights (B,), the off counts before it (B,) and the episode steps before it (B,): the step's termination probability
    (B,) and the new off counts (B,), MuscleVAE's in the forward pass."""
    count = off_count_step(straight_through(head_height_error(S, head_z), ERR_THRESHOLD), count)
    fired = torch.sigmoid((count - ERR_LENGTH + 0.5) / FIRE_TAU) * (ep_step >= MIN_LENGTH).to(count.dtype)
    blown = straight_through(observation_max(S), OBS_LIMIT)
    return 1.0 - (1.0 - fired) * (1.0 - blown), count


def termination(S, head_z: torch.Tensor, count: torch.Tensor, ep_step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The hard rule (bool (B,), integer counts (B,)), for checks: MuscleSim's done_flags on a state."""
    off = head_height_error(S, head_z) > ERR_THRESHOLD
    count = torch.where(off, count + 1, (count - 1).clamp(min=0))
    finite = torch.isfinite(torch.cat((S[0], S[2], S[3]), -1)).flatten(-2).all(-1)
    return ~finite | (observation_max(S) > OBS_LIMIT) | ((ep_step >= MIN_LENGTH) & (count >= ERR_LENGTH)), count


# ---- the reference
class MuscleReference(NamedTuple):
    """Per-frame reference tables (F frames, MuscleSim's Reference order), on one device."""

    body_p: torch.Tensor  # (F, 23, 3)
    body_R: torch.Tensor  # (F, 23, 3, 3)
    body_v: torch.Tensor  # (F, 23, 3) backward differences
    body_w: torch.Tensor  # (F, 23, 3) rotation rates (backward differences)
    head_z: torch.Tensor  # (F,) their placement's head height (the termination's)
    action: torch.Tensor  # (F, 284) the reference action l_ref / l_Tpose - 1 (MuscleSim.reference_action)
    clip_last: torch.Tensor  # (F,) the last frame of each frame's clip

    @property
    def state(self):
        return self.body_p, self.body_R, self.body_v, self.body_w


def build_reference(names: tuple[str, ...], lift: torch.Tensor, head_z: torch.Tensor, length: torch.Tensor, rest_length: torch.Tensor,
                    char: dict | None = None, device: str | torch.device = "cpu") -> MuscleReference:  # fmt: skip
    """The tables of the named clips (module docstring) from MuscleSim's Reference: ``lift``, ``head_z`` and the muscle
    lengths ``length`` (F, 284) of every frame; forward kinematics in double precision on the CPU."""
    char = mc.load_character() if char is None else char
    rows = {k: [] for k in ("p", "R", "v", "w", "last")}
    total = 0
    for name in names:
        clip = clip_source.load_clip(os.path.join(mc.CLIPS_DIR, f"{name}.npz"))
        if clip.fps != mc.CONTROL_HZ:
            raise ValueError(f"{name}: {clip.fps} fps, expected the control rate {mc.CONTROL_HZ}")
        p, R = mc.forward_kinematics(clip.root_p.double(), clip.root_quat.double(), clip.joints.double(), char)
        T = p.shape[0]
        v = (p[1:] - p[:-1]) / DT
        w = so3_log(R[1:] @ R[:-1].transpose(-1, -2)) / DT
        rows["v"].append(torch.cat((v[:1], v))), rows["w"].append(torch.cat((w[:1], w)))
        rows["p"].append(p), rows["R"].append(R)
        rows["last"].append(torch.full((T,), total + T - 1, dtype=torch.long))
        total += T
    if total != lift.shape[0] or total != head_z.shape[0] or total != length.shape[0]:
        raise ValueError(f"the clips have {total} frames, the simulator's reference {lift.shape[0]}: build it from the same clips")
    p = torch.cat(rows["p"])
    p[..., 2] += lift.double().cpu()[:, None]
    f32 = lambda x: x.to(device=device, dtype=torch.float32)  # noqa: E731
    action = length.to(device) / rest_length.to(device) - 1.0
    return MuscleReference(f32(p), f32(torch.cat(rows["R"])), f32(torch.cat(rows["v"])), f32(torch.cat(rows["w"])), f32(head_z), f32(action),
                           torch.cat(rows["last"]).to(device))  # fmt: skip


def reference_from_sim(sim) -> MuscleReference:
    """:func:`build_reference` for a MuscleSim's clips."""
    r = sim.ref
    return build_reference(r.names, r.lift, r.head_z, r.length, sim.muscles.rest_length, sim.char, sim.device)


# ---- SuperMimic's reward
@dataclass
class MuscleRewardCfg:
    """SuperMimic's tracking-kernel widths and the G1 task's action-rate weight (tasks.tracking.RewardCfg)."""

    body_pos_std: float = RewardCfg.body_pos_std
    body_ori_std: float = RewardCfg.body_ori_std
    body_linvel_std: float = RewardCfg.body_linvel_std
    body_angvel_std: float = RewardCfg.body_angvel_std
    w_action_rate: float = RewardCfg.w_action_rate

    def scaled(self, k: float) -> MuscleRewardCfg:
        """The widths times ``k`` (``--reward_std_scale``, training only)."""
        widths = (k * getattr(self, f) for f in ("body_pos_std", "body_ori_std", "body_linvel_std", "body_angvel_std"))
        return MuscleRewardCfg(*widths, w_action_rate=self.w_action_rate)


MAX_STEP_REWARD = 4 * DT  # the four tracking terms' maximum
FULL_TRACKING_SCORE = 4.0  # per unit time, as tasks.tracking's 5 (SuperTrack's penalty ratio)


def action_rate(a: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
    """The G1 task's action rate (tasks.tracking.MDP.reward: the squared setpoint changes summed over its 29 actuators) of
    muscle targets (..., 284) in G1 action units: sum((da / ACTION_UNIT)^2) x 29 / 284."""
    return ((a - a_prev) / ACTION_UNIT).square().sum(-1) * (G1_ACTUATORS / mc.NUM_MUSCLES)


def anchored_reference(S, ref: MuscleReference, f: torch.Tensor):
    """The reference bodies of frames ``f`` (B,) re-anchored to the states' roots: the reference turned by the heading
    difference dR = yaw(R_root R_ref_root^T) about its root, which is placed at the character's horizontal position and
    the reference's height (tasks.tracking.MDP.reward); its velocities turned by dR."""
    p, R = S[0], S[1]
    rp, rR = ref.body_p[f], ref.body_R[f]
    dR = yaw_mat(mat_mat(R[..., 0, :, :], rR[..., 0, :, :].transpose(-1, -2))).unsqueeze(-3)
    anchor = torch.cat((p[..., 0, :2], rp[..., 0, 2:3]), -1).unsqueeze(-2)
    return (anchor + mat_vec(dR, rp - rp[..., :1, :]), mat_mat(dR, rR), mat_vec(dR, ref.body_v[f]), mat_vec(dR, ref.body_w[f]))


def reward(S, ref: MuscleReference, f: torch.Tensor, cfg: MuscleRewardCfg, rate: torch.Tensor | None = None, terms: bool = False):
    """SuperMimic's reward (module docstring) of states S against reference frames ``f``, with the action-rate penalty of
    ``rate`` (``action_rate``) if given, times dt: (B,), and with ``terms`` the mean squared errors."""
    rp, rR, rv, rw = anchored_reference(S, ref, f)
    p, R, v, w = S
    e_pos = ((rp - p) ** 2).sum(-1).mean(-1)
    e_ori = rot_angle_sq(rR, R).mean(-1)
    e_lv = ((rv - v) ** 2).sum(-1).mean(-1)
    e_av = ((rw - w) ** 2).sum(-1).mean(-1)
    r = (torch.exp(-e_pos / cfg.body_pos_std**2) + torch.exp(-e_ori / cfg.body_ori_std**2) + torch.exp(-e_lv / cfg.body_linvel_std**2)
         + torch.exp(-e_av / cfg.body_angvel_std**2))  # fmt: skip
    if rate is not None:
        r = r + cfg.w_action_rate * rate
    r = r * DT
    if terms:
        return r, {"e_body_pos": e_pos, "e_body_ori": e_ori, "e_body_linvel": e_lv, "e_body_angvel": e_av}
    return r


def root_frame_mpjpe(S, ref: MuscleReference, f: torch.Tensor) -> torch.Tensor:
    """Mean body position error (B,) in the root's frame, relative to the root (tools/replay_musclevae.py's)."""
    p, R = S[0], S[1]
    rp, rR = ref.body_p[f], ref.body_R[f]
    mine = mat_vec(R[..., :1, :, :].transpose(-1, -2), p - p[..., :1, :])
    theirs = mat_vec(rR[..., :1, :, :].transpose(-1, -2), rp - rp[..., :1, :])
    return (mine - theirs).norm(dim=-1).mean(-1)


__all__ = ["ACTION_UNIT", "MuscleReference", "MuscleRewardCfg", "action_rate", "body_states", "build_reference", "flatten_bodies", "reward", "soft_termination", "termination",
           "their_angular_velocity", "unflatten_bodies"]  # fmt: skip
