"""Animal motion tracking: an animal body (robots.animals) follows its animal's recorded kinematics (motion.sources.mimic_mjx).

State tuple ``S = (qpos, qvel, act)``: MuJoCo's generalized coordinates of the body (a free root and hinges) and its actuators'
activations (the rat's filters; empty for the fly and the stick insect); flat, their concatenation. Observation, goal, reward
and termination are analytic functions of the state, the bodies' positions ``fk`` (AnimalKinematics of qpos, computed once
per state) and the reference frame, shared by the simulator's transitions and the world model's imagined rollouts. The
policy's inputs are in the root's frame and its heading's, so they do not depend on the position or the heading.

The reference is clips one after the other at the control rate; a step taken at frame ``f`` tracks frame ``f + 1`` of the
same clip, and imagined rollouts stay on a clip's last frame once they reach it (``AnimalReference.end``).

Lengths are in the body's units (the fly's centimetres, the others' metres): the kernels' widths are fractions of the body's
size ``length`` (the median over the reference of the bodies' largest distance from the root), and the terminations are
MIMIC-MJX's per animal: the root farther than ``term_dist`` from its reference, turned more than ``term_deg`` away, or the
joints' L2 distance from the reference's above ``term_joint_l2``. The action is one value in [-1, 1] per actuator: an offset
about the reference's joint position for a position servo (the fly's), the control over its range for a torque actuator (the
rat's, through its activation filter; the stick insect's).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from ..core.rotations import mat_to_6d, quat_to_mat, rot_angle_sq, yaw_mat
from ..worldmodels.mjcf import MjcfLayout, MjcfWorldModel

# MIMIC-MJX's terminations (vnl-playground's imitation environments): root distance (body units), rotation (deg), joint L2 (rad)
TERMINATION = {"rat": (0.1, 120.0, 4.5), "fly": (0.5, 30.0, 20.0), "stick": (0.05, 120.0, 4.5)}


class AnimalKinematics(nn.Module):
    """The bodies' world positions from qpos: MjcfWorldModel's forward kinematics (``body_poses``) on the layout's tree."""

    fused = False
    _init_kinematics = MjcfWorldModel._init_kinematics
    body_poses = MjcfWorldModel.body_poses
    _fk_lists = MjcfWorldModel._fk_lists

    def __init__(self, lay: MjcfLayout, bodies: list[int]):
        super().__init__()
        self.lay = lay
        self._init_kinematics(lay)
        self.register_buffer("bodies", torch.tensor(bodies, dtype=torch.long), persistent=False)

    def forward(self, qpos: torch.Tensor) -> torch.Tensor:
        """(..., len(bodies), 3): the tracked bodies' positions."""
        return self.body_poses(qpos)[0][..., self.bodies, :]


class AnimalReference(NamedTuple):
    """Clips at the control rate, (T, ...) per field: qpos, qvel, the tracked bodies' positions, and the first and the last
    frame of each frame's clip."""

    qpos: torch.Tensor
    qvel: torch.Tensor
    bodies: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor

    @property
    def frames(self) -> int:
        return self.qpos.shape[0]

    def next_frame(self, f: torch.Tensor) -> torch.Tensor:
        return torch.minimum(f + 1, self.end[f])

    def ahead(self, f: torch.Tensor, k: int) -> torch.Tensor:
        """The frame ``k`` steps after ``f``, at most its clip's last."""
        return torch.minimum(f + k, self.end[f])


@torch.no_grad()
def build_reference(clips: list[tuple[torch.Tensor, torch.Tensor]], kin: AnimalKinematics, device: str) -> AnimalReference:
    """The reference of ``clips`` ((qpos, qvel) pairs at the control rate), one after the other."""
    fields, offset = [], 0
    for q, v in clips:
        n = q.shape[0]
        if n < 3:
            raise ValueError("a clip needs at least three frames")
        fields.append((q, v, torch.full((n,), offset), torch.full((n,), offset + n - 1)))
        offset += n
    q, v, start, end = (torch.cat(x).to(device) for x in zip(*fields, strict=True))
    bodies = torch.cat([kin(q[i : i + 8192]) for i in range(0, q.shape[0], 8192)])
    return AnimalReference(q, v, bodies, start, end)


@dataclass
class RewardCfg:
    """Kernel widths: lengths as fractions of the body's size, angles in rad, the joints' velocity as a multiple of the
    reference's RMS joint speed."""

    root_pos_std: float = 0.1
    body_pos_std: float = 0.1
    end_eff_std: float = 0.1
    root_rot_std: float = 0.4
    joint_std: float = 0.4  # RMS over the joints
    joint_vel_std: float = 1.0
    w_root_pos: float = 1.0
    w_root_rot: float = 1.0
    w_joint: float = 1.0
    w_joint_vel: float = 0.25
    w_body_pos: float = 1.0
    w_end_eff: float = 1.0
    w_action_rate: float = -0.05
    w_ctrl: float = -0.02

    def scaled(self, k: float) -> RewardCfg:
        """The tracking kernels' widths scaled by ``k`` (training's --reward_std_scale)."""
        widths = ("root_pos_std", "body_pos_std", "end_eff_std", "root_rot_std", "joint_std", "joint_vel_std")
        return RewardCfg(**{**self.__dict__, **{n: getattr(self, n) * k for n in widths}})


class AnimalTask:
    """``model``: the body's MjModel (with its activation filters, robots.animals.build_model); ``kin``: the tracked bodies'
    kinematics, ``end_effectors``: their indices among them; ``ctrl_dt`` the control step, ``act_r`` and ``act_rbar`` the
    filters' factors over it (robots.animals.activation_filter); ``action_scale``: a position servo's offset at action 1 (rad)."""

    def __init__(self, animal: str, model, kin: AnimalKinematics, ref: AnimalReference, end_effectors: list[int], ctrl_dt: float,
                 act_r, act_rbar, cfg: RewardCfg = RewardCfg(), action_scale: float = 0.5, goal_stride: int = 1):  # fmt: skip
        import numpy as np

        if goal_stride < 1:
            raise ValueError(f"goal_stride must be at least 1, got {goal_stride}")
        lay = kin.lay
        if lay.anchor is None or lay.balls or lay.slides:
            raise ValueError("AnimalTask: a free root and hinges")
        self.animal, self.kin, self.ref, self.cfg, self.ctrl_dt, self.goal_stride = animal, kin, ref, cfg, ctrl_dt, goal_stride
        dev = ref.qpos.device
        self.nq, self.nv, self.na, self.nu = model.nq, model.nv, model.na, model.nu
        self.state_dim = self.nq + self.nv + self.na
        self.rq, self.rd = lay.anchor  # the root's qpos and qvel addresses
        self.jq = torch.tensor([s[0] for s in lay.scalars], device=dev)  # the hinges' qpos and qvel addresses
        self.jd = torch.tensor([s[1] for s in lay.scalars], device=dev)
        self.nj = len(lay.scalars)
        self.ee = torch.tensor(end_effectors, device=dev)
        # the body's size: the median over the reference of the tracked bodies' largest distance from the root
        reach = torch.linalg.vector_norm(ref.bodies - ref.qpos[:, None, self.rq : self.rq + 3], dim=-1).amax(-1)
        self.length = float(reach.median())
        jv = ref.qvel[:, self.jd]
        self.joint_speed = float(jv.pow(2).mean().sqrt())  # the reference's RMS joint speed
        self.term_dist, self.term_deg, self.term_joint_l2 = TERMINATION[animal]
        # the actions: offsets about the reference's joint for position servos (affine bias), the control range otherwise
        servo = model.actuator_biastype == 1
        lo = np.where(model.actuator_ctrllimited > 0, model.actuator_ctrlrange[:, 0], -1.0)
        hi = np.where(model.actuator_ctrllimited > 0, model.actuator_ctrlrange[:, 1], 1.0)
        joint_trn = model.actuator_trntype == 0
        if (servo & ~joint_trn).any():
            raise ValueError("AnimalTask: a position servo on a joint transmission only (its action is about the reference's joint)")
        self.servo = torch.tensor(servo, device=dev)
        # each servo's joint (0 for the torque actuators, a tendon's among them, whose action is their control range)
        self.act_joint_q = torch.tensor(np.where(joint_trn, model.jnt_qposadr[np.where(joint_trn, model.actuator_trnid[:, 0], 0)], 0), device=dev)
        self.u_lo, self.u_hi = torch.tensor(lo, dtype=torch.float32, device=dev), torch.tensor(hi, dtype=torch.float32, device=dev)
        self.action_scale = action_scale
        # the filtered actuators, in their activations' order (MuJoCo's actuator_actadr), with the filters' factors
        filt = [i for i in np.argsort(model.actuator_actadr) if model.actuator_actadr[i] >= 0]
        if [int(model.actuator_actadr[i]) for i in filt] != list(range(self.na)) or (model.actuator_actnum[filt] != 1).any():
            raise ValueError("AnimalTask: one activation per filtered actuator")
        self.filt = torch.tensor(filt, dtype=torch.long, device=dev)
        act_of = np.zeros(self.nu, dtype=np.int64)  # each actuator's activation (0 where it has none: masked by is_filt)
        act_of[filt] = np.arange(len(filt))
        self.act_of = torch.tensor(act_of, device=dev)
        self.is_filt = torch.tensor(np.isin(np.arange(self.nu), filt), device=dev)
        self.act_r = torch.as_tensor(act_r, device=dev)[self.filt]
        self.act_rbar = torch.as_tensor(act_rbar, device=dev)[self.filt]
        n_ee = len(end_effectors)
        self.obs_dim = 1 + 6 + 3 + 3 + 2 * self.nj + self.na + 3 * n_ee + self.nu
        self.goal_frame_dim = 3 + 6 + self.nj + 3 + 3 * n_ee
        self.goal_offsets = 1 + goal_stride * torch.arange(64, device=dev)
        c = cfg
        self.max_step_reward = (c.w_root_pos + c.w_root_rot + c.w_joint + c.w_joint_vel + c.w_body_pos + c.w_end_eff) * ctrl_dt

    # ---------------------------------------------------------------- state
    def unflatten(self, x: torch.Tensor):
        return x[..., : self.nq], x[..., self.nq : self.nq + self.nv], x[..., self.nq + self.nv :]

    @staticmethod
    def flatten(S) -> torch.Tensor:
        return torch.cat(S, -1)

    def root(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return qpos[..., self.rq : self.rq + 3], quat_to_mat(qpos[..., self.rq + 3 : self.rq + 7])

    # ---------------------------------------------------------------- actuation
    def setpoint(self, a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """The controls for policy output ``a`` in [-1, 1] at frame ``f``."""
        target = self.ref.qpos[self.ref.next_frame(f)][..., self.act_joint_q] + self.action_scale * a
        span = self.u_lo + 0.5 * (a + 1) * (self.u_hi - self.u_lo)
        return torch.clamp(torch.where(self.servo, target, span), self.u_lo, self.u_hi)

    def activation_step(self, act: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The filters over a control step under ``u``: the activations it ends at and the means its physics steps see."""
        uf = u[..., self.filt]
        return uf + (act - uf) * self.act_r, uf + (act - uf) * self.act_rbar

    def wm_control(self, act: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """The world model's control input (worldmodels.animal): the controls, the filtered actuators' replaced by their mean
        activations, and the start activations beside them."""
        if not self.na:
            return u
        mean = self.activation_step(act, u)[1][..., self.act_of]
        return torch.cat((torch.where(self.is_filt, mean, u), act), -1)

    # ---------------------------------------------------------------- policy's inputs
    def _local(self, R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (x.unsqueeze(-2) @ R).squeeze(-2)  # rows x^T R = (R^T x)^T

    def obs(self, S, fk: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """The root's height, its tilt (the rotation about its heading), its velocities in its frame, the joints, the
        activations, the end effectors about the root in its frame, the previous action."""
        qpos, qvel, act = S
        p, R = self.root(qpos)
        L = self.length
        H = yaw_mat(R)
        ee = self._local(R.unsqueeze(-3), fk[..., self.ee, :] - p.unsqueeze(-2)) / L
        return torch.cat((p[..., 2:3] / L, mat_to_6d(H.transpose(-1, -2) @ R), self._local(R, qvel[..., self.rd : self.rd + 3]) / L,
                          qvel[..., self.rd + 3 : self.rd + 6], qpos[..., self.jq], qvel[..., self.jd] / self.joint_speed, act, ee.flatten(-2),
                          a_prev), -1)  # fmt: skip

    def goal(self, S, f: torch.Tensor, goal_frames: int) -> torch.Tensor:
        """``goal_frames`` reference frames ahead, about the state in the root's frame: the root's offset, the relative
        rotation, the joints' offsets, the reference's root velocity, and its end effectors about its root in its frame."""
        qpos, _, _ = S
        ref, L = self.ref, self.length
        p, R = self.root(qpos)
        g = torch.minimum(f.unsqueeze(-1) + self.goal_offsets[:goal_frames], ref.end[f].unsqueeze(-1))  # (..., k)
        gq = ref.qpos[g]
        gp, gR = self.root(gq)
        rows = R.unsqueeze(-3)
        ee = self._local(gR.unsqueeze(-3), ref.bodies[g][..., self.ee, :] - gp.unsqueeze(-2)) / L
        parts = (self._local(rows, gp - p.unsqueeze(-2)) / L, mat_to_6d(rows.transpose(-1, -2) @ gR), gq[..., self.jq] - qpos[..., self.jq].unsqueeze(-2),
                 self._local(rows, ref.qvel[g][..., self.rd : self.rd + 3]) / L, ee.flatten(-2))  # fmt: skip
        return torch.cat(parts, -1).flatten(-2)

    def critic_features(self, S, fk: torch.Tensor, f: torch.Tensor, a_prev: torch.Tensor, goal_frames: int) -> torch.Tensor:
        return torch.cat((self.obs(S, fk, a_prev), self.goal(S, f, goal_frames)), -1)

    # ---------------------------------------------------------------- tracking
    def errors(self, S, fk: torch.Tensor, f: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tracking errors against frame ``f``: the root's position (body units), its squared rotation angle, the joints'
        RMS and L2 errors (rad) and their velocities' RMS error, and each tracked body's and end effector's distance."""
        qpos, qvel, _ = S
        ref = self.ref
        p, R = self.root(qpos)
        rp, rR = self.root(ref.qpos[f])
        dq = qpos[..., self.jq] - ref.qpos[f][..., self.jq]
        body = torch.linalg.vector_norm(fk - ref.bodies[f], dim=-1)
        return {
            "root_pos": torch.linalg.vector_norm(p - rp, dim=-1),
            "root_rot_sq": rot_angle_sq(R, rR),
            "joint": dq.pow(2).mean(-1).sqrt(),
            "joint_l2": torch.linalg.vector_norm(dq, dim=-1),
            "joint_vel": (qvel[..., self.jd] - ref.qvel[f][..., self.jd]).pow(2).mean(-1).sqrt(),
            "body": body,
            "end_eff": body[..., self.ee],
        }

    def reward(self, S, fk: torch.Tensor, f: torch.Tensor, a: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """The tracking reward of reaching ``S`` at frame ``f`` with action ``a`` after ``a_prev``."""
        c, e, L = self.cfg, self.errors(S, fk, f), self.length
        r = (
            c.w_root_pos * torch.exp(-e["root_pos"].pow(2) / (c.root_pos_std * L) ** 2)
            + c.w_root_rot * torch.exp(-e["root_rot_sq"] / c.root_rot_std**2)
            + c.w_joint * torch.exp(-e["joint"].pow(2) / c.joint_std**2)
            + c.w_joint_vel * torch.exp(-e["joint_vel"].pow(2) / (c.joint_vel_std * self.joint_speed) ** 2)
            + c.w_body_pos * torch.exp(-e["body"].pow(2).mean(-1) / (c.body_pos_std * L) ** 2)
            + c.w_end_eff * torch.exp(-e["end_eff"].pow(2).mean(-1) / (c.end_eff_std * L) ** 2)
            + c.w_action_rate * (a - a_prev).pow(2).mean(-1)
            + c.w_ctrl * a.pow(2).mean(-1)
        )
        return r * self.ctrl_dt

    def _fail_margins(self, S, f: torch.Tensor) -> tuple[torch.Tensor, ...]:
        qpos = S[0]
        p, R = self.root(qpos)
        rp, rR = self.root(self.ref.qpos[f])
        dq = qpos[..., self.jq] - self.ref.qpos[f][..., self.jq]
        deg = torch.rad2deg(rot_angle_sq(R, rR).sqrt())
        return (torch.linalg.vector_norm(p - rp, dim=-1) - self.term_dist, deg - self.term_deg,
                torch.linalg.vector_norm(dq, dim=-1) - self.term_joint_l2)  # fmt: skip

    def termination(self, S, f: torch.Tensor) -> torch.Tensor:
        d, r, j = self._fail_margins(S, f)
        return (d > 0) | (r > 0) | (j > 0)

    def soft_termination(self, S, f: torch.Tensor) -> torch.Tensor:
        """The termination's probability with the thresholds as sigmoids (widths a tenth of each, 5 degrees), differentiable."""
        d, r, j = self._fail_margins(S, f)
        alive = (1 - torch.sigmoid(d / (0.1 * self.term_dist))) * (1 - torch.sigmoid(r / 5.0)) * (1 - torch.sigmoid(j / (0.1 * self.term_joint_l2)))
        return 1 - alive
