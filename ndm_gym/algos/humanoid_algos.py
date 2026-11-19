"""SuperTrack and SuperMimic for humanoid motion tracking (tasks.humanoid_tracking): the quadruped task's agents
(algos.quadruped_algos), which read the task's sizes and state layout, with the world model of the humanoid's generalized
coordinates (worldmodels.mjcf). SuperMimic's is the quadruped's as it is (the task gives the reward, the termination and
the critic's inputs); SuperTrack's minimizes the L1 errors of the quantities the humanoid task tracks (``TERMS``), or, with
``--st_terms g1``, the library G1's SuperTrack terms (``G1_TERMS``)."""

from __future__ import annotations

import torch

from ..core.rotations import rot_angle_sq, so3_log
from .quadruped_algos import QuadrupedSuperMimic, QuadrupedSuperTrack
from .supertrack import LOCAL_TERMS, local_features

G1_TERMS = LOCAL_TERMS + ("rpos", "rrot")  # algos.supertrack.TRACK_TERMS


class HumanoidSuperMimic(QuadrupedSuperMimic):
    """SuperMimic on the humanoid task (QuadrupedSuperMimic's policy and critic updates on its reward)."""


class HumanoidSuperTrack(QuadrupedSuperTrack):
    """SuperTrack on the humanoid task. The policy minimizes, over the imagined window, the L1 errors of ``TERMS``: the
    tracked bodies about the re-anchored reference (the reward's) and in the world, their rotations, linear and angular
    velocities, the pelvis' position, height and rotation, the joints and their velocities; each term weighted once to 1
    at calibration, or with ``--st_task_weights`` by the task's reward weight over its kernel's width (``task_weights``),
    plus QuadrupedSuperTrack's failure penalty, action regularizer and imagined action noise.

    ``--st_terms g1``: the library G1's terms instead (algos.supertrack, SuperTrack's Eq. 1-7 with the world-space root
    terms of its §4.2): every body's position, velocity, rotation (6D) and angular velocity in the pelvis' frame, every
    body's height and the pelvis' up vector, and the pelvis' world position and rotation (its rotation vector against the
    reference's), each summed over its elements and calibrated to 1; ``--st_balance_w`` weighs the heights and the up
    vector on top, as the G1's."""

    TERMS = ("body_pos", "body_world", "body_ori", "body_lin_vel", "body_ang_vel", "root_pos", "root_height", "root_ori", "joint_pos", "joint_vel")

    def __init__(self, args, model, kin, task, dev: str):
        self.g1_terms = getattr(args, "st_terms", "task") == "g1"
        if self.g1_terms:
            if getattr(args, "st_task_weights", False):
                raise ValueError("--st_terms g1 calibrates its terms: not with --st_task_weights")
            self.TERMS = G1_TERMS
        super().__init__(args, model, kin, task, dev)
        if self.g1_terms:
            self.ref_local = self._reference_local(task.ref)
            self.term_mult[[self.TERMS.index("hei"), self.TERMS.index("up")]] = float(getattr(args, "st_balance_w", 1.0))

    @torch.no_grad()
    def _reference_local(self, ref, chunk: int = 65536) -> tuple[torch.Tensor, ...]:
        """``local_features`` of every body at every reference frame (algos.supertrack.ReferenceLocal's)."""
        parts = None
        for s in range(0, ref.frames, chunk):
            qpos, qvel = ref.qpos[s : s + chunk], ref.qvel[s : s + chunk]
            fk = self.kin.forward(qpos)
            feats = local_features(*fk, *self.kin.velocities(fk, qvel))
            parts = [[x] for x in feats] if parts is None else [p + [x] for p, x in zip(parts, feats, strict=True)]
        return tuple(torch.cat(p) for p in parts)

    @staticmethod
    def task_weights(c) -> tuple[dict, float]:
        """--st_task_weights: the G1 task's six tracked quantities (tasks.humanoid_tracking.RewardCfg) each weighs its reward
        weight over its kernel's width; a step's worth of loss is their weights' sum."""
        per_unit = {"body_pos": c.w_body_pos / c.body_pos_std, "body_ori": c.w_body_ori / c.body_ori_std, "body_lin_vel": c.w_body_linvel / c.body_linvel_std,
                    "body_ang_vel": c.w_body_angvel / c.body_angvel_std, "root_pos": c.w_root_pos / c.root_pos_std,
                    "root_ori": c.w_root_ori / c.root_ori_std}  # fmt: skip
        return per_unit, c.w_body_pos + c.w_body_ori + c.w_body_linvel + c.w_body_angvel + c.w_root_pos + c.w_root_ori

    def _errors(self, S, fk, f: torch.Tensor, vel=None) -> torch.Tensor:
        """(B, terms) L1 errors of the state ``S`` against the reference's frame ``f``: per body or joint the mean, per vector
        the sum of its components' absolute values, per rotation its angle (``--st_terms g1``: ``_g1_errors``). ``vel``: the
        bodies' velocities, if computed."""
        qpos, qvel = S
        V, W = self.kin.velocities(fk, qvel) if vel is None else vel
        if self.g1_terms:
            return self._g1_errors(S, fk, (V, W), f)
        task, ref = self.task, self.task.ref
        bp, bR = task.tracked(fk)
        t = self.kin.tracked
        ref_bp, ref_bR, _ = task.anchored(S, fk, f)
        angle = lambda a, b: (rot_angle_sq(a, b) + 1e-8).sqrt()  # noqa: E731  (the gradient finite at 0)
        errs = [(bp - ref_bp).abs().sum(-1).mean(-1), (bp - ref.body_p[f]).abs().sum(-1).mean(-1), angle(bR, ref_bR).mean(-1),
                (V[..., t, :] - ref.body_v[f]).abs().sum(-1).mean(-1), (W[..., t, :] - ref.body_w[f]).abs().sum(-1).mean(-1),
                (qpos[..., :3] - ref.qpos[f, :3]).abs().sum(-1), (qpos[..., 2] - ref.qpos[f, 2]).abs(), angle(fk[1][..., 0, :, :], ref.R[f]),
                (qpos[..., 7:] - ref.qpos[f, 7:]).abs().mean(-1), (qvel[..., 6:] - ref.qvel[f, 6:]).abs().mean(-1)]  # fmt: skip
        return torch.stack(errs, -1)

    def _g1_errors(self, S, fk, vel, f: torch.Tensor) -> torch.Tensor:
        """(B, G1_TERMS) the library G1's SuperTrack errors (algos.supertrack.body_errors): the absolute errors of every
        element of each term, summed within the term."""
        qpos = S[0]
        errs = [(x - r[f]).abs().flatten(1).sum(-1) for x, r in zip(local_features(*fk, *vel), self.ref_local, strict=True)]
        errs.append((qpos[..., :3] - self.task.ref.qpos[f, :3]).abs().sum(-1))
        errs.append(so3_log(self.task.ref.R[f].transpose(-1, -2) @ fk[1][..., 0, :, :]).abs().sum(-1))
        return torch.stack(errs, -1)

    def _st_step(self, S, fk, f: torch.Tensor, a_prev: torch.Tensor, noise: torch.Tensor):
        """QuadrupedSuperTrack's imagined step, the bodies' velocities computed once for the errors and the reward."""
        task = self.task
        a = self.pi(task.obs(S, fk, a_prev), task.goal(S, fk, f, self.goal_frames))
        S = self.wm.step(*S, task.setpoint(a + noise, f))
        fk, f = self.kin.forward(S[0]), task.ref.next_frame(f)
        vel = self.kin.velocities(fk, S[1])
        return S, fk, f, a, self._errors(S, fk, f, vel), task.soft_termination(S, fk, f), task.reward(S, fk, f, a, a_prev, vel)


HUMANOID_ALGOS = {"supertrack": HumanoidSuperTrack, "supermimic": HumanoidSuperMimic}
