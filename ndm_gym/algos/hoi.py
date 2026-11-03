"""Imagination for HDMI's human-object interaction tasks (docs/hoi_design.md §3): HDMI's step input (HDMIInput) rebuilt
from a neural dynamics model's prediction, so that ``ndm_gym.tasks.hdmi``'s own rewards, observations and terminations
run on imagined rollouts.

From the predicted robot state (G1 order), object state, contact forces and the per-env context: the 16 keypoints by our
forward kinematics, their linear velocities as centre-of-mass velocities (HDMI's), joints in HDMI's order, a hinged
object's contact body through its joint frames, the end-effector points and contact targets. Approximations, as stated
in the design: the joints after the third physics substep are interpolated at 3/4 of the step; the actuators' torque
is the PD law at that interpolated state with the last substep's target, and the hinge torque HDMI's friction and
damping law at the interpolated rate (both are computed before the last substep).
"""

from __future__ import annotations

import torch

from ..core.kinematics import mat_mat, mat_vec
from ..core.rotations import axis_angle_to_mat
from ..robots import g1
from ..robots.g1 import G1_JOINT_NAMES, NUM_J, motor_of
from ..tasks.hdmi import (DECIMATION, FUTURE_STEPS, JOINT_NAMES, KEYPOINT_NAMES, NUM_ACTIONS, HDMIClip, HDMIInput, HDMIState, contact_points,
                          reference_window)  # fmt: skip
from ..tasks.hdmi_tasks import HDMITaskCfg
from ..worldmodels.hoi import ContextLayout

WRISTS = ("left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint",
          "right_wrist_yaw_joint")  # fmt: skip
SUB_FRACTION = (DECIMATION - 1) / DECIMATION  # where the third substep ends, as a fraction of the control step


class HOIImagination:
    """``constants``: the task's ``ndm_gym.sim.rigid.hdmi_sim.ndm_constants``; ``hinge``: its ``HDMISim.hinge_frames``
    for a hinged object, else None; ``layout``: the context's layout."""

    def __init__(self, cfg: HDMITaskCfg, clip: HDMIClip, constants: dict, hinge: dict | None, layout: ContextLayout, device):
        if (hinge is None) != (cfg.object_joint is None):
            raise ValueError(f"{cfg.name}: hinge frames are needed exactly for a hinged object")
        self.cfg, self.clip, self.layout, self.device = cfg, clip, layout, device
        self.hinged = hinge is not None
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.kin = g1.kinematics(device, tracked=list(KEYPOINT_NAMES))
        self.future_steps = torch.tensor(FUTURE_STEPS, device=device)
        self.j_hdmi = torch.tensor([G1_JOINT_NAMES.index(n) for n in JOINT_NAMES], device=device)  # HDMI order -> G1 index
        self.eef_kp = [KEYPOINT_NAMES.index(b) for b in cfg.contact_eef_bodies]
        self.eef_offsets, self.target_offsets = t(cfg.contact_eef_offsets), t(cfg.contact_target_offsets)
        self.q0 = t(constants["q0"])[self.j_hdmi]  # HDMI order
        self.scale = t(constants["act_scale"])  # (23,) HDMI's action order = its first 23 joints
        motors = [motor_of(n) for n in JOINT_NAMES]
        self.kp = t([cfg.wrist_kp if (cfg.wrist_kp is not None and n in WRISTS) else g1.kp(m[2]) for n, m in zip(JOINT_NAMES, motors)])
        self.kd, self.effort = t([g1.kd(m[2]) for m in motors]), t([m[0] for m in motors])
        if self.hinged:
            self.p_p, self.R_p, self.axis = t(hinge["p_p"]), t(hinge["R_p"]), t(hinge["axis"])
            self.p_c, self.R_c = t(hinge["p_c"]), t(hinge["R_c"])

    def _ctx(self, ctx, i: int) -> torch.Tensor:
        """The reset scalar ``i`` after the 29 joint offsets: 0 delay, 1 filter constant, 2 hinge friction, 3 damping."""
        k = self.layout.startup + NUM_J + i
        return ctx[..., k]

    def contact_body(self, So, ctx):
        """World pose of the object's contact body: a free object's root; a hinge's child, base X_p Rot(axis, q) X_c^-1."""
        if not self.hinged:
            return So[0], So[1]
        bp, bR = self.layout.poses(ctx)[0]
        Rq = mat_mat(self.R_p, axis_angle_to_mat(self.axis, So[0][..., 0]))
        Rc_t = self.R_c.transpose(-1, -2)
        p_rel = self.p_p + mat_vec(Rq, -mat_vec(Rc_t, self.p_c))
        return bp + mat_vec(bR, p_rel), mat_mat(bR, mat_mat(Rq, Rc_t))

    def state(self, S, So, f_eef, ctx, kp_com_b, obj_scale) -> HDMIState:
        """HDMIState of a robot state S (G1 order), object state So, end-effector forces (N, E, 3) in the contact body's
        frame, contexts (N, C), keypoint centres of mass (N, 16, 3) in their link frames and object scales (N, 3)."""
        P, R, V, W = self.kin.forward(*S)
        vcom = V + torch.cross(W, mat_vec(R, kp_com_b), dim=-1)
        q, qd = S[4][..., self.j_hdmi], S[5][..., self.j_hdmi]
        cp, cR = self.contact_body(So, ctx)
        if self.hinged:
            op, oR = self.layout.poses(ctx)[0]
            hq, hqd = So[0][..., 0], So[1][..., 0]
        else:
            op, oR, hq, hqd = So[0], So[1], None, None
        ep, tp = contact_points(P[:, self.eef_kp], R[:, self.eef_kp], self.eef_offsets, cp, cR, self.target_offsets * obj_scale.unsqueeze(-2))
        return HDMIState(P, R, vcom, W, q, qd, op, oR, ep, tp, f_eef, hq, hqd)

    def step_input(self, now: HDMIState, prev: HDMIState, t_prev, u_sub, action_buf, ctx, f_feet) -> HDMIInput:
        """HDMIInput of an imagined control step from the state ``prev`` to ``now`` (HDMIState), with the frame counter
        before the step ``t_prev`` (N,), the substeps' filtered actions ``u_sub`` (N, 4, 23), the raw-action buffer after
        the step (N, 23, 3) and the feet's forces (N, 4, 2, 3)."""
        off = ctx[..., self.layout.startup : self.layout.startup + NUM_J][..., self.j_hdmi]  # HDMI order
        q3 = prev.joint_pos + SUB_FRACTION * (now.joint_pos - prev.joint_pos)
        qd3 = prev.joint_vel + SUB_FRACTION * (now.joint_vel - prev.joint_vel)
        target = self.q0 + off
        target = torch.cat((target[..., :NUM_ACTIONS] + self.scale * u_sub[:, -1], target[..., NUM_ACTIONS:]), -1)
        tau = torch.minimum(torch.maximum(self.kp * (target - q3) - self.kd * qd3, -self.effort), self.effort)
        hinge_tau = None
        if self.hinged:
            rate = prev.obj_joint_vel + SUB_FRACTION * (now.obj_joint_vel - prev.obj_joint_vel)
            f, c = self._ctx(ctx, 2), self._ctx(ctx, 3)
            hinge_tau = (-torch.sign(rate) * (rate.abs() > 0.01) * f - c * rate).clamp(-100.0, 100.0)
        return HDMIInput(now, prev, reference_window(self.clip, t_prev - 1, self.future_steps), reference_window(self.clip, t_prev, self.future_steps),
                         torch.stack((q3, now.joint_pos), 1), torch.stack((qd3, now.joint_vel), 1), off, u_sub[:, -1], tau, action_buf, f_feet,
                         hinge_tau)  # fmt: skip
