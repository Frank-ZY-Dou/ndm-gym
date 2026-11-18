"""SuperTrack objective adapted to native articulated Adroit Door coordinates.

Uses the upstream acceleration MJCF model and fixed LossBalancer calibration.
The reference-control residual parameterization is explicit; this is not an
XHand algorithm-only comparison. Physical final-policy evaluation is required.
"""

import math

import torch


from .door import DoorResidual
from .supertrack import LossBalancer


class DoorSuperTrack(DoorResidual):
    """Model-through-time L1 tracking, without reward optimization or a critic."""

    model_contract = "accel-squared-v2"

    def __init__(self, *args, upstream_policy_optimizer=False, correction_reg=0.,
                 correction_imag_noise=0., task_weights=False, **kwargs):
        if any(not math.isfinite(x) or x < 0 for x in (correction_reg, correction_imag_noise)):
            raise ValueError("Correction regularization and noise must be finite and nonnegative")
        if type(task_weights) is not bool:
            raise ValueError("task_weights must be boolean")
        self.task_weights = task_weights
        self.correction_reg = float(correction_reg)
        self.correction_imag_noise = float(correction_imag_noise)
        if type(upstream_policy_optimizer) is not bool:
            raise ValueError("upstream_policy_optimizer must be boolean")
        self.upstream_policy_optimizer = upstream_policy_optimizer
        super().__init__(*args, **kwargs)
        if self.upstream_policy_optimizer:
            self.opt_pi.defaults["betas"] = (.9, .95)
            for group in self.opt_pi.param_groups:
                group["betas"] = (.9, .95)
        device = self.reference_actions.device
        hidden = kwargs.get("wm_hidden", (64, 64))
        self.wm = self.world_model_cls(self.layout, hidden=hidden, amp=False, kind="accel").to(device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=kwargs.get("lr", 1e-3))
        self.bal = LossBalancer(len(self.task.tracking_widths)).to(device)
        if self.task_weights:
            # HandSuperTrack --st_task_weights: reward weight / kernel width / H.
            # Door has four equally weighted tracking kernels in native units.
            widths = torch.tensor(self.task.tracking_widths, device=device)
            self.bal.w.copy_(widths.reciprocal() / self.horizon)
            self.bal.scale.copy_(widths)
            self.bal.ready.fill_(True)

    def wm_loss(self, qpos, qvel, physical_ctrl, valid=None):
        """HandSuperTrack's squared normalized multistep error, mean over windows.

        HandSuperTrack inherits HandSuperMimic._wm_rollout. Its loss includes
        positions and velocities, sums steps, and divides by the batch size.
        Keep the generic MJCF L1 loss unchanged for other callers.
        """
        q, v = qpos[0], qvel[0]
        total = q.new_zeros(())
        for t in range(physical_ctrl.shape[0]):
            q, v = self.wm.step(q, v, physical_ctrl[t])
            error = self.wm.error(q, v, qpos[t+1], qvel[t+1]).square().mean(-1)
            mask = torch.ones_like(error) if valid is None else valid[:t+1].all(0).to(error.dtype)
            total = total + (error * mask).sum() / qpos.shape[1]
        return total

    def policy_update(self, state, frame, previous_action):
        """One upstream-style calibrated tracking update through a frozen model."""
        if frame.ndim != 1 or torch.any(frame < 0) or torch.any(frame + self.horizon >= self.task.frames):
            raise ValueError("Imagined window must remain within the reference")
        state = tuple(x.detach() for x in state)
        previous_action = previous_action.detach()
        flags = [p.requires_grad for p in self.wm.parameters()]
        try:
            self.wm.requires_grad_(False)
            errors, corrections = [], []
            for _ in range(self.horizon):
                obs, goal = self.task.obs(state, previous_action), self.task.goal(state, frame)
                if self.correction_reg or self.correction_imag_noise:
                    correction = self.pi.correction(obs, goal)
                    corrections.append(correction)
                    action = (self.reference_actions[frame] + self.pi.residual_scale * correction).clamp(-1, 1)
                    noise = (self.correction_imag_noise * torch.randn_like(correction)
                             if self.correction_imag_noise else torch.zeros_like(correction))
                    imagined = (self.reference_actions[frame]
                                + self.pi.residual_scale * (correction + noise)).clamp(-1, 1)
                else:
                    action = self.pi(obs, goal, self.reference_actions[frame])
                    imagined = action
                state = self.wm.step(*state, self.setpoint(imagined), self.task.dt)
                frame = frame + 1
                errors.append(self.task.tracking_terms(state, frame))
                previous_action = action
            terms = torch.stack(errors).sum(0).mean(0)
            if not torch.isfinite(terms).all():
                raise FloatingPointError("Nonfinite Door tracking terms")
            if not self.bal.calibrated:
                self.bal.calibrate(terms.detach().clamp(min=1e-6), torch.ones_like(terms))
            loss = (terms * self.bal.w).sum()
            if self.correction_reg:
                offsets = torch.stack(corrections)
                loss = loss + self.correction_reg * (offsets.square().mean() + offsets.abs().mean())
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1., error_if_nonfinite=True)
            if self.upstream_policy_optimizer:
                torch.nn.utils.clip_grad_value_(self.pi.parameters(), .1)
            self.opt_pi.step()
            return loss.detach(), grad.detach()
        finally:
            for parameter, flag in zip(self.wm.parameters(), flags):
                parameter.requires_grad_(flag)

    def state_dict(self):
        return {**super().state_dict(), "task_weights": self.task_weights, "upstream_policy_optimizer": self.upstream_policy_optimizer, "correction_reg": self.correction_reg, "correction_imag_noise": self.correction_imag_noise, "method": "supertrack", "model_contract": self.model_contract, "bal": self.bal.state_dict()}

    def load_state_dict(self, state):
        if state.get("task_weights", False) != self.task_weights:
            raise ValueError("Door tracking weight configuration mismatch")
        if (state.get("correction_reg", 0.) != self.correction_reg
                or state.get("correction_imag_noise", 0.) != self.correction_imag_noise):
            raise ValueError("Door correction configuration mismatch")
        if state.get("upstream_policy_optimizer", False) != self.upstream_policy_optimizer:
            raise ValueError("Door policy optimizer configuration mismatch")
        if state.get("model_contract") != self.model_contract:
            raise ValueError("Door model contract mismatch; legacy checkpoints require historical source")
        if state.get("method") != "supertrack":
            raise ValueError("Door method mismatch")
        super().load_state_dict(state)
        self.bal.load_state_dict(state["bal"])
