"""Opt-in native Door algorithm; eager fp32 correctness path.

Uses the shared Algo method contract with articulated MJCF states, like the
hand algorithms. Imagined reward is not evidence of physical opening.
"""

from dataclasses import asdict

import torch

from ..policies.adroit_door import DoorPolicy, DoorResidualPolicy
from ..policies.door_imagination import residual_update
from ..worldmodels.mjcf import MjcfWorldModel


class DoorResidual:
    """Reference-control prior and bounded learned correction, hand actuators only.

    Caller supplies replay windows in physical actuator units, fits normalizers
    from training data only, and separately evaluates the final policy in physics.
    This is not a SuperMimic or SuperTrack implementation.
    """

    world_model_cls = MjcfWorldModel
    residual = True

    def __init__(self, task, layout, reference_actions, *, hidden=(256, 256),
                 wm_hidden=(64, 64), horizon=4, lr=1e-3, device="cpu"):
        if not isinstance(horizon, int) or horizon < 1 or horizon >= task.frames:
            raise ValueError("horizon must fit within reference")
        actions = torch.as_tensor(reference_actions, dtype=torch.float32, device=device)
        if actions.shape != (task.frames - 1, task.action_dim):
            raise ValueError("Expected one reference action per transition")
        if not torch.isfinite(actions).all() or torch.any(actions.abs() > 1):
            raise ValueError("Reference actions must be finite and normalized")
        if layout.nq != 30 or layout.nv != 30 or layout.nu != 28 or layout.dt != task.dt:
            raise ValueError("Expected native Door layout and control timestep")
        self.task, self.horizon = task, horizon
        self.layout = layout
        self.reference_actions = actions.detach().clone()
        self.wm = self.world_model_cls(layout, hidden=wm_hidden, amp=False, kind="increment").to(device)
        self.pi = DoorResidualPolicy(goal_frames=task.goal_frames, hidden=hidden).to(device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)

    def setpoint(self, action):
        return self.task.setpoint(action)

    def wm_loss(self, qpos, qvel, physical_ctrl, valid=None):
        """Shared MJCF multistep loss; controls are physical, not normalized."""
        return self.wm.window_loss(qpos, qvel, physical_ctrl, valid)

    def policy_update(self, state, frame, previous_action):
        return residual_update(self.pi, self.opt_pi, self.wm, self.task, state,
                               frame, previous_action, self.reference_actions, self.horizon)

    def supervised_update(self, obs, goal, correction):
        """Fit bounded teacher corrections, scaled before the residual multiplier.

        Targets are normalized corrections in [-1, 1], not physical controls.
        Caller freezes collection and train/evaluation splits before fitting.
        Normalizers are fitted separately on training inputs only.
        """
        if (obs.ndim != 2 or goal.ndim != 2 or correction.ndim != 2
                or obs.shape[0] == 0 or goal.shape[0] != obs.shape[0]
                or correction.shape != (obs.shape[0], self.task.action_dim)):
            raise ValueError("Expected aligned nonempty supervised batches")
        if (not all(torch.isfinite(x).all() for x in (obs, goal, correction))
                or torch.any(correction.abs() > 1)):
            raise ValueError("Teacher corrections must be finite and bounded")
        prediction = DoorPolicy.forward(self.pi, obs, goal)
        loss = (prediction - correction).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite supervised Door loss")
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), 1., error_if_nonfinite=True)
        self.opt_pi.step()
        return loss.detach()

    def task_identity(self):
        """Deployment contract, independent of device; validated before recovery."""
        return {**({"goal_frames": self.task.goal_frames} if self.task.goal_frames != 1 else {}),
                "q": self.task.q.detach().cpu().clone(),
                "v": self.task.v.detach().cpu().clone(),
                "ctrl_mid": self.task.ctrl_mid.detach().cpu().clone(),
                "ctrl_half": self.task.ctrl_half.detach().cpu().clone(),
                "dt": self.task.dt, "reward": asdict(self.task.cfg),
                "layout": asdict(self.layout)}

    def state_dict(self):
        """Both optimizers and normalizers, with reference identity for recovery."""
        return {"wm": self.wm.state_dict(), "pi": self.pi.state_dict(),
                "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(),
                "reference_actions": self.reference_actions.clone(), "horizon": self.horizon,
                "task_identity": self.task_identity()}

    def load_state_dict(self, state):
        if state["horizon"] != self.horizon or not torch.equal(
                state["reference_actions"].to(self.reference_actions), self.reference_actions):
            raise ValueError("Door checkpoint reference or horizon mismatch")
        expected = self.task_identity()
        actual = state.get("task_identity", {})
        if set(actual) != set(expected) or any(
                not torch.equal(actual[k].cpu(), value) if isinstance(value, torch.Tensor)
                else actual[k] != value for k, value in expected.items()):
            raise ValueError("Door checkpoint task identity mismatch")
        self.wm.load_state_dict(state["wm"])
        self.pi.load_state_dict(state["pi"])
        self.opt_wm.load_state_dict(state["opt_wm"])
        self.opt_pi.load_state_dict(state["opt_pi"])
