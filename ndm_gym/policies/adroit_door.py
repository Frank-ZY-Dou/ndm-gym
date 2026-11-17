"""Native 28-actuator Door policy using upstream MLP and normalization paths."""

import torch
from torch import nn

from ..core.nets import mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..tasks.adroit_door import DoorTask


class DoorPolicy(nn.Module):
    def __init__(self, goal_frames=1, hidden=(256, 256)):
        super().__init__()
        if not isinstance(goal_frames, int) or goal_frames < 1:
            raise ValueError("goal_frames must be positive")
        n = DoorTask.obs_dim + DoorTask.goal_frame_dim * goal_frames
        self.norm = RunningNorm(n)
        self.net = mlp(n, list(hidden), DoorTask.action_dim)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, obs, goal):
        return torch.tanh(run_mlp(self.net, self.norm(torch.cat((obs, goal), -1))))


class DoorResidualPolicy(DoorPolicy):
    """Explicit reference-control prior plus learned hand-control correction.

    Requires a time-aligned normalized reference action from the caller. Zero
    initialization reproduces that action exactly; it is an open-loop baseline,
    not evidence of learned tracking. Corrections actuate only the native hand.
    This opt-in policy does not change DoorPolicy or simulator behavior.
    """

    def __init__(self, goal_frames=1, hidden=(256, 256), residual_scale=0.1):
        import math

        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be finite and positive")
        super().__init__(goal_frames, hidden)
        self.residual_scale = residual_scale
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def correction(self, obs, goal):
        """Dimensionless bounded offset before the native-control scale."""
        return super().forward(obs, goal)

    def forward(self, obs, goal, reference_action):
        correction = self.correction(obs, goal)
        if reference_action.shape != correction.shape:
            raise ValueError("Reference action shape must match policy output")
        return (reference_action + self.residual_scale * correction).clamp(-1, 1)
