"""Policy for manipulation demonstration tracking: an MLP from the observation and the goal (tasks.robomimic_tracking) to one
action per actuator in [-1, 1], an offset of its position target about the reference's."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import mlp, run_mlp
from ..core.normalizer import RunningNorm


class RobomimicPolicy(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, num_actions: int, hidden=(1024, 512, 256)):
        super().__init__()
        self.net = mlp(obs_dim + goal_dim, list(hidden), num_actions)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        self.norm = RunningNorm(obs_dim + goal_dim)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.from_features(torch.cat((obs, goal), -1))

    def from_features(self, x: torch.Tensor) -> torch.Tensor:
        """The action from the observation and the goal as one tensor (RobomimicTask.critic_features: the critic's input too)."""
        return torch.tanh(run_mlp(self.net, self.norm(x)))
