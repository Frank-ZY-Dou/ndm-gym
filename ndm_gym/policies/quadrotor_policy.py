"""Policy for quadrotor tracking: an MLP from the observation and the goal (tasks.quadrotor_tracking) to the action in
[-1, 1], offsets of the thrust and the body rates about the reference's."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..robots.quadrotor import NUM_U


class QuadPolicy(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, hidden=(512, 256, 128)):
        super().__init__()
        self.net = mlp(obs_dim + goal_dim, list(hidden), NUM_U)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        self.norm = RunningNorm(obs_dim + goal_dim)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.from_features(torch.cat((obs, goal), -1))

    def from_features(self, x: torch.Tensor) -> torch.Tensor:
        """The action from the observation and the goal as one tensor (QuadTask.critic_features: the critic's input too)."""
        return torch.tanh(run_mlp(self.net, self.norm(x)))
