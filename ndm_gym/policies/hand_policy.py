"""Policy for hand-object tracking: an MLP from the observation and the goal (tasks.hand_tracking) to one action per
joint in [-1, 1], an offset of the position target about the reference pose."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import mlp, run_mlp
from ..core.normalizer import RunningNorm


class HandPolicy(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, num_joints: int, hidden=(1024, 512, 256)):
        super().__init__()
        self.net = mlp(obs_dim + goal_dim, list(hidden), num_joints)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        self.norm = RunningNorm(obs_dim + goal_dim)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(run_mlp(self.net, self.norm(torch.cat((obs, goal), -1))))
