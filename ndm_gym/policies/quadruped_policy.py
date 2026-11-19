"""Policy for quadruped motion tracking: an MLP from the observation and the goal (tasks.quadruped_tracking) to one action
per joint in [-1, 1], an offset of the position target about the reference pose. ``bound`` (train_humanoid --action_bound):
the actions in [-bound, bound] instead, ``bound * tanh(y / bound)`` of the MLP's output y (1: tanh(y), as before)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..robots.go2 import NUM_J


class QuadrupedPolicy(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, hidden=(1024, 512, 256), num_actions: int = NUM_J, bound: float = 1.0):
        super().__init__()
        self.bound = float(bound)
        self.net = mlp(obs_dim + goal_dim, list(hidden), num_actions)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()
        self.norm = RunningNorm(obs_dim + goal_dim)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        y = run_mlp(self.net, self.norm(torch.cat((obs, goal), -1)))
        return torch.tanh(y) if self.bound == 1.0 else self.bound * torch.tanh(y / self.bound)
