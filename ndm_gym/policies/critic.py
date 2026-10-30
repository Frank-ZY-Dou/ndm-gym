"""State-value critic (paper Table A.4)."""

from __future__ import annotations

import torch.nn as nn

from ..core.nets import mlp, run_mlp
from ..core.normalizer import RunningNorm


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
