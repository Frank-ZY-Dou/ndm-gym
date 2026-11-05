"""Running mean/std normalizer for network inputs and outputs."""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    # train.py --no_sync (InstantTrack, BlinkMimic): the freeze test on the device (torch.where) instead of a host-side
    # `if`, and the statistics written in place (the same values; a CUDA graph reads the tensors it captured)
    no_sync: bool = False

    @torch.no_grad()
    def update(self, x: torch.Tensor, gate: torch.Tensor | None = None):
        """``gate`` (--no_sync only; train_muscles' graphable updates): a 0-dim bool, the statistics unchanged where it is false."""
        if gate is not None and not RunningNorm.no_sync:
            raise ValueError("RunningNorm.update's gate needs RunningNorm.no_sync (--no_sync)")
        x = x.reshape(-1, x.shape[-1]).float()
        if not RunningNorm.no_sync and self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        if RunningNorm.no_sync:
            live = self.count < self.max_count if gate is None else (self.count < self.max_count) & gate
            var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
            self.mean.copy_(torch.where(live, self.mean + delta * n / tot, self.mean))
            self.var.copy_(torch.where(live, var, self.var))
            self.count.copy_(torch.where(live, tot, self.count))
            return
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)
