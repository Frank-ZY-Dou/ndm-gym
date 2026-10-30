"""MLP building blocks shared by world models, policies and critics (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

# bf16 autocast for MLP matmuls only (state integration, FK, rewards stay fp32); set by the entry points
AMP_DTYPE: torch.dtype | None = None

ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU, "elu": nn.ELU, "mish": nn.Mish, "relu": nn.ReLU}


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def run_mlp(net: nn.Module, x: torch.Tensor, amp: bool = True) -> torch.Tensor:
    """net(x), under bf16 autocast when the entry point set AMP_DTYPE, unless ``amp`` is False."""
    if AMP_DTYPE is None or not amp:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()
