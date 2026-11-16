"""MLP building blocks shared by world models, policies and critics (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ---------------------------------------------------------------- deferred weight gradients (XMimic's dense(defer=True))
def linears(net: nn.Sequential) -> list[nn.Linear]:
    """The MLP's Linear layers, in order."""
    return [m for m in net if isinstance(m, nn.Linear)]


def _autocasts(device) -> bool:
    """Whether ``run_mlp``'s autocast acts on tensors of ``device``: it is CUDA's."""
    return torch.device(device).type == "cuda"


def deferred_dtype(device, amp: bool = True) -> torch.dtype:
    """The dtype of an MLP's pre-activations on ``device`` as ``run_mlp`` computes them: the autocast's where it applies
    (AMP_DTYPE set, ``amp``, a device it acts on), else fp32."""
    return AMP_DTYPE if AMP_DTYPE is not None and amp and _autocasts(device) else torch.float32


def mlp_taps(net: nn.Sequential, lead: tuple, device, amp: bool = True) -> list[torch.Tensor]:
    """Per Linear layer of ``net`` a zero tensor ``lead + (out,)`` that requires grad, in the pre-activations' dtype
    (``deferred_dtype``): ``run_mlp_deferred`` adds a slice of each to the layer's output at every application, so that after
    a backward pass its ``.grad`` holds the gradients at those pre-activations (``deferred_grads``)."""
    dt = deferred_dtype(device, amp)
    return [torch.zeros(*lead, m.out_features, device=device, dtype=dt, requires_grad=True) for m in linears(net)]


def run_mlp_deferred(net: nn.Sequential, x: torch.Tensor, taps: list[torch.Tensor], amp: bool = True) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """``run_mlp`` with each Linear layer's weight and bias detached and ``taps[i]`` (zeros, the layer's output shape; one
    application's slice of ``mlp_taps``) added to the i-th Linear's output. The values are ``run_mlp``'s; no gradient reaches
    the Linear layers' parameters, the pre-activations' gradients land in the taps instead. Also returns each Linear's input
    (detached; under the autocast as the product reads it, in bf16), from which ``deferred_grads`` forms the parameters'
    gradients over all applications at once."""
    cast = AMP_DTYPE if AMP_DTYPE is not None and amp and _autocasts(x.device) else None  # where run_mlp's autocast acts

    def forward(h):
        ins, i = [], 0
        for m in net:
            if isinstance(m, nn.Linear):
                if cast is not None:
                    h = h.to(cast)  # the autocast's own cast of the product's input
                ins.append(h.detach())  # values only: no gradient goes through the stored inputs
                h = F.linear(h, m.weight.detach(), m.bias.detach()) + taps[i]
                i += 1
            else:
                h = m(h)
        return h, ins

    if cast is None:
        return forward(x)
    with torch.autocast("cuda", dtype=cast):
        y, ins = forward(x)
    return y.float(), ins


@torch.no_grad()
def deferred_grads(net: nn.Sequential, ins: list[torch.Tensor], taps: list[torch.Tensor], bf16_product: bool = False):
    """The Linear layers' gradients of ``run_mlp_deferred``'s applications, added to their parameters' ``.grad`` (set where
    it is None): for layer i with the stacked inputs ``ins[i]`` (..., in) and the taps' gradients ``taps[i].grad`` (..., out)
    over all applications, dW = sum dy^T x as one matrix product and db = sum dy, where backpropagation forms one product per
    application inside the backward chain (XMimic's deferred_grads). Operands in bf16 (the autocast's) are multiplied in
    fp32, which holds them exactly, so the result is the fp32 sum (on the GPU a TF32 product under allow_tf32, which also
    holds bf16 exactly); ``bf16_product``: the bf16 product instead, its result rounded to bf16 as each application's is in
    backpropagation, without the casts."""
    for m, x, t in zip(linears(net), ins, taps, strict=True):
        if t.grad is None:
            raise RuntimeError("deferred_grads: no gradient reached a layer's pre-activation")
        dy, x = t.grad.flatten(0, -2), x.flatten(0, -2)
        if bf16_product and dy.dtype == torch.bfloat16:
            dw = (dy.T @ x).float()
        else:
            dw = dy.float().T @ x.float()
        db = dy.float().sum(0)
        for prm, g in ((m.weight, dw), (m.bias, db)):
            if prm.grad is None:
                prm.grad = g
            else:
                prm.grad.add_(g)
