"""The policy gradient of a BPTT update taken apart by rollout (and by imagined step), for ``--pi_rollout_clip`` and
``tools/grad_stats.py --anatomy``.

The policy gradient of an update is g = sum_{t,b} J_tb^T abar_tb over the imagined steps t and the rollouts b: abar_tb is
the gradient that reaches the policy's output at step t of rollout b from all its later uses (the world model, the
observations, the losses), the output's adjoint, and J_tb the Jacobian of that output in the policy's parameters. For a
policy of linear layers, layer l's part is sum_{t,b} D_tb X_tb^T (weight) and sum_{t,b} D_tb (bias), with X the layer's
input and D the gradient of its output when the policy runs again on the steps' inputs with abar as its output's
gradient. ``RolloutGrads`` does that rerun and keeps X and D, from which come, without forming the per-rollout parts
G_b = sum_t J_tb^T abar_tb, their squared norms sum_l sum_{t,t'} (D_tb . D_t'b) (X_tb . X_t'b + 1), their dot products
with a vector, and any weighted sum sum_b w_b G_b.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..policies.tracking_policy import TrackingPolicy


class RolloutGrads:
    """The rerun of ``pi`` on the inputs of H imagined steps of B rollouts, obs histories ``Ob`` (H, B, hist, obs) and
    goals ``Gl`` (H, B, goal), with the output adjoints ``A`` (H, B, J) as the output's gradient, ``chunk`` rollouts
    at a time (as the update ran it: matmuls under the bf16 autocast of ``run_mlp`` when it is set). Every parameter of
    ``pi`` must be a linear layer's weight or bias. Gradients are flat, the parameters in ``pi.parameters()``'s order."""

    def __init__(self, pi: TrackingPolicy, Ob: torch.Tensor, Gl: torch.Tensor, A: torch.Tensor, chunk: int = 512):
        self.lins = [m for m in pi.modules() if isinstance(m, nn.Linear)]
        self.params = list(pi.parameters())
        self.offs, o = {}, 0
        for p in self.params:
            self.offs[p], o = slice(o, o + p.numel()), o + p.numel()
        if set(self.offs) != {p for m in self.lins for p in (m.weight, m.bias)}:
            raise ValueError("RolloutGrads takes a policy of linear layers with biases and no other parameters")
        self.H, self.B = A.shape[:2]
        self.kept = []  # per chunk: (c0, c1, [(X, D) of every layer]), rows (t, b) with t outer
        xs, ds = {}, {}

        def fhook(m, inp, out):
            xs[m] = inp[0].detach()
            out.register_hook(lambda d: ds.__setitem__(m, d.detach()))

        handles = [m.register_forward_hook(fhook) for m in self.lins]
        try:
            for c0 in range(0, self.B, chunk):
                c1 = min(self.B, c0 + chunk)
                with torch.enable_grad():
                    a = TrackingPolicy.forward(pi, Ob[:, c0:c1].flatten(0, 1), Gl[:, c0:c1].flatten(0, 1))
                    torch.autograd.grad(a, self.params, A[:, c0:c1].flatten(0, 1).to(a.dtype))
                self.kept.append((c0, c1, [(xs[m], ds[m]) for m in self.lins]))
        finally:
            for h in handles:
                h.remove()

    def _layers(self):
        """(c0, c1, layer, X, D) with X and D in fp32 as (H, rollouts of the chunk, width)."""
        for c0, c1, layers in self.kept:
            for m, (X, D) in zip(self.lins, layers):
                yield c0, c1, m, X.float().unflatten(0, (self.H, c1 - c0)), D.float().unflatten(0, (self.H, c1 - c0))

    def rollout_sq_norms(self) -> torch.Tensor:
        """|G_b|^2 (B,)."""
        n2 = torch.zeros(self.B, device=self.params[0].device)
        for c0, c1, _, X, D in self._layers():
            Xe, De = X.transpose(0, 1), D.transpose(0, 1)  # (rollouts, H, width)
            n2[c0:c1] += ((De @ De.transpose(1, 2)) * (Xe @ Xe.transpose(1, 2) + 1)).sum((1, 2))
        return n2

    def rollout_dots(self, v: torch.Tensor) -> torch.Tensor:
        """G_b . v (B,) for a flat vector v."""
        out = torch.zeros(self.B, device=v.device)
        for c0, c1, m, X, D in self._layers():
            Xe, De = X.transpose(0, 1), D.transpose(0, 1)
            out[c0:c1] += ((Xe @ v[self.offs[m.weight]].view_as(m.weight).T) * De).sum((1, 2)) + De.sum(1) @ v[self.offs[m.bias]]
        return out

    def weighted(self, weights: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        """sum_b w_b G_b for every weight vector of ``weights`` (name -> (B,)), and each chunk's own sum of G_b."""
        acc, groups = {k: {} for k in weights}, []
        for c0, c1, layers in self.kept:
            grp = {}
            for m, (X, D) in zip(self.lins, layers):
                X, D = X.float(), D.float()
                grp[m.weight], grp[m.bias] = D.T @ X, D.sum(0)
                for k, w in weights.items():
                    Dw = D * w[c0:c1].repeat(self.H)[:, None]
                    acc[k][m.weight] = acc[k].get(m.weight, 0) + Dw.T @ X
                    acc[k][m.bias] = acc[k].get(m.bias, 0) + Dw.sum(0)
            groups.append(self.flat(grp))
        return {k: self.flat(a) for k, a in acc.items()}, groups

    def step_parts(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per layer, every step's part summed over the rollouts: weights (H, out, in) and biases (H, out)."""
        W = [torch.zeros(self.H, *m.weight.shape, device=m.weight.device) for m in self.lins]
        bias = [torch.zeros(self.H, m.out_features, device=m.weight.device) for m in self.lins]
        idx = {m: i for i, m in enumerate(self.lins)}
        for _, _, m, X, D in self._layers():
            W[idx[m]] += D.transpose(1, 2) @ X
            bias[idx[m]] += D.sum(1)
        return W, bias

    def flat(self, part: dict) -> torch.Tensor:
        """A gradient given per parameter (parameter -> tensor) as one flat vector."""
        return torch.cat([part[p].flatten() for p in self.params])
