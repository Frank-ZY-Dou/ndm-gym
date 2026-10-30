"""SuperMimic world model: predicts frame-local increments of every state component (paper §3.2, Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import so3_exp, so3_log, yaw_mat
from ..robots.g1 import NUM_J
from ..state import INV_DIM, XI_DIM, invariant_features
from .rigidformer import RigidFormerNet

WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J


class IncrementWorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments."""

    def __init__(self, hidden=(2048, 2048, 2048), act: str = "silu", layer_norm: bool = True, rigidformer: dict | None = None):
        super().__init__()
        if rigidformer is None:
            self.in_dim = INV_DIM + NUM_J + XI_DIM
            self.net = mlp(self.in_dim, list(hidden), WM_OUT, act=ACTIVATIONS[act], layer_norm=layer_norm)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.in_norm = RunningNorm(self.in_dim)
        else:  # (dp, dth, dv, dw) from the pelvis token, (dq, dqd) from the joint tokens
            self.net = RigidFormerNet(root_out=12, joint_out=2, **rigidformer)
            self.in_dim, self.in_norm = self.net.in_dim, self.net.make_input_norm()
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)
        self.amp = True  # bf16 autocast in step (if enabled); --wm_bptt_fp32 turns it off during the policy's BPTT

    def inputs(self, S, a, xi_n):
        if isinstance(self.net, RigidFormerNet):
            return self.net.inputs(S, a, xi_n)
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ R1),
                (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n, input_noise: float = 0.0):
        """``input_noise``: std of Gaussian noise on the normalized inputs (only while fitting, as a regularizer)."""
        p, R, v, w, q, qd = S
        x = self.in_norm(self.inputs(S, a, xi_n))
        if input_noise != 0:
            x = x + input_noise * torch.randn_like(x)
        y = run_mlp(self.net, x, amp=self.amp)
        d = y * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                (H.transpose(-1, -2) @ (ph - p).unsqueeze(-1)).squeeze(-1),
                so3_log(Rt @ Rh),
                (Rt @ (vh - v).unsqueeze(-1)).squeeze(-1),
                (Rt @ (wh - w).unsqueeze(-1)).squeeze(-1),
                qh - q,
                qdh - qd,
            ),
            -1,
        )
