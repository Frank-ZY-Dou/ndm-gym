"""SuperTrack world model: predicts accelerations and integrates them (Fussell et al. 2021, §3.4, Eq. 8-12)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import so3_exp
from ..robots.g1 import NUM_J
from ..state import CONTROL_DT, INV_DIM, XI_DIM, invariant_features
from .rigidformer import RigidFormerNet

ACC_OUT = 3 + 3 + NUM_J


class AccelWorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting the root's linear/angular accelerations in its
    own frame and the joint accelerations, integrated over one control step with semi-implicit Euler.

    ``substeps`` n integrates the (constant) predicted acceleration as n semi-implicit Euler substeps of dt / n, in
    closed form: velocities gain dt * a, positions dt * (v + (n + 1) / (2 n) * dt * a). n = 1 is the paper's single
    step (Eq. 9-12); n = 4 matches the simulator's four 5 ms substeps.
    """

    def __init__(self, hidden=(2048, 2048, 2048), substeps: int = 1, act: str = "silu", layer_norm: bool = True,
                 rigidformer: dict | None = None):
        super().__init__()
        self.pos_coef = (substeps + 1) / (2 * substeps) * CONTROL_DT
        if rigidformer is None:
            self.in_dim = INV_DIM + NUM_J + XI_DIM
            self.net = mlp(self.in_dim, list(hidden), ACC_OUT, act=ACTIVATIONS[act], layer_norm=layer_norm)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.in_norm = RunningNorm(self.in_dim)
        else:  # body-token Transformer: (av, aw) from the pelvis token, joint accelerations from the joint tokens
            self.net = RigidFormerNet(root_out=6, joint_out=1, **rigidformer)
            self.in_dim, self.in_norm = self.net.in_dim, self.net.make_input_norm()
        # same floor as the increment model's (1e-4 per step), expressed per second
        self.out_norm = RunningNorm(ACC_OUT, eps=1e-4 / CONTROL_DT)
        self.amp = True  # bf16 autocast in step (if enabled); --wm_bptt_fp32 turns it off during the policy's BPTT

    def inputs(self, S, a, xi_n):
        if isinstance(self.net, RigidFormerNet):
            return self.net.inputs(S, a, xi_n)
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Mean accelerations from S to S1 over one control step (the regression targets)."""
        p, R, v, w, q, qd = S
        v1, w1, qd1 = S1[2], S1[3], S1[5]
        Rt = R.transpose(-1, -2)
        dv = (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1)
        dw = (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1)
        return torch.cat((dv, dw, qd1 - qd), -1) / CONTROL_DT

    def step(self, S, a, xi_n, input_noise: float = 0.0):
        """``input_noise``: std of Gaussian noise on the normalized inputs (only while fitting, as a regularizer)."""
        p, R, v, w, q, qd = S
        x = self.in_norm(self.inputs(S, a, xi_n))
        if input_noise != 0:
            x = x + input_noise * torch.randn_like(x)
        y = run_mlp(self.net, x, amp=self.amp)
        acc = y * self.out_norm.std() + self.out_norm.mean
        av, aw, aq = acc.split((3, 3, NUM_J), -1)
        av, aw = (R @ av.unsqueeze(-1)).squeeze(-1), (R @ aw.unsqueeze(-1)).squeeze(-1)
        v1, w1, qd1 = v + CONTROL_DT * av, w + CONTROL_DT * aw, qd + CONTROL_DT * aq
        c = self.pos_coef
        return (p + CONTROL_DT * (v + c * av), so3_exp(CONTROL_DT * (w + c * aw)) @ R, v1, w1, q + CONTROL_DT * (qd + c * aq), qd1)
