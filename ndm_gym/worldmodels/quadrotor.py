"""World model of a quadrotor on the tracking task's state (tasks.quadrotor_tracking), in the two parameterizations of the
LAFAN1 world models. ``accel`` (SuperTrack's, worldmodels.accel): an MLP predicts the linear acceleration in the body's frame
and the body rates' derivative, integrated over one control step with semi-implicit Euler in closed form for the simulator's
``substeps`` (velocities gain dt a, positions dt (v + (n + 1) / (2 n) dt a)). ``increment`` (SuperMimic's,
worldmodels.increment): it predicts the increments of every state component, in the body's frame.

A rigid body in free flight: its dynamics do not depend on the position or the heading, so the network reads the world's up
axis and the linear velocity in the body's frame, the body rates and the command (13 numbers). Both parameterizations are fit
with the same error, the next state's in units of the spread of one true step's increments."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import so3_exp, so3_log
from ..robots.quadrotor import CONTROL_DT, NUM_U

KINDS = ("increment", "accel")
WM_IN = 3 + 3 + 3 + NUM_U
WM_OUT = 3 + 3 + 3 + 3  # increments: position, rotation, linear velocity (body frame), body rates
WM_OUT_ACCEL = 3 + 3  # the linear acceleration (body frame), the body rates' derivative


def _rot(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (R @ x.unsqueeze(-1)).squeeze(-1)


class QuadWorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, u_t), u_t the command (thrust, body rates)."""

    def __init__(self, hidden=(512, 512, 512), act: str = "silu", layer_norm: bool = True, kind: str = "increment", substeps: int = 4):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.pos_coef = (substeps + 1) / (2 * substeps) * CONTROL_DT
        out = WM_OUT if kind == "increment" else WM_OUT_ACCEL
        self.net = mlp(WM_IN, list(hidden), out, act=ACTIVATIONS[act], layer_norm=layer_norm)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(WM_IN)
        self.out_norm = RunningNorm(out, eps=1e-4 if kind == "increment" else 1e-4 / CONTROL_DT)
        # accel: the spread of one true step's increments, the error's units (increment: the output normalizer's)
        self.err_norm = RunningNorm(WM_OUT, eps=1e-4) if kind == "accel" else None
        # floors on the error normalization: position (m), rotation (rad), linear (m/s) and angular velocity (rad/s)
        self.register_buffer("err_floor", torch.cat((torch.full((3,), 5e-4), torch.full((3,), 1e-3), torch.full((3,), 1e-2),
                                                     torch.full((3,), 5e-2))))  # fmt: skip

    def inputs(self, S, u: torch.Tensor) -> torch.Tensor:
        p, R, v, w = S
        return torch.cat((R[..., 2, :], _rot(R.transpose(-1, -2), v), w, u), -1)

    @staticmethod
    def increments(S, S1) -> torch.Tensor:
        """The increments from S to S1 in S's body frame."""
        p, R, v, w = S
        p1, R1, v1, w1 = S1
        Rt = R.transpose(-1, -2)
        return torch.cat((_rot(Rt, p1 - p), so3_log(Rt @ R1), _rot(Rt, v1 - v), w1 - w), -1)

    def targets(self, S, S1) -> torch.Tensor:
        """The regression targets, what ``out_norm`` normalizes: the increments from S to S1, or (accel) the mean
        accelerations over the step."""
        if self.kind == "increment":
            return self.increments(S, S1)
        return torch.cat((_rot(S[1].transpose(-1, -2), S1[2] - S[2]), S1[3] - S[3]), -1) / CONTROL_DT

    def step(self, S, u: torch.Tensor):
        """The next state from ``S`` and the command ``u``."""
        p, R, v, w = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, u)))
        d = y * self.out_norm.std() + self.out_norm.mean
        if self.kind == "accel":
            av, aw = d.split((3, 3), -1)
            av, c = _rot(R, av), self.pos_coef
            return p + CONTROL_DT * (v + c * av), R @ so3_exp(CONTROL_DT * (w + c * aw)), v + CONTROL_DT * av, w + CONTROL_DT * aw
        dp, dth, dv, dw = d.split((3, 3, 3, 3), -1)
        return p + _rot(R, dp), R @ so3_exp(dth), v + _rot(R, dv), w + dw

    def error(self, S_hat, S) -> torch.Tensor:
        """The prediction's error in the increments' layout (the true state's body frame), in units of the increments' spread."""
        std = self.out_norm.std() if self.kind == "increment" else self.err_norm.std()
        return self.increments(S, S_hat) / torch.maximum(std, self.err_floor)
