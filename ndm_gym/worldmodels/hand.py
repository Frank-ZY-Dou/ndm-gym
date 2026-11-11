"""World model of a hand and the object it handles, on the hand-object state (tasks.hand_tracking), in the two
parameterizations of the LAFAN1 world models. ``increment`` (SuperMimic's, worldmodels.increment): an MLP predicts the
increments of every state component. ``accel`` (SuperTrack's, worldmodels.accel): it predicts the accelerations of the
joints and of the object (linear and angular, world frame), integrated with semi-implicit Euler over one control step.
Both read the hand's joints, the object about the palm and in the world (its height and tilt decide the contact with the
floor), and the position targets' offsets from the joints, and both are fit with the same error: the next state's, in
units of the spread of one true step's increments."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import mat_to_6d, so3_exp, so3_log
from ..robots.xhand import NUM_J
from ..state import CONTROL_DT

KINDS = ("increment", "accel")
# the palm's translations are left out: the dynamics do not depend on them
WM_IN = (NUM_J - 3) + NUM_J + 3 + 6 + 3 + 3 + 1 + 6 + NUM_J
WM_OUT = NUM_J + NUM_J + 3 + 3 + 3 + 3
WM_OUT_ACCEL = NUM_J + 3 + 3  # the joints' accelerations, the object's linear and angular


class HandWorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, u_t), u_t the joints' position targets."""

    def __init__(self, hidden=(1024, 1024, 1024), act: str = "silu", layer_norm: bool = True, kind: str = "increment"):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        out = WM_OUT if kind == "increment" else WM_OUT_ACCEL
        self.net = mlp(WM_IN, list(hidden), out, act=ACTIVATIONS[act], layer_norm=layer_norm)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(WM_IN)
        self.out_norm = RunningNorm(out, eps=1e-4 if kind == "increment" else 1e-4 / CONTROL_DT)
        # accel: the spread of one true step's increments, the error's units (increment: the output normalizer's)
        self.err_norm = RunningNorm(WM_OUT, eps=1e-4) if kind == "accel" else None
        # floors on the error normalization: joint positions and velocities, the object's position, rotation, velocities
        self.register_buffer("err_floor", torch.cat((torch.full((NUM_J,), 1e-3), torch.full((NUM_J,), 2e-2), torch.full((3,), 5e-4),
                                                     torch.full((3,), 5e-3), torch.full((3,), 1e-2), torch.full((3,), 5e-2))))  # fmt: skip

    def inputs(self, S, fk, u: torch.Tensor) -> torch.Tensor:
        q, qd, p, R, v, w = S
        keypoints, palm_R = fk
        Rt = palm_R.transpose(-1, -2)
        return torch.cat((q[..., 3:], qd, (Rt @ (p - keypoints[..., 0, :]).unsqueeze(-1)).squeeze(-1), mat_to_6d(Rt @ R), v, w, p[..., 2:3],
                          mat_to_6d(R), u - q), -1)  # fmt: skip

    @staticmethod
    def increments(S, S1) -> torch.Tensor:
        """The increments from S to S1: the object's rotation in its own frame."""
        q, qd, p, R, v, w = S
        q1, qd1, p1, R1, v1, w1 = S1
        return torch.cat((q1 - q, qd1 - qd, p1 - p, so3_log(R.transpose(-1, -2) @ R1), v1 - v, w1 - w), -1)

    def targets(self, S, S1) -> torch.Tensor:
        """The regression targets, what ``out_norm`` normalizes: the increments from S to S1, or (accel) the mean
        accelerations over the step of the joints and of the object."""
        if self.kind == "increment":
            return self.increments(S, S1)
        return torch.cat((S1[1] - S[1], S1[4] - S[4], S1[5] - S[5]), -1) / CONTROL_DT

    def step(self, S, fk, u: torch.Tensor):
        """The next state from ``S``, its forward kinematics ``fk`` (HandKinematics.forward) and the targets ``u``."""
        q, qd, p, R, v, w = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, fk, u)))
        d = y * self.out_norm.std() + self.out_norm.mean
        if self.kind == "accel":  # semi-implicit Euler over the control step; the angular velocity is the world frame's
            aq, av, aw = d.split((NUM_J, 3, 3), -1)
            qd1, v1, w1 = qd + CONTROL_DT * aq, v + CONTROL_DT * av, w + CONTROL_DT * aw
            return q + CONTROL_DT * qd1, qd1, p + CONTROL_DT * v1, so3_exp(CONTROL_DT * w1) @ R, v1, w1
        dq, dqd, dp, dth, dv, dw = d.split((NUM_J, NUM_J, 3, 3, 3, 3), -1)
        return q + dq, qd + dqd, p + dp, R @ so3_exp(dth), v + dv, w + dw

    def error(self, S_hat, S) -> torch.Tensor:
        """The prediction's error in the increments' layout, in units of the increments' spread."""
        q, qd, p, R, v, w = S
        qh, qdh, ph, Rh, vh, wh = S_hat
        e = torch.cat((qh - q, qdh - qd, ph - p, so3_log(R.transpose(-1, -2) @ Rh), vh - v, wh - w), -1)
        std = self.out_norm.std() if self.kind == "increment" else self.err_norm.std()
        return e / torch.maximum(std, self.err_floor)
