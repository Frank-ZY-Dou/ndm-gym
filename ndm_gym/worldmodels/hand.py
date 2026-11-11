"""World model of a hand and the object it handles: SuperMimic's increment model (worldmodels.increment) on the
hand-object state (tasks.hand_tracking). An MLP predicts the increments of every state component from the hand's
joints, the object about the palm and in the world (its height and tilt decide the contact with the floor), and the
position targets' offsets from the joints."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import mat_to_6d, so3_exp, so3_log
from ..robots.xhand import NUM_J

# the palm's translations are left out: the dynamics do not depend on them
WM_IN = (NUM_J - 3) + NUM_J + 3 + 6 + 3 + 3 + 1 + 6 + NUM_J
WM_OUT = NUM_J + NUM_J + 3 + 3 + 3 + 3


class HandWorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, u_t), u_t the joints' position targets."""

    def __init__(self, hidden=(1024, 1024, 1024), act: str = "silu", layer_norm: bool = True):
        super().__init__()
        self.net = mlp(WM_IN, list(hidden), WM_OUT, act=ACTIVATIONS[act], layer_norm=layer_norm)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(WM_IN)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)
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
    def targets(S, S1) -> torch.Tensor:
        """The increments from S to S1 (the regression targets): the object's rotation in its own frame."""
        q, qd, p, R, v, w = S
        q1, qd1, p1, R1, v1, w1 = S1
        return torch.cat((q1 - q, qd1 - qd, p1 - p, so3_log(R.transpose(-1, -2) @ R1), v1 - v, w1 - w), -1)

    def step(self, S, fk, u: torch.Tensor):
        """The next state from ``S``, its forward kinematics ``fk`` (HandKinematics.forward) and the targets ``u``."""
        q, qd, p, R, v, w = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, fk, u)))
        d = y * self.out_norm.std() + self.out_norm.mean
        dq, dqd, dp, dth, dv, dw = d.split((NUM_J, NUM_J, 3, 3, 3, 3), -1)
        return q + dq, qd + dqd, p + dp, R @ so3_exp(dth), v + dv, w + dw

    def error(self, S_hat, S) -> torch.Tensor:
        """The prediction's error in the targets' layout, in units of the increments' spread."""
        q, qd, p, R, v, w = S
        qh, qdh, ph, Rh, vh, wh = S_hat
        e = torch.cat((qh - q, qdh - qd, ph - p, so3_log(R.transpose(-1, -2) @ Rh), vh - v, wh - w), -1)
        return e / torch.maximum(self.out_norm.std(), self.err_floor)
