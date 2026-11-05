# Ported from BlinkMimic (https://github.com/fuyichen0725/blinkmimic @ d264658, src/blinkmimic/networks.py) under its MIT
# License:
#
# Copyright (c) 2026 Fuyi Chen
# Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""BlinkMimic's semi-implicit world model (``--wm_structured``): velocity increments are predicted, positions integrated.

The network predicts the velocity increments and, for the positions, corrections on top of the semi-implicit drift: the
position change that the old and new velocities imply over the control step, dt ((1 - c) v + c v'). c = (n + 1) / (2 n) =
0.625 for the simulator's n = 4 substeps. This is the idea of SuperTrack's acceleration world model (Fussell et al. 2021,
§3.4; ``accel.py``): velocities from the network, positions by semi-implicit integration. With zero corrections a step
equals ``AccelWorldModel``'s with ``--st_substeps 4`` (tests/test_blinkmimic.py). SuperMimic's world model
(``increment.py``) predicts every increment, positions included, directly.
"""

from __future__ import annotations

import torch

from ..core.nets import run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import so3_exp, yaw_mat
from ..robots.g1 import NUM_J
from ..state import CONTROL_DT
from .increment import WM_OUT, IncrementWorldModel


class SemiImplicitIncrementWorldModel(IncrementWorldModel):
    """``IncrementWorldModel`` whose network outputs (the ``targets`` layout) are the velocity increments and the position
    corrections on top of the drift. The fit's loss is normalized by the plain increments' scale (``inc_norm``)."""

    structured = True  # the fused inference copy (g1_infer.inference_copy) integrates plain increments and refuses it
    # weight of the new velocity in the drift: n semi-implicit substeps with a linearly changing velocity give
    # sum_k k / n^2 = (n + 1) / (2 n); 0.625 for n = 4 (BlinkMimic measured the residual std of the joint-position
    # increment at 16% with 0.5, 11% with 0.625 and 17% with 1.0)
    DRIFT_W = 0.625

    def __init__(self, hidden=(2048, 2048, 2048), act: str = "silu", layer_norm: bool = True, rigidformer: dict | None = None):
        if rigidformer is not None:
            raise ValueError("--wm_structured: the MLP world model only")
        super().__init__(hidden, act, layer_norm)
        self.inc_norm = RunningNorm(WM_OUT, eps=1e-4)

    increments = staticmethod(IncrementWorldModel.targets)  # the plain frame-local increments

    def drift(self, S, v1, w1, qd1):
        """The semi-implicit position change from S with the new velocities, in the increments' frames: the heading frame
        for the root position, the body frame for the rotation vector."""
        p, R, v, w, q, qd = S
        a, b = CONTROL_DT * (1 - self.DRIFT_W), CONTROL_DT * self.DRIFT_W
        Ht, Rt = yaw_mat(R).transpose(-1, -2), R.transpose(-1, -2)
        return (Ht @ (a * v + b * v1).unsqueeze(-1)).squeeze(-1), (Rt @ (a * w + b * w1).unsqueeze(-1)).squeeze(-1), a * qd + b * qd1

    def targets(self, S, S1):
        """The velocity increments and the position corrections on top of the drift from S to S1."""
        dp, dth, dv, dw, dq, dqd = self.increments(S, S1).split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        gp, gth, gq = self.drift(S, S1[2], S1[3], S1[5])
        return torch.cat((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)

    def integrate(self, S, d):
        """The next state from denormalized network outputs ``d`` (the ``targets`` layout)."""
        p, R, v, w, q, qd = S
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        qd1 = qd + dqd
        gp, gth, gq = self.drift(S, v1, w1, qd1)
        dp, dth, dq = dp + gp, dth + gth, dq + gq
        return (p + (yaw_mat(R) @ dp.unsqueeze(-1)).squeeze(-1), R @ so3_exp(dth), v1, w1, q + dq, qd1)

    def step(self, S, a, xi_n, input_noise: float = 0.0):
        """``input_noise``: std of Gaussian noise on the normalized inputs (only while fitting, as a regularizer)."""
        x = self.in_norm(self.inputs(S, a, xi_n))
        if input_noise != 0:
            x = x + input_noise * torch.randn_like(x)
        y = run_mlp(self.net, x, amp=self.amp)
        return self.integrate(S, y * self.out_norm.std() + self.out_norm.mean)
