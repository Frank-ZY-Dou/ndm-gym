"""Differentiable contact dynamics for two Wuji hands and an 88-key piano."""

import itertools
import math

import torch
from torch import nn

from ndm_gym.core.nets import mlp

NUM_ACTUATORS = 52
KEY_COUNT = 88


class TipGeometry(nn.Module):
    """Evaluate per-digit quadratic fits and rotate each fingertip into the keyboard frame.

    The coefficient tensor is indexed by hand, digit, polynomial term and xyz (2 × 5 × 15 × 3).
    Origins, key centers and key half-sizes use meters. The six wrist scales convert normalized
    translation and rotation commands to meters and radians.
    """

    def __init__(self, coeff, origin, scales, centers, sizes):
        super().__init__()
        pairs = list(itertools.combinations_with_replacement(range(4), 2))
        self.register_buffer("ii", torch.tensor([i for i, _ in pairs]), persistent=False)
        self.register_buffer("jj", torch.tensor([j for _, j in pairs]), persistent=False)
        for name, values in (("coeff", coeff), ("origin", origin), ("scales", scales), ("centers", centers), ("sizes", sizes)):
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.float32))

    def forward(self, q):
        hands = q.reshape(-1, 2, 26)
        digits = hands[:, :, 6:].reshape(-1, 2, 5, 4)
        poly = torch.cat((torch.ones_like(digits[..., :1]), digits, digits[..., self.ii] * digits[..., self.jj]), -1)
        local = (poly[..., None, :] @ self.coeff).squeeze(-2)
        pose = hands[:, :, :6] * self.scales
        r = pose[:, :, None, 3:].expand_as(local)
        theta = r.norm(dim=-1, keepdim=True)
        cross = torch.cross(r, local, dim=-1)
        rot = local + torch.sinc(theta / math.pi) * cross + 0.5 * torch.sinc(theta / (2 * math.pi)).square() * torch.cross(r, cross, dim=-1)
        return (rot + (self.origin + pose[:, :, :3])[:, :, None, :]).flatten(1, 2)

    def distances(self, tips):
        diff = tips[:, :, None, :] - self.centers
        # Approximate distance to reachable points on every key surface. Any of the ten tips can approach any key.
        dx = (diff[..., 0].abs() - self.sizes[:, 0] * 0.80).clamp_min(0)
        dy = diff[..., 1]
        dz = diff[..., 2] - self.sizes[:, 2]
        return dx, dy, dz


class PianoDynamics(nn.Module):
    """Predict one 20 Hz control step from the measured state and 52 normalized actuator targets.

    The 192 state values are actuator positions, their changes since the preceding step and the
    88 key depths as fractions of full travel. Both hands share the key surface geometry.
    """

    def __init__(self, geometry: TipGeometry):
        super().__init__()
        self.geom = geometry
        self.actuator = mlp(3 * NUM_ACTUATORS, [128, 128], NUM_ACTUATORS, act=nn.SiLU)
        self.key_residual = mlp(3 * NUM_ACTUATORS + KEY_COUNT, [192, 128], KEY_COUNT, act=nn.SiLU)
        self.tip_cal = nn.Parameter(torch.zeros(10))
        for net in (self.actuator, self.key_residual):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self, state, action):
        q, v, last = torch.split(state, [NUM_ACTUATORS, NUM_ACTUATORS, KEY_COUNT], -1)
        features = torch.cat((q, v, action), -1)
        vel = 0.5 * (action - q) + 0.12 * v + self.actuator(features)
        q1 = q + vel
        tips = self.geom(q1 + 0.18 * (action - q1))
        dx, dy, dz = self.geom.distances(tips)
        yerr = (dy.abs() - self.geom.sizes[:, 1] * 0.50).clamp_min(0)
        align = torch.exp(-((yerr / 0.007).square() + (dx / 0.012).square()).clamp_max(35))
        depth = torch.sigmoid((-dz - 0.004 + self.tip_cal[:, None] * 0.012) / 0.0027)
        contact = (align * depth).amax(1)
        keys = 0.91 * contact + 0.09 * last + self.key_residual(torch.cat((features, last), -1))
        return torch.cat((q1, vel, keys), -1)


class FingerDynamics(nn.Module):
    """Shared acceleration MLP conditioned on digit identity. Units are normalized to one control step."""

    def __init__(self, digits: int):
        super().__init__()
        self.digits = digits
        self.net = mlp(5 + digits, [64, 64], 2, act=nn.SiLU)
        self.register_buffer("identity", torch.eye(digits))

    def forward(self, state, action):
        q, k, v, w = state.split(self.digits, -1)
        features = torch.stack((q, k, v, w, action), -1)
        accel = self.net(torch.cat((features, self.identity.expand(len(q), -1, -1)), -1))
        v1 = v + accel[:, :, 0]
        w1 = w + accel[:, :, 1]
        return torch.cat((q + v1, k + w1, v1, w1), dim=-1)
