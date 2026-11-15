"""Native articulated Door tracking heads for ``MjcfWorldModel``'s (qpos, qvel).

This module does not import a simulator or actuate the door. Policies produce 28
normalized hand controls; the native controller maps them to actuator ranges.
Frame f denotes the current state: a transition from f is scored against f + 1.
No hold frames or fabricated velocities are added to the physical reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..motion.adroit_door import DoorReference


@dataclass(frozen=True)
class DoorRewardCfg:
    """Separate native units: ARTz in metres, remaining joints in radians."""

    translation_std: float = 0.02
    hand_std: float = 0.3
    door_std: float = 0.2
    latch_std: float = 0.3
    action_rate_weight: float = 0.01
    opening_rad: float = 1.35

    def __post_init__(self):
        import math

        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0 or (name.endswith("std") and value == 0):
                raise ValueError(f"Invalid Door reward setting: {name}")


class DoorTask:
    """Analytic heads shared by physical transitions and learned rollouts.

    No error-based early termination: report the entire reference horizon,
    including failed tracking. Nonfinite states terminate and are failures.
    Actual opening is independent of the dense tracking reward.
    """

    obs_dim = 88  # qpos30 + scaled qvel30 + previous controls28
    goal_frame_dim = 60
    action_dim = 28

    def __init__(self, reference: DoorReference, ctrl_range, device="cpu", cfg: DoorRewardCfg | None = None):
        self.cfg = cfg or DoorRewardCfg()
        self.q = torch.tensor(reference.qpos, dtype=torch.float32, device=device)
        self.v = torch.tensor(reference.qvel, dtype=torch.float32, device=device)
        self.dt, self.frames = reference.metadata["control_dt"], reference.frames
        ranges = torch.tensor(ctrl_range, dtype=torch.float32, device=device)
        if ranges.shape != (28, 2) or not torch.isfinite(ranges).all() or not (ranges[:, 1] > ranges[:, 0]).all():
            raise ValueError("Expected finite increasing native actuator ranges (28, 2)")
        self.ctrl_mid = ranges.mean(-1)
        self.ctrl_half = (ranges[:, 1] - ranges[:, 0]) / 2

    def next_frame(self, frame):
        return (frame + 1).clamp(max=self.frames - 1)

    def obs(self, state, a_prev):
        q, v = state
        return torch.cat((q, 0.1 * v, a_prev), -1)

    def goal(self, state, frame, goal_frames=1):
        q, v = state
        parts = []
        for k in range(1, goal_frames + 1):
            f = (frame + k).clamp(max=self.frames - 1)
            parts.extend((self.q[f] - q, 0.1 * (self.v[f] - v)))
        return torch.cat(parts, -1)

    def setpoint(self, action):
        """Physical native actuator controls for BOTH MuJoCo and MjcfWorldModel.

        The world model's servo features require physical controls, not normalized
        policy actions. Reward's action-rate cost still uses normalized actions.
        """
        return self.ctrl_mid + action.clamp(-1, 1) * self.ctrl_half

    def errors(self, state, frame):
        delta = state[0] - self.q[frame]
        return {"translation_m": delta[..., 0].abs(),
                "hand_rad": delta[..., 1:28].square().mean(-1).sqrt(),
                "door_rad": delta[..., 28].abs(), "latch_rad": delta[..., 29].abs()}

    def reward(self, state, frame, action, a_prev):
        c = self.cfg
        # Squared errors directly: sqrt at an exact match has an undefined gradient.
        d = state[0] - self.q[frame]
        r = (torch.exp(-d[..., 0].square() / c.translation_std**2)
             + torch.exp(-d[..., 1:28].square().mean(-1) / c.hand_std**2)
             + torch.exp(-d[..., 28].square() / c.door_std**2)
             + torch.exp(-d[..., 29].square() / c.latch_std**2)
             - c.action_rate_weight * (action - a_prev).square().mean(-1))
        return r * self.dt

    def opened(self, state):
        q, v = state
        return torch.isfinite(q).all(-1) & torch.isfinite(v).all(-1) & (q[..., 28] >= self.cfg.opening_rad)

    def termination(self, state, frame):
        q, v = state
        return (frame >= self.frames - 1) | ~torch.isfinite(q).all(-1) | ~torch.isfinite(v).all(-1)
