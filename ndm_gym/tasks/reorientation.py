"""Simulator-independent fixed-palm object reorientation objectives.

Quaternions are scalar-first (wxyz). Goals distinguish cube orientations (no
symmetry reduction); a marked face therefore has a unique target orientation.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReorientationCfg:
    """SI units; a goal must be held continuously before success."""

    tolerance: float = 0.2
    hold_steps: int = 10
    episode_steps: int = 250
    drop_distance: float = 0.15
    action_penalty: float = 0.001

    def __post_init__(self):
        if not 0 < self.tolerance < torch.pi:
            raise ValueError("tolerance must be in (0, pi)")
        if not 0 < self.hold_steps <= self.episode_steps:
            raise ValueError("hold_steps must be positive and <= episode_steps")
        if self.drop_distance <= 0 or self.action_penalty < 0:
            raise ValueError("invalid distance or penalty")


def orientation_error(q: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Shortest angular distance in radians; invariant to quaternion sign.

    Inputs are nonzero quaternions of shape (..., 4). atan2 avoids acos's
    singular gradient at a matching orientation.
    """
    q, goal = F.normalize(q, dim=-1), F.normalize(goal, dim=-1)
    w, v = q[..., :1], q[..., 1:]
    gw, gv = goal[..., :1], goal[..., 1:]
    scalar = (w * gw + (v * gv).sum(-1, keepdim=True)).squeeze(-1)
    vector = w * gv - gw * v - torch.linalg.cross(v, gv, dim=-1)
    return 2 * torch.atan2(vector.norm(dim=-1), scalar.abs())


def sample_goals(n: int, *, generator: torch.Generator, device="cpu") -> torch.Tensor:
    """Uniform SO(3) goals from normalized isotropic Gaussian quaternions."""
    return F.normalize(torch.randn(n, 4, generator=generator, device=device), dim=-1)


def reward(q, goal, action, cfg: ReorientationCfg):
    """Bounded orientation score minus mean squared actuator effort proxy."""
    return torch.exp(-orientation_error(q, goal)) - cfg.action_penalty * action.square().mean(-1)


class GoalTracker:
    """Batched episode bookkeeping; callers reset only after storing transitions.

    Drops take precedence over success. Time limits are truncations, distinct
    from terminal success/drop transitions for value bootstrapping.
    """

    def __init__(self, n: int, cfg: ReorientationCfg, device="cpu"):
        self.cfg = cfg
        self.steps = torch.zeros(n, dtype=torch.long, device=device)
        self.held = torch.zeros_like(self.steps)

    def reset(self, ids):
        self.steps[ids] = 0
        self.held[ids] = 0

    def step(self, error, distance):
        self.steps += 1
        dropped = (~torch.isfinite(error)) | (~torch.isfinite(distance)) | (distance > self.cfg.drop_distance)
        inside = (error <= self.cfg.tolerance) & ~dropped
        self.held = torch.where(inside, self.held + 1, 0)
        success = (self.held >= self.cfg.hold_steps) & ~dropped
        terminated = success | dropped
        truncated = (self.steps >= self.cfg.episode_steps) & ~terminated
        return {"success": success, "dropped": dropped, "terminated": terminated, "truncated": truncated}
