"""Quadrotor trajectory tracking: a racing quadrotor (ndm_gym.robots.quadrotor) follows flown trajectories.

State tuple ``S = (p, R, v, w)``: the position, the rotation matrix, the world linear velocity and the angular velocity in
the body's frame (the body rates); flat, MuJoCo's qpos and qvel of the free joint. Observation, goal, reward and termination
are analytic functions of the state and the reference frame, shared by the simulator's transitions and the world model's
imagined rollouts. The policy's inputs are in the body's frame, so they do not depend on the position or the heading.

The reference is one or more flights, one after the other; a step taken at frame ``f`` tracks frame ``f + 1`` of the same
flight, and imagined rollouts stay on a flight's last frame once they reach it (``QuadReference.end``). The action is an
offset about the reference's feedforward command (``thrust``: the thrust along the reference's body z that gives its
acceleration against gravity and drag; its body rates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..core.rotations import mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq, so3_log
from ..robots.quadrotor import CONTROL_DT, GRAVITY, NUM_U, QuadrotorSpec

STATE_DIM = 3 + 4 + 3 + 3
OBS_DIM = 3 + 3 + 3 + NUM_U
GOAL_FRAME_DIM = 3 + 6 + 3


def flatten_state(S) -> torch.Tensor:
    p, R, v, w = S
    return torch.cat((p, mat_to_quat(R), v, w), -1)


def unflatten_state(x: torch.Tensor):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13]


def _to_body(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (x.unsqueeze(-2) @ R).squeeze(-2)  # rows x^T R = (R^T x)^T


class QuadReference(NamedTuple):
    """Flights at the control rate, (T, ...) per field: the state's components, the feedforward thrust, and the first and
    the last frame of each frame's flight."""

    p: torch.Tensor
    R: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    thrust: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor

    @property
    def frames(self) -> int:
        return self.p.shape[0]

    def state(self, f: torch.Tensor):
        return self.p[f], self.R[f], self.v[f], self.w[f]

    def next_frame(self, f: torch.Tensor) -> torch.Tensor:
        return torch.minimum(f + 1, self.end[f])

    def ahead(self, f: torch.Tensor, k: int) -> torch.Tensor:
        """The frame ``k`` steps after ``f``, at most its flight's last."""
        return torch.minimum(f + k, self.end[f])


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0], g[-1] = (x[1] - x[0]) / dt, (x[-1] - x[-2]) / dt
    return g


def _body_rates(R: torch.Tensor, dt: float) -> torch.Tensor:
    """The angular velocity in the body's frame of a rotation sequence (T, 3, 3), by central differences."""
    w = torch.zeros(R.shape[0], 3, dtype=R.dtype)
    w[1:-1] = so3_log(R[:-2].transpose(-1, -2) @ R[2:]) / (2 * dt)
    w[0], w[-1] = so3_log(R[0].T @ R[1]) / dt, so3_log(R[-2].T @ R[-1]) / dt
    return w


def build_reference(flights: list[dict[str, torch.Tensor]], spec: QuadrotorSpec, device: str) -> QuadReference:
    """The reference of ``flights`` (motion.sources.drone_racing.load_flight, sampled every CONTROL_DT), one after the other;
    velocities by central differences within each flight."""
    fields, offset = [], 0
    for fl in flights:
        p, R = fl["p"], quat_to_mat(fl["quat"])
        n = p.shape[0]
        if n < 3:
            raise ValueError("a flight needs at least three frames")
        v = _gradient(p, CONTROL_DT)
        acc = _gradient(v, CONTROL_DT)
        force = spec.mass * (acc + torch.tensor([0.0, 0.0, GRAVITY])) + spec.drag * v  # what the thrust has to give
        thrust = (force * R[..., 2]).sum(-1).clamp(0.0, spec.max_thrust)
        fields.append((p, R, v, _body_rates(R, CONTROL_DT), thrust, torch.full((n,), offset), torch.full((n,), offset + n - 1)))
        offset += n
    return QuadReference(*(torch.cat(x).to(device) for x in zip(*fields, strict=True)))


@dataclass
class RewardCfg:
    pos_std: float = 0.3  # m
    rot_std: float = 0.5  # rad
    vel_std: float = 2.0  # m/s
    ang_std: float = 4.0  # rad/s
    w_pos: float = 2.0
    w_rot: float = 1.0
    w_vel: float = 0.5
    w_ang: float = 0.25
    w_action_rate: float = -0.05
    term_pos: float = 1.0  # m, the position's distance from its reference

    def scaled(self, k: float) -> RewardCfg:
        """The tracking kernels' widths scaled by ``k`` (training's --reward_std_scale)."""
        widths = ("pos_std", "rot_std", "vel_std", "ang_std")
        return RewardCfg(**{**self.__dict__, **{n: getattr(self, n) * k for n in widths}})


class QuadTask:
    """``goal_stride``: the goal's frames are 1, 1 + stride, 1 + 2 stride, ... control steps ahead."""

    def __init__(self, ref: QuadReference, spec: QuadrotorSpec = QuadrotorSpec(), cfg: RewardCfg = RewardCfg(), goal_stride: int = 5):
        if goal_stride < 1:
            raise ValueError(f"goal_stride must be at least 1, got {goal_stride}")
        self.ref, self.spec, self.cfg, self.goal_stride = ref, spec, cfg, goal_stride
        dev = ref.p.device
        self.scale = torch.tensor((spec.thrust_scale, *spec.rate_scale), device=dev)
        self.u_lo = torch.tensor((0.0, -spec.rate_limit, -spec.rate_limit, -spec.rate_limit), device=dev)
        self.u_hi = torch.tensor((spec.max_thrust, spec.rate_limit, spec.rate_limit, spec.rate_limit), device=dev)
        self.max_step_reward = (cfg.w_pos + cfg.w_rot + cfg.w_vel + cfg.w_ang) * CONTROL_DT
        self.goal_offsets = 1 + goal_stride * torch.arange(64, device=dev)  # the goal's frames, in control steps ahead

    def setpoint(self, a: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """The command (thrust, body rates) for policy output ``a`` in [-1, 1] at frame ``f``: offsets about the reference's."""
        ff = torch.cat((self.ref.thrust[f].unsqueeze(-1), self.ref.w[f]), -1)
        return torch.clamp(ff + self.scale * a, self.u_lo, self.u_hi)

    def obs(self, S, a_prev: torch.Tensor) -> torch.Tensor:
        """The world's up axis and the linear velocity in the body's frame, the body rates, the previous action."""
        p, R, v, w = S
        return torch.cat((R[..., 2, :], _to_body(R, v), w, a_prev), -1)

    def goal(self, S, f: torch.Tensor, goal_frames: int) -> torch.Tensor:
        """``goal_frames`` reference frames ahead, about the state in the body's frame: the position's offset, the relative
        rotation, the reference's velocity."""
        p, R, _, _ = S
        ref = self.ref
        # the frames (..., goal_frames), each at most its flight's last; the frames' terms in one batch of operations
        g = torch.minimum(f.unsqueeze(-1) + self.goal_offsets[:goal_frames], ref.end[f].unsqueeze(-1))
        rows = R.unsqueeze(-3)  # (..., 1, 3, 3): rows x^T R = (R^T x)^T
        parts = ((ref.p[g] - p.unsqueeze(-2)).unsqueeze(-2) @ rows, mat_to_6d(rows.transpose(-1, -2) @ ref.R[g]).unsqueeze(-2),
                 ref.v[g].unsqueeze(-2) @ rows)  # fmt: skip
        return torch.cat(parts, -1).flatten(-3)

    def errors(self, S, f: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tracking errors against frame ``f``: position (m), squared rotation angle, linear (m/s) and angular velocity (rad/s)."""
        p, R, v, w = S
        return {
            "pos": torch.linalg.vector_norm(p - self.ref.p[f], dim=-1),
            "rot_sq": rot_angle_sq(R, self.ref.R[f]),
            "vel": torch.linalg.vector_norm(v - self.ref.v[f], dim=-1),
            "ang": torch.linalg.vector_norm(w - self.ref.w[f], dim=-1),
        }

    def reward(self, S, f: torch.Tensor, a: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        """The tracking reward of reaching ``S`` at frame ``f`` with action ``a`` after ``a_prev``."""
        c, e = self.cfg, self.errors(S, f)
        r = (
            c.w_pos * torch.exp(-e["pos"].pow(2) / c.pos_std**2)
            + c.w_rot * torch.exp(-e["rot_sq"] / c.rot_std**2)
            + c.w_vel * torch.exp(-e["vel"].pow(2) / c.vel_std**2)
            + c.w_ang * torch.exp(-e["ang"].pow(2) / c.ang_std**2)
            + c.w_action_rate * (a - a_prev).pow(2).mean(-1)
        )
        return r * CONTROL_DT

    def termination(self, S, f: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(S[0] - self.ref.p[f], dim=-1) > self.cfg.term_pos

    def soft_termination(self, S, f: torch.Tensor, width: float = 0.1) -> torch.Tensor:
        """The termination's probability with the threshold as a sigmoid, differentiable in the state."""
        return torch.sigmoid((torch.linalg.vector_norm(S[0] - self.ref.p[f], dim=-1) - self.cfg.term_pos) / width)

    def critic_features(self, S, f: torch.Tensor, a_prev: torch.Tensor, goal_frames: int) -> torch.Tensor:
        return torch.cat((self.obs(S, a_prev), self.goal(S, f, goal_frames)), -1)
