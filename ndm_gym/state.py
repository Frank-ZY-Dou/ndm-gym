"""State layout, sim context and control rate shared by simulators, world models, tasks and motion libraries.

State tuple ``S = (p, R, v, w, q, qd)``: pelvis position, rotation matrix, world linear/angular
velocity of the pelvis link frame, joint positions and velocities. Sim context
``xi = [friction, joint calibration offsets (NUM_J), torso center-of-mass offset (3)]``.
"""

from __future__ import annotations

import torch

from .core.rotations import mat_to_6d, mat_to_quat, quat_to_mat, yaw_mat
from .robots.g1 import NUM_J

CONTROL_DT = 0.02
OBJ_DIM = 3 + 4 + 3 + 3  # object state: position, quaternion (wxyz), world linear and angular velocity
STATE_DIM = 3 + 4 + 3 + 3 + NUM_J + NUM_J
INV_DIM = 1 + 6 + 3 + 3 + NUM_J + NUM_J
XI_DIM = 1 + NUM_J + 3


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    return (x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + NUM_J],
            x[..., 13 + NUM_J : 13 + 2 * NUM_J])  # fmt: skip


def flatten_object(So) -> torch.Tensor:
    """Object state ``So = (p, R, v, w)`` as (..., OBJ_DIM)."""
    p, R, v, w = So
    return torch.cat((p, mat_to_quat(R), v, w), -1)


def unflatten_object(x: torch.Tensor):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1 : 1 + NUM_J] / 0.01, xi[..., 1 + NUM_J : XI_DIM] / 0.05), -1)
