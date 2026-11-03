"""Policy for tracking with MuscleVAE's muscle-actuated character (docs/musclevae_design.md §4): the LAFAN1 tracking
policy's structure and sizes (TrackingPolicy: a motion encoder of the goal and a decoder, 2048-1024-512-256 each, a
32-value latent through tanh) on MuscleVAE's inputs (ndm_gym.policies.musclevae_pretrained): the observation of the state
(371 values) and the muscle state (15) into the decoder, the tracked reference frame's observation (371) through the
encoder; 284 muscle targets out. MuscleVAE's policy sees no history and one target frame, and neither does this one."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..robots.musclevae_char import NUM_MUSCLES
from ..tasks.muscle_tracking import MuscleReference, their_angular_velocity
from . import musclevae_pretrained as mp

OBS_DIM, GOAL_DIM = mp.POLICY_OBS_DIM, mp.OBS_DIM  # 386, 371


def policy_observation(S, F: torch.Tensor) -> torch.Tensor:
    """MuscleVAE's policy input (..., 386) of a state (tasks.muscle_tracking): the observation (MU:204-234) with their
    backward-difference angular velocity, then the muscle state."""
    p, R, v, w = S
    return torch.cat((mp.observation(p, R, v, their_angular_velocity(w)), F), -1)


def goal_table(ref: MuscleReference, chunk: int = 4096) -> torch.Tensor:
    """The observation (F, 371) of every reference frame: the goal of a step that tracks it, as their policy's target."""
    return torch.cat([mp.observation(ref.body_p[i : i + chunk], ref.body_R[i : i + chunk], ref.body_v[i : i + chunk],
                                     their_angular_velocity(ref.body_w[i : i + chunk])) for i in range(0, ref.body_p.shape[0], chunk)])  # fmt: skip


class MusclePolicy(nn.Module):
    """a = dec(tanh enc(goal), obs): deterministic, latent z in R^32."""

    def __init__(self, z_dim: int = 32):
        super().__init__()
        self.enc = mlp(GOAL_DIM, [2048, 1024, 512, 256], z_dim)
        self.dec = mlp(z_dim + OBS_DIM, [2048, 1024, 512, 256], NUM_MUSCLES)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm, self.goal_norm = RunningNorm(OBS_DIM), RunningNorm(GOAL_DIM)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        return run_mlp(self.dec, torch.cat((z, self.obs_norm(obs)), -1))


__all__ = ["GOAL_DIM", "OBS_DIM", "MusclePolicy", "goal_table", "policy_observation"]
