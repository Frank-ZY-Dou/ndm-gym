"""Policy for HDMI's human-object interaction tasks: the LAFAN1 tracking policy's structure and sizes (TrackingPolicy)
on HDMI's observation groups (ndm_gym.tasks.hdmi): the command group (future reference) through the goal encoder, the
proprioceptive and object groups into the decoder. Its output is HDMI's residual over the reference: the action is
``(q_ref - q0) / scale`` of the current frame plus this output (the teacher's residual, §3.1). With ``priv_dim``
(train_hoi ``--priv_obs``), the decoder also reads HDMI's privileged group, encoded as HDMI's PPO teacher encodes it
(an MLP of the privileged and object groups to 256 features; ndm_gym.algos.ppo_hoi)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core.nets import mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..tasks.hdmi import NUM_ACTIONS


PRIV_FEATURES = 256  # HDMI's teacher encoder's output (ndm_gym.algos.ppo_hoi.PPOConfig.latent_dim)


class HOIPolicy(nn.Module):
    """a = dec(tanh enc(command), policy, object[, penc(priv, object)]): deterministic, latent z in R^32."""

    def __init__(self, command_dim: int, policy_dim: int, object_dim: int, z_dim: int = 32, priv_dim: int = 0):
        super().__init__()
        self.priv_dim, self.policy_dim = priv_dim, policy_dim
        self.enc = mlp(command_dim, [2048, 1024, 512, 256], z_dim)
        self.dec = mlp(z_dim + policy_dim + object_dim + (PRIV_FEATURES if priv_dim else 0), [2048, 1024, 512, 256], NUM_ACTIONS)
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.command_norm, self.obs_norm = RunningNorm(command_dim), RunningNorm(policy_dim + object_dim)
        if priv_dim:  # the teacher's encoder: Linear, LayerNorm, Mish, Linear (ppo_hoi: make_mlp(q + o, (L,)), Linear(L, L))
            self.priv_enc = mlp(priv_dim + object_dim, [PRIV_FEATURES], PRIV_FEATURES, act=nn.Mish, layer_norm=True)
            self.priv_norm = RunningNorm(priv_dim)

    def forward(self, obs: dict) -> torch.Tensor:
        z = torch.tanh(run_mlp(self.enc, self.command_norm(obs["command"])))
        o = self.obs_norm(torch.cat((obs["policy"], obs["object"]), -1))
        if not self.priv_dim:
            return run_mlp(self.dec, torch.cat((z, o), -1))
        q = run_mlp(self.priv_enc, torch.cat((self.priv_norm(obs["priv"]), o[..., self.policy_dim :]), -1))
        return run_mlp(self.dec, torch.cat((z, o, q), -1))

    @torch.no_grad()
    def update_norms(self, obs: dict):
        self.command_norm.update(obs["command"])
        self.obs_norm.update(torch.cat((obs["policy"], obs["object"]), -1))
        if self.priv_dim:
            self.priv_norm.update(obs["priv"])
