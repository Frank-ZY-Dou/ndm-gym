"""RigidFormer-style body-token Transformer backbone for the world models of :mod:`ndm_gym.worldmodels`.

Structure after RigidFormer (Z. Dou et al., "RigidFormer: Learning Rigid Dynamics using Transformers", arXiv 2605.09196),
re-implemented for an articulated body rather than copied: one token per rigid body instead of one per free object,
analytic forward kinematics instead of the point-cloud encoder, 3-D rotary attention on the body positions, pre-norm
blocks with RMS and QK normalization, gated attention and SwiGLU, register tokens, FiLM conditioning on the global sim
context, and per-body read-out heads in reduced coordinates (exact joint constraints, the articulated counterpart of
RigidFormer's Kabsch projection). Design and measurements: ``docs/rigidformer_backbone.md``.

Shapes (N rollouts, B = 30 bodies in kinematic-tree order, J = 29 joints in G1 order, F = token width):
    net.inputs(S, a [N,J], xi_n [N,XI_DIM], f=None) -> x [N, D], D = B*F + GLOBAL_DIM  raw features, fit by in_norm
    in_norm(x) (TokenInputNorm)                      -> [N, D + 3B]  normalized x, then body positions / POS_SCALE
    net(in_norm(x))                                  -> y [N, root_out + J*joint_out]  normalized outputs
The world models' ``step`` is unchanged: its optional input noise (normalized units) also jitters the rotary positions,
by input_noise * POS_SCALE metres.
``y`` holds the root components first, then each joint component over the J joints: (6, 29 x 1) is ``ACC_OUT`` of the
acceleration model, (12, 29 x 2) is ``WM_OUT`` of the increment model.

Token features of body b (F = TOKEN_DIM, + WRENCH_DIM with ``ext_wrench``), in the root's heading (yaw) frame with the
origin at the pelvis's ground projection:
    [0:3] position (height kept absolute)   [3:9] 6-D rotation   [9:12] linear velocity   [12:15] angular velocity
    [15:21] the body's parent joint: q, qd, setpoint u, PD error q0 + offset + alpha u - q, PD torque / effort limit
            (clipped), calibration offset (normalized xi); zeros on the pelvis
    [21:27] optional external wrench [F; tau about the link origin] (``f.wrench_h[:, b]`` of ``inputs``)
followed by the global context (friction and torso center-of-mass offset, normalized xi) that conditions every block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..core.kinematics import ArticulationKinematics
from ..core.kinematics import mat_mat as _mm
from ..core.kinematics import mat_vec as _mv
from ..core.normalizer import RunningNorm
from ..core.rotations import axis_angle_to_mat, mat_to_6d, yaw_mat
from ..robots.g1 import G1_JOINT_NAMES, NUM_J, action_scale, default_joint_pos_list, kd, kinematics, kp, motor_of

TOKEN_DIM = 3 + 6 + 3 + 3 + 6
WRENCH_DIM = 6
GLOBAL_DIM = 1 + 3
XI_OFFSET_SCALE = 0.01  # ndm_gym.state.normalize_xi divides the joint calibration offsets by this
POS_SCALE = 0.1  # m; scale of the body positions TokenInputNorm passes through for the rotary attention


def tree_hops(parent: list[int]) -> torch.Tensor:
    """Number of joints on the path between every pair of bodies of a kinematic tree: (B, B) long."""
    anc = []
    for i in range(len(parent)):
        path, j = [], i
        while j >= 0:
            path.append(j)
            j = parent[j]
        anc.append({b: n for n, b in enumerate(path)})
    B = len(parent)
    hops = torch.zeros(B, B, dtype=torch.long)
    for i in range(B):
        for j in range(B):
            hops[i, j] = min(anc[i][b] + n for b, n in anc[j].items() if b in anc[i])
    return hops


class TreeKinematics(nn.Module):
    """ArticulationKinematics.forward(..., all_bodies=True) with its constants as non-persistent buffers, so they follow
    the module's device (tests/test_rigidformer_backbone.py)."""

    def __init__(self, kin: ArticulationKinematics):
        super().__init__()
        self.parent, self.joint_idx = list(kin.parent), list(kin.joint_idx)
        for name in ("axes", "off_p", "off_R"):
            self.register_buffer(name, getattr(kin, name).detach().cpu().clone(), persistent=False)

    def forward(self, p, R, v, w, q, qd):
        """World positions (N,B,3), rotations (N,B,3,3), link-origin linear and angular velocities (N,B,3)."""
        P, Rs, V, W = [], [], [], []
        for i, par in enumerate(self.parent):
            if par < 0:
                P.append(p), Rs.append(R), V.append(v), W.append(w)
                continue
            j = self.joint_idx[i]
            pi = P[par] + _mv(Rs[par], self.off_p[i])
            R_off = _mm(Rs[par], self.off_R[i])
            Rs.append(_mm(R_off, axis_angle_to_mat(self.axes[i], q[..., j])))
            W.append(W[par] + _mv(R_off, self.axes[i]) * qd[..., j : j + 1])
            V.append(V[par] + torch.cross(W[par], pi - P[par], dim=-1))
            P.append(pi)
        return torch.stack(P, -2), torch.stack(Rs, -3), torch.stack(V, -2), torch.stack(W, -2)


class Rope3d(nn.Module):
    """3-D rotary position embedding (RigidFormer's anchor RoPE with one point per token). Channel pairs (2m, 2m+1) of
    the first ``dim`` channels of every head turn by x_c * omega_f, for the three coordinates c and ``dim / 6``
    log-spaced frequencies omega_f in [1, max_freq] rad/m, so query-key products depend on relative positions only."""

    def __init__(self, dim: int, max_freq: float):
        super().__init__()
        assert dim > 0 and dim % 6 == 0, dim
        self.register_buffer("freqs", torch.logspace(0.0, math.log10(max_freq), dim // 6), persistent=False)

    def forward(self, pos: torch.Tensor):
        """pos (N, T, 3) -> cos, sin (N, 1, T, dim / 2) in fp32."""
        ang = (pos.float().unsqueeze(-1) * self.freqs).flatten(-2)
        return ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)


def rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading 2 * cos.shape[-1] channels of x (N, H, T, hd) in fp32; the other channels pass."""
    d = 2 * cos.shape[-1]
    x0, x1 = x[..., :d].float().unflatten(-1, (-1, 2)).unbind(-1)
    r = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).flatten(-2)
    return torch.cat((r.to(x.dtype), x[..., d:]), -1)


class Block(nn.Module):
    """Pre-norm Transformer block of RigidFormer's decoder over a token set (any tokens: bodies, particles):
    x + W_o(sigmoid(gate) * Attn(RMSNorm(x))) with QK-RMSNorm, rotary positions and an optional additive bias, then
    FiLM x * (1 + gamma(c)) + beta(c) (zero-initialized), then x + SwiGLU(RMSNorm(x)). Attention runs through
    ``F.scaled_dot_product_attention`` (flash kernel without a bias, memory-efficient kernel with one).

    x (N, T, D) fp32 residual stream; cos, sin (N, 1, T, r) or None; bias (1 or N, H, T, T) or None; c (N, C) or None.
    """

    def __init__(self, dim: int, heads: int, ffn_mult: float, ctx_dim: int):
        super().__init__()
        assert dim % heads == 0, (dim, heads)
        self.heads, self.hd = heads, dim // heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.qkvg = nn.Linear(dim, 4 * dim)
        self.q_norm, self.k_norm = nn.RMSNorm(self.hd, eps=1e-6), nn.RMSNorm(self.hd, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.film = None
        if ctx_dim:
            self.film = nn.Linear(ctx_dim, 2 * dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        hidden = int(ffn_mult * dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x, cos=None, sin=None, bias=None, c=None):
        N, T, D = x.shape
        q, k, v, g = self.qkvg(self.norm1(x)).view(N, T, 4, self.heads, self.hd).unbind(2)
        q, k, v = self.q_norm(q.float()).transpose(1, 2), self.k_norm(k.float()).transpose(1, 2), v.transpose(1, 2)
        if cos is not None:
            q, k = rope_rotate(q, cos, sin), rope_rotate(k, cos, sin)
        q, k = q.to(v.dtype), k.to(v.dtype)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(v.dtype))
        x = x + self.proj(a.transpose(1, 2).reshape(N, T, D) * torch.sigmoid(g.reshape(N, T, D)))
        if self.film is not None:
            gamma, beta = self.film(c).unsqueeze(1).chunk(2, -1)
            x = x * (1 + gamma) + beta
        u, z = self.w12(self.norm2(x)).chunk(2, -1)
        return x + self.w3(u * F.silu(z))


class TokenInputNorm(RunningNorm):
    """RunningNorm of the flat token features whose forward also passes the body positions through, unnormalized up to
    the fixed POS_SCALE, for the rotary attention: x (N, D) -> (N, D + 3B). ``update`` and the state dict are
    RunningNorm's."""

    def __init__(self, dim: int, pos_index: torch.Tensor):
        super().__init__(dim)
        self.register_buffer("pos_index", pos_index, persistent=False)

    def forward(self, x):
        return torch.cat((super().forward(x), x[..., self.pos_index] / POS_SCALE), -1)


class RigidFormerNet(nn.Module):
    """Body-token Transformer: per-body features -> shared token embedding + learned body embedding (+ registers) ->
    ``layers`` blocks with rotary positions, a kinematic-tree attention bias and FiLM on the global context -> a SwiGLU
    fusion of the token features of ``head_layers`` (0 = embedding) -> root head on the pelvis token, shared joint head
    on the other tokens. Both heads start at zero, like the MLP's last layer.

    Args:
        kin: kinematics of the articulation (body 0 the floating root; body b >= 1 owns one joint).
        root_out, joint_out: outputs per root (6 accel / 12 increment) and per joint (1 / 2).
        dim, layers, heads, ffn_mult, registers: Transformer size; ``registers`` extra learned tokens at the origin.
        rope, rope_max_freq: rotary attention on the body positions (frequencies 1..max_freq rad/m).
        rope_detach: no gradient through the rotary positions while the parameters do not require gradients (the
            policy's BPTT, as ``checkpoint="frozen"``); the positions still reach the token features, so only the
            attention pattern's dependence on them is held fixed in the backward pass. Off by default.
        tree_bias: learned per-head bias on the number of joints between two bodies (registers: one extra class).
        film: global context by FiLM in every block (True) or appended to every token (False).
        ext_wrench: reserve the per-body external-wrench channel (``f.wrench_h`` of ``inputs``, then required there).
        checkpoint: activation checkpointing of the trunk (recomputed in the backward pass): "never", "always", or
            "frozen" = only while the parameters do not require gradients, i.e. in the policy's BPTT through the frozen
            world model, where H steps of activations would not fit (docs/rigidformer_backbone.md, measurements).
    """

    def __init__(self, kin: ArticulationKinematics, root_out: int, joint_out: int, dim: int = 128, layers: int = 3,
                 heads: int = 4, ffn_mult: float = 2.5, registers: int = 2, rope: bool = True, rope_max_freq: float = 10.0,
                 rope_detach: bool = False, tree_bias: bool = True, film: bool = True, head_layers: tuple[int, ...] | None = None,
                 ext_wrench: bool = False, checkpoint: str = "frozen"):  # fmt: skip
        super().__init__()
        B = kin.num_bodies
        assert kin.parent[0] == -1 and all(p >= 0 for p in kin.parent[1:]), kin.parent
        assert sorted(kin.joint_idx[1:]) == list(range(NUM_J)), kin.joint_idx
        self.n_body, self.n_reg, self.root_out, self.joint_out = B, registers, root_out, joint_out
        assert checkpoint in ("never", "always", "frozen"), checkpoint
        if rope_detach and not rope:
            raise ValueError("rope_detach needs the rotary positions (rope=True)")
        self.ext_wrench, self.checkpoint, self.rope_detach = ext_wrench, checkpoint, rope_detach
        self.tok_dim = TOKEN_DIM + (WRENCH_DIM if ext_wrench else 0)
        self.in_dim = B * self.tok_dim + GLOBAL_DIM
        self.kin = TreeKinematics(kin)
        joint_of_body = torch.tensor(kin.joint_idx[1:])
        self.register_buffer("joint_of_body", joint_of_body, persistent=False)  # (B-1,) joint of tokens 1..B-1
        self.register_buffer("body_of_joint", torch.argsort(joint_of_body), persistent=False)  # (J,) token - 1 of joints
        motors = [motor_of(n) for n in G1_JOINT_NAMES]
        consts = {
            "q0": default_joint_pos_list(),
            "alpha": [action_scale(n) for n in G1_JOINT_NAMES],
            "kp": [kp(m[2]) for m in motors],
            "kd": [kd(m[2]) for m in motors],
            "tau_max": [m[0] for m in motors],
        }
        for name, val in consts.items():
            self.register_buffer(name, torch.tensor(val, dtype=torch.float32), persistent=False)
        self.register_buffer("pos_index", (torch.arange(B).unsqueeze(-1) * self.tok_dim + torch.arange(3)).flatten(),
                             persistent=False)  # fmt: skip

        self.ctx_net = nn.Sequential(nn.Linear(GLOBAL_DIM, 64), nn.SiLU()) if film else None
        self.embed = nn.Linear(self.tok_dim + (0 if film else GLOBAL_DIM), dim)
        self.body_emb = nn.Parameter(0.5 * torch.randn(B, dim))
        self.registers = nn.Parameter(0.02 * torch.randn(1, registers, dim)) if registers else None
        self.blocks = nn.ModuleList(Block(dim, heads, ffn_mult, 64 if film else 0) for _ in range(layers))
        hd = dim // heads
        self.rope = Rope3d(6 * ((3 * hd // 4) // 6), rope_max_freq) if rope else None
        self.rel_bias = None
        if tree_bias:
            hops = tree_hops(kin.parent)
            T = B + registers
            rel = torch.full((T, T), int(hops.max()) + 1, dtype=torch.long)  # register pairs: their own class
            rel[:B, :B] = hops
            self.register_buffer("rel", rel, persistent=False)
            self.rel_bias = nn.Parameter(torch.zeros(heads, int(hops.max()) + 2))
        self.head_layers = tuple(head_layers) if head_layers is not None else (0, layers)
        assert all(0 <= i <= layers for i in self.head_layers), self.head_layers
        hidden = int(ffn_mult * dim)
        self.head_norm = nn.RMSNorm(dim * len(self.head_layers), eps=1e-6)
        self.head_w12 = nn.Linear(dim * len(self.head_layers), 2 * hidden)
        self.head_w3 = nn.Linear(hidden, dim)
        self.root_head, self.joint_head = nn.Linear(dim, root_out), nn.Linear(dim, joint_out)
        for m in (self.root_head, self.joint_head):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    @property
    def out_dim(self) -> int:
        return self.root_out + NUM_J * self.joint_out

    def make_input_norm(self) -> TokenInputNorm:
        return TokenInputNorm(self.in_dim, self.pos_index.clone())

    def inputs(self, S, a, xi_n, f=None):
        """Flat raw features (N, in_dim) of states S, normalized setpoints a (N, J) and normalized sim context xi_n.

        f: the external forces, whose ``wrench_h`` (N, B, 6) holds every body's force and torque about its link origin
        in the root's heading frame; required with ``ext_wrench``, must be None without.
        """
        p, R, v, w, q, qd = S
        P, Rb, V, W = self.kin(p, R, v, w, q, qd)
        Ht = yaw_mat(R).transpose(-1, -2).unsqueeze(-3)
        origin = torch.cat((p[..., :2], torch.zeros_like(p[..., 2:])), -1).unsqueeze(-2)
        pos, lin, ang = _mv(Ht, P - origin), _mv(Ht, V), _mv(Ht, W)
        off_n = xi_n[..., 1 : 1 + NUM_J]
        err = self.q0 + XI_OFFSET_SCALE * off_n + self.alpha * a - q  # the simulator's target is q0 + offset + alpha u
        tau = torch.minimum(torch.maximum(self.kp * err - self.kd * qd, -self.tau_max), self.tau_max) / self.tau_max
        joint = torch.stack((q, qd, a, err, tau, off_n), -1).index_select(-2, self.joint_of_body)
        joint = torch.cat((torch.zeros_like(joint[..., :1, :]), joint), -2)
        parts = [pos, mat_to_6d(_mm(Ht, Rb)), lin, ang, joint]
        if self.ext_wrench:
            if f is None:
                raise ValueError("ext_wrench=True needs the external forces f (f.wrench_h of shape (N, B, 6))")
            if f.wrench_h.shape != pos.shape[:-1] + (WRENCH_DIM,):
                shape = tuple(pos.shape[:-1]) + (WRENCH_DIM,)
                raise ValueError(f"wrench_h has shape {tuple(f.wrench_h.shape)}, expected {shape}")
            parts.append(f.wrench_h)
        elif f is not None:
            raise ValueError("ext_wrench=False: f must be None")
        tok = torch.cat(parts, -1)
        return torch.cat((tok.flatten(-2), xi_n[..., :1], xi_n[..., 1 + NUM_J :]), -1)

    def forward(self, x):
        """x (N, in_dim + 3B) from TokenInputNorm -> y (N, out_dim)."""
        x, pos = x[:, : self.in_dim], POS_SCALE * x[:, self.in_dim :].unflatten(-1, (self.n_body, 3))
        frozen = not self.embed.weight.requires_grad
        recompute = self.checkpoint == "always" or (self.checkpoint == "frozen" and frozen)
        if self.rope_detach and frozen:
            pos = pos.detach()
        if recompute and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._trunk, x, pos, use_reentrant=False, preserve_rng_state=False)
        return self._trunk(x, pos)

    def _trunk(self, x, pos):
        N, B = x.shape[0], self.n_body
        tok = x[:, : B * self.tok_dim].unflatten(-1, (B, self.tok_dim))
        ctx = x[:, B * self.tok_dim :]
        c = None
        if self.ctx_net is None:
            tok = torch.cat((tok, ctx.unsqueeze(1).expand(-1, B, -1)), -1)
        else:
            c = self.ctx_net(ctx)
        h = self.embed(tok).float() + self.body_emb
        if self.n_reg:
            h = torch.cat((h, self.registers.expand(N, -1, -1)), 1)
            pos = torch.cat((pos, pos.new_zeros(N, self.n_reg, 3)), 1)
        cos, sin = self.rope(pos) if self.rope is not None else (None, None)
        bias = self.rel_bias[:, self.rel].unsqueeze(0) if self.rel_bias is not None else None
        hs = [h]
        for blk in self.blocks:
            h = blk(h, cos, sin, bias, c)
            hs.append(h)
        z = torch.cat([hs[i][:, :B] for i in self.head_layers], -1)
        u, g = self.head_w12(self.head_norm(z)).chunk(2, -1)
        z = self.head_w3(u * F.silu(g))
        y_joint = self.joint_head(z[:, 1:]).index_select(1, self.body_of_joint)  # (N, J, joint_out), G1 joint order
        return torch.cat((self.root_head(z[:, 0]), y_joint.transpose(1, 2).flatten(1)), -1)


def rigidformer_kwargs(a: dict) -> dict | None:
    """RigidFormerNet keyword arguments for training arguments ``a`` (``--wm_backbone``, ``--rf_*``), None for the MLP.
    The MLP's own flags must stay at their defaults with this backbone (they would otherwise be silently ignored)."""
    if a["wm_backbone"] == "mlp":
        if a.get("rf_rope_detach", False):
            raise ValueError("--rf_rope_detach configures the RigidFormer backbone, not the MLP")
        return None
    if a["wm_backbone"] != "rigidformer":
        raise ValueError(f"unknown world-model backbone {a['wm_backbone']!r}")
    if list(a["wm_hidden"]) != [2048, 2048, 2048] or a["wm_act"] != "silu" or a["wm_no_ln"]:
        raise ValueError("--wm_hidden/--wm_act/--wm_no_ln configure the MLP backbone; the RigidFormer backbone takes --rf_*")
    return dict(kin=kinematics("cpu"), dim=a["rf_dim"], layers=a["rf_layers"], heads=a["rf_heads"], registers=a["rf_registers"],
                tree_bias=not a["rf_no_tree_bias"], rope=not a["rf_no_rope"], rope_detach=a.get("rf_rope_detach", False))  # fmt: skip

