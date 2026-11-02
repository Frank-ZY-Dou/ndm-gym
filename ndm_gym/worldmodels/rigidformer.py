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
With K > 1 anchors per body (``anchors``, RigidFormer's anchor RoPE), x also holds P rotary points of every body after
the global context, which in_norm passes through instead of the body positions (D + 3BP and D + 6BP): with
``anchor_mode="mean"`` (ARoPE, the mean of the anchors' rotary angles) P = 1, the anchors' centroid, whose angles that
mean is, the angles being linear in the position; with "concat" (every anchor coordinate its own channels) P = K, the
anchors; see ``Rope3d``. The rotary positions are those of the inputs (heading frame, origin at the pelvis's ground
projection, where the registers sit), or with ``rope_center`` relative to their centroid over all bodies (and anchors).
The world models' ``step`` is unchanged: its optional input noise (normalized units) also jitters the rotary positions,
by input_noise * POS_SCALE metres (each rotary point independently).
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
    """3-D rotary position embedding (RigidFormer's anchor RoPE). Channel pairs (2m, 2m+1) of the first ``dim``
    channels of every head turn by x_c * omega_f, for the three coordinates c and ``dim / 6`` log-spaced frequencies
    omega_f in [1, max_freq] rad/m, so query-key products depend on relative positions only.

    RigidFormer's PhysFormerRoPE (official code, github.com/Frank-ZY-Dou/Dynamics-Modeling, RigidFormer/src/models/
    physformer_rope.py) gives a token of K anchor points its angles in one of two ways:
      ARoPE (paper Sec. 3.3; _forward_sa_rope, physformer_rope.py:414-460): every anchor's angles as above, mean-pooled
          over the anchors. The angles are linear in the position, so this is the one point above at the anchors'
          centroid, as that file says (lines 289-294 and 423-430); RigidFormerNet passes that centroid.
      ``points`` P > 1 (forward_multi_point with use_set_aggregation=False, physformer_rope.py:338-400): pos (N, T, P, 3),
          whose 3P coordinates, point by point (p0x, p0y, p0z, p1x, ...), each turn their own dim / 3P channels by
          dim / 6P log-spaced frequencies in [1, max_freq] rad/m. dim must be a multiple of 6P; other settings are
          refused, as there (lines 349-372), not truncated.
    Both channels of a pair turn by the same angle, as there with INTERLEAVED_PAIRING (physformer_rope.py:106-120);
    its default half-split layout gives them different angles when a coordinate has more than one frequency."""

    def __init__(self, dim: int, max_freq: float, points: int = 1):
        super().__init__()
        assert dim > 0 and dim % 6 == 0, dim
        if not isinstance(points, int) or points < 1 or dim % (6 * points):
            raise ValueError(f"Rope3d turns {dim // 2} channel pairs, which the {3 * points} coordinates of {points} points per token "
                             f"cannot share equally: it needs a multiple of {6 * points} channels")  # fmt: skip
        self.points = points
        self.register_buffer("freqs", torch.logspace(0.0, math.log10(max_freq), dim // (6 * points)), persistent=False)

    def forward(self, pos: torch.Tensor):
        """pos (N, T, 3), or (N, T, P, 3) with P = ``points`` > 1 -> cos, sin (N, 1, T, dim / 2) in fp32."""
        if self.points > 1:  # every point's coordinates their own channels
            if pos.dim() != 4 or pos.shape[-2] != self.points:
                raise ValueError(f"expected positions (N, T, {self.points}, 3), got {tuple(pos.shape)}")
            pos = pos.flatten(-2)
        ang = (pos.float().unsqueeze(-1) * self.freqs).flatten(-2)
        return ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)


def rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading 2 * cos.shape[-1] channels of x (N, H, T, hd) in fp32; the other channels pass."""
    d = 2 * cos.shape[-1]
    x0, x1 = x[..., :d].float().unflatten(-1, (-1, 2)).unbind(-1)
    r = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), -1).flatten(-2)
    return torch.cat((r.to(x.dtype), x[..., d:]), -1)


def rope_rotate_fused(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``rope_rotate`` as one elementwise expression over all channels, x * c + (x with the two channels of every pair
    swapped) * s, with c = (cos, cos) and s = (-sin, sin) on a pair and c = 1, s = 0 on the channels that pass. Without
    the stack and cat of ``rope_rotate``, torch.compile fuses the QK normalization, the rotation and the cast into one
    kernel per block instead of nine (A100, 4096 x 33 tokens, 4 heads: 0.18 instead of 0.78 ms per block). Equal to
    ``rope_rotate`` up to float rounding (the compiled kernels contract the products and sums differently).

    It also avoids a miscompilation of ``rope_rotate`` by torch 2.11's Inductor: with 33 tokens, the last of them at
    zero positions concatenated in the graph, RMS-normalized queries (keys) and the angles computed in the same graph,
    the kernel that stacks the rotated halves reads its permuted, padded input (strides [1600, 12, 48, 1] for 4 heads)
    in logical order and scrambles heads and tokens (HOIRigidFormerNet: 31 tokens and 2 registers). The G1's 32 tokens
    are not affected (checked at the study's shapes)."""
    pad = x.shape[-1] - 2 * cos.shape[-1]
    c = F.pad(cos.repeat_interleave(2, -1), (0, pad), value=1.0)
    s = F.pad(torch.stack((-sin, sin), -1).flatten(-2), (0, pad))
    x = x.float()
    return x * c + x.unflatten(-1, (-1, 2)).flip(-1).flatten(-2) * s


class Block(nn.Module):
    """Pre-norm Transformer block of RigidFormer's decoder over a token set (any tokens: bodies, particles):
    x + W_o(sigmoid(gate) * Attn(RMSNorm(x))) with QK-RMSNorm, rotary positions and an optional additive bias, then
    FiLM x * (1 + gamma(c)) + beta(c) (zero-initialized), then x + SwiGLU(RMSNorm(x)). Attention runs through
    ``F.scaled_dot_product_attention`` (flash kernel without a bias, memory-efficient kernel with one), or with
    ``attention = "token"`` through ``ndm_gym.core.token_attention`` (inference only: no backward pass).

    x (N, T, D) fp32 residual stream; cos, sin (N, 1, T, r) or None; bias (1 or N, H, T, T) or None; c (N, C) or None.
    fused_rope: ``rope_rotate_fused`` instead of ``rope_rotate`` (equal up to float rounding, faster when compiled).
    """

    def __init__(self, dim: int, heads: int, ffn_mult: float, ctx_dim: int, fused_rope: bool = False):
        super().__init__()
        assert dim % heads == 0, (dim, heads)
        self.heads, self.hd = heads, dim // heads
        self.fused_rope, self.attention = fused_rope, "sdpa"
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
        dt = self.q_norm.weight.dtype  # fp32 (bf16 only in inference copies with bf16 parameters)
        q, k, v = self.q_norm(q.to(dt)).transpose(1, 2), self.k_norm(k.to(dt)).transpose(1, 2), v.transpose(1, 2)
        if cos is not None:
            rot = rope_rotate_fused if self.fused_rope else rope_rotate
            q, k = rot(q, cos, sin), rot(k, cos, sin)
        q, k = q.to(v.dtype), k.to(v.dtype)
        if self.attention == "token":
            from ..core.token_attention import token_attention

            if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, bias)):
                raise RuntimeError('attention="token" has no backward pass; use it under torch.no_grad() (inference)')
            a = token_attention(q, k, v, None if bias is None else bias.to(v.dtype))
        elif self.attention == "sdpa":
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(v.dtype))
        else:
            raise ValueError(f"attention must be 'sdpa' or 'token', got {self.attention!r}")
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
            world model, where H steps of activations would not fit (docs/rigidformer_backbone.md, measurements);
            "frozen_readout" = while frozen, only the read-out (the blocks' activations are kept, which fused_train
            makes small: two (N, T, dim) tensors per block).
        fused_rope: the blocks' rotary embedding as one elementwise expression (``rope_rotate_fused``): the same
            function up to float rounding, faster under torch.compile. Off by default, which keeps the numerics of the
            runs trained before the flag.
        fused_train: the blocks' passes under bf16 autocast (the world-model fit, the policy's BPTT) as the fused Triton
            kernels of ``ndm_gym.kernels.rf_train`` (CUDA; bf16 products with fp32 accumulation, everything else fp32):
            the same function up to float rounding. Passes without autocast (fp32: ``--wm_bptt_fp32``, the penalty of
            ``--wm_smooth_w``) run the blocks as without the option. Off by default.
        keep_ops: with fused_train, the fused forward passes whose parameters train (the world-model fit) also keep the
            bf16 operands their backward passes would recompute (``rf_block_train(keep_ops=True)``; about 100 MB more per
            block at 2048 windows); the same function up to float rounding. Off by default.
        anchors: K anchor points per body for the rotary attention (RigidFormer's ARoPE): K > 1 places them by
            RigidFormer's farthest point sampling over the body's visual meshes (``ArticulationKinematics.anchors``) and
            moves them with the body; 1 (default) keeps one point per body, its link origin (either mode).
        anchor_mode: how the K anchors give the rotary angles (``Rope3d``): "mean" (default) mean-pools their angles
            (ARoPE), which are the angles of their centroid: the inputs carry that body-fixed point, and the blocks see
            one rotary point per body as with K = 1; "concat" gives each of their 3K coordinates its own channels. The
            rotary channels per head are 6 * floor(head width / 8), 24 at the head width 32 of RF-S and RF-M; "concat"
            needs a multiple of 6K of them: K = 4 fits 24 (one frequency, 1 rad/m, per coordinate), K = 8 needs 48
            (head width 64). Either mode changes only the rotary tables the blocks take, so it runs with fused_train.
        rope_center: the rotary positions relative to their centroid over all bodies (and anchors) of the sample, as
            RigidFormer's model.rope_center_positions, so that the zero-angle registers sit at that centroid instead
            of the pelvis's ground projection; the body tokens' query-key products do not change (rotary products
            depend on differences), so only the registers' attention does. Off by default.
    """

    def __init__(self, kin: ArticulationKinematics, root_out: int, joint_out: int, dim: int = 128, layers: int = 3,
                 heads: int = 4, ffn_mult: float = 2.5, registers: int = 2, rope: bool = True, rope_max_freq: float = 10.0,
                 rope_detach: bool = False, tree_bias: bool = True, film: bool = True, head_layers: tuple[int, ...] | None = None,
                 ext_wrench: bool = False, checkpoint: str = "frozen", fused_rope: bool = False, fused_train: bool = False, anchors: int = 1,
                 anchor_mode: str = "mean", rope_center: bool = False, keep_ops: bool = False):  # fmt: skip
        super().__init__()
        B = kin.num_bodies
        assert kin.parent[0] == -1 and all(p >= 0 for p in kin.parent[1:]), kin.parent
        assert sorted(kin.joint_idx[1:]) == list(range(NUM_J)), kin.joint_idx
        self.n_body, self.n_reg, self.root_out, self.joint_out = B, registers, root_out, joint_out
        assert checkpoint in ("never", "always", "frozen", "frozen_readout"), checkpoint
        if rope_detach and not rope:
            raise ValueError("rope_detach needs the rotary positions (rope=True)")
        if fused_train and (fused_rope or not film):
            raise ValueError("fused_train needs FiLM conditioning (film=True) and excludes fused_rope, which it would leave unused")
        if keep_ops and not fused_train:
            raise ValueError("keep_ops keeps the fused blocks' operands: it needs fused_train")
        self.ext_wrench, self.checkpoint, self.rope_detach, self.fused_train = ext_wrench, checkpoint, rope_detach, fused_train
        self.keep_ops = keep_ops
        self.tok_dim = TOKEN_DIM + (WRENCH_DIM if ext_wrench else 0)
        if not isinstance(anchors, int) or anchors < 1 or (anchors > 1 and not rope):
            raise ValueError(f"anchors must be an integer >= 1 (more than one needs the rotary positions), got {anchors!r}")
        if anchor_mode not in ("mean", "concat"):
            raise ValueError(f"anchor_mode must be 'mean' or 'concat', got {anchor_mode!r}")
        if rope_center and not rope:
            raise ValueError("rope_center needs the rotary positions (rope=True)")
        self.n_anchor, self.anchor_mode, self.rope_center = anchors, anchor_mode, rope_center
        # rotary points per body in the inputs after the global context: none with K = 1 (the link origin, a token
        # feature), the anchors' centroid with "mean", the K anchors with "concat"
        self.n_rot_pts = 0 if anchors == 1 else anchors if anchor_mode == "concat" else 1
        self.in_dim = B * self.tok_dim + GLOBAL_DIM + 3 * B * self.n_rot_pts
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
        if anchors > 1:  # each body's K anchors in its link frame (saved with the model); in_norm passes the rotary points on
            self.register_buffer("anchor_pts", kin.anchors(anchors).cpu())  # (B, K, 3)
            self.register_buffer("pos_index", torch.arange(B * self.tok_dim + GLOBAL_DIM, self.in_dim), persistent=False)
        else:
            self.register_buffer("pos_index", (torch.arange(B).unsqueeze(-1) * self.tok_dim + torch.arange(3)).flatten(),
                                 persistent=False)  # fmt: skip

        self.ctx_net = nn.Sequential(nn.Linear(GLOBAL_DIM, 64), nn.SiLU()) if film else None
        self.embed = nn.Linear(self.tok_dim + (0 if film else GLOBAL_DIM), dim)
        self.body_emb = nn.Parameter(0.5 * torch.randn(B, dim))
        self.registers = nn.Parameter(0.02 * torch.randn(1, registers, dim)) if registers else None
        self.blocks = nn.ModuleList(Block(dim, heads, ffn_mult, 64 if film else 0, fused_rope=fused_rope) for _ in range(layers))
        hd = dim // heads
        n_rot = 6 * ((3 * hd // 4) // 6)  # rotary channels per head
        if rope and anchor_mode == "concat" and anchors > 1 and n_rot % (6 * anchors):
            raise ValueError(f"anchor_mode 'concat' with {anchors} anchors needs a multiple of {6 * anchors} rotary channels per head "
                             f"(2 per anchor coordinate at least); head width {hd} gives {n_rot}")  # fmt: skip
        self.rope = Rope3d(n_rot, rope_max_freq, max(self.n_rot_pts, 1)) if rope else None
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
        if self.n_anchor > 1:  # every body's rotary points moved with it, in the heading frame as the token positions
            pts = self.anchor_pts if self.anchor_mode == "concat" else self.anchor_pts.mean(1, keepdim=True)  # (B, P, 3)
            rp = _mv(Ht.unsqueeze(-3), P.unsqueeze(-2) + _mv(Rb.unsqueeze(-3), pts) - origin.unsqueeze(-2))
            return torch.cat((tok.flatten(-2), xi_n[..., :1], xi_n[..., 1 + NUM_J :], rp.flatten(-3)), -1)
        return torch.cat((tok.flatten(-2), xi_n[..., :1], xi_n[..., 1 + NUM_J :]), -1)

    def forward(self, x):
        """x (N, in_dim + 3B), or (N, in_dim + 3BP) with P rotary points per body, from TokenInputNorm -> y (N, out_dim)."""
        pts = (self.n_body, self.n_rot_pts, 3) if self.n_rot_pts > 1 else (self.n_body, 3)
        x, pos = x[:, : self.in_dim], POS_SCALE * x[:, self.in_dim :].unflatten(-1, pts)
        if self.rope_center:  # RigidFormer's rope_center_positions (RigidFormer/src/models/decoder.py:239-255): registers at the centroid
            pos = pos - pos.mean(tuple(range(1, pos.dim() - 1)), keepdim=True)
        frozen = not self.embed.weight.requires_grad
        recompute = self.checkpoint == "always" or (self.checkpoint == "frozen" and frozen)
        if self.rope_detach and frozen:
            pos = pos.detach()
        if recompute and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._trunk, x, pos, use_reentrant=False, preserve_rng_state=False)
        if self.checkpoint == "frozen_readout" and frozen and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._readout, *self._blocks(x, pos), use_reentrant=False, preserve_rng_state=False)
        return self._trunk(x, pos)

    def _trunk(self, x, pos):
        return self._readout(*self._blocks(x, pos))

    def _blocks(self, x, pos):
        """The tokens' embedding and the blocks -> the token features of the ``head_layers`` ((N, T, dim) each)."""
        N, B = x.shape[0], self.n_body
        tok = x[:, : B * self.tok_dim].unflatten(-1, (B, self.tok_dim))
        ctx = x[:, B * self.tok_dim : B * self.tok_dim + GLOBAL_DIM] if self.n_anchor > 1 else x[:, B * self.tok_dim :]
        c = None
        if self.ctx_net is None:
            tok = torch.cat((tok, ctx.unsqueeze(1).expand(-1, B, -1)), -1)
        else:
            c = self.ctx_net(ctx)
        h = self.embed(tok).float() + self.body_emb
        if self.n_reg:
            h = torch.cat((h, self.registers.expand(N, -1, -1)), 1)
            pos = torch.cat((pos, pos.new_zeros(N, self.n_reg, *pos.shape[2:])), 1)
        cos, sin = self.rope(pos) if self.rope is not None else (None, None)
        bias = self.rel_bias[:, self.rel].unsqueeze(0) if self.rel_bias is not None else None
        hs = [h]
        if self.fused_train and torch.is_autocast_enabled("cuda"):  # the bf16 passes; fp32 passes run the blocks below
            from ..kernels.rf_train import rf_block_train

            for blk in self.blocks:
                h = rf_block_train(blk, h, cos, sin, bias, c, keep_ops=self.keep_ops)
                hs.append(h)
        else:
            for blk in self.blocks:
                h = blk(h, cos, sin, bias, c)
                hs.append(h)
        return [hs[i] for i in self.head_layers]

    def _readout(self, *hs):
        """The read-out of the head layers' token features: SwiGLU fusion of the body tokens, root and joint heads."""
        B = self.n_body
        z = torch.cat([h[:, :B] for h in hs], -1)
        u, g = self.head_w12(self.head_norm(z)).chunk(2, -1)
        z = self.head_w3(u * F.silu(g))
        y_joint = self.joint_head(z[:, 1:]).index_select(1, self.body_of_joint)  # (N, J, joint_out), G1 joint order
        return torch.cat((self.root_head(z[:, 0]), y_joint.transpose(1, 2).flatten(1)), -1)


def rigidformer_kwargs(a: dict) -> dict | None:
    """RigidFormerNet keyword arguments for training arguments ``a`` (``--wm_backbone``, ``--rf_*``), None for the MLP.
    The MLP's own flags must stay at their defaults with this backbone (they would otherwise be silently ignored)."""
    if a["wm_backbone"] == "mlp":
        for flag in ("rf_rope_detach", "rf_fused_rope", "rf_fused_train", "rf_bptt_keep_blocks", "rf_fit_keep_ops"):
            if a.get(flag, False):
                raise ValueError(f"--{flag} configures the RigidFormer backbone, not the MLP")
        if a.get("rf_anchors", 1) != 1 or a.get("rf_anchor_mode", "mean") != "mean" or a.get("rf_rope_center", False):
            raise ValueError("--rf_anchors/--rf_anchor_mode/--rf_rope_center configure the RigidFormer backbone, not the MLP")
        return None
    if a["wm_backbone"] != "rigidformer":
        raise ValueError(f"unknown world-model backbone {a['wm_backbone']!r}")
    if list(a["wm_hidden"]) != [2048, 2048, 2048] or a["wm_act"] != "silu" or a["wm_no_ln"]:
        raise ValueError("--wm_hidden/--wm_act/--wm_no_ln configure the MLP backbone; the RigidFormer backbone takes --rf_*")
    fused_train, keep_blocks = a.get("rf_fused_train", False), a.get("rf_bptt_keep_blocks", False)
    if fused_train and a.get("amp", "bf16") != "bf16":
        raise ValueError("--rf_fused_train fuses the blocks' passes under bf16 autocast (--amp bf16); with --amp none there are none")
    if a.get("rf_fit_keep_ops", False) and not fused_train:
        raise ValueError("--rf_fit_keep_ops keeps the fused blocks' operands in the world-model fit: it needs --rf_fused_train")
    if keep_blocks and (not fused_train or a.get("wm_bptt_fp32", False)):
        raise ValueError("--rf_bptt_keep_blocks keeps the fused blocks' tensors through the policy's BPTT: it needs --rf_fused_train, and "
                         "excludes --wm_bptt_fp32, whose fp32 BPTT runs the unfused blocks (their activations do not fit)")  # fmt: skip
    return dict(kin=kinematics("cpu"), dim=a["rf_dim"], layers=a["rf_layers"], heads=a["rf_heads"], registers=a["rf_registers"],
                tree_bias=not a["rf_no_tree_bias"], rope=not a["rf_no_rope"], rope_detach=a.get("rf_rope_detach", False),
                fused_rope=a.get("rf_fused_rope", False), fused_train=fused_train, checkpoint="frozen_readout" if keep_blocks else "frozen",
                anchors=a.get("rf_anchors", 1), anchor_mode=a.get("rf_anchor_mode", "mean"), rope_center=a.get("rf_rope_center", False),
                keep_ops=a.get("rf_fit_keep_ops", False))  # fmt: skip

