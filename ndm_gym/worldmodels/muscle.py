"""World models of MuscleVAE's muscle-actuated character (docs/musclevae_design.md §4), in maximal coordinates: the next
state of the 23 bodies and the muscle state (ndm_gym.tasks.muscle_tracking) from the state, the muscle state and the
action (the 284 muscle targets). The character's 14 ball joints have no single-angle coordinates, so the model works on
the bodies, as SuperTrack's and MuscleVAE's do.

Inputs, in the root's heading frame H (the yaw of the pelvis's rotation): the root's height, and per body its position
relative to the root, its rotation (6-D), velocity and rotation rate; the muscle state (15); the action (284): 645 values.

Outputs, per body in the same frame (``kind``):
* "accel" (SuperTrack, Fussell et al. 2021 §3.4, Eq. 8-12; MuscleVAE's world model predicts the same, each body's change
  of velocity and angular velocity): the linear and angular accelerations over the control step, integrated by
  semi-implicit Euler, v' = v + dt H a_v, w' = w + dt H a_w, p' = p + dt v', R' = exp(dt w') R, which is exact for the
  state's backward-difference velocities; and the muscle state's rate of change, F' = F + dt dF/dt.
* "increment" (SuperMimic §3.2): the increments of position, rotation (R' = exp(H dth) R), velocity and rotation rate,
  and of the muscle state.

The network is the G1 world models' MLP (core.nets, zero-initialized output) between running normalizers (core.normalizer),
or with ``rigidformer`` the token Transformer ``MuscleRigidFormerNet``, one token per body and one per muscle, or one
per body that also carries the muscles' analytic wrench (docs/musclevae_design.md §5); the outputs' normalizer floor
is the G1 models' 1e-4 per step (per second for the accelerations and rates). All 3 x 3 products are written
elementwise (core.kinematics.mat_vec, mat_mat), so no TF32 matmul rounds them.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_
import torch.utils.checkpoint

from ..core.kinematics import mat_mat, mat_vec
from ..core.nets import ACTIVATIONS, mlp, run_mlp
from ..core.normalizer import RunningNorm
from ..core.rotations import mat_to_6d, so3_exp, so3_log, yaw_mat
from ..robots import muscles as mu
from ..robots.musclevae_char import BODY_PARENTS, NUM_MUSCLES
from ..tasks.muscle_tracking import DT, FATIGUE_DIM, NB

KINDS = ("accel", "increment")
IN_DIM = 1 + NB * (3 + 6 + 3 + 3) + FATIGUE_DIM + NUM_MUSCLES  # 645
OUT_DIM = {"accel": NB * 6 + FATIGUE_DIM, "increment": NB * 12 + FATIGUE_DIM}  # 153, 291


def heading(S) -> torch.Tensor:
    """The root's heading rotation H (..., 1, 3, 3), broadcast over the bodies."""
    return yaw_mat(S[1][..., 0, :, :]).unsqueeze(-3)


def _in(H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """H^T x."""
    return mat_vec(H.transpose(-1, -2), x)


def increments(S, F, S1, F1) -> torch.Tensor:
    """From (S, F) to (S1, F1), per body in S's heading frame: position, rotation (log of R1 R^T), velocity and rotation
    rate increments, then the muscle state's: (..., 291)."""
    p, R, v, w = S
    p1, R1, v1, w1 = S1
    H = heading(S)
    body = torch.cat((_in(H, p1 - p), _in(H, so3_log(mat_mat(R1, R.transpose(-1, -2)))), _in(H, v1 - v), _in(H, w1 - w)), -1)
    return torch.cat((body.flatten(-2), F1 - F), -1)


TOKENS = ("all", "body")  # MuscleRigidFormerNet's token sets
BODY_TOK = 3 + 6 + 3 + 3 + 6 + 3  # MuscleRigidFormerNet's body token features
MUSCLE_TOK = 13 + 3 + 3 + 3  # and muscle token features
MID = 13  # the muscle token's path midpoint: its rotary position
MUSCLE_SCALARS = 13  # with tokens="body": the muscle features before the path's geometry, the muscle encoder's input
WRENCH = 6  # and the body token's muscle wrench (force, torque about the body's origin) after BODY_TOK


def body_hops() -> np.ndarray:
    """Number of joints between every pair of the character's bodies: (NB, NB)."""
    from .rigidformer import tree_hops

    return tree_hops(list(BODY_PARENTS)).numpy()


def relation_classes(muscle_bodies: list[set[int]], registers: int) -> tuple[torch.Tensor, int]:
    """The attention bias's class of every token pair (bodies, then muscles, then registers): (T, T) long, and the number
    of classes. Two bodies: the joints between them, h; a muscle and a body: h + K from the nearest body the muscle's
    anchors move with; two muscles: h + 2 K between their nearest bodies; any pair with a register: 3 K, with K = the
    largest h + 1."""
    hops = body_hops()
    K = int(hops.max()) + 1
    member = np.zeros((len(muscle_bodies), NB), dtype=bool)
    for m, bodies in enumerate(muscle_bodies):
        member[m, sorted(bodies)] = True
    big = np.iinfo(np.int64).max
    mb = np.where(member[:, :, None], hops[None], big).min(1)  # (M, NB): from the muscle's nearest body
    mm = np.where(member[None, :, :], mb[:, None, :], big).min(-1)  # (M, M): between the nearest bodies
    M, T = len(muscle_bodies), NB + len(muscle_bodies) + registers
    rel = np.full((T, T), 3 * K, dtype=np.int64)
    rel[:NB, :NB] = hops
    rel[NB : NB + M, :NB], rel[:NB, NB : NB + M] = mb + K, mb.T + K
    rel[NB : NB + M, NB : NB + M] = mm + 2 * K
    return torch.from_numpy(rel), 3 * K + 1


@torch.compiler.disable
def _geometry_tables(mm: mu.MuscleModel) -> list[torch.Tensor]:
    """kernels.muscle_geometry's tables of a fp32 muscle model, built eagerly (numpy loops)."""
    from ..kernels.muscle_geometry import MuscleTables

    return MuscleTables(mm).tensors


@torch.compiler.disable
def _muscle_model(table, params, device, dtype) -> mu.MuscleModel:
    """robots.muscles.MuscleModel, built eagerly: traced by torch.compile, its constructor's Python loops (the initial
    fatigue curve's 4800 steps) would become a graph that Inductor cannot lower (RecursionError)."""
    return mu.MuscleModel(table, params, device, dtype)


class MuscleRigidFormerNet(nn.Module):
    """RigidFormer for MuscleVAE's character (docs/musclevae_design.md §5), with one of two token sets (``tokens``):

    "all": one token per body (23, the tree's order), one per muscle (284) and register tokens; 3-D rotary attention on
    the tokens' positions (a body's center of mass, a muscle's path midpoint; registers at the origin), a learned
    per-head bias by relation (``relation_classes``), FiLM on the muscle state (the 15 part fatigue values), and
    read-out heads on the body tokens (each body's accelerations or increments of ``MuscleWorldModel``) and on each body
    part's muscle tokens, pooled with the part fatigue's f0 weights (the part's three fatigue rates or increments).

    "body": the body tokens and the registers only. Each body token also carries the force and the torque about its
    origin that the muscles' forces (the clipped PD force below) exert on it through the muscle model's anchors
    (``robots.muscles.MuscleModel.body_wrench``), and the mean, over the muscles attached to it (``muscle_bodies``), of
    the muscle encodings SiLU(Linear(the muscle's MUSCLE_SCALARS features) + a learned per-muscle embedding), each
    ``muscle_dim`` wide (default 32), mapped linearly to the token width. The attention bias classes are the joints
    between two bodies, and one more for any pair with a register. The fatigue read-out has one token per body part:
    the f0-weighted mean of its muscles' encodings, mapped linearly, plus the mean of the body tokens over the bodies
    its muscles attach to (f0-weighted, each muscle's weight shared equally by its bodies); the part tokens go through
    the body tokens' read-out to the fatigue head.

    The blocks are RigidFormerNet's (``rigidformer.Block``); the heads start at zero, like the MLP's last layer. With
    ``fused_train`` (body tokens, 23 bodies and registers a power of two: registers=9), the blocks' passes under bf16
    autocast run as ``kernels.rf_train``'s fused kernels, as RigidFormerNet's ``fused_train`` (the same function up to float
    rounding; passes without autocast run the blocks). With ``fused_features`` (body tokens), the muscles' lengths, length
    rates and wrench on CUDA in fp32 come from ``kernels.muscle_geometry``'s fused kernels (the same functions up to float
    rounding; elsewhere from ``robots.muscles``).

    Token features, in the root's heading frame H with the origin at the pelvis's ground projection (heights absolute):
      body (BODY_TOK): position, 6-D rotation, velocity, rotation rate, and relative to its parent its rotation (6-D) and
        rotation rate in the parent's frame (zeros on the pelvis); with "body", then the muscle wrench (WRENCH);
      muscle (MUSCLE_TOK), from the bodies by ``robots.muscles`` (``muscle_model``'s parameters): length over rest
        length, normalized length, length rate over rest length (1/s), action, PD error over rest length, and what the
        first substep's muscle step gives with the muscle's part fatigue standing in for its own (the world model's state
        has no per-muscle fatigue): the clipped PD force over f0max, the activation, the force-length, force-velocity and
        passive factors; then its part's fatigue (MA, MR, MF), its path midpoint and the unit directions of its first and
        last segments. With "body", the first MUSCLE_SCALARS of them (up to the part's fatigue).
    ``inputs`` returns them flat with the muscle state; ``forward`` takes them through ``TokenInputNorm`` (positions passed
    on for the rotary attention) and returns the flat output in ``MuscleWorldModel``'s layout.

    ``checkpoint``: activation checkpointing of the world model's whole step, the features included, recomputed in the
    backward pass (``MuscleWorldModel.step``): "never", "always", or "frozen" = only while the parameters do not
    require gradients (the policy's BPTT). Default (None): "always" for the muscle tokens (with 309 tokens the
    activations of a window fit or of the policy's BPTT do not fit otherwise), "frozen" for the body tokens (a window
    fit's activations fit; the BPTT's 32 steps at 4096 rollouts do not).
    """

    def __init__(self, kind: str, muscle_model: str = "released", dim: int = 128, layers: int = 3, heads: int = 4, ffn_mult: float = 2.5,
                 registers: int = 2, rope: bool = True, rope_max_freq: float = 10.0, tree_bias: bool = True, checkpoint: str | None = None,
                 tokens: str = "all", muscle_dim: int | None = None, fused_train: bool = False, fused_features: bool = False):  # fmt: skip
        super().__init__()
        from ..robots.musclevae_char import load_muscle_table
        from .rigidformer import Block, Rope3d, TokenInputNorm

        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if tokens not in TOKENS:
            raise ValueError(f"tokens must be one of {TOKENS}, got {tokens!r}")
        if checkpoint is None:
            checkpoint = "always" if tokens == "all" else "frozen"
        if checkpoint not in ("never", "always", "frozen"):
            raise ValueError(f"checkpoint must be 'never', 'always' or 'frozen', got {checkpoint!r}")
        if fused_features and tokens != "body":
            raise ValueError("fused_features computes the body tokens' muscle features (tokens='body')")
        if fused_train and (tokens != "body" or (NB + registers) & (NB + registers - 1) or NB + registers < 16):
            raise ValueError(f"fused_train runs the body tokens' blocks (tokens='body') on a power of two of tokens: {NB} bodies and "
                             f"{registers} registers make {NB + registers}; registers=9 makes 32")  # fmt: skip
        if tokens == "all" and muscle_dim is not None:
            raise ValueError("muscle_dim is the width of the muscle encodings of tokens='body'")
        self.table, self.params = load_muscle_table(), mu.muscle_params(muscle_model)
        self.kind, self.n_reg, self.checkpoint, self.tokens, self.fused_train = kind, registers, checkpoint, tokens, fused_train
        self.fused_features = fused_features
        self._models = {}  # robots.muscles.MuscleModel per device (``muscles``)
        self._tables = {}  # kernels.muscle_geometry.MuscleTables' tensors per device (``geometry_tables``)
        M = self.table.num_muscles
        self.n_muscle = M
        self.tok_dim = BODY_TOK if tokens == "all" else BODY_TOK + WRENCH
        if tokens == "all":
            self.in_dim = NB * BODY_TOK + M * MUSCLE_TOK + FATIGUE_DIM
            pos = [b * BODY_TOK + torch.arange(3) for b in range(NB)] + [NB * BODY_TOK + m * MUSCLE_TOK + MID + torch.arange(3) for m in range(M)]
        else:
            self.in_dim = NB * self.tok_dim + M * MUSCLE_SCALARS + FATIGUE_DIM
            pos = [b * self.tok_dim + torch.arange(3) for b in range(NB)]
        self.register_buffer("pos_index", torch.cat(pos), persistent=False)
        self.register_buffer("parents", torch.tensor([max(b, 0) for b in BODY_PARENTS]), persistent=False)
        self.register_buffer("part", torch.as_tensor(np.asarray(self.table.part, dtype=np.int64)), persistent=False)
        f0, part = np.asarray(self.table.f0, dtype=np.float64), np.asarray(self.table.part, dtype=np.int64)
        part_weight = np.eye(mu.NUM_PARTS)[part] * (f0 / np.bincount(part, weights=f0, minlength=mu.NUM_PARTS)[part])[:, None]
        self.register_buffer("part_weight", torch.as_tensor(part_weight.T, dtype=torch.float32), persistent=False)  # (parts, M)
        self._norm_cls = TokenInputNorm
        self.ctx_net = nn.Sequential(nn.Linear(FATIGUE_DIM, 64), nn.SiLU())
        if tokens == "all":
            self.embed_body, self.embed_muscle = nn.Linear(BODY_TOK, dim), nn.Linear(MUSCLE_TOK, dim)
            self.tok_emb = nn.Parameter(0.5 * torch.randn(NB + M, dim))  # learned per body and per muscle (its static attributes)
        else:
            muscle_dim = 32 if muscle_dim is None else muscle_dim
            self.embed_body = nn.Linear(self.tok_dim, dim)
            self.muscle_in = nn.Linear(MUSCLE_SCALARS, muscle_dim)
            self.muscle_emb = nn.Parameter(0.5 * torch.randn(M, muscle_dim))  # learned per muscle (its static attributes)
            self.muscle_out, self.part_in = nn.Linear(muscle_dim, dim), nn.Linear(muscle_dim, 2 * dim)
            self.tok_emb = nn.Parameter(0.5 * torch.randn(NB, dim))
            attach = np.zeros((M, NB))
            for m, bodies in enumerate(self.muscles("cpu").muscle_bodies()):
                attach[m, sorted(bodies)] = 1.0
            # (NB, M): the mean over a body's muscles; (parts, NB): each part's f0 weights, a muscle's shared by its bodies;
            # (NB + parts, M): both poolings of the muscle encodings, one product
            self.register_buffer("attach", torch.as_tensor(attach.T / attach.sum(0)[:, None], dtype=torch.float32), persistent=False)
            part_body = part_weight.T @ (attach / attach.sum(1, keepdims=True))
            self.register_buffer("part_body", torch.as_tensor(part_body, dtype=torch.float32), persistent=False)
            self.register_buffer("pool", torch.cat((self.attach, self.part_weight)), persistent=False)
        self.registers = nn.Parameter(0.02 * torch.randn(1, registers, dim)) if registers else None
        self.blocks = nn.ModuleList(Block(dim, heads, ffn_mult, 64, fused_rope=not fused_train) for _ in range(layers))
        self.chunk = 64 if int(ffn_mult * dim) % 64 == 0 else 32  # the fused kernels' SwiGLU columns per step
        self.rope = Rope3d(6 * ((3 * (dim // heads) // 4) // 6), rope_max_freq) if rope else None
        self.rel_bias = None
        if tree_bias:
            if tokens == "all":
                rel, n_cls = relation_classes(self.muscles("cpu").muscle_bodies(), registers)
            else:  # the joints between two bodies; any pair with a register one more class
                hops = torch.as_tensor(body_hops())
                n_cls = int(hops.max()) + 2
                rel = torch.full((NB + registers, NB + registers), n_cls - 1, dtype=torch.long)
                rel[:NB, :NB] = hops
            self.register_buffer("rel", rel, persistent=False)
            self.rel_bias = nn.Parameter(torch.zeros(heads, n_cls))
        hidden = int(ffn_mult * dim)
        self.head_norm = nn.RMSNorm(2 * dim, eps=1e-6)
        self.head_w12, self.head_w3 = nn.Linear(2 * dim, 2 * hidden), nn.Linear(hidden, dim)
        self.body_head = nn.Linear(dim, 6 if kind == "accel" else 12)
        self.fatigue_head = nn.Linear(dim, 3)
        for h in (self.body_head, self.fatigue_head):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def muscles(self, device, dtype=torch.float32) -> mu.MuscleModel:
        """The muscle model on ``device`` in ``dtype`` (built once for each, outside any compiled graph)."""
        key = (str(torch.device(device)), dtype)
        if key not in self._models:
            self._models[key] = _muscle_model(self.table, self.params, device, dtype)
        return self._models[key]

    def geometry_tables(self, device) -> list[torch.Tensor]:
        """The fused geometry kernels' tables on ``device`` (built once, outside any compiled graph)."""
        key = str(torch.device(device))
        if key not in self._tables:
            self._tables[key] = _geometry_tables(self.muscles(device))
        return self._tables[key]

    def make_input_norm(self):
        return self._norm_cls(self.in_dim, self.pos_index.clone())

    def hidden_matrices(self) -> list[nn.Parameter]:
        """Muon's parameters with ``--wm_opt muon`` (``core.muon.MuonAdam``), as the G1 RigidFormer's: every block's weight
        matrices as stored (``rigidformer.block_matrices``). Adam keeps the rest: the token embeddings and the context
        network, the per-token and register embeddings, with "body" the muscle encoder and its maps to the token and
        read-out widths, the relation bias, the norms, the biases, FiLM, and the read-out's SwiGLU fusion and heads."""
        from .rigidformer import block_matrices

        return block_matrices(self.blocks)

    def inputs(self, S, F, a) -> torch.Tensor:
        p, R, v, w = S
        mm = self.muscles(p.device, p.dtype)
        H = heading(S)
        Ht = H.transpose(-1, -2)
        origin = torch.cat((p[..., :1, :2], torch.zeros_like(p[..., :1, 2:])), -1)
        Rp, wp = R.index_select(-3, self.parents), w.index_select(-2, self.parents)
        Rpt = Rp.transpose(-1, -2)
        not_root = torch.ones_like(p[..., :1]).index_fill(-2, torch.zeros(1, dtype=torch.long, device=p.device), 0.0)
        rel = torch.cat((mat_to_6d(mat_mat(Rpt, R)), mat_vec(Rpt, w - wp)), -1) * not_root
        body = torch.cat((_in(H, p - origin), mat_to_6d(mat_mat(Ht, R)), _in(H, v), _in(H, w), rel), -1)
        fused = self.fused_features and p.is_cuda and p.dtype == torch.float32
        if fused:  # the same lengths, rates and wrench as robots.muscles', up to float rounding
            from ..kernels import muscle_geometry as mg

            tables, frames = self.geometry_tables(p.device), mg.frames(p, R, v, w)
            length, dl = (x.reshape(p.shape[:-2] + (-1,)) for x in mg.muscle_lengths(frames, tables))
        else:
            anchors = mm.anchor_positions(p, R)
            length = mm.lengths(anchors)
        l_norm = mm.normalized_length(length)
        if not fused:
            dl = mm.length_rates(anchors, mm.anchor_velocities(p, R, v, w))
        rate, rate_norm = mm.rates(length, length - dl / self.params.sim_hz)
        fl, fv, fpe = mu.hill_curves(l_norm, rate_norm, self.params)
        P = mu.NUM_PARTS
        MA, MR, MF = (F.index_select(-1, k * P + self.part) for k in range(3))
        lo, hi = mu.activation_bounds(MA, MR, self.params)
        target = mm.target_length(a)
        force = mu.pd_force(length, target, rate, mm.kp, mm.kd, mm.f0max, fl, fv, fpe, lo, hi)[0]
        fv_inv = torch.ones_like(fv) if self.params.inversion_rate == "zero" else fv
        # robots.muscles.invert_hill, with its denominator kept off zero: far outside the physical range of lengths the
        # force-length factor underflows to 0 and 0 / 0 would make the feature NaN (any positive force over 0 gives hi)
        a_unclipped = (force - fpe * mm.f0max).clamp(min=0.0) / (fl * mm.f0max * fv_inv).clamp(min=1e-12)
        activation = torch.where(a_unclipped > hi, hi, a_unclipped)
        activation = torch.where(activation < lo, lo, activation)
        rest = mm.rest_length
        scalars = (length / rest, l_norm, dl / rest, a, (length - target) / rest, force / mm.f0max, activation, fl, fv, fpe, MA, MR, MF)
        if self.tokens == "body":
            if fused:
                wrench = mg.muscle_wrench(frames, force.reshape(-1, force.shape[-1]).contiguous(), tables).reshape(p.shape[:-1] + (6,))
                force_b, torque_b = wrench[..., :3], wrench[..., 3:]
            else:
                force_b, torque_b = mm.body_wrench(anchors, mm.anchor_forces(anchors, force), p)
            body = torch.cat((body, _in(H, force_b), _in(H, torque_b)), -1)
            return torch.cat((body.flatten(-2), torch.stack(scalars, -1).flatten(-2), F), -1)
        mid, first, last = mm.path_points(anchors)
        muscle = torch.cat((torch.stack(scalars, -1), _in(H, mid - origin), _in(H, first), _in(H, last)), -1)
        return torch.cat((body.flatten(-2), muscle.flatten(-2), F), -1)

    def recompute(self) -> bool:
        """Whether a step with gradients is recomputed in the backward pass (``checkpoint``)."""
        frozen = not self.embed_body.weight.requires_grad
        return torch.is_grad_enabled() and (self.checkpoint == "always" or (self.checkpoint == "frozen" and frozen))

    def forward(self, x):
        from .rigidformer import POS_SCALE

        T = NB + self.n_muscle if self.tokens == "all" else NB
        feats, pos = x[:, : self.in_dim], POS_SCALE * x[:, self.in_dim :].unflatten(-1, (T, 3))
        N, M = feats.shape[0], self.n_muscle
        i = NB * self.tok_dim
        body = self.embed_body(feats[:, :i].unflatten(-1, (NB, self.tok_dim)))
        if self.tokens == "all":
            muscle = self.embed_muscle(feats[:, i : i + M * MUSCLE_TOK].unflatten(-1, (M, MUSCLE_TOK)))
            c = self.ctx_net(feats[:, i + M * MUSCLE_TOK :])
            h = torch.cat((body, muscle), 1).float() + self.tok_emb
        else:
            # the muscle encodings muscle-major (M, N, width), so that both poolings are one product over the muscles
            xm = feats[:, i : i + M * MUSCLE_SCALARS].unflatten(-1, (M, MUSCLE_SCALARS)).transpose(0, 1)
            e = F_.silu(self.muscle_in(xm) + self.muscle_emb.unsqueeze(1))
            pooled = (self.pool.to(e.dtype) @ e.flatten(1)).unflatten(-1, (N, -1)).transpose(0, 1)  # (N, NB + parts, width)
            c = self.ctx_net(feats[:, i + M * MUSCLE_SCALARS :])
            h = (body + self.muscle_out(pooled[:, :NB])).float() + self.tok_emb
        if self.n_reg:
            h = torch.cat((h, self.registers.expand(N, -1, -1)), 1)
            pos = torch.cat((pos, pos.new_zeros(N, self.n_reg, 3)), 1)
        cos, sin = self.rope(pos) if self.rope is not None else (None, None)
        bias = self.rel_bias[:, self.rel].unsqueeze(0) if self.rel_bias is not None else None
        h0 = h
        fused = self.fused_train and torch.is_autocast_enabled("cuda")  # the bf16 passes; fp32 passes run the blocks
        if fused:
            from ..kernels.rf_train import rf_block_train
        for blk in self.blocks:
            h = rf_block_train(blk, h, cos, sin, bias, c, chunk=self.chunk) if fused else blk(h, cos, sin, bias, c)
        if self.tokens == "body":
            hb = torch.cat((h0[:, :NB], h[:, :NB]), -1)
            # (N, parts, 2 dim); a product with 23 terms per output, as one reduction rather than a batched matrix product
            hp = (self.part_body.to(hb.dtype)[None, :, :, None] * hb[:, None]).sum(2) + self.part_in(pooled[:, NB:])
            u, g = self.head_w12(self.head_norm(torch.cat((hb, hp.to(hb.dtype)), 1))).chunk(2, -1)
            z = self.head_w3(u * F_.silu(g))
            return torch.cat((self.body_head(z[:, :NB]).flatten(1), self.fatigue_head(z[:, NB:]).transpose(1, 2).flatten(1)), -1)
        u, g = self.head_w12(self.head_norm(torch.cat((h0[:, :T], h[:, :T]), -1))).chunk(2, -1)
        z = self.head_w3(u * F_.silu(g))
        parts = torch.einsum("pm,nmd->npd", self.part_weight.to(z.dtype), z[:, NB:])  # (N, parts, dim)
        fatigue = self.fatigue_head(parts).transpose(1, 2).flatten(1)  # (N, 15): MA of the parts, MR, MF, as the muscle state
        return torch.cat((self.body_head(z[:, :NB]).flatten(1), fatigue), -1)


def rigidformer_kwargs(args) -> dict | None:
    """``MuscleRigidFormerNet``'s keyword arguments for training arguments ``args`` (``--wm_backbone``, ``--rf_*``,
    ``--muscle_model``; ndm_gym.train_muscles), None for the MLP. Runs from before ``--rf_muscle_tokens`` have "all"."""
    backbone = getattr(args, "wm_backbone", "mlp")
    if backbone == "mlp":
        return None
    if backbone != "rigidformer":
        raise ValueError(f"unknown world-model backbone {backbone!r}")
    kw = dict(muscle_model=args.muscle_model, dim=args.rf_dim, layers=args.rf_layers, heads=args.rf_heads, registers=args.rf_registers,
              tree_bias=not args.rf_no_tree_bias, rope=not args.rf_no_rope)  # fmt: skip
    if getattr(args, "rf_muscle_tokens", "all") == "body":
        kw.update(tokens="body", muscle_dim=args.rf_muscle_dim)
    if getattr(args, "rf_fused_train", False):
        kw.update(fused_train=True)
    if getattr(args, "rf_fused_features", False):
        kw.update(fused_features=True)
    return kw


class MuscleWorldModel(nn.Module):
    """Deterministic dynamics (S', F') = f(S, F, a) of the muscle character's bodies and muscle state (module docstring).
    ``rigidformer``: keyword arguments of ``MuscleRigidFormerNet`` for the token Transformer instead of the MLP (None)."""

    def __init__(self, kind: str, hidden=(2048, 2048, 2048), act: str = "silu", layer_norm: bool = True, rigidformer: dict | None = None):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.out_dim = OUT_DIM[kind]
        if rigidformer is None:
            self.in_dim = IN_DIM
            self.net = mlp(self.in_dim, list(hidden), self.out_dim, act=ACTIVATIONS[act], layer_norm=layer_norm)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.in_norm = RunningNorm(self.in_dim)
        else:
            self.net = MuscleRigidFormerNet(kind, **rigidformer)
            self.in_dim, self.in_norm = self.net.in_dim, self.net.make_input_norm()
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4 / DT if kind == "accel" else 1e-4)
        self.amp = True  # bf16 autocast in step (if the entry point enables it)

    def inputs(self, S, F, a) -> torch.Tensor:
        if isinstance(self.net, MuscleRigidFormerNet):
            return self.net.inputs(S, F, a)
        p, R, v, w = S
        H = heading(S)
        body = torch.cat((_in(H, p - p[..., :1, :]), mat_to_6d(mat_mat(H.transpose(-1, -2), R)), _in(H, v), _in(H, w)), -1)
        return torch.cat((p[..., 0, 2:3], body.flatten(-2), F, a), -1)

    def targets(self, S, F, S1, F1) -> torch.Tensor:
        """The regression targets from (S, F) to (S1, F1): the outputs that :meth:`integrate` turns into (S1, F1)."""
        if self.kind == "increment":
            return increments(S, F, S1, F1)
        H = heading(S)
        body = torch.cat((_in(H, S1[2] - S[2]), _in(H, S1[3] - S[3])), -1) / DT
        return torch.cat((body.flatten(-2), (F1 - F) / DT), -1)

    def integrate(self, S, F, y: torch.Tensor):
        """The next state and muscle state from the outputs ``y`` (denormalized)."""
        p, R, v, w = S
        H = heading(S)
        body, dF = y[..., :-FATIGUE_DIM].unflatten(-1, (NB, -1)), y[..., -FATIGUE_DIM:]
        if self.kind == "accel":
            av, aw = body.split(3, -1)
            v1, w1 = v + DT * mat_vec(H, av), w + DT * mat_vec(H, aw)
            return (p + DT * v1, mat_mat(so3_exp(DT * w1, elementwise=True), R), v1, w1), F + DT * dF
        dp, dth, dv, dw = body.split(3, -1)
        R1 = mat_mat(so3_exp(mat_vec(H, dth), elementwise=True), R)
        return (p + mat_vec(H, dp), R1, v + mat_vec(H, dv), w + mat_vec(H, dw)), F + dF

    def step(self, S, F, a, input_noise: float = 0.0):
        """One control step; ``input_noise``: std of Gaussian noise on the normalized inputs (fitting only)."""
        if isinstance(self.net, MuscleRigidFormerNet):
            return self._rf_step(S, F, a, input_noise)
        y = self._outputs(*S, F, a, input_noise)
        return self.integrate(S, F, y * self.out_norm.std() + self.out_norm.mean)

    @torch.compiler.disable(recursive=False)
    def _rf_step(self, S, F, a, input_noise: float):
        """The RigidFormer's step, in a frame of its own: a caller's torch.compile compiles the functions it calls one step
        at a time, since a window of steps as one graph nests the features' gathers too deeply for Inductor's lowering
        (RecursionError). With ``checkpoint``, the step is recomputed in the backward pass: for the muscle tokens from
        this frame, so a compiled caller's recomputation runs eagerly; for the body tokens from ``_rf_outputs``, whose
        compiled graph holds the recomputation."""
        if self.net.tokens == "body":
            y = self._rf_outputs(*S, F, a, input_noise)
        elif self.net.recompute():
            y = torch.utils.checkpoint.checkpoint(self._outputs, *S, F, a, input_noise, use_reentrant=False, preserve_rng_state=input_noise != 0)
        else:
            y = self._outputs(*S, F, a, input_noise)
        return self._integrate_y(S, F, y)

    def _rf_outputs(self, p, R, v, w, F, a, input_noise: float):
        """The body-token RigidFormer's normalized outputs of a step, with ``checkpoint`` recomputed in the backward pass."""
        if self.net.recompute():
            return torch.utils.checkpoint.checkpoint(self._outputs, p, R, v, w, F, a, input_noise, use_reentrant=False,
                                                     preserve_rng_state=input_noise != 0)  # fmt: skip
        return self._outputs(p, R, v, w, F, a, input_noise)

    def _integrate_y(self, S, F, y):
        return self.integrate(S, F, y * self.out_norm.std() + self.out_norm.mean)

    def _outputs(self, p, R, v, w, F, a, input_noise: float):
        """The network's normalized outputs of a step."""
        x = self.in_norm(self.inputs((p, R, v, w), F, a))
        if input_noise != 0:
            x = x + input_noise * torch.randn_like(x)
        return run_mlp(self.net, x, amp=self.amp)


__all__ = ["BODY_TOK", "IN_DIM", "KINDS", "MUSCLE_SCALARS", "MUSCLE_TOK", "OUT_DIM", "TOKENS", "WRENCH", "MuscleRigidFormerNet", "MuscleWorldModel",
           "heading", "increments", "relation_classes", "rigidformer_kwargs"]  # fmt: skip
