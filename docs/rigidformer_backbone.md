# RigidFormer world-model backbone

## 1. What it is

A Transformer backbone for SuperTrack's acceleration world model (`AccelWorldModel`) and SuperMimic's increment world
model (`IncrementWorldModel`), in `ndm_gym/worldmodels/rigidformer.py`. It replaces the MLP and its input features; the
targets, output normalizer and integration are unchanged. The backbone follows the structure of RigidFormer [1] and is
reimplemented here for articulated bodies: one token per rigid body instead of one per free object, analytic forward
kinematics instead of a point-cloud encoder, 3-D rotary attention on the body positions, pre-norm blocks with RMS and QK
normalization, gated attention and SwiGLU, register tokens and FiLM conditioning. The read-out in reduced coordinates
(root and joints) keeps the joint constraints exact, the articulated counterpart of RigidFormer's Kabsch projection.

## 2. Inputs and tokens

One token per G1 body (30, pelvis first, in kinematic-tree order; every other body owns one joint) plus `--rf_registers`
learned register tokens. Vectors are in the root's heading (yaw) frame with the origin at the pelvis's ground projection,
so the features do not change under horizontal translation or rotation about the vertical axis; heights stay absolute.

| slots | features of a body |
|---|---|
| 0:15 | position of the link origin (z: height above the ground), 6-D rotation, linear velocity of the link origin, angular velocity |
| 15:21 | its joint: q, q̇, setpoint u, PD error e, torque ratio τ̂, calibration offset (normalized); zeros on the pelvis |
| 21:27 | optional external wrench: force, and torque about the link origin (`ext_wrench=True`; off by default) |

u is the normalized joint setpoint sent to the simulator (for both algorithms), e = q0 + δ + α·u − q (δ: calibration
offset) is the simulator's PD target minus q, and τ̂ = clip(kp·e − kd·q̇, ±τmax) / τmax, with the simulator's constants
from `ndm_gym/robots/g1.py`. Four global values follow the tokens: friction and the torso center-of-mass offset.
`TokenInputNorm` normalizes the 634 = 30 × 21 + 4 raw features and appends the 90 body-position coordinates divided by
0.1 m for the rotary attention, so the input noise of `step` (`--wm_input_noise`) moves those positions by noise × 0.1 m.

## 3. Architecture

```
h    = W_e · features_b + E_b for the 30 bodies, then the registers       shared embedding W_e, learned body embedding E_b
c    = SiLU(W_c · context)                                                 64 values
L blocks:
  h <- h + W_o (sigmoid(g) * Attn(RoPE(RMSNorm(q)), RoPE(RMSNorm(k)), v, tree bias))    q, k, v, g = W_qkvg · RMSNorm(h)
  h <- h * (1 + gamma_l(c)) + beta_l(c)                                                 FiLM, zero-initialized
  h <- h + SwiGLU(RMSNorm(h))                                                           hidden width 2.5 d
z_b  = SwiGLU(RMSNorm([h_b after the embedding | h_b after the last block]))            the 30 body tokens
y    = [root head(z_pelvis) | joint head(z_b) in G1 joint order]                        heads zero-initialized
```

- Rotary attention turns the first 6·⌊(3·hd/4)/6⌋ channels of each head by the body position times log-spaced
  frequencies in [1, 10] rad/m (24 of 32 channels, 4 frequencies per coordinate at the sizes below), so query-key products
  depend only on relative positions; registers sit at the origin. The tree bias is a learned, zero-initialized bias per
  head and per number of joints between two bodies (0 to 16; pairs involving a register form one more class).
- Output layout: the root components, then each joint component over the 29 joints. With both heads at zero, the first
  prediction is the output normalizer's mean, as with the MLP's zero-initialized last layer.
- The residual stream is fp32; matrix multiplications run under the training's bf16 autocast (`--amp bf16`). The default
  `checkpoint="frozen"` recomputes the trunk in the backward pass only while the parameters are frozen, as in the
  policy's BPTT. `ext_wrench=True` makes `inputs` require forces (`wrench_h`, (N, 30, 6)); the world models pass none yet.

## 4. Use in SuperTrack-RF and SuperMimic-RF

`--wm_backbone rigidformer` selects the backbone (default `mlp`). Both algorithms use the same trunk and features; only
the heads differ, as only the last layer differs between the two MLPs.

| run | flags | heads: root + per joint | outputs |
|---|---|---|---|
| SuperTrack-RF | `--algo supertrack --wm_backbone rigidformer` | 6 + 1: root linear and angular acceleration; joint acceleration | 35 |
| SuperMimic-RF | `--algo supermimic --wm_backbone rigidformer` | 12 + 2: root Δp, Δθ, Δv, Δω; joint Δq, Δq̇ | 70 |

Size flags, with the defaults of `ndm_gym/train.py`: `--rf_dim 128` (token width), `--rf_layers 3` (blocks),
`--rf_heads 4`, `--rf_registers 2`, and the switches `--rf_no_tree_bias` and `--rf_no_rope` (off); the other constructor
arguments keep their defaults. Against policy-gradient spikes in the BPTT through the frozen world model (off by
default): `--rf_rope_detach` (no gradient through the rotary positions; the positions still reach the token features),
and for either backbone `--bptt_clip_k k` (each rollout's state gradient after every imagined step capped at k times the
batch median) and `--wm_bptt_fp32` (no bf16 autocast in the world model during the BPTT). `tools/grad_stats.py` measures
their effect on a checkpoint at learning rate 0. The MLP's flags `--wm_hidden`, `--wm_act` and `--wm_no_ln` must stay at
their defaults; `rigidformer_kwargs` raises an error instead of ignoring them. Targets, output normalizers, integration
and losses are those of the MLP world models. The world-model fit compiles one step instead of the whole k-step rollout,
which as one graph takes about 10 min to compile (one step: about 2 min). Benchmark settings and results:
[README](../README.md).

## 5. Sizes

| size | `--rf_dim` / `--rf_layers` / `--rf_heads` | parameters |
|---|---|---|
| RF-XS | 96 / 2 / 3 | 0.38 M |
| RF-S (default) | 128 / 3 / 4 | 0.88 M |
| RF-M | 192 / 4 / 6 | 2.43 M |
| RF-L | 256 / 4 / 8 | 4.26 M |
| MLP 2048 × 3 (`--wm_backbone mlp`) | | 8.75 M (SuperTrack), 8.82 M (SuperMimic) |

All four have head width 32, two registers, the tree bias and RoPE; both algorithms' counts round to these values.

## 6. Speed and memory

Measured on one A100-SXM4-40GB (PyTorch 2.11.0+cu126, bf16 autocast, TF32, default `torch.compile`) with noisy LAFAN1
reference states: medians of 10 timed calls after warm-up and compilation (3-5 for (b)), about ±5% between repeats; peak
memory from `torch.cuda.max_memory_allocated` (model, optimizer state, data). (a) is one `wm_update` (8-step open-loop
rollout of 2048 windows, loss, backward pass, gradient clipping, Adam step), run 10 times per iteration (`--wm_batches`);
(b) is the world-model part of a policy update, BPTT through 32 frozen world-model steps for 4096 rollouts.

| SuperTrack | (a) world-model update | (b) BPTT, world-model part |
|---|---|---|
| MLP 2048 × 3 | 58 ms, 0.7 GB | 95 ms, 2.3 GB |
| RF-XS | 133 ms, 6.5 GB | 452 ms, 2.3 GB |
| RF-S | 145 ms, 11.9 GB | 778 ms, 3.2 GB |
| RF-M | 225 ms, 22.6 GB | 1744 ms, 5.1 GB |
| RF-L | 254 ms, 30.1 GB | 1918 ms, 6.5 GB |

SuperMimic: MLP 39 ms, 0.6 GB in (a) and 104 ms, 2.4 GB in (b); RF-S 130 ms, 11.8 GB in (a) and 780 ms, 3.2 GB in (b).
As in training, (a) compiles the MLP's whole rollout and RigidFormer's single step. In (b), RigidFormer recomputes its
trunk (`checkpoint="frozen"`); without recomputation, RF-XS and RF-S run out of the 40 GB. With `checkpoint="always"`,
RF-S takes 170 ms and 1.6 GB in (a). Without the tree bias, RoPE or registers, RF-S takes 158, 149 and 140 ms in (a).

## 7. Validation

`python -m pytest tests/test_rigidformer_backbone.py` runs 15 test cases on CPU (`-m "not slow"` skips the two slow
ones); like the other tests, they need the G1 assets and the LAFAN1 motions in `data/`. They check:

- features: forward kinematics and offset scale, shapes, pelvis token, absolute heights, each body's own joint, PD error
  (simulator's PD target minus q), torque ratio in [-1, 1], global context, invariance to yaw and horizontal translation;
- read-out: the layout, joint j read from the token of the body that owns it, and `step` equal to an MLP world model's
  while the heads are at zero (both algorithms);
- training: gradients to the setpoints, context, velocities and all parameters; checkpointing (same outputs and
  gradients; by default only while frozen); input noise; the training-flag mapping and its errors; the wrench channel;
- learning (slow): each world model fits the one-step dynamics of a random teacher MLP world model on 128 states.

## Reference

[1] Z. Dou, M. Guo, H. Wu, D. Roble, T. Stuyck, W. Matusik. RigidFormer: Learning Rigid Dynamics using Transformers.
[arXiv:2605.09196](https://arxiv.org/abs/2605.09196), 2026.
