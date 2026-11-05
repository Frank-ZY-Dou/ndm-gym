# RigidFormer world-model backbone

## 1. What it is

`ndm_gym/worldmodels/rigidformer.py` replaces the MLP and input features in SuperTrack's acceleration world model
(`AccelWorldModel`) and SuperMimic's increment world model (`IncrementWorldModel`). Targets, output normalizer,
integration and losses stay unchanged. The backbone follows the structure of RigidFormer [1] and is reimplemented here
for articulated bodies: body tokens replace free-object tokens, and analytic forward kinematics replaces the point-cloud
encoder. The reduced-coordinate read-out (root and
joints) enforces joint constraints, the articulated counterpart of RigidFormer's Kabsch projection.

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
With anchors (section 3) the raw features end with every body's rotary points (the anchors' centroid with `mean`, 90
values; the anchors with `concat`, 90 K), which take the body positions' place in the rotary attention.

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
- Anchor RoPE (RigidFormer's ARoPE, off by default): `--rf_anchors K` gives every body K anchor points, the first K of
  RigidFormer's farthest point sampling (rule `centroid_farthest`) over its visual-mesh vertices in the link frame,
  which move with the body. `--rf_anchor_mode mean` (the default) mean-pools the anchors' rotary angles; the angles
  being linear in the position, these are the angles of the anchors' centroid, so the inputs carry that body-fixed point
  (3 values per body after the global context) and the blocks see one rotary point per body, as without anchors.
  `concat` gives each of the 3K anchor coordinates its own channel pairs (K = 4 at head width 32: one frequency, 1 rad/m,
  per coordinate; a K whose coordinates cannot share the channels equally is refused). `--rf_rope_center` measures the
  rotary positions from their centroid over the bodies (and anchors), where the zero-angle registers then sit; the body
  tokens' query-key products do not change. RigidFormer's v1 configuration is `--rf_anchors 4 --rf_rope_center` (mode
  `mean`, interleaved channel pairs as here). Both modes change only the rotary tables the blocks take, so they combine
  with every speed option of section 6.
- Output layout: the root components, then each joint component over the 29 joints. With both heads at zero, the first
  prediction is the output normalizer's mean, as with the MLP's zero-initialized last layer.
- The residual stream is fp32; matrix multiplications run under the training's bf16 autocast (`--amp bf16`). The default
  `checkpoint="frozen"` recomputes the trunk in the backward pass only while the parameters are frozen, as in the
  policy's BPTT. `ext_wrench=True` makes `inputs` require forces (`wrench_h`, (N, 30, 6)); the world models pass none yet.

## 4. Use in SuperTrack-RF and SuperMimic-RF

`--wm_backbone rigidformer` selects the backbone (default `mlp`). Both algorithms share the trunk and features; only
their heads differ.

| run | flags | heads: root + per joint | outputs |
|---|---|---|---|
| SuperTrack-RF | `--algo supertrack --wm_backbone rigidformer` | 6 + 1: root linear and angular acceleration; joint acceleration | 35 |
| SuperMimic-RF | `--algo supermimic --wm_backbone rigidformer` | 12 + 2: root Δp, Δθ, Δv, Δω; joint Δq, Δq̇ | 70 |

Defaults in `ndm_gym/train.py`: `--rf_dim 128` (token width), `--rf_layers 3` (blocks), `--rf_heads 4`,
`--rf_registers 2`; `--rf_no_tree_bias` and `--rf_no_rope` are off. Other constructor arguments keep their defaults.
Options for policy-gradient spikes during BPTT through the frozen world model (off by default): `--rf_rope_detach`
(no gradient through rotary positions, which still enter the token features),
and for either backbone `--bptt_clip_k k` (each rollout's state gradient after every imagined step capped at k times the
batch median), `--wm_bptt_fp32` (no bf16 autocast in the world model during the BPTT), `--pi_rollout_clip k` (each
rollout's part of the policy gradient over the horizon at most k times the median over the rollouts, sized by the
gradient that reaches the policy's output) and `--bptt_decay lam` (the gradient through every imagined state times lam).
`tools/grad_stats.py` measures their effect on a checkpoint at learning rate 0; with `--anatomy` it also takes every
update's gradient apart by imagined step and by rollout (`ndm_gym/algos/rollout_grads.py`) and evaluates robust
aggregations of the rollouts on the same batches. The MLP's flags `--wm_hidden`, `--wm_act` and `--wm_no_ln` must stay at
their defaults; `rigidformer_kwargs` rejects other values. Fitting compiles one step rather than the whole k-step rollout,
which takes about 10 min to compile as one graph (one step: about 2 min). Benchmark settings and results:
[README](../README.md).

Offline, RigidFormer has 7-16% lower open-loop error than the MLP in distribution and 11-21% out of it at a learning rate
of 4e-4 picked on the benchmark's test sets. At the shared default 2e-4, changes are -4% / -11% (SuperTrack's model) and +8% / +0%
(SuperMimic's). See the [offline ablation study](../benchmarks/g1_lafan1_tracking/ablations.md#summary).

## 5. Sizes

| size | `--rf_dim` / `--rf_layers` / `--rf_heads` | parameters |
|---|---|---|
| RF-XS | 96 / 2 / 3 | 0.38 M |
| RF-S (default) | 128 / 3 / 4 | 0.88 M |
| RF-M | 192 / 4 / 6 | 2.43 M |
| RF-L | 256 / 4 / 8 | 4.26 M |
| MLP 2048 × 3 (`--wm_backbone mlp`) | | 8.75 M (SuperTrack), 8.82 M (SuperMimic) |

All four have head width 32, two registers, the tree bias and RoPE; both algorithms' counts round to these values.

Fewer tokens (`--rf_merge`, default `none`: one token per body). A link may share its parent's token when the parent is
not the pelvis and has no other child (`token_groups`). `coincident`: when the link's origin lies within 5 cm of the
parent's, i.e. serial hinges whose axes nearly meet: the G1's wrists (roll, pitch and yaw links; origins 3.8 and 4.6 cm
apart), ankles (pitch and roll links, 1.8 cm), the waist's yaw and roll links with the torso (4.4 and 0 cm) and each
shoulder's pitch and roll links (4.0 cm): 20 tokens (the next closest pair, a hip's pitch and roll links, is 6.0 cm
apart). `distal`: only the chains that end at a leaf, the wrists and ankles: 24 tokens. `joints`: one token per
anatomical joint, the hinges named `<joint>_{pitch,roll,yaw}_joint` along a chain: the pelvis, the waist (with the
torso), and per side the hip, knee, ankle, shoulder, elbow and wrist: 14 tokens, 16 with the two registers (a power of
two, as the fused kernels need). A token's input is its members' token features side by side (zero-padded to three),
through one shared embedding; its rotary position is its last member's link origin; the tree bias counts the joints
between tokens in the tree of tokens. In the read-out every body takes its token's features plus a learned embedding of
the body (bodies that share a token only), then the per-body SwiGLU fusion and heads as before, so every joint keeps its
own output. The inputs, the step and the outputs do not change. The fused training kernels (`--rf_fused_train`) need
token counts, widths and head widths that are powers of two; other settings are refused when the network is built.

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

Speed options (`add_speed_args`, off by default):

- `--rf_fused_train`: each block's forward and backward passes as fused Triton kernels (`ndm_gym/kernels/rf_train.py`).
  The forward pass is one kernel, which also keeps the residual after the attention; the backward pass is two kernels
  between cuBLAS products with qkvg's and the output projection's weights, and the parameters' gradients are bf16
  products over all tokens with fp32 outputs. The products round their inputs to bf16 as autocast does; RMSNorms,
  softmax, gates and FiLM stay in fp32. One RF-S block, forward and backward, GPU time: 2.75 instead of 3.63 ms with the
  parameters trained (2048 windows), 2.87 instead of 4.65 ms frozen (4096 rollouts). A block keeps two (N, T, d) tensors
  for its backward pass, which lowers the runs' peak memory from 17 to 12 GB. It launches more and smaller kernels than
  the compiled block and is faster only with `--wm_cudagraphs` and `--pi_cudagraphs`. The kernels replace the blocks'
  passes under bf16 autocast; passes without autocast (the BPTT of `--wm_bptt_fp32`, the fp32 penalty of
  `--wm_smooth_w`) run the blocks as without the option. Refused with `--amp none`.
- `--rf_bptt_keep_blocks` (with `--rf_fused_train`, not with `--wm_bptt_fp32`): the policy's BPTT keeps the fused
  blocks' saved tensors and recomputes only the read-out (`checkpoint="frozen_readout"`), with the same results and
  about 12 GB more at 4096 rollouts × 32 steps.
- `--combo_kernels`: inductor launches independent small kernels of a compiled graph as one
  (`torch._inductor.config.combo_kernels`); RigidFormer's world-model step at 2048 windows: 93 forward and 307 backward
  kernels instead of 223 and 498.
- `--compile_eager` (either backbone): the forward kinematics the training otherwise runs eagerly, as compiled functions:
  the collection's reward (`TrackingSim.reward_fn`), the world model's normalizer features (`Algo.norm_inputs`: for
  RigidFormer its token features with their forward kinematics, for the MLP its invariant features) and SuperMimic's
  critic input in the policy loss. Eagerly each is several hundred small launches, which left the collection CPU-bound
  (SuperTrack-RF: 125 ms per iteration, 48 ms of it on the GPU; with the option 65 and 28 ms) and the GPU of
  SuperMimic's policy update partly idle (SuperMimic-RF: 496 ms per iteration, 450 ms with the option). Equal up to
  float rounding.
- `--rf_fit_keep_ops` (with `--rf_fused_train`): the world-model fit's fused forward passes also keep the bf16 operands
  their backward passes would recompute (the normalized input and FiLM output, the q/k/v/gate products: about 100 MB
  per block at 2048 windows, 2.4 GB during the fit), and the backward passes read them and run the SwiGLU half as a
  hidden-chunk kernel, a cuBLAS product and a row kernel. One block, forward and backward at 2048 windows: 2.46 instead
  of 2.71 ms. Equal up to float rounding; the BPTT's passes (frozen parameters) are unchanged.

Training iterations per second (A100-SXM4-40GB; medians of 40 iterations after compilation, from the study's
checkpoints; "existing": `--wm_cudagraphs --pi_cudagraphs --fused_adam --elementwise_3x3`, with `--critic_cudagraphs` for
SuperMimic and `--rf_fused_rope` for RigidFormer, which `--rf_fused_train` replaces; each column adds its option to the
previous ones):

| | default | existing | + `--rf_fused_train` | + `--combo_kernels` | + `--rf_bptt_keep_blocks` |
|---|---|---|---|---|---|
| SuperTrack-RF | 0.452 | 0.535 | 0.706 | 0.739 | 0.787 |
| SuperMimic-RF | 0.407 | 0.522 | 0.682 | 0.708 | 0.755 |
| SuperTrack (MLP) | 1.188 | 3.126 | – | 3.196 | – |
| SuperMimic (MLP) | 1.264 | 2.747 | – | 2.799 | – |

Peak memory of SuperTrack-RF: 17.1 GB by default, 12.0 GB with `--rf_fused_train`, 26.2 GB with `--rf_bptt_keep_blocks`
as well (SuperMimic-RF: 16.1, 11.6, 25.6 GB).

Anchor RoPE in RigidFormer's v1 configuration (`--rf_anchors 4 --rf_rope_center`), with every option in the last column,
measured alongside runs without it: SuperTrack-RF 0.775 against 0.772 iterations per
second, SuperMimic-RF 0.741 against 0.742; GPU time per iteration 1210 against 1207 ms and 1236 against 1233 ms; the same
peak memory.

`--compile_eager` and `--rf_fit_keep_ops`, measured in the same session. Columns add options cumulatively, starting from
all applicable options above. RigidFormer was remeasured: 0.772 and 0.742 rather than 0.787 and 0.755.

| | options above | + `--compile_eager` | + `--rf_fit_keep_ops` |
|---|---|---|---|
| SuperTrack-RF | 0.772 | 0.816 | 0.852 |
| SuperMimic-RF | 0.742 | 0.811 | 0.848 |
| SuperTrack (MLP) | 3.192 | 3.366 | – |
| SuperMimic (MLP) | 2.802 | 3.259 | – |

The `--rf_fit_keep_ops` column also has two exact changes to `--rf_fused_train`'s backward pass, which no longer
zero-fills a buffer it overwrites or the gradients of outputs that no loss reads (the same bits as before); with
`--compile_eager` they make 0.823 for SuperTrack-RF and 0.818 for SuperMimic-RF. `--rf_fit_keep_ops` shortens the
world-model fit's backward pass (GPU time per iteration, SuperTrack-RF: 526 to 473 ms) and lengthens its forward pass
(171 to 182 ms). With the v1 anchor RoPE and both options: SuperTrack-RF 0.848, SuperMimic-RF 0.847. Peak memory changes
by at most 0.1 GB (the policy update's BPTT sets it). Without `--rf_bptt_keep_blocks`, `--compile_eager` takes
SuperTrack-RF from 0.731 to 0.763 iterations per second and SuperMimic-RF from 0.697 to 0.759.

Gradients against fp32 (the default path without autocast and TF32; 8 batches of 512 windows and of 1024 rollouts from
replays of the checkpoints' policies, median relative error of all parameters' gradients): the world-model fit 0.14 with
`--rf_fused_train` and 0.16 by default for SuperTrack-RF, 0.10 and 0.11 for SuperMimic-RF. The policy update's
gradient through 32 imagined steps: 0.66 and 0.75 for SuperTrack-RF, 0.75 and 0.70 for SuperMimic-RF; with MLP world
models 0.38 (SuperTrack) and 0.73 (SuperMimic); through 8 steps (SuperMimic-RF) 0.24 and 0.23, and 0.03 between the two
paths. The BPTT amplifies rounding: a 10⁻⁶ relative change of the start velocities moves the fp32 gradient by
10⁻⁵-10⁻² (once 0.22, SuperTrack-RF), and the difference stays at 0.66-0.74 with the world model in fp32
(`--wm_bptt_fp32`), with the policy in fp32, or with both (TF32 products).

Trainings of 400 iterations from scratch (seed 1) differ from the default run about as much as a repeat of the default
run does. Mean relative differences per logged iteration from iteration 100, for `--rf_fused_train` alone, all the speed
options above, and the repeat: world-model loss 1.9%, 4.4% and 4.3% (SuperTrack-RF), 5.8%, 3.8% and 7.3%
(SuperMimic-RF); normalized episode return 35%, 37% and 45%, and 12%, 18% and 16%. With all the options these runs made
0.778 and 0.733 iterations per second, by default 0.393 and 0.404.

The same trainings with `--compile_eager` and `--rf_fit_keep_ops` added to all the options, against those four runs of
each algorithm (in parentheses the range among the four): SuperTrack-RF, two runs, world-model loss 1.3-6.3% (1.9-4.4%),
policy loss 1.7-3.6% (2.0-4.0%), normalized episode return 51-85% (35-46%); with `--compile_eager` alone (one run)
1.2-3.8%, 1.1-3.4% and 41-57%, with `--rf_fit_keep_ops` alone (two runs) 1.5-5.2%, 2.1-4.5% and 25-50%. SuperMimic-RF,
two runs: world-model loss 2.5-6.2% (3.3-7.3%), policy loss 2.6-11.6% (3.4-9.6%), normalized episode return 5-19%
(12-18%). Single runs reach either side of the four runs' range: over iterations 100-400, the mean episode length of
SuperTrack-RF was 47.6 and 46.8 with both options, 53.2 with `--compile_eager` alone and 62.2 and 52.2 with
`--rf_fit_keep_ops` alone, against 51.7-60.4; that of SuperMimic-RF 62.9 and 81.5 against 71.5-91.3.

As a simulator (inference only), the G1 world models take HDMI's inference paths (`worldmodels/g1_infer.py`,
`worldmodels/g1_fused.py`, `kernels/rf_block.py`; `wmbench/speed_pair.py` times them): an inference copy with the network
in bf16 end to end or in fp32 under autocast, token attention and the relation bias gathered once; the step's features
(the forward kinematics and token features, or the MLP's invariant features) and integration as two Triton kernels; the
blocks as fused kernels (`FusedBlock`, the model as trained or its fp32 copy; a model trained with `--rf_fused_train` runs
them instead of the training kernels); the whole network in one kernel (`FusedG1RigidFormerNet`, refused on an H100 while
Triton 3.6 miscompiles it there). Each path is compared with the model on the GPU before use and refused on a mismatch:
the fused features and integration within float rounding of `inputs` and `step`, the network no further from its fp32
values than the model under bf16 autocast (the bf16 copy: twice its mean and four times its largest error).

## 7. Validation

`python -m pytest tests/test_rigidformer_backbone.py` runs 30 CPU tests (`-m "not slow"` skips the two slow ones).
They need G1 assets and LAFAN1 motions in `data/` and check:

- features: forward kinematics and offset scale, shapes, pelvis token, absolute heights, each body's own joint, PD error
  (simulator's PD target minus q), torque ratio in [-1, 1], global context, invariance to yaw and horizontal translation;
- read-out: the layout, joint j read from the token of the body that owns it, and `step` equal to an MLP world model's
  while the heads are at zero (both algorithms);
- training: gradients to the setpoints, context, velocities and all parameters; checkpointing (same outputs and
  gradients; by default only while frozen; `frozen_readout` the same bits); input noise; the training-flag mapping and
  its errors; the wrench channel;
- anchor RoPE: the farthest point sampling (ties, order independence, duplicates, prefix) and the body anchors; the
  rotary tables the blocks take against RigidFormer's definitions computed directly in float64 (one point per body, 4
  anchors with `mean` and `concat`, each with and without `--rf_rope_center`); translation invariance of the query-key
  products; `--rf_anchors 1` equal to the network without anchors; both modes' inputs, invariances and refusals; the
  centring;
- learning (slow): each world model fits the one-step dynamics of a random teacher MLP world model on 128 states.

`python -m pytest tests/test_rf_train.py` checks `--rf_fused_train` on a GPU (skipped without one): a block's output
and every gradient (input, rotary tables, attention bias, FiLM context, parameters) against the fp32 block, next to the
bf16-autocast block's own errors, over token counts, batch sizes, options, tile settings and frozen parameters, also
with `--rf_fit_keep_ops` (its output the same bits as without; operands kept only while the parameters train); bitwise
determinism; the whole network eager and compiled, trained and frozen, and with `--rf_bptt_keep_blocks` (the same
bits), also with the anchor RoPE (RigidFormer's v1 configuration and `concat`) and `--rf_fit_keep_ops`; both world
models' 4-step rollouts with the anchor RoPE and with `--rf_fit_keep_ops`, fit and BPTT; the refusals. With
`NDM_RF_TRAIN_CKPT` (a LAFAN1 RigidFormer checkpoint) and `NDM_RF_TRAIN_DATA` (a recording of its policy,
`python -m ndm_gym.wmbench.collect`) set, also the world-model fit on recorded windows.

`tests/test_lafan1_training.py` checks `--compile_eager`: which functions it compiles (both algorithms, on CPU, with the
same bits when the compiler is replaced by the identity) and, on a GPU, the compiled reward, normalizer features
(RigidFormer and MLP) and SuperMimic's critic input with its gradients against the eager ones.

## Reference

[1] Z. Dou, M. Guo, H. Wu, D. Roble, T. Stuyck, W. Matusik. RigidFormer: Learning Rigid Dynamics using Transformers.
[arXiv:2605.09196](https://arxiv.org/abs/2605.09196), 2026.
