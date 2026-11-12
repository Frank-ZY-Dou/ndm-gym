# Neural dynamics models as simulators: HDMI's scenes, and the G1 against differentiable simulators

One otherwise idle H100 80GB. Env-steps per second use 20 ms control steps: HDMI's four 5 ms physics substeps in one
Isaac Lab 3 / Newton/MuJoCo-Warp call, without task terms. Values are medians of three repeats of 200 steps after 20
warm-up steps. Scenes start at the clip's first frame, without contacts between props. Scripts: `run_speed.sh`,
`run_speed_rf.sh` (throughput), `run_fits.sh`, `run_score_rf.sh` (accuracy).

## Physics, all 13 tasks (N = 4096)

| task | env-steps/s |
|---|---|
| carry_and_place_bread_box | 312,269 |
| carry_box_over_shoulder | 395,549 |
| move_foam | 383,658 |
| move_stool-climb | 290,208 |
| move_suitcase | 354,683 |
| open_door-feet | 443,236 |
| open_foldchair-sit | 438,576 |
| push_box | 332,151 |
| push_door-hand | 440,205 |
| roll_ball-hand | 379,027 |
| topple_wood_board_and_cross | 282,865 |
| traverse_door | 441,302 |
| truman | 358,328 |

## NDM against physics on three tasks

RF: RigidFormer RF-S, (a) as trained, (b) bf16 inference copy with Triton step kernels, (c) fused Triton blocks.
MLP: bf16 inference copy (accuracy not scored). M = million env-steps per second; parentheses show the ratio to physics.
Physics runs out of memory beyond N = 16384.

**move_suitcase**

| N | physics | RF (a) | RF (b) | RF (c) | MLP copy |
|---|---|---|---|---|---|
| 256 | 0.05 M | 0.38 M (7.3x) | 0.70 M (13.7x) | 0.90 M (17.5x) | 2.04 M (39.7x) |
| 1024 | 0.16 M | 0.85 M (5.3x) | 1.30 M (8.1x) | 1.69 M (10.5x) | 7.54 M (47.0x) |
| 4096 | 0.35 M | 1.23 M (3.5x) | 1.65 M (4.6x) | 2.30 M (6.5x) | 16.28 M (45.9x) |
| 16384 | 0.48 M | 1.43 M (3.0x) | 1.85 M (3.9x) | 2.63 M (5.5x) | 20.65 M (43.5x) |
| 65536 | - | 1.45 M | 1.90 M | 2.71 M | 21.15 M |

**open_door-feet**

| N | physics | RF (a) | RF (b) | RF (c) | MLP copy |
|---|---|---|---|---|---|
| 256 | 0.06 M | 0.39 M (5.9x) | 0.72 M (11.0x) | 0.90 M (13.9x) | 2.29 M (35.4x) |
| 1024 | 0.20 M | 0.86 M (4.2x) | 1.30 M (6.4x) | 1.69 M (8.3x) | 7.96 M (38.9x) |
| 4096 | 0.44 M | 1.23 M (2.8x) | 1.66 M (3.8x) | 2.31 M (5.2x) | 16.47 M (37.1x) |
| 16384 | 0.61 M | 1.44 M (2.4x) | 1.86 M (3.1x) | 2.64 M (4.3x) | 21.09 M (34.8x) |
| 65536 | - | 1.46 M | 1.90 M | 2.71 M | 20.96 M |

**truman**

| N | physics | RF (a) | RF (b) | RF (c) | MLP copy |
|---|---|---|---|---|---|
| 256 | 0.05 M | 0.38 M (7.3x) | 0.70 M (13.5x) | 0.88 M (17.0x) | 2.11 M (40.6x) |
| 1024 | 0.16 M | 0.84 M (5.2x) | 1.29 M (8.0x) | 1.68 M (10.4x) | 7.65 M (47.1x) |
| 4096 | 0.36 M | 1.23 M (3.4x) | 1.65 M (4.6x) | 2.30 M (6.4x) | 16.21 M (45.3x) |
| 16384 | 0.48 M | 1.43 M (3.0x) | 1.85 M (3.9x) | 2.63 M (5.5x) | 20.97 M (43.7x) |
| 65536 | - | 1.45 M | 1.90 M | 2.71 M | 21.14 M |

## Accuracy of RigidFormer's inference paths

Keypoint error (mm) after k open-loop steps on held-out data, and each path's largest relative difference from the eager
bf16 model over all metrics (keypoints, joints, object, contact forces). "fp32 vs bf16" is bf16's own difference.

| task | k | eager fp32 | eager bf16 | (a) compiled | (b) bf16 copy | (c) fused blocks | fp32 vs bf16, largest | (a) largest | (b) largest | (c) largest |
|---|---|---|---|---|---|---|---|---|---|---|
| move_suitcase | 8 | 13.45 | 13.46 | 13.46 | 13.47 | 13.46 | 0.09% | 0.27% | 0.31% | 0.20% |
| move_suitcase | 32 | 66.77 | 66.56 | 66.54 | 66.25 | 66.58 | 0.53% | 0.20% | 0.92% | 0.06% |
| open_door-feet | 8 | 11.55 | 11.56 | 11.56 | 11.53 | 11.55 | 0.60% | 0.29% | 1.03% | 0.24% |
| open_door-feet | 32 | 63.58 | 63.80 | 63.83 | 63.73 | 63.75 | 0.52% | 0.26% | 1.50% | 0.31% |
| truman | 8 | 18.88 | 18.90 | 18.90 | 18.89 | 18.88 | 0.15% | 0.32% | 0.38% | 0.11% |
| truman | 32 | 92.49 | 92.48 | 92.50 | 92.70 | 92.34 | 0.49% | 0.18% | 0.55% | 0.34% |

Largest differences from eager bf16: 1.5% for path (b) on open_door-feet at 32 steps; 0.32% for compiled kernels alone
(path (a), the same network).

## Accuracy of the NDMs on the 13 tasks

Each NDM uses 20000 offline updates on 4096 environments x 400 steps (seed 0), scored on held-out data (seed 1).
The collection policy follows reference joints plus noise. One fit per model; differences of a few percent are within
fit-to-fit noise (about ±5% on LAFAN1's offline benchmark). topple_wood_board_and_cross's data exclude contacts between props.

Keypoint error (mm) after 8 / 32 open-loop steps:

| task | MLP accel | MLP increment | RF accel | RF increment |
|---|---|---|---|---|
| carry_and_place_bread_box | 13.1 / 68.3 | 14.0 / 68.0 | 11.8 / 61.7 | 12.0 / 63.6 |
| carry_box_over_shoulder | 11.4 / 55.7 | 12.6 / 59.6 | 10.7 / 59.0 | 10.6 / 54.3 |
| move_foam | 12.2 / 59.6 | 13.2 / 58.7 | 10.5 / 50.8 | 11.4 / 55.4 |
| move_stool-climb | 16.5 / 84.4 | 17.5 / 88.6 | 14.9 / 76.8 | 15.8 / 76.3 |
| move_suitcase | 14.9 / 70.0 | 15.5 / 64.3 | 13.5 / 66.6 | 14.3 / 67.3 |
| open_door-feet | 12.8 / 81.3 | 13.6 / 75.5 | 11.6 / 63.8 | 12.4 / 78.9 |
| open_foldchair-sit | 12.9 / 62.5 | 12.9 / 57.4 | 10.2 / 45.7 | 10.5 / 46.7 |
| push_box | 9.2 / 45.9 | 9.2 / 38.9 | 8.3 / 42.2 | 8.4 / 39.4 |
| push_door-hand | 11.1 / 66.6 | 12.0 / 55.7 | 9.8 / 51.7 | 10.1 / 58.4 |
| roll_ball-hand | 13.4 / 79.9 | 14.6 / 80.7 | 12.2 / 72.2 | 12.6 / 72.8 |
| topple_wood_board_and_cross | 20.3 / 118.7 | 21.4 / 117.4 | 18.5 / 96.1 | 19.0 / 103.3 |
| traverse_door | 12.7 / 63.0 | 13.5 / 66.5 | 11.0 / 56.4 | 11.4 / 61.0 |
| truman | 20.8 / 102.0 | 21.2 / 94.4 | 18.9 / 92.5 | 18.9 / 94.4 |

Object error after 32 steps: position (mm), or hinge angle (mrad) for doors and the folding chair:

| task | MLP accel | MLP increment | RF accel | RF increment |
|---|---|---|---|---|
| carry_and_place_bread_box | 39.8 | 37.3 | 30.1 | 28.1 |
| carry_box_over_shoulder | 43.5 | 42.7 | 35.9 | 36.3 |
| move_foam | 53.1 | 50.7 | 39.6 | 36.4 |
| move_stool-climb | 50.1 | 47.2 | 43.9 | 43.5 |
| move_suitcase | 46.7 | 44.3 | 41.5 | 40.5 |
| open_door-feet | 20.2 mrad | 17.2 mrad | 13.9 mrad | 13.9 mrad |
| open_foldchair-sit | 17.1 mrad | 15.6 mrad | 11.7 mrad | 11.5 mrad |
| push_box | 12.7 | 12.2 | 11.5 | 11.1 |
| push_door-hand | 28.8 mrad | 25.9 mrad | 17.4 mrad | 20.5 mrad |
| roll_ball-hand | 81.6 | 79.9 | 66.6 | 67.0 |
| topple_wood_board_and_cross | 91.2 | 78.5 | 74.9 | 71.2 |
| traverse_door | 31.0 mrad | 29.9 mrad | 15.1 mrad | 17.5 mrad |
| truman | 5.2 | 3.0 | 2.5 | 2.1 |

RigidFormer / MLP error ratios with the same parameterization (median over tasks):

| | keypoints, 8 steps | keypoints, 32 steps | object, 32 steps | hand-object force, 8 steps | feet force, 8 steps |
|---|---|---|---|---|---|
| accel | 0.90 (13 of 13) | 0.90 (12 of 13) | 0.76 (13 of 13) | 0.74 (13 of 13) | 0.90 (13 of 13) |
| increment | 0.86 (13 of 13) | 0.93 (8 of 13) | 0.81 (13 of 13) | 0.74 (13 of 13) | 0.90 (13 of 13) |

## Differentiable simulators and the G1 world models

One otherwise idle A100-SXM4-40GB (driver 565.57.01), 2026-11-12. The G1 with primitive colliders (the mjlab model
NablaEngine bundles); a control step is 20 ms of four 5 ms substeps with PD position targets; float32. The simulators start
from standing states with smooth random targets near the default pose. Forward: control steps per second over 200 steps
(four segments of 50, each from the standing states), the median of three repeats after a warm-up, compilation excluded.
Gradient: the forward and reverse-mode gradient of a 32-step rollout's loss with respect to its 32 target vectors, ms per
control step and peak GPU memory. The world model is SuperTrack's MLP from the LAFAN1 study, `rfd_st_mlp_s7` (last
checkpoint): its step from the clip's frame-0 reference states, as trained (bf16, compiled) and on its fastest inference
path, and the gradient through the model as trained, the call one CUDA graph. Scripts: [`diffsim/`](diffsim/README.md) for
the simulators, `ndm_gym.wmbench.speed_pair` for the world model (`--grad_horizon 32`: the gradient).

Forward, control steps per second (all worlds):

| | N = 1,024 | N = 4,096 | N = 16,384 |
|---|---|---|---|
| NablaEngine, as published (10 iterations) | 0.95 M | 1.16 M | 1.28 M |
| NablaEngine, tuned (5) | 0.98 M | 1.17 M | 1.29 M |
| MuJoCo Warp + PR 1535, as published (10, line search 20) | 0.16 M | 0.30 M | 0.38 M |
| MuJoCo Warp + PR 1535, tuned (5, 8) | 0.16 M | 0.31 M | 0.39 M |
| MJX, as published (every contact pair padded) | 0.016 M | 0.018 M | 0.019 M |
| MJX, tuned (capacities calibrated on MuJoCo C) | 0.046 M | 0.074 M | 0.079 M |
| MLP world model, as trained | 3.00 M | 5.74 M | 7.70 M |
| MLP, fastest path | 3.40 M | 7.22 M | 9.18 M |

Gradient of a 32-step rollout's loss with respect to its 32 target vectors, ms per control step (peak GPU memory):

| | N = 1,024 | N = 4,096 | N = 16,384 |
|---|---|---|---|
| NablaEngine, as published | 1.64 (0.8 GiB) | 5.48 (3.1 GiB) | 20.06 (12.5 GiB) |
| NablaEngine, tuned | 2.02 (0.8 GiB) | 5.44 (3.1 GiB) | 19.92 (12.5 GiB) |
| MuJoCo Warp + PR 1535, as published | 25.32 (0.5 GiB) | 56.35 (2.0 GiB) | 184.59 (8.0 GiB) |
| MuJoCo Warp + PR 1535, tuned | 22.76 (0.5 GiB) | 50.84 (1.8 GiB) | 168.23 (7.3 GiB) |
| MJX, as published | out of memory | out of memory | out of memory |
| MJX, tuned, its solver as a fixed scan of 5 iterations | 113.45 (3.5 GiB) | 290.45 (13.1 GiB) | out of memory on the 40 GB card |
| MLP world model | 0.69 (1.0 GiB) | 1.59 (2.3 GiB) | 4.84 (7.3 GiB) |

Ratios at equal N (forward: the MLP's fastest path; gradient: as trained), against each simulator's faster setting:

| | N = 1,024 | N = 4,096 | N = 16,384 |
|---|---|---|---|
| MLP forward / NablaEngine | 3.5x | 6.2x | 7.1x |
| MLP forward / MuJoCo Warp | 20.9x | 23.2x | 23.4x |
| MLP gradient / NablaEngine | 2.37x | 3.42x | 4.11x |
| MLP gradient / MuJoCo Warp | 32.9x | 32.0x | 34.7x |
| MLP forward / MJX | 74x | 98x | 116x |
| MLP gradient / MJX | 164x | 183x | - |

- The two simulators simulate the same thing: the 32-step loss agrees to five digits at every N (3.34579 / 3.34578,
  3.33408 / 3.33408, 3.34197 / 3.34197). No world fell or overflowed a capacity. MuJoCo Warp reached its line-search cap in
  some substep in every world, in both settings. With 5 Newton iterations, 80-83% of MuJoCo Warp's worlds and 10% of
  NablaEngine's world-steps reached the iteration cap (with 10: none, and 0.001-0.002%).
- Tuning: NablaEngine's published setting already runs a control step as one compiled call, and 5 iterations change its
  times by 3% at most (its gradient at 1,024 worlds is 23% slower). MuJoCo Warp's tuned setting is 4% faster forward and
  10-11% faster in the gradient.
- The simulators are exact and the world models are not: on the LAFAN1 study's offline benchmark the body-position error
  after 32 open-loop steps on the SuperTrack policy's held-out set is 59.2 mm for the MLP
  ([ablations](../g1_lafan1_tracking/ablations.md)).
- NablaEngine's own timing kit on the same GPU (its scripts unmodified; `diffsim/run_paper_bench.sh`), 4,096 worlds, ms per control
  step: NablaEngine forward 3.17, forward with its tape plus the adjoint 4.90; MuJoCo Warp 3.17.0 forward 12.8; with pull
  request 1535 forward 13.7 and adjoint 39.3; MJX forward 372.8 (no reverse mode natively); Genesis 1.4.5 forward 27.6
  (performance mode); MuJoCo C on 16 CPU threads 74.1.
- MJX (mujoco-mjx 3.17.0, jax 0.12.0.1): stock MJX differentiates its solver only with one iteration
  ([issue #2259](https://github.com/google-deepmind/mujoco/issues/2259)); with one, the rollout's loss is not finite on this
  model (40.47, 103.45 and 381.41 ms per control step), so the gradient row replaces the solver's outer loop by a fixed scan
  of its 5 iterations, which gives its forward loss (3.34473 and 3.33268 at 1,024 and 4,096 worlds, against its forward
  3.34477 and 3.33267; NablaEngine and MuJoCo Warp read 3.34579 and 3.33408). Its as-published setting pads every contact pair and
  runs out of memory in reverse mode at every N; the tuned one sizes the capacities from MuJoCo C on 256 worlds.
