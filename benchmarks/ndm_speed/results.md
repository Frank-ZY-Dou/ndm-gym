# Neural dynamics models as simulators: HDMI's human-object interaction scenes

One H100 80GB running nothing else. Env-steps per second with 20 ms control steps (physics runs HDMI's four 5 ms
substeps in Isaac Lab 3 with Newton/MuJoCo-Warp, its best case: one call per control step, no task terms), the median of
three repeats of 200 steps after 20 warm-up steps. The physics scenes start at the clip's first frame, without contacts
between props. Scripts: `run_speed.sh`, `run_speed_rf.sh` (throughput), `run_fits.sh`, `run_score_rf.sh` (accuracy).

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

RF: RigidFormer RF-S, (a) as trained, (b) bf16 inference copy with the step as Triton kernels, (c) blocks as fused
Triton kernels. MLP: its bf16 inference copy (accuracy not scored). M = million env-steps per second; in parentheses,
the ratio to physics. Physics runs out of memory beyond N = 16384.

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

Every path stays within 1.5% of the eager bf16 model on every metric (path (b), open_door-feet, 32 steps); compiled
kernels alone (path (a), the same network) differ by up to 0.32%.

## Accuracy of the NDMs on the 13 tasks

Each NDM fit offline for 20000 updates on 4096 environments x 400 steps (seed 0), scored on a held-out collection
(seed 1); the data policy is the reference's joints plus noise. One fit per model, so differences of a few percent are
within fit-to-fit noise (about ±5% on LAFAN1's offline benchmark). topple_wood_board_and_cross's data were collected
without contacts between its props.

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

RigidFormer against the MLP in the same parameterization, as the ratio of the errors (median over tasks):

| | keypoints, 8 steps | keypoints, 32 steps | object, 32 steps | hand-object force, 8 steps | feet force, 8 steps |
|---|---|---|---|---|---|
| accel | 0.90 (13 of 13) | 0.90 (12 of 13) | 0.76 (13 of 13) | 0.74 (13 of 13) | 0.90 (13 of 13) |
| increment | 0.86 (13 of 13) | 0.93 (8 of 13) | 0.81 (13 of 13) | 0.74 (13 of 13) | 0.90 (13 of 13) |
