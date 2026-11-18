# SuperMimic on Eden’s Ode to Joy

Three training seeds controlled two floating Wuji hands on the same 90 BPM arrangement as the SuperTrack run. Each final checkpoint played the whole song from 256 held-out starts. The stored starting perturbations and goal masks matched element for element between algorithms. All aggregate uncertainties below are sample standard deviations across the three training seeds.

## Comparison at the same training budget

| Final measurement | SuperMimic | SuperTrack |
|---|---:|---:|
| Eden F1 | 0.4123 ± 0.2090 | 0.8490 ± 0.0705 |
| Precision | 0.7539 ± 0.1757 | 0.9124 ± 0.0253 |
| Recall | 0.3900 ± 0.1870 | 0.8421 ± 0.0778 |
| Two-note frames with both due pitches down (%) | 10.5 ± 12.7 | 69.4 ± 10.9 |
| Two-note frames with exactly the intended keys down (%) | 10.1 ± 12.6 | 65.4 ± 11.7 |
| Exact keyboard match across all frames (%) | 23.7 ± 11.9 | 71.7 ± 10.6 |
| Note events reached per episode (of 19) | 8.30 ± 7.83 | 17.76 ± 0.73 |
| Training time per seed (seconds) | 361.3 ± 19.1 | 323.1 ± 16.3 |
| Collected control transitions per training second | 283.9 ± 14.6 | 317.4 ± 15.6 |
| Paired attacks, within 50 ms | 50/3,072 (1.6%) | 1,374/3,072 (44.7%) |
| Episodes reaching every note | 0/768 | 169/768 |
| Episodes exact for the whole song | 0/768 | 0/768 |

A paired attack counts when both new presses fall within ±50 ms of their written onset and within 50 ms of one another. Held-note accuracy allows an already depressed note to carry across frames. Exact two-note accuracy also requires every other key to be up. [SuperTrack benchmark](ode_to_joy_results.md)

## Results by training seed

| Seed | Untrained F1 | Final F1 | Precision | Recall | Both down | Exact two-note frames | Timely attacks | All 19 reached | Training | Final evaluation | Time to .95 validation F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1304 | 0.4893 | 0.7865 | 0.4571 | 6.9% | 6.1% | 36/1,024 (3.5%) | 0/256 | 350.4 s | 64.7 s | Unreached |
| 1 | 0.1304 | 0.5719 | 0.9111 | 0.5342 | 24.7% | 24.1% | 14/1,024 (1.4%) | 0/256 | 383.4 s | 54.1 s | Unreached |
| 2 | 0.1304 | 0.1758 | 0.5642 | 0.1787 | 0.0% | 0.0% | 0/1,024 (0.0%) | 0/256 | 350.3 s | 53.9 s | Unreached |

Total timed training was **18.07 GPU-minutes**. The complete command took approximately 21.8 wall-clock minutes including scene initialization, untrained and trained evaluations, and checkpoint I/O. CUDA-synchronized training time includes validation after every round. Each final evaluation column measures the held-out physical simulation pass.

| Training component | Seconds per seed, mean ± SD |
|---|---:|
| Physics collection | 174.1 ± 2.5 |
| World model | 9.5 ± 0.2 |
| Actor and critic together | 26.9 ± 0.1 |
| Validation | 150.7 ± 16.6 |

## Training and evaluation protocol

- One NVIDIA GeForce RTX 3060 (12 GiB, driver 595.99.02) and Intel i7-12700F, with Python 3.12.16, PyTorch 2.13.0+cu129, MuJoCo 3.15.0, MuJoCo-Warp 3.15.0 and Warp 1.17.0. NDM Gym base commit was `173fd778456d2743ff03ff71861f5b6c96d4dda0` plus the local piano task adapter.
- Each policy controls 52 independent axes, with free six-axis wrists and 20 finger joints per hand. Its input contains measured actuator positions and their last changes, 88 key depths, the previous 52 actions, and 11 frames of the target keyboard. The actor has hidden widths 192/128. Both algorithms use the same learned piano dynamics, including the approximate fingertip geometry and the actuator and key residual networks. Physics runs at 200 Hz and the policy at 20 Hz.
- The task adapter passes real replay states and score-conditioned model steps to NDM Gym’s hand SuperMimic update. Its per-step reward is `1 − due-key error − 0.85 × largest unintended key depth − 0.20 × reach error`. Due-key error averages distance from full travel across the due pitches. Reach error averages the distance from each due key to the closest of the ten tips, in units of 35 mm. The weights are those of the SuperTrack run. The actor receives gradients through three model steps and the critic’s estimate of the remaining return. The last score frame terminates the return. Both hands can reach any pitch. Training uses neither finger assignments nor a motion demonstration. [NDM Gym actor and critic updates](../../ndm_gym/algos/hand_algos.py)
- Discount γ = 0.99, λ = 0.95, full gradients through the imagined states. The normalized value network has hidden widths 256/128. Each actor update fits the critic for five epochs, four minibatches of 72 per epoch, on the 288 imagined state targets. The critic and world-model learning rates were 0.0008 throughout. The actor used 0.0008 for the first ten rounds and decayed linearly to 0.00016. All three used fused Adam. Actor betas were (0.9, 0.95) and world-model and critic betas were (0.9, 0.999).
- Seeds 0, 1 and 2 each ran 20 rounds of 32 worlds × 160 collected control steps, or 102,400 real transitions per seed. Every round ran 90 two-step world-model updates and 50 three-step actor updates, both with batch size 96. This gives 1,800 world-model, 1,000 actor and 20,000 critic optimizer steps per seed. The replay buffer holds the six most recent collection rounds. After the first all-random round, 24 worlds explore around the policy and eight use random actions. Training starts vary within ±0.14 of each action scale.
- Final evaluation uses 256 fixed starts (RNG 789, one nominal, perturbation up to ±0.20 of each action scale) with a full unbroken deterministic policy playback per start. The time-to-target is the first of the 20 round-end validations (32 starts, RNG 456, ±0.15 of action scale) reaching mean F1 ≥ 0.95. Generalization here measures initial-pose variation on this single song. [PR #7 benchmark protocol](../quadrotor_racing_tracking/README.md)
- Eden’s framewise precision, recall and F1 compare all 88 physical key states with the score. A key counts as down at 87% travel. A zero denominator contributes 1 for the corresponding metric. The 230 frames include 164 with two pitches due, 36 with one and 30 with none. There are 19 note events and four written two-key attacks. Finger or pad contact does not enter the score. [Eden scorer](https://github.com/embodied-ai-nexus/Eden/blob/148d972beaee94b801c6a55808f00445bedb381b/eden/tasks/piano/custom_terms/metrics.py)

## What the hands played

The saved physical key masks separate missed pitches from extra presses. Counts per frame average over all frames and held-out starts, weighting each training seed equally.

| Measurement | SuperMimic | SuperTrack |
|---|---:|---:|
| Missing due pitches per frame | 1.112 ± 0.343 | 0.283 ± 0.132 |
| Extra pitches per frame | 0.373 ± 0.258 | 0.167 ± 0.043 |
| Frames with an extra pitch (%) | 28.1 ± 17.6 | 15.7 ± 3.2 |
| Rest frames with all keys released (%) | 94.1 ± 4.5 | 99.6 ± 0.3 |

## Validation and critic diagnostics

| Seed | Best validation F1 (round) | Final validation F1 | Final world-model loss | Final critic MSE | Final imagined reward per step |
|---|---:|---:|---:|---:|---:|
| 0 | 0.6434 (3) | 0.4938 | 0.4987 | 2.0874 | 0.0881 |
| 1 | 0.6313 (14) | 0.5776 | 0.5296 | 0.9529 | 0.1562 |
| 2 | 0.6048 (17) | 0.1760 | 0.4215 | 13.6036 | 0.0776 |

Losses and imagined reward above are from the last optimizer update of round 20. Physical keyboard metrics use the final checkpoints.

| Collection round | Real transitions per seed | SuperMimic validation F1 | SuperTrack validation F1 |
|---:|---:|---:|---:|
| 1 | 5,120 | 0.1725 ± 0.0728 | 0.3832 ± 0.0209 |
| 5 | 25,600 | 0.1537 ± 0.1476 | 0.5687 ± 0.0778 |
| 10 | 51,200 | 0.2899 ± 0.0892 | 0.7603 ± 0.1322 |
| 15 | 76,800 | 0.3454 ± 0.0622 | 0.8303 ± 0.0775 |
| 20 | 102,400 | 0.4158 ± 0.2118 | 0.8401 ± 0.0778 |

The following check uses the nominal physical rollout from each final checkpoint. At each score frame the critic sees the recorded state, previous action and score. The measured return sums the same continuous piano reward on the subsequent observed states, discounted by 0.99 to the end of the song. Positive bias means the critic forecast more reward than the policy earned. All four columns use reward units.

| Seed | Critic at song start | Measured discounted return at start | Mean critic bias across frames | Mean absolute critic error |
|---|---:|---:|---:|---:|
| 0 | 35.52 | 26.70 | +11.00 | 11.04 |
| 1 | 34.74 | 41.87 | +14.46 | 15.39 |
| 2 | 46.86 | -24.97 | +60.47 | 60.47 |

## Recorded rollout

The recording replays the final checkpoint of seed 1, selected by highest final validation F1. On the nominal start in a fresh process it scored **0.6181 F1**, reached **17/19** note events, matched **99/230** full keyboard frames, and played **0/4** paired attacks within the timing tolerance. The recorded audio uses the measured key onsets. The fresh nominal key mask differed from the original by 15 of 20,240 key/frame entries. The MP4 and WebM both decoded to 230 frames with audio.

- Left hand contacts on correctly depressed pitches were recorded on C3. The contacting links were left/l_thumb_distal, left/l_thumb_tip_sensor_frame.
- Right hand contacts on correctly depressed pitches were recorded on C4, D4, E4, F4, G4. The contacting links were right/r_middle_finger_distal, right/r_middle_finger_tip_sensor_frame.

52/52 commanded axes varied during the recorded episode. The following model prediction errors use the same replay and its recorded actions. Depth errors are fractions of full key travel.

| Open-loop horizon | All-key depth MAE | Due-key depth MAE |
|---|---:|---:|
| 1 (50 ms) | 0.0477 | 0.2143 |
| 3 (150 ms) | 0.0647 | 0.3313 |
| 10 (500 ms) | 0.0873 | 0.4695 |

To reproduce this recording, replay `supermimic/seed1/last.pt` with the benchmark runner and render its nominal trajectory.

## Verification

Precision, recall, F1 and two-note frame accuracy were independently recomputed from the saved 88-key masks. All three final checkpoints contain the trained critic. The hand and piano test suite passed 30 tests in the sandbox and skipped seven GPU-only checks there. Four additional tests of CUDA-graph training updates failed on this host in both the working tree and the unmodified base commit. This benchmark uses eager actor, critic and world-model updates (`update_graphs=False`).

## Reproduction

Run the training command with `--algo supermimic` and a fresh output directory. Each seed directory contains `history.json` with round timings and critic diagnostics, `results.json` with settings and held-out metrics, `evaluation.npz` with every measured mask and start, `rollout.npz` with nominal motion, and the best-validation and final checkpoints. Each checkpoint includes actor, critic and world-model weights. [Training, evaluation and replay commands](README.md#eden-ode-to-joy-with-52-independent-controls)
