# Two G1s boxing: results

Two G1s track InterHuman 1566 (11 s at 50 Hz) in one simulation with robot-robot contacts. The runs train on one
A100-SXM4-80GB, apart from the second and third training seeds of the selected configurations (A100-SXM4-40GB,
[below](#training-seeds-0-2)). A run's best checkpoint on its learning curve (one evaluation per checkpoint) is evaluated five
times from the clip's first frame, both robots at once. Success means
neither robot terminates; returns and errors average both robots. Global MPJPE uses the world frame; root-relative MPJPE
measures each body relative to its robot's root.

Setup:

- Data: [InterHuman](https://github.com/tr3e/InterGen) (CC BY-NC-SA 4.0, not redistributed), clip 1566: two people
  boxing (punches, dodges, feints, counters). Both people are retargeted to the 29-DoF G1 with HoloMotion's HoloRetarget
  (`tools/convert_interhuman.py`; it needs a licensed SMPL model), in their common world frame, horizontal positions scaled once by 0.810, each
  robot raised once so that its lowest foot point is on the floor.
- Scene (`--motions interhuman --paired`): two G1s per environment, each tracking its side with the LAFAN1 task; a pair
  resets and ends together; each robot's world model sees only its own state. 4096 robots in 2048 environments.
- Configurations: the rounds use seed 0 and MLP world models. LAFAN1's selections, with every speed option: SuperTrack `--algo
  supertrack --st_fail_w 5 --st_balance_w 4`; SuperMimic `--algo supermimic --reward_std_scale 0.7 --action_noise 0.3
  --imag_start_perturb 0.3`. The selections on the clip are below ([commands](../../README.md#two-g1s-one-two-person-clip)); they and
  their RigidFormer counterparts then train with [three seeds](#training-seeds-0-2). The speed
  tables' RigidFormer is SuperTrack-RigidFormer's (`--algo supertrack --st_fail_w 5 --st_balance_w 2 --wm_backbone rigidformer
  --wm_lr 4e-4 --wm_batches 5`).

## Selected on the clip

Two rounds with equal run counts start from LAFAN1's selections, those of its tuning rounds (SuperMimic's later 3 h
configuration on LAFAN1, with action noise 0.1, was not run on the clip). Five evaluations of each run's best checkpoint determine
the ranking. Learning curves use one evaluation per checkpoint to select that checkpoint (the selections: one checkpoint per
1,000 iterations).

| | SuperTrack | SuperMimic |
|---|---|---|
| configuration | LAFAN1's selection with `--st_com`, BlinkMimic's recipe and `--pi_batch 4096` | LAFAN1's selection with BlinkMimic's recipe, `--contact_reward 0.25 --ref_contact_fix --sm_residual --wm_structured` and `--pi_lr_mult 2` |
| training time of the selected run | 0.5 h | 0.3 h |
| best checkpoint, of the run's iterations | 7,000 of 9,396 | 6,264 of 6,264 |
| pair success | 5 / 5 | 5 / 5 |
| normalized return | 0.893 | 0.905 |
| global MPJPE (mm) | 39 | 40 |
| root-relative MPJPE (mm) | 14.2 | 16.4 |
| last checkpoint: return, global, root-relative MPJPE (mm) | 0.892, 57, 14.3 | 0.905, 40, 16.4 |

On the learning curves, the selected SuperTrack run holds 0.889-0.894 from 2,000 iterations on; the selected SuperMimic run reaches
0.905 at its end. Round 1's SuperTrack holds 0.888-0.895 from 3,000 iterations to 1 h; round 1's SuperMimic holds 0.901-0.906 from 4,000
to 14,000 iterations and falls to 0.83 at 1 h. Ranking uses each run's best checkpoint because BlinkMimic's preset trains for 3,500 iterations.
The time to return 0.88 is measured on its own, [below](#time-to-return-088).

Best-checkpoint return, global and root-relative MPJPE (mm); selections in bold:

| round | SuperTrack | SuperMimic |
|---|---|---|
| start, LAFAN1's selection, 3 h | 0.891, 79, 12.2 | 0.895, 54, 20.0 |
| 1, 1 h | `--st_com`: 0.887, 69, 14.4 | the recipe: **0.906, 52, 15.7** |
| | `--st_com` with the recipe: **0.895, 46, 13.7** | the recipe with `--reward_std_scale 0.5`: 0.901, 61, 14.0 |
| 2, 0.3-0.5 h | `--pi_batch 4096` (0.5 h): **0.893, 39, 14.2** | `--pi_batch 4096` (0.5 h): 0.905, 53, 16.4 |
| | `--pi_lr_mult 2` (0.3 h): 0.894, 51, 13.6 | `--pi_lr_mult 2` (0.3 h): **0.905, 40, 16.4** |
| | `--st_balance_w 2` (0.3 h): 0.893, 45, 14.5 | `--action_noise 0.2` (0.3 h): 0.905, 50, 16.4 |

Both rounds select the highest return, or within 0.005 of it the lowest global MPJPE. Round 2 compares its shorter runs with round 1's
selection.

## Training seeds 0-2

The two selections and their RigidFormer counterparts, SuperTrack-RF and SuperMimic-RF, three training seeds each. The
RigidFormer configurations start from round 1's selections without the MLP's options, use a small RigidFormer (`--rf_merge
joints --rf_dim 128 --rf_layers 2 --rf_heads 4 --rf_registers 2`) and were chosen on the clip with seed 0 in six further rounds
of two 0.5 h runs per algorithm, by the rule above (the MLP: the two rounds above). SuperTrack-RF's start has LAFAN1's
SuperTrack-RF balance weight (`--st_balance_w 2`) in place of round 1's 4; SuperMimic-RF's keeps round 1's kicks
(`--imag_start_perturb 0.3`; LAFAN1's SuperMimic-RF has 0.4). The four are named configurations:
`ndm-train --config two-g1-supertrack`, `two-g1-supertrack-rf`,
`two-g1-supermimic`, `two-g1-supermimic-rf`. All pairs succeed in every evaluation.

| configuration | return, seeds 0 / 1 / 2 | mean ± sd | global MPJPE (mm), seeds 0 / 1 / 2 | mean ± sd | root-relative MPJPE (mm), seeds 0 / 1 / 2 | mean ± sd |
|---|---|---|---|---|---|---|
| SuperTrack, MLP | 0.8932 / 0.8931 / 0.8913 | 0.8925 ± 0.0011 | 39.4 / 44.9 / 54.2 | 46.2 ± 7.5 | 14.2 / 13.8 / 14.2 | 14.1 ± 0.2 |
| SuperTrack-RF | 0.8919 / 0.8946 / 0.8948 | 0.8938 ± 0.0016 | 47.5 / 52.2 / 41.9 | 47.2 ± 5.2 | 14.8 / 14.5 / 14.7 | 14.7 ± 0.2 |
| SuperMimic, MLP | 0.9047 / 0.9052 / 0.9025 | 0.9042 ± 0.0015 | 40.1 / 45.3 / 57.0 | 47.5 ± 8.6 | 16.4 / 16.4 / 15.6 | 16.1 ± 0.4 |
| SuperMimic-RF | 0.9048 / 0.9029 / 0.9036 | 0.9038 ± 0.0010 | 43.0 / 47.4 / 47.1 | 45.8 ± 2.5 | 16.6 / 17.8 / 16.7 | 17.0 ± 0.7 |

How it was read: a run's best checkpoint on its learning curve (one checkpoint per 1,000 iterations and the last one, one
evaluation each) is evaluated five times from the clip's first frame; a cell is the mean of the five, and ± is the standard
deviation over the three seeds. Seed 0 trained on an A100-SXM4-80GB for 0.5 h (SuperMimic with the MLP 0.3 h, its best
checkpoint its last). Seeds 1 and 2 trained on A100-SXM4-40GB GPUs to seed 0's iteration counts (9,396, 4,373, 6,264 and
5,718, in the table's order), so training times are not compared across the seeds.

The RigidFormer pair's budget is equal in time, not in iterations: in 0.5 h SuperTrack-RF runs 4,373 iterations and
SuperMimic-RF 5,718, with two world-model batches per iteration against five. The six rounds' runs (seed 0, 0.5 h each):

| round | SuperTrack-RF | SuperMimic-RF |
|---|---|---|
| 1 | the small RigidFormer (kept); RF-S | the small RigidFormer (kept); RF-S |
| 2 | `--wm_batches 3`; `--st_fail_w 3` (selected) | `--wm_batches 3` (selected); `--pi_ema 0.99` |
| 3 | `--wm_lr 8e-4` (selected); `--wm_lr 1.6e-3` | `--wm_lr 8e-4`; `--wm_lr 1.6e-3` (selected) |
| 4 | `--wm_opt muon`; `--wm_opt muon --wm_lr 3.2e-3` | `--wm_opt muon`; `--wm_opt muon --wm_lr 3.2e-3` (selected) |
| 5 | `--st_fail_w 2`; `--st_fail_w 2 --wm_opt muon --wm_lr 3.2e-3` (selected) | `--contact_reward 0.15`; `--wm_batches 2` (selected) |
| 6 | `--wm_lr 3e-3`; `--wm_lr 3.5e-3` (selected) | `--wm_lr 3e-3` (selected); `--wm_lr 3.5e-3` |

Two world-model batches and the policy average 0.99 were tried for SuperMimic-RF only; the failure-penalty weights are
SuperTrack's own option, the foot-contact reward SuperMimic's.

### SuperTrack-MLP and SuperMimic-MLP at one budget

SuperMimic-MLP trained again to SuperTrack-MLP's 9,396 iterations (seeds 0-2, A100-SXM4-40GB), read as above: the same
number of collection steps and updates for both. Its 6,264-iteration rows above are its runs for 0.3 h.

| configuration | iterations | return, seeds 0 / 1 / 2 | mean ± sd | global MPJPE (mm), seeds 0 / 1 / 2 | mean ± sd | root-relative MPJPE (mm), seeds 0 / 1 / 2 | mean ± sd |
|---|---|---|---|---|---|---|---|
| SuperTrack, MLP | 9,396 | 0.8932 / 0.8931 / 0.8913 | 0.8925 ± 0.0011 | 39.4 / 44.9 / 54.2 | 46.2 ± 7.5 | 14.2 / 13.8 / 14.2 | 14.1 ± 0.2 |
| SuperMimic, MLP | 9,396 | 0.9053 / 0.9045 / 0.9067 | 0.9055 ± 0.0011 | 41.8 / 45.9 / 48.3 | 45.3 ± 3.3 | 16.7 / 15.8 / 15.5 | 16.0 ± 0.7 |

SuperTrack-MLP minus SuperMimic-MLP, Welch 95% intervals: return -0.0129 [-0.0154, -0.0105], global MPJPE +0.8 mm
[-15.0, +16.7], root-relative MPJPE -2.0 mm [-3.4, -0.5]. SuperMimic maximizes the return directly and SuperTrack
minimizes the pose errors directly; each leads on the measure it optimizes, and the global MPJPE is level.

## LAFAN1's selections, 3 h

Last checkpoints, evaluated five times:

| configuration | iterations in 3 h | it/s (world model only, then) | pair success | normalized return | global MPJPE (mm) | root-relative MPJPE (mm) |
|---|---|---|---|---|---|---|
| SuperTrack, MLP | 31,428 | 4.90, 3.09 | 5 / 5 | 0.887 (0.885-0.888) | 106 (103-109) | 12.5 (12.5-12.5) |
| SuperMimic, MLP | 29,768 | 5.19, 2.90 | 5 / 5 | 0.886 (0.885-0.886) | 96 (94-101) | 20.0 (19.9-20.1) |

Means over the five evaluations, their range in parentheses. Per robot (normalized return): SuperTrack-MLP 0.881 and
0.892, SuperMimic-MLP 0.879 and 0.892. SuperTrack-MLP's evaluation ran on an A100 40GB, SuperMimic-MLP's on the A100 80GB
it trained on.

## Time to return 0.88

Minutes of training from iteration 50 to the first checkpoint whose evaluations (the mean of two) return at least 0.88; seeds 0-2;
a checkpoint every 250 iterations, evaluated after training; A100 40GB SXM4 GPUs that ran nothing else. Times to the target, not
engineered.

| configuration | GPUs | minutes to 0.88, seeds 0 / 1 / 2 | mean ± sd | iterations to 0.88 | ms per iteration, 500-2000 |
|---|---|---|---|---|---|
| `two-g1-supertrack` | 1 | 3.66 / 3.66 / 3.66 | 3.66 ± 0.00 | 1,250, 1,250, 1,250 | 182.4 |
| `two-g1-supermimic` | 1 | 3.26 / 3.26 / 3.26 | 3.26 ± 0.00 | 1,250, 1,250, 1,250 | 162.1 |
| `two-g1-supertrack-3gpu` | 3 | 1.73 / 2.08 / 1.73 | 1.85 ± 0.21 | 1,250, 1,500, 1,250 | 86.3 |
| `two-g1-supermimic-3gpu` | 3 | 1.51 / 1.52 / 1.83 | 1.62 ± 0.18 | 1,250, 1,250, 1,500 | 75.3 |

At that checkpoint, from the same evaluations, mean ± sd over seeds 0-2:

| configuration | GPUs | return | global MPJPE (mm) | root-relative MPJPE (mm) |
|---|---|---|---|---|
| `two-g1-supertrack` | 1 | 0.8823 ± 0.0010 | 79.9 ± 4.6 | 16.8 ± 0.4 |
| `two-g1-supermimic` | 1 | 0.8886 ± 0.0042 | 67.8 ± 7.2 | 19.9 ± 1.4 |
| `two-g1-supertrack-3gpu` | 3 | 0.8840 ± 0.0004 | 72.2 ± 2.0 | 16.1 ± 0.5 |
| `two-g1-supermimic-3gpu` | 3 | 0.8861 ± 0.0021 | 81.4 ± 7.6 | 20.0 ± 0.4 |

Time per iteration by phase, seed 0, iterations 200-1000, ms:

| configuration | total | collection | world-model updates | policy update with the critic |
|---|---|---|---|---|
| `two-g1-supertrack` | 182.2 | 62.0 | 35.0 | 85.0 |
| `two-g1-supermimic` | 162.5 | 65.0 | 26.0 | 71.0 |

On three GPUs (`--layout AWL`) each device's steps, ms:

| configuration | total | actor: collection | world-model learner: updates | policy learner: update |
|---|---|---|---|---|
| `two-g1-supertrack-3gpu` | 86.3 | 48.9 | 35.1 | 84.9 |
| `two-g1-supermimic-3gpu` | 75.5 | 52.0 | 26.7 | 70.0 |

- Both reach 0.88 in the same iterations on average: SuperTrack with the lower root-relative MPJPE (16.8 against 19.9 mm on one
  GPU, 16.1 against 20.0 mm on three), SuperMimic sooner, as its iteration is shorter. There the global MPJPE is lower for
  SuperMimic on one GPU and for SuperTrack on three, with a wide spread between seeds, and the return is slightly higher for
  SuperMimic.
- On one GPU the policy update is the largest phase: 85 against 71 ms. SuperTrack's policy batch is twice SuperMimic's
  (`--pi_batch` 4,096 against 2,048).
- On three GPUs the collection and the world-model updates run beside the policy update, so the iteration is the policy update
  (84.9 and 70.0 ms): about half the time of one GPU, at 5.5 and 4.9 GPU-minutes against 3.66 and 3.26.
- The goal encoded once (`--goal_table --pi_encode_once`) computes each policy update's values up to float rounding; it shortens
  the update by 14 ms (SuperTrack) and 12 ms (SuperMimic), because the goal encoder runs once over all imagined steps.

## Speed

Robot-steps per second (a pair-step is two), one otherwise idle A100-SXM4-80GB (driver 565.57.01); median of three repeats
of 200 timed steps after 20 warm-up steps. Physics: Isaac Lab 3
with Newton/MuJoCo-Warp, the four 5 ms substeps in one call, robot-robot contacts on, no task terms. The world models
step each robot on its own predictions (`torch.compile`, bf16).

| pairs (robots) | physics | MLP world model | RigidFormer world model |
|---|---|---|---|
| 256 (512) | 80,880 | 2,031,353 (25.1x) | 533,037 (6.6x) |
| 1024 (2048) | 204,963 | 4,044,639 (19.7x) | 1,040,959 (5.1x) |
| 4096 (8192) | 312,231 | 7,208,713 (23.1x) | 1,345,609 (4.3x) |

Inference paths, measured the same way on the same GPU with SuperTrack's checkpoints: RigidFormer uses an fp32 copy,
Triton kernels for features and integration, and one kernel for the network. The MLP uses a bf16 copy with the fused step.

| pairs (robots) | physics | RigidFormer, as trained | RigidFormer, inference path | MLP, as trained | MLP, inference path |
|---|---|---|---|---|---|
| 256 (512) | 80,849 | 531,992 (6.6x) | 970,471 (12.0x) | 1,723,078 (21.3x) | 2,633,946 (32.6x) |
| 1024 (2048) | 205,021 | 1,041,206 (5.1x) | 1,628,768 (7.9x) | 3,467,402 (16.9x) | 5,228,805 (25.5x) |
| 4096 (8192) | 312,157 | 1,348,128 (4.3x) | 1,875,278 (6.0x) | 7,169,604 (23.0x) | 8,418,466 (27.0x) |
