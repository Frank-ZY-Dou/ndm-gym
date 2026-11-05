# Two G1s boxing: results

One two-person clip, InterHuman 1566 (11 s at 50 Hz, both people retargeted to the G1 in their common world frame),
tracked by two G1s in one simulation with robot-robot contacts on. All runs train on one A100-SXM4-80GB. A checkpoint is
evaluated five times from the clip's first frame, both robots at once. A pair succeeds when neither robot terminates; returns
and errors are means over the two robots. Global MPJPE is in the world frame; root-relative MPJPE takes each body relative to its
robot's root.

Setup:
- Data: [InterHuman](https://github.com/tr3e/InterGen) (CC BY-NC-SA 4.0, not redistributed), clip 1566: two people
  boxing (punches, dodges, feints, counters). Both people are retargeted to the 29-DoF G1 with HoloMotion's HoloRetarget
  (`tools/convert_interhuman.py`; it needs a licensed SMPL model), their horizontal positions scaled once by 0.810, each
  robot raised once so that its lowest foot point is on the floor.
- Scene (`--motions interhuman --paired`): two G1s per environment, each tracking its side with the LAFAN1 task; a pair
  resets and ends together; each robot's world model sees only its own state. 4096 robots in 2048 environments.
- Configurations: seed 0 throughout, MLP world models. LAFAN1's selections, with every speed option: SuperTrack `--algo
  supertrack --st_fail_w 5 --st_balance_w 4`; SuperMimic `--algo supermimic --reward_std_scale 0.7 --action_noise 0.3
  --imag_start_perturb 0.3`. The selections on the clip are below ([commands](../../README.md#two-g1s-one-two-person-clip)). The speed
  tables' RigidFormer is SuperTrack-RigidFormer's (`--algo supertrack --st_fail_w 5 --st_balance_w 2 --wm_backbone rigidformer
  --wm_lr 4e-4 --wm_batches 5`).

## Selected on the clip

Each algorithm's configuration on this clip comes from two rounds with the same number of runs, starting from LAFAN1's
selection and ranked by five evaluations of each run's best checkpoint. Every checkpoint is evaluated once on the learning curve,
which gives the best checkpoint and the time to return 0.88 (the selections: one checkpoint per 1,000 iterations).

| | SuperTrack | SuperMimic |
|---|---|---|
| configuration | LAFAN1's selection with `--st_com` and BlinkMimic's recipe | LAFAN1's selection with BlinkMimic's recipe and `--contact_reward 0.25 --ref_contact_fix --sm_residual --wm_structured` |
| best checkpoint, of the iterations in 1 h | 15,000 of 20,482 | 10,000 of 21,188 |
| pair success | 5 / 5 | 5 / 5 |
| normalized return | 0.895 | 0.906 |
| global MPJPE (mm) | 46 | 52 |
| root-relative MPJPE (mm) | 13.7 | 15.7 |
| training to return 0.88 (h) | 0.20 (2,000 iterations) | 0.05 (1,000 iterations) |
| last checkpoint: return, global, root-relative MPJPE (mm) | 0.895, 45, 13.7 | 0.843, 118, 34.6 |

On the learning curves, SuperTrack holds 0.888-0.895 from 3,000 iterations on. SuperMimic holds 0.901-0.906 from 4,000 to 14,000
iterations and falls to 0.83 at 1 h. BlinkMimic's recipe is built for short runs (its preset trains 3,500 iterations), so the
ranking uses each run's best checkpoint, for both algorithms.

Return, global and root-relative MPJPE (mm) of each run's best checkpoint, in bold each round's selection:

| round | SuperTrack | SuperMimic |
|---|---|---|
| start, LAFAN1's selection, 3 h | 0.891, 79, 12.2 | 0.895, 54, 20.0 |
| 1, 1 h | `--st_com`: 0.887, 69, 14.4 | the recipe: **0.906, 52, 15.7** |
| | `--st_com` with the recipe: **0.895, 46, 13.7** | the recipe with `--reward_std_scale 0.5`: 0.901, 61, 14.0 |
| 2, 0.3-0.5 h | `--pi_batch 4096`: 0.893, 39, 14.2 | `--pi_batch 4096`: 0.905, 53, 16.4 |
| | `--pi_lr_mult 2`: 0.894, 51, 13.6 | `--pi_lr_mult 2`: 0.905, 40, 16.4 |
| | `--st_balance_w 2`: 0.893, 45, 14.5 | `--action_noise 0.2`: 0.905, 50, 16.4 |

Round 1 selects the highest return, or within 0.005 the lower global MPJPE. Round 2 trains its changes for less time and keeps round
1's selections, since no change raises the return by more than 0.005.

## LAFAN1's selections, 3 h

Each trains for 3 h; its last checkpoint is evaluated five times.

| configuration | iterations in 3 h | it/s (world model only, then) | pair success | normalized return | global MPJPE (m) | root-relative MPJPE (mm) |
|---|---|---|---|---|---|---|
| SuperTrack, MLP | 31,428 | 4.90, 3.09 | 5 / 5 | 0.887 (0.885-0.888) | 0.106 (0.103-0.109) | 12.5 (12.5-12.5) |
| SuperMimic, MLP | 29,768 | 5.19, 2.90 | 5 / 5 | 0.886 (0.885-0.886) | 0.096 (0.094-0.101) | 20.0 (19.9-20.1) |

Means over the five evaluations, their range in parentheses. Per robot (normalized return): SuperTrack-MLP 0.881 and
0.892, SuperMimic-MLP 0.879 and 0.892. SuperTrack-MLP's evaluation ran on an A100 40GB, SuperMimic-MLP's on the A100 80GB
it trained on.

## Speed

Simulation throughput of the two-robot scene, robot-steps per second (a pair-step is two), one A100-SXM4-80GB that ran
nothing else (driver 565.57.01); 200 timed steps after 20 warm-up steps, the median of three repeats. Physics: Isaac Lab 3
with Newton/MuJoCo-Warp, the four 5 ms substeps in one call, robot-robot contacts on, no task terms. The world models
step each robot on its own predictions (`torch.compile`, bf16).

| pairs (robots) | physics | MLP world model | RigidFormer world model |
|---|---|---|---|
| 256 (512) | 80,880 | 2,031,353 (25.1x) | 533,037 (6.6x) |
| 1024 (2048) | 204,963 | 4,044,639 (19.7x) | 1,040,959 (5.1x) |
| 4096 (8192) | 312,231 | 7,208,713 (23.1x) | 1,345,609 (4.3x) |

The world models' inference paths, measured as the table on the same GPU: the
RigidFormer as an fp32 copy with the step's features and integration as Triton kernels and the whole network as one
kernel, the MLP as a bf16 copy with the fused step, with SuperTrack's trained checkpoints.

| pairs (robots) | physics | RigidFormer, as trained | RigidFormer, inference path | MLP, as trained | MLP, inference path |
|---|---|---|---|---|---|
| 256 (512) | 80,849 | 531,992 (6.6x) | 970,471 (12.0x) | 1,723,078 (21.3x) | 2,633,946 (32.6x) |
| 1024 (2048) | 205,021 | 1,041,206 (5.1x) | 1,628,768 (7.9x) | 3,467,402 (16.9x) | 5,228,805 (25.5x) |
| 4096 (8192) | 312,157 | 1,348,128 (4.3x) | 1,875,278 (6.0x) | 7,169,604 (23.0x) | 8,418,466 (27.0x) |
