# Two G1s boxing: results

One two-person clip, InterHuman 1566 (11 s at 50 Hz, both people retargeted to the G1 in their common world frame),
tracked by two G1s in one simulation with robot-robot contacts on. Each configuration trains for 3 h on
one A100-SXM4-80GB; its last checkpoint is evaluated five times from the clip's first frame, both robots at once.
A pair succeeds when neither robot terminates; returns and errors are means over the two robots.

Setup:
- Data: [InterHuman](https://github.com/tr3e/InterGen) (CC BY-NC-SA 4.0, not redistributed), clip 1566: two people
  boxing (punches, dodges, feints, counters). Both people are retargeted to the 29-DoF G1 with HoloMotion's HoloRetarget
  (`tools/convert_interhuman.py`; it needs a licensed SMPL model), their horizontal positions scaled once by 0.810, each
  robot raised once so that its lowest foot point is on the floor.
- Scene (`--motions interhuman --paired`): two G1s per environment, each tracking its side with the LAFAN1 task; a pair
  resets and ends together; each robot's world model sees only its own state. 4096 robots in 2048 environments.
- Configurations: README's final ones with every speed option, `--seed 0`. SuperTrack-MLP `--algo supertrack --st_fail_w 5
  --st_balance_w 4`; SuperTrack-RigidFormer `--algo supertrack --st_fail_w 5 --st_balance_w 2 --wm_backbone rigidformer
  --wm_lr 4e-4 --wm_batches 5`; SuperMimic-MLP `--algo supermimic --reward_std_scale 0.7 --action_noise 0.3
  --imag_start_perturb 0.3`.

## Tracking

| configuration | iterations in 3 h | it/s (world model only, then) | pair success | normalized return | global MPJPE (m) | local MPJPE (mm) |
|---|---|---|---|---|---|---|
| SuperTrack, MLP | 31,428 | 4.90, 3.09 | 5 / 5 | 0.887 (0.885-0.888) | 0.106 (0.103-0.109) | 12.5 (12.5-12.5) |
| SuperTrack, RigidFormer | 12,424 | 2.48, 1.21 | 5 / 5 | 0.839 (0.790-0.853) | 0.196 (0.156-0.312) | 23.7 (19.2-40.6) |
| SuperMimic, MLP | 29,768 | 5.19, 2.90 | 5 / 5 | 0.886 (0.885-0.886) | 0.096 (0.094-0.101) | 20.0 (19.9-20.1) |

Means over the five evaluations, their range in parentheses. Per robot (normalized return): SuperTrack-MLP 0.881 and
0.892, SuperTrack-RigidFormer 0.863 and 0.815, SuperMimic-MLP 0.879 and 0.892. SuperMimic ran with the MLP world model
only. SuperTrack-MLP's evaluation ran on an A100 40GB, the others' on the A100 80GB they trained on.

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
