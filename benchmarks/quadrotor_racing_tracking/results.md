# Quadrotor trajectory tracking: results

Contributed by [Zhengming Yu](https://github.com/Yzmblog) ([#7](https://github.com/Frank-ZY-Dou/ndm-gym/pull/7)).

SuperTrack and SuperMimic, with MLP world models, track flown trajectories of Race Against the Machine with a racing
quadrotor ([the task](../../docs/tasks/quadrotor.md), [the rules](README.md#rules-fixed-before-the-runs)).

## Setup

- **Reference:** the motion-captured flight of the dataset's drone (release v3.0.0), its part above 0.5 m, at 100 Hz: the
  autonomous flights, flown by the dataset's controller, and the piloted ones, flown by human pilots.
- **Training:** `ndm_gym.train_quadrotor` in 2,048 worlds with the systems options (`--fused_adam --reset_masked --no_sync
  --collect_graph --update_graphs`), one GPU per run: an A100 40GB SXM4 for
  [the named configurations](#the-named-configurations-on-an-a100-40gb-sxm4), an H200 for
  [the contributed study](#the-contributed-study-on-an-h200) (6,000 iterations with the learning rate's decay over them).
- **Evaluation:** `ndm_gym.evaluate_quadrotor` on the last checkpoint: 256 rollouts of the whole flight from perturbed
  starts (evaluation seed 0), without exploration noise and without resets. A rollout succeeds if it never terminates (the
  position within 1 m of the reference). Errors are means over the flight's frames and the rollouts; the largest position
  error is taken after the flight's first second (the perturbed start); mean ± sample SD over three training seeds.
  Reference's command, open loop: the same rollouts with the reference's feedforward thrust and body rates and no policy.
- **Data:** the flights are downloaded and converted by `tools/download_drone_racing.py` and are not in this repository.

| task | flight | frames (100 Hz) | length | top speed |
|---|---|---|---|---|
| `ellipse` | `flight-01a-ellipse`, two laps | 1,735 | 17.3 s | 21.7 m/s |
| `lemniscate` | `flight-07a-lemniscate`, two laps | 2,631 | 26.3 s | 10.0 m/s |
| `race_track` | `flight-13a-trackRATM`, three laps of seven gates over 5 m of height | 4,154 | 41.5 s | 21.2 m/s |
| `all` | the 18 autonomous flights, six of each shape | 49,297 | 492.8 s | 21.7 m/s |
| `ellipse_piloted` | `flight-01p-ellipse`, a human pilot's 14 laps | 10,208 | 102.1 s | 8.3 m/s |
| `lemniscate_piloted` | `flight-07p-lemniscate`, a human pilot's 12 laps | 10,365 | 103.6 s | 8.3 m/s |
| `race_track_piloted` | `flight-13p-trackRATM`, a professional pilot's 8 laps | 4,201 | 42.0 s | 19.3 m/s |

```bash
python tools/download_drone_racing.py --set both                # the autonomous and the piloted flights into data/drone_racing
# train and evaluate a task and seed: SuperTrack's most accurate configuration, then SuperMimic's
benchmarks/quadrotor_racing_tracking/run.sh <task> <seed> logs/quad_st --config quadrotor-supertrack-accurate
benchmarks/quadrotor_racing_tracking/run.sh <task> <seed> logs/quad_sm --config quadrotor-supermimic-accurate
# the time to the target: a fast configuration, then its checkpoints in order, the target held for 500 iterations
benchmarks/quadrotor_racing_tracking/run.sh <task> <seed> logs/quad_st_fast --config quadrotor-supertrack-fast
benchmarks/quadrotor_racing_tracking/time_to.sh logs/quad_st_fast/<task>/seed_<seed> 1500 500
# the tables
python benchmarks/quadrotor_racing_tracking/summarize.py SuperTrack=logs/quad_st SuperMimic=logs/quad_sm
```

## The named configurations, on an A100 40GB SXM4

The most accurate configurations (`quadrotor-supertrack-accurate`, `quadrotor-supermimic-accurate`) and the fast ones
(`quadrotor-supertrack-fast`, `quadrotor-supermimic-fast`), one A100 40GB SXM4 per run.

### The most accurate configurations

6,000 iterations with the learning rate's cosine over them; the last checkpoint evaluated as in [the setup](#setup), 256
perturbed starts per flight; mean ± SD over seeds 0-2 (on the three autonomous flights a checkpoint every 250 iterations
was also evaluated on the run's GPU while it trained).

| task | | seeds | success | position | position, largest after 1 s | rotation | velocity | training | iterations/s |
|---|---|---|---|---|---|---|---|---|---|
| `ellipse` | reference's command, open loop | | 0.0 % | (in the air for 1.5 s) | | | | | |
|  | SuperTrack, `quadrotor-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 3.06 ± 0.01 mm | 7 ± 0 mm | 2.10 ± 0.00 deg | 0.030 ± 0.000 m/s | 13.2 ± 0.0 min | 7.4 ± 0.0  |
|  | SuperMimic, `quadrotor-supermimic-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 5.71 ± 0.15 mm | 20 ± 1 mm | 1.94 ± 0.00 deg | 0.043 ± 0.000 m/s | 17.6 ± 0.0 min | 5.5 ± 0.0  |
| `lemniscate` | reference's command, open loop | | 0.0 % | (in the air for 1.5 s) | | | | | |
|  | SuperTrack, `quadrotor-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 1.89 ± 0.01 mm | 5 ± 0 mm | 3.25 ± 0.00 deg | 0.018 ± 0.000 m/s | 13.3 ± 0.0 min | 7.3 ± 0.0  |
|  | SuperMimic, `quadrotor-supermimic-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 3.95 ± 0.09 mm | 22 ± 0 mm | 3.17 ± 0.00 deg | 0.027 ± 0.000 m/s | 17.8 ± 0.0 min | 5.4 ± 0.0  |
| `race_track` | reference's command, open loop | | 0.0 % | (in the air for 1.5 s) | | | | | |
|  | SuperTrack, `quadrotor-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 1.98 ± 0.01 mm | 28 ± 0 mm | 1.37 ± 0.00 deg | 0.034 ± 0.000 m/s | 13.4 ± 0.1 min | 7.3 ± 0.0  |
|  | SuperMimic, `quadrotor-supermimic-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 4.08 ± 0.05 mm | 31 ± 1 mm | 1.31 ± 0.00 deg | 0.043 ± 0.000 m/s | 17.8 ± 0.0 min | 5.4 ± 0.0  |
| `all` | reference's command, open loop | | 0.0 % | (in the air for 1.5 s) | | | | | |
|  | SuperTrack, `quadrotor-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 2.31 ± 0.00 mm | 12 ± 0 mm | 2.27 ± 0.00 deg | 0.028 ± 0.000 m/s | 13.4 ± 0.0 min | 7.2 ± 0.0  |
|  | SuperMimic, `quadrotor-supermimic-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 4.94 ± 0.12 mm | 23 ± 0 mm | 2.15 ± 0.00 deg | 0.038 ± 0.000 m/s | 18.0 ± 0.2 min | 5.4 ± 0.0  |
| `ellipse_piloted` | reference's command, open loop | | 0.0 % | (in the air for 1.6 s) | | | | | |
|  | SuperTrack, `quadrotor-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 1.11 ± 0.01 mm | 28 ± 0 mm | 2.77 ± 0.00 deg | 0.016 ± 0.000 m/s | 13.4 ± 0.1 min | 7.2 ± 0.0  |
|  | SuperMimic, `quadrotor-supermimic-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 4.48 ± 0.04 mm | 66 ± 3 mm | 2.62 ± 0.00 deg | 0.032 ± 0.000 m/s | 17.8 ± 0.1 min | 5.4 ± 0.0  |
| `lemniscate_piloted` | reference's command, open loop | | 0.0 % | (in the air for 1.6 s) | | | | | |
|  | SuperTrack, `quadrotor-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 0.99 ± 0.00 mm | 8 ± 0 mm | 2.40 ± 0.00 deg | 0.015 ± 0.000 m/s | 13.2 ± 0.1 min | 7.3 ± 0.0  |
|  | SuperMimic, `quadrotor-supermimic-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 4.00 ± 0.05 mm | 23 ± 1 mm | 2.28 ± 0.00 deg | 0.028 ± 0.000 m/s | 17.6 ± 0.1 min | 5.5 ± 0.0  |
| `race_track_piloted` | reference's command, open loop | | 0.0 % | (in the air for 1.3 s) | | | | | |
|  | SuperTrack, `quadrotor-supertrack-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 2.75 ± 0.03 mm | 55 ± 1 mm | 3.33 ± 0.01 deg | 0.060 ± 0.000 m/s | 13.4 ± 0.1 min | 7.3 ± 0.0  |
|  | SuperMimic, `quadrotor-supermimic-accurate` | 0, 1, 2 | 100.0 ± 0.0 % | 6.63 ± 0.06 mm | 96 ± 1 mm | 3.00 ± 0.00 deg | 0.086 ± 0.000 m/s | 17.9 ± 0.1 min | 5.4 ± 0.0  |

| configuration | options beside the systems options and `--iterations 6000 --lr_decay_iters 6000` |
|---|---|
| `quadrotor-supertrack-accurate` | `--algo supertrack --lr 8e-4 --st_fail_w 2 --st_task_weights --horizon 100 --wm_hidden 256 256 256` and HOI's optimizer options, `--pi_opt muon --wm_opt muon --pi_ema 0.995 --pi_lr_mult 2 --lr_decay_ret 0.95 --lr_decay_by 1400 --lr_decay_len 300` |
| `quadrotor-supermimic-accurate` | `--algo supermimic --lr 4e-4 --horizon 100 --wm_hidden 256 256 256` and HOI's optimizer options |

No rollout of any run terminates. SuperTrack's three seeds end within 0.07 mm of each other on every task, SuperMimic's
within 0.31 mm. SuperMimic has the lower rotation error on all seven tasks (1.31 to 3.17 against 1.37 to 3.33 deg): where
the reference's attitude and its acceleration do not agree for the model, a policy trades the one error for the other
([the drone](../../docs/tasks/quadrotor.md#the-drone)).

**The 18 flights' policy on the pilots' flights, zero-shot.** The policy trained on all 18 autonomous flights (`all`), flown on
the three piloted flights it never saw (mean ± SD over its training seeds 0-2, each seed in brackets; 256 perturbed starts):

| | ellipse, piloted | lemniscate, piloted | race track, piloted |
|---|---|---|---|
| success, SuperTrack / SuperMimic | 100 / 100 % | 100 / 100 % | 100 / 100 % |
| position, SuperTrack | 4.74 ± 1.35 mm (3.58 / 6.22 / 4.43) | 4.08 ± 1.41 mm (3.12 / 5.70 / 3.44) | 18.09 ± 2.32 mm (15.42 / 19.41 / 19.45) |
| position, SuperMimic | 11.05 ± 0.97 mm (11.90 / 9.99 / 11.26) | 10.85 ± 0.53 mm (10.49 / 10.62 / 11.46) | 35.73 ± 1.29 mm (34.52 / 35.59 / 37.09) |
| position, largest after 1 s, SuperTrack / SuperMimic | 23 / 36 mm | 20 / 52 mm | 88 / 125 mm |

No rollout terminates. The professional pilot's race track, faster than the laps the policy trained on, costs both the most.

### The time to the target

The training time from which a run keeps a success of at least 0.99 with a mean position error below 20 mm: the trainer's
own clock at the first checkpoint from which every checkpoint for the next 500 iterations meets the target. A checkpoint
every 50 iterations, each evaluated from 64 perturbed starts on the run's GPU while it trains (`time_to.sh <run> 1500 500`
evaluates them afterwards); warm compile caches. Time to the target, not engineered. Mean ± SD over seeds 0-2, each seed
in brackets:

| | ellipse | lemniscate | race track | the target from iteration | iterations/s |
|---|---|---|---|---|---|
| SuperTrack, `quadrotor-supertrack-fast` | 0.38 ± 0.04 min (0.36 / 0.43 / 0.35) | 0.32 ± 0.02 min (0.31 / 0.34 / 0.31) | 0.36 ± 0.00 min (0.36 / 0.37 / 0.36) | 450-500 | 19.4 |
| SuperMimic, `quadrotor-supermimic-fast` | 0.71 ± 0.03 min (0.75 / 0.69 / 0.69) | 0.69 ± 0.02 min (0.71 / 0.68 / 0.67) | 0.67 ± 0.00 min (0.67 / 0.67 / 0.67) | 600-650 | 10.9 |

Every run keeps the target from the first checkpoint that meets it. The position error there, mean over the seeds:
16.3 / 13.4 / 15.2 mm (SuperTrack) and 18.2 / 13.3 / 18.6 mm (SuperMimic).

| configuration | options beside the systems options and `--iterations 1500 --lr_decay_iters 6000 --ckpt_every 50 --ckpt_keep` |
|---|---|
| `quadrotor-supertrack-fast` | `--algo supertrack --lr 4e-4 --st_fail_w 2 --st_task_weights --horizon 64 --wm_hidden 256 256 256 --pi_opt muon --wm_opt muon --pi_ema 0 --pi_lr_mult 2 --lr_decay_ret 0.95 --lr_decay_by 1400 --lr_decay_len 300 --pi_batch 1024 --wm_batch 1024` |
| `quadrotor-supermimic-fast` | `--algo supermimic --lr 4e-4 --horizon 64 --wm_hidden 256 256 256 --pi_opt muon --wm_opt muon --pi_batch 2048` |

### What each option does

Both algorithms were tried with the same changes: the imagined window, the learning rate, the batches, the world model's
width, HOI's optimizer options together and one at a time, and for accuracy two options of each algorithm's own. The
comparisons below are on the three autonomous flights, each option against the configuration without it; "nine runs" are
seeds 0-2 on each flight.

- **The policy average costs time while the policy improves.** `--pi_ema 0.995` evaluates an exponential average of the
  policy's weights, which trails the trained policy by about 1/(1 - 0.995) = 200 updates. The target is met about that
  much later: SuperTrack holds it from iteration 650-800 with the average and from 450-500 without (0.93 ± 0.09 against
  0.48 ± 0.04 min, nine runs each, at the default batches); SuperMimic with it first meets it at iteration 800-900
  instead of 600-650 (seed 0). At the end of 6,000 iterations it costs nothing: on the last checkpoints of the most
  accurate configurations (nine runs each) the trained policy's error differs from the average's by +0.003 mm (SuperTrack)
  and -0.008 mm (SuperMimic) on average, per run by at most 0.02 and 0.10 mm, either way.
- **Muon keeps the target once met.** With Muon for both networks (`--pi_opt muon --wm_opt muon`) 0 of 9 SuperMimic runs
  lose the target after meeting it, against 5 of 9 with Adam: it holds it from 0.86 ± 0.08 min against 1.55 ± 0.75 min,
  though it first meets it no sooner (0.86 against 0.84 ± 0.11 min, iterations 550-650). With the policy batch of 2,048, 0
  of 9 against 7 of 9 (held from 0.69 ± 0.03 against 1.29 ± 0.45 min). SuperTrack's time changes
  little with Adam in Muon's place (0.87 against 0.91 min, seed 0). At the end of 6,000 iterations Muon lowers both errors:
  SuperMimic's from 6.16 to 5.59 mm with Muon alone, SuperTrack's is 2.84 mm with Muon and 2.97 mm with Adam in its place
  (seed 0, the mean of the ellipse and the race track).
- **A longer window counts more of a correction.** A body rate moves the position only through the attitude and then the
  acceleration, so a correction's effect on the position keeps growing after it. Over 100 imagined steps (1 s) instead of
  64 (0.64 s) the policy's objective counts more of it, and at the end of 6,000 iterations SuperTrack's error falls from
  2.57 to 2.51 mm and SuperMimic's from 5.66 to 5.05 mm (the mean of nine runs each), at about 1.4 times the training
  time. With 16 or 32 steps SuperTrack, which has no critic to count what follows its window, does not learn the task (no
  rollout succeeds); SuperMimic's critic counts it, and it learns with them, less accurately (11.55 and 7.73 mm against
  6.16 mm with 64; seed 0, the ellipse and the race track).
- **Smaller batches make an iteration shorter.** An iteration is mostly the policy update through the imagined window,
  whose cost grows with the policy's batch of imagined rollouts ([the parts](#an-iteration-of-the-fast-configurations)).
  With batches of 1,024 for the policy and the world model SuperTrack runs 19.4 instead of 10.4 iterations/s and holds the
  target from the same iterations, 450-500; SuperMimic with a policy batch of 2,048 runs 10.9 instead of 7.8 iterations/s
  and holds it from 600-650 instead of 550-650 (with Muon, nine runs each).
- **A doubled learning rate.** Under the same cosine over 6,000 iterations, a peak of 8e-4 instead of 4e-4 lowers
  SuperTrack's error from 2.57 to 2.42 mm (nine runs), and with the 100-step window to 2.31 mm. SuperMimic's falls with it
  from 6.16 to 5.57 mm (seed 0, the ellipse and the race track), as with the window (5.56 mm) and HOI's optimizer options
  (5.49 mm), whose pair gives its most accurate configuration (4.58 mm, nine runs).

### An iteration of the fast configurations

`QUAD_PROFILE=1` (the device synchronized around each part), 1,000 iterations on the ellipse, medians after iteration 400:

| | collection | world-model updates | policy update | iterations/s |
|---|---|---|---|---|
| SuperTrack, `quadrotor-supertrack-fast` | 2.6 ms | 12.5 ms | 35.3 ms | 19.8 |
| SuperMimic, `quadrotor-supermimic-fast` | 2.6 ms | 18.1 ms | 69.9 ms | 11.0 |

SuperMimic's policy update has twice SuperTrack's batch and also trains its critic. The timed runs, whose checkpoints are
evaluated on the same GPU while they train, run 19.4 and 10.9 iterations/s.

## The contributed study, on an H200

The pull request's configurations, selected on the ellipse ([selection](#selection)) and run on one H200 per run.

| task | | seeds | success | position | position, largest after 1 s | rotation | velocity | training | iterations/s |
|---|---|---|---|---|---|---|---|---|---|
| `ellipse` | reference's command, open loop | | 0.0 % | (in the air for 1.5 s) | | | | | |
|  | SuperTrack, the hand task's options | 0, 1, 2 | 0.0 ± 0.0 % |  |  |  |  | 3.2 ± 0.1 min | 32.6 ± 1.1  |
|  | SuperTrack, selected | 0, 1, 2 | 100.0 ± 0.0 % | 3.38 ± 0.03 mm | 8 ± 0 mm | 2.10 ± 0.00 deg | 0.031 ± 0.000 m/s | 5.6 ± 0.0 min | 18.3 ± 0.0  |
|  | SuperMimic, the hand task's options | 0, 1, 2 | 100.0 ± 0.0 % | 15.90 ± 1.22 mm | 35 ± 2 mm | 2.26 ± 0.09 deg | 0.058 ± 0.003 m/s | 3.8 ± 0.1 min | 27.0 ± 1.1  |
|  | SuperMimic, selected | 0, 1, 2 | 100.0 ± 0.0 % | 6.55 ± 0.39 mm | 22 ± 3 mm | 1.97 ± 0.02 deg | 0.045 ± 0.000 m/s | 7.0 ± 0.2 min | 14.1 ± 0.5  |
| `lemniscate` | reference's command, open loop | | 0.0 % | (in the air for 1.4 s) | | | | | |
|  | SuperTrack, the hand task's options | 0, 1, 2 | 0.0 ± 0.0 % |  |  |  |  | 3.2 ± 0.2 min | 32.7 ± 1.4  |
|  | SuperTrack, selected | 0, 1, 2 | 100.0 ± 0.0 % | 2.02 ± 0.02 mm | 6 ± 0 mm | 3.26 ± 0.00 deg | 0.018 ± 0.000 m/s | 5.8 ± 0.2 min | 17.4 ± 0.8  |
|  | SuperMimic, the hand task's options | 0, 1, 2 | 100.0 ± 0.0 % | 12.62 ± 1.25 mm | 80 ± 9 mm | 3.40 ± 0.02 deg | 0.045 ± 0.004 m/s | 3.8 ± 0.0 min | 27.6 ± 1.1  |
|  | SuperMimic, selected | 0, 1, 2 | 100.0 ± 0.0 % | 4.96 ± 0.03 mm | 20 ± 1 mm | 3.18 ± 0.01 deg | 0.027 ± 0.000 m/s | 7.1 ± 0.2 min | 14.1 ± 0.6  |
| `race_track` | reference's command, open loop | | 0.0 % | (in the air for 1.4 s) | | | | | |
|  | SuperTrack, the hand task's options | 0, 1, 2 | 0.0 ± 0.0 % |  |  |  |  | 3.3 ± 0.1 min | 31.9 ± 0.1  |
|  | SuperTrack, selected | 0, 1, 2 | 100.0 ± 0.0 % | 2.19 ± 0.02 mm | 29 ± 0 mm | 1.38 ± 0.00 deg | 0.035 ± 0.000 m/s | 5.9 ± 0.1 min | 16.9 ± 0.0  |
|  | SuperMimic, the hand task's options | 0, 1, 2 | 100.0 ± 0.0 % | 16.92 ± 3.73 mm | 62 ± 13 mm | 1.80 ± 0.14 deg | 0.070 ± 0.004 m/s | 3.7 ± 0.1 min | 27.6 ± 1.1  |
|  | SuperMimic, selected | 0, 1, 2 | 100.0 ± 0.0 % | 5.08 ± 0.16 mm | 30 ± 1 mm | 1.39 ± 0.07 deg | 0.045 ± 0.000 m/s | 7.3 ± 0.1 min | 13.5 ± 0.0  |

Every seed is a training of its own (other initial networks, other logs); SuperTrack's three end within 0.05 mm of each
other on a task. The rotation error is about the same for both algorithms and all seeds: the policy gives the rotation up
for the position where the reference's attitude and its acceleration do not agree for the model (the motion-captured body
axis is a few degrees off the thrust that the acceleration needs; [the drone](../../docs/tasks/quadrotor.md#the-drone)).

| configuration | options beside the systems options and `--iterations 6000 --lr_decay_iters 6000` |
|---|---|
| SuperTrack, the hand task's options | `--algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights` (16 imagined steps, a world model of 3 × 1024) |
| SuperTrack, selected | `--algo supertrack --lr 4e-4 --st_fail_w 2 --st_task_weights --horizon 64 --wm_hidden 256 256 256 --pi_opt muon --wm_opt muon --pi_ema 0.995 --pi_lr_mult 2 --lr_decay_ret 0.95 --lr_decay_by 1400 --lr_decay_len 300` |
| SuperMimic, the hand task's options | none (16 imagined steps, a learning rate of 2e-4, a world model of 3 × 1024) |
| SuperMimic, selected | `--algo supermimic --lr 4e-4 --horizon 64 --wm_hidden 256 256 256` |

**SuperTrack with the hand task's options does not learn the task**: every rollout terminates (in the air for 3.6 s on the
ellipse; its training error stays at 0.43 m). Its 16 imagined steps are 0.16 s here. A body rate moves the position only
through the attitude and the acceleration, so within that window correcting a position error costs more rotation and
velocity error than it gains; SuperTrack has no critic to count what follows the window. With 64 steps it tracks to 2 to
3.4 mm. SuperMimic, whose critic does, learns with 16 steps and is 2.4 to 3.3 times more accurate with 64.

### One policy on all 18 flights

| task | | seeds | success | position | position, largest after 1 s | rotation | velocity | training | iterations/s |
|---|---|---|---|---|---|---|---|---|---|
| `all` | reference's command, open loop | | 0.0 % | (in the air for 1.5 s) | | | | | |
|  | SuperTrack, selected | 0, 1, 2 | 100.0 ± 0.0 % | 2.54 ± 0.02 mm | 12 ± 0 mm | 2.27 ± 0.00 deg | 0.028 ± 0.000 m/s | 5.8 ± 0.0 min | 16.9 ± 0.0  |
|  | SuperMimic, selected | 0, 1, 2 | 100.0 ± 0.0 % | 6.22 ± 0.34 mm | 24 ± 0 mm | 2.19 ± 0.01 deg | 0.040 ± 0.000 m/s | 7.2 ± 0.1 min | 13.8 ± 0.6  |

Per shape, the policy on all flights against the policy on one flight of the shape (mean over the shape's six flights and
the seeds): SuperTrack 2.97 mm on the ellipses (3.38 mm on one), 2.58 mm on the lemniscates (2.02 mm) and 2.08 mm on the
race tracks (2.19 mm); SuperMimic 7.41 (6.55), 5.83 (4.96) and 5.40 mm (5.08 mm). The hardest single flight of the 18 is
tracked to 3.3 mm (SuperTrack) and 7.5 to 8.3 mm (SuperMimic), and no rollout of any flight terminates. These runs are the
selected configurations as they are.

Four of the race track's flights have frames that the motion capture dropped (gaps of 51 to 84 ms; one flight has six).
The loader fills them in before it smooths the poses (`ndm_gym/motion/sources/drone_racing.py`); a filter over the
samples as they are bends the path at a gap, and gave one flight eight frames at up to 34 m/s. The three tasks' flights have
no gaps.

### The piloted flights

The same configurations, without a selection of their own, on the first piloted flight of each shape. The pilots' lines
differ on every lap and dip to 0.1 m over the floor; the professional pilot's race track reaches 9 g, and at its peaks the
thrust the reference needs (up to 77 N) exceeds the model's largest (64 N): the feedforward command is clamped there, and
the policy's thrust cannot exceed it either.

| task | | seeds | success | position | position, largest after 1 s | rotation | velocity | training | iterations/s |
|---|---|---|---|---|---|---|---|---|---|
| `ellipse_piloted` | reference's command, open loop | | 0.0 % | (in the air for 1.6 s) | | | | | |
|  | SuperTrack | 0, 1, 2 | 100.0 ± 0.0 % | 1.34 ± 0.01 mm | 27 ± 0 mm | 2.76 ± 0.00 deg | 0.018 ± 0.000 m/s | 5.9 ± 0.0 min | 16.9 ± 0.0  |
|  | SuperMimic | 0, 1, 2 | 100.0 ± 0.0 % | 5.51 ± 0.26 mm | 62 ± 6 mm | 2.67 ± 0.03 deg | 0.034 ± 0.001 m/s | 7.3 ± 0.1 min | 13.4 ± 0.1  |
| `lemniscate_piloted` | reference's command, open loop | | 0.0 % | (in the air for 1.6 s) | | | | | |
|  | SuperTrack | 0, 1, 2 | 100.0 ± 0.0 % | 1.20 ± 0.00 mm | 9 ± 0 mm | 2.39 ± 0.00 deg | 0.016 ± 0.000 m/s | 5.8 ± 0.2 min | 17.3 ± 0.9  |
|  | SuperMimic | 0, 1, 2 | 100.0 ± 0.0 % | 4.71 ± 0.11 mm | 24 ± 1 mm | 2.31 ± 0.02 deg | 0.029 ± 0.000 m/s | 7.2 ± 0.1 min | 14.1 ± 0.5  |
| `race_track_piloted` | reference's command, open loop | | 0.0 % | (in the air for 1.3 s) | | | | | |
|  | SuperTrack | 0, 1, 2 | 100.0 ± 0.0 % | 3.08 ± 0.03 mm | 52 ± 1 mm | 3.32 ± 0.00 deg | 0.062 ± 0.000 m/s | 5.9 ± 0.1 min | 17.3 ± 0.8  |
|  | SuperMimic | 0, 1, 2 | 100.0 ± 0.0 % | 8.13 ± 0.27 mm | 98 ± 6 mm | 3.04 ± 0.01 deg | 0.090 ± 0.002 m/s | 7.2 ± 0.2 min | 14.1 ± 0.5  |

The pilots' flights are tracked as closely as the controller's; the errors' largest values are at the race track's peaks.

**The autonomous flights' policy on the pilots' flights, zero-shot.** The policy trained on all 18 autonomous flights
(`all`), flown on the three piloted flights it never saw (mean ± SD over its training seeds 0-2; 256 perturbed starts):

| | ellipse, piloted | lemniscate, piloted | race track, piloted |
|---|---|---|---|
| success, SuperTrack / SuperMimic | 100 / 100 % | 100 / 100 % | 100 / 100 % |
| position, SuperTrack | 5.90 ± 1.71 mm | 5.46 ± 1.21 mm | 22.33 ± 3.46 mm |
| position, SuperMimic | 12.40 ± 1.50 mm | 13.24 ± 1.64 mm | 38.14 ± 4.24 mm |
| position, largest after 1 s, SuperTrack / SuperMimic | 31 / 50 mm | 24 / 60 mm | 120 / 131 mm |

No rollout terminates. The policies are conditioned on the reference's frames ahead and on its feedforward command, so a
line they never saw is a new sequence of goals, not a new task; the professional pilot's race track, faster and harder than
the controller's laps the policies trained on, costs them the most.

## Selection

On the ellipse with training seed 0, 6,000 iterations: among the configurations whose 256 rollouts all succeed, the lowest
position error (all of these succeed). The three SuperTrack configurations with 64 steps are within the seeds' spread of
each other; the rule takes the lowest.

| | configuration | position | training |
|---|---|---|---|
| SuperTrack | `--horizon 64` | 3.8 mm | 5.8 min |
| | `--horizon 64 --st_imag_noise 0.2` | 3.8 mm | 5.7 min |
| | `--horizon 100` | 3.6 mm | 8.0 min |
| | **`--horizon 64` with HOI's optimizer options** (Muon, the policy average, the learning-rate ramp) | **3.4 mm** | 5.8 min |
| SuperMimic | `--horizon 32` | 8.6 mm | 4.5 min |
| | `--horizon 32 --reward_std_scale 0.5` | 8.8 mm | 4.7 min |
| | `--horizon 32 --action_noise 0.3 --bptt_decay 0.95` with HOI's optimizer options | 14.7 mm | 4.5 min |
| | **`--horizon 64`** | **7.0 mm** | 6.8 min |

HOI's optimizer options were tried at 64 steps for SuperTrack only, where they are part of its selected configuration;
SuperMimic's one run with them had 32 steps and two more changes, and its selected configuration is without them.

Before the selection, in runs of 4,000 iterations on the ellipse: SuperTrack with calibrated term weights (the default
without `--st_task_weights`) stays at 0.33 m with 64 steps, the body rates' term, whose first error is small, holding the
policy at the feedforward command; with the task's weights and 32 steps it stays at 0.31 m; a world model of 3 × 256 in the
place of 3 × 1024 changes SuperMimic's error from 13.0 to 14.1 mm (one seed each).

## Training time

The training time at which a run first has a success of at least 0.99 with a mean position error below 20 mm
(`time_to.sh`: its checkpoints every 250 iterations, each evaluated from 64 perturbed starts), mean ± SD over the training
seeds 0-2, on one H200:

| | ellipse | lemniscate | race track |
|---|---|---|---|
| SuperTrack, selected | 0.79 ± 0.17 min | 0.68 ± 0.09 min | 0.75 ± 0.15 min |
| SuperMimic, selected | 0.93 ± 0.17 min | 0.78 ± 0.10 min | 0.89 ± 0.14 min |

Every run reaches it at iteration 750 or 1,000.

### Where an iteration's time goes

`QUAD_PROFILE=1`, the device synchronized around each part, on the ellipse on an H200 with the systems options:

| | collection | world-model updates | policy update |
|---|---|---|---|
| SuperMimic, 32 steps, world model of 3 × 1024 (the hand task's) | 2.4 ms | 15.7 ms | 49.4 ms |
| world model of 3 × 256 | 2.2 ms | 10.6 ms | 45.1 ms |
| the goal's frames in one batch of operations, the features of a step reused by the next | 2.2 ms | 10.6 ms | 32.1 ms |
| policy and world-model batches of 1,024 (not used in the results) | 2.2 ms | 9.3 ms | 18.0 ms |
| SuperTrack, 64 steps, world model of 3 × 256 | 2.3 ms | 10.3 ms | 51.2 ms |
| the goal's frames in one batch of operations | 2.3 ms | 10.3 ms | 45.8 ms |

The updates are bound by the number of small operations in an imagined step, not by the networks' size: the smaller
world model alone saves little. The goal (five reference frames about the state) was computed frame by frame, and
SuperMimic computed it twice per step, for the policy and for the critic; both now compute it once, in one batch of
operations, with the same values.

## The world model as the simulator

A checkpoint's world model against MuJoCo-Warp (`ndm_gym.wmbench.speed_quadrotor`, 4,096 worlds on an H200, the ellipse's
seed 0): control steps per second, each a CUDA graph, and the open-loop error of the world model rolled out from a
simulator state under the simulator's commands, with the policy's exploration noise.

| | MuJoCo-Warp | world model, fp32 | world model, bf16 |
|---|---|---|---|
| SuperTrack's acceleration model | 4.07 M steps/s | 13.2 M steps/s (3.2x) | 12.4 M steps/s |
| SuperMimic's increment model | 3.87 M steps/s | 12.2 M steps/s (3.2x) | 11.4 M steps/s |

| position error after | 1 step | 8 | 32 | 64 | 100 steps (1 s) |
|---|---|---|---|---|---|
| SuperTrack's acceleration model | 0.02 mm | 0.22 mm | 1.41 mm | 4.04 mm | 8.46 mm |
| SuperMimic's increment model | 1.78 mm | 4.03 mm | 6.12 mm | 10.77 mm | 20.17 mm |

The simulator is one rigid body without contacts, four physics steps per control step: there is little for a world model
to save, and bf16 does not help a network of this size. The acceleration model integrates the position from the velocity,
as the simulator does; the increment model predicts it. The two models also differ in their training: each is the world
model of its algorithm's run on the ellipse, SuperTrack's trained with Muon and SuperMimic's with Adam, each on its own
policy's data, so the open-loop errors do not compare the parameterizations alone.

On an A100 40GB SXM4, with the step fused: [results_a100.md](results_a100.md#the-world-model-as-the-simulator).

## The model's estimates

The drone's inertia and drag are estimates ([the drone](../../docs/tasks/quadrotor.md#the-drone)). SuperTrack's selected
policies of the ellipse, trained on the model as it is, flown on models with other values
(`evaluate_quadrotor --inertia_scale`, `--drag_scale`, `--mass_scale`; the reference's feedforward command as trained;
seeds 0, 1, 2):

| the simulated drone | success | position | position, largest after 1 s |
|---|---|---|---|
| as trained | 100 % | 3.40 / 3.35 / 3.40 mm | 8 / 8 / 8 mm |
| half the inertia | 100 % | 3.87 / 3.82 / 4.12 mm | 17 / 13 / 18 mm |
| twice the inertia | 100 % | 9.97 / 6.76 / 7.07 mm | 265 / 114 / 92 mm |
| no drag | 100 % | 28.05 / 28.55 / 29.75 mm | 156 / 166 / 157 mm |
| twice the drag | 100 % | 25.58 / 25.19 / 26.03 mm | 144 / 142 / 140 mm |
| 1.2 times the mass, the same largest thrust | 100 % | 7.81 / 7.64 / 7.90 mm | 57 / 55 / 53 mm |

No rollout terminates. The drag matters most: at 20 m/s it is 7 N of the model's forces, and a policy trained with one
value flies another with errors of 25 to 30 mm.
