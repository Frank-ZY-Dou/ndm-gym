# Quadrotor trajectory tracking: results

Contributed by [Zhengming Yu](https://github.com/Yzmblog) ([#7](https://github.com/Frank-ZY-Dou/ndm-gym/pull/7)).

SuperTrack and SuperMimic, with MLP world models, track flown trajectories of Race Against the Machine with a racing
quadrotor ([the task](../../docs/tasks/quadrotor.md), [the rules](README.md#rules-fixed-before-the-runs)).

## Setup

- **Reference:** the motion-captured flight of the dataset's drone (release v3.0.0), its part above 0.5 m, at 100 Hz: the
  autonomous flights, flown by the dataset's controller, and the piloted ones, flown by human pilots.
- **Training:** `ndm_gym.train_quadrotor`, 6,000 iterations in 2,048 worlds with the learning rate's decay over them, on one
  H200 per run, with the systems options (`--fused_adam --reset_masked --no_sync --collect_graph --update_graphs`).
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
# train and evaluate a task and seed: SuperTrack's selected configuration, then SuperMimic's
benchmarks/quadrotor_racing_tracking/run.sh <task> <seed> logs/quad_st --config quadrotor-supertrack-accurate
benchmarks/quadrotor_racing_tracking/run.sh <task> <seed> logs/quad_sm --config quadrotor-supermimic-accurate
```

## Results

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
| SuperTrack, selected (`quadrotor-supertrack-accurate`) | `--algo supertrack --lr 4e-4 --st_fail_w 2 --st_task_weights --horizon 64 --wm_hidden 256 256 256 --pi_opt muon --wm_opt muon --pi_ema 0.995 --pi_lr_mult 2 --lr_decay_ret 0.95 --lr_decay_by 1400 --lr_decay_len 300` |
| SuperMimic, the hand task's options | none (16 imagined steps, a learning rate of 2e-4, a world model of 3 × 1024) |
| SuperMimic, selected (`quadrotor-supermimic-accurate`) | `--algo supermimic --lr 4e-4 --horizon 64 --wm_hidden 256 256 256` |

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
named configurations as they are (`--config quadrotor-supertrack-accurate`, `quadrotor-supermimic-accurate`).

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

Every run reaches it at iteration 750 or 1,000. `quadrotor-supertrack-fast` and `quadrotor-supermimic-fast` stop there.

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
as the simulator does; the increment model predicts it.

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
