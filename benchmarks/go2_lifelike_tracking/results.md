# Quadruped motion tracking: results

SuperTrack and SuperMimic, with the library's MLP world model of generalized coordinates, make a Unitree Go2 track the
retargeted motion capture of a Labrador ([the task](../../docs/tasks/quadruped.md), [the plan and rules](README.md)).

## Setup

- **Reference:** the 65 clips that `tools/convert_lifelike.py` cuts from the 33 files of the Lifelike motion capture
  (488 s at the robot's time scale, 50 Hz). The data are downloaded by `tools/download_lifelike.sh` and are not in this
  repository.
- **Training:** `ndm_gym.train_quadruped` with the learning rate's decay over the run, a checkpoint every 500 iterations
  (`--keep_ckpts`), one run per GPU, three runs at a time on three RTX 6000 Ada (48 GB) of one machine, no other job on
  those GPUs. All clips: 10,000 iterations; one clip: 6,000.
- **Evaluation:** `ndm_gym.evaluate_quadruped` on the last checkpoint: 252 rollouts per clip (all clips) or 256 (one
  clip) of the whole clip from perturbed starts (evaluation seed 0), without exploration noise and without resets. A
  rollout succeeds if it never terminates. Errors are means over the frames before a termination. Failed clips: the mean
  number of clips failed by a pass of one rollout per clip. Pushes: the same rollouts with a horizontal velocity of 1 or
  2 m/s added to the trunk every 2 s in a random direction (no pushes in training).
- **Time to target:** every checkpoint of the run evaluated the same way (without pushes); the training wall time of the
  first checkpoint from which every later one has a success rate of at least 0.99 and the trunk's error below 20 mm.
- Mean ± sample SD over three training seeds (0, 1, 2).

| task | clip | length | trunk speed, mean / top |
|---|---|---|---|
| `walk` | `dog_quad_walk_002_1` | 6.0 s | 0.88 / 1.25 m/s |
| `run` | `dog_quad_run_002_0` | 5.5 s | 1.63 / 2.95 m/s |
| `play` | `dog_play_002_0` | 10.1 s | 0.87 / 2.15 m/s |
| `all` | the 65 clips, one policy | 488 s | |

Configurations (`ndm_gym/configs.py`):

- **SuperMimic, selected** (`--config go2-supermimic`, `go2-supermimic-clip`): the hand task's systems options, HOI's
  optimizer options (Muon for the policy and the world model, the policy's EMA, the learning rate's decay on the return),
  `--lr 4e-4 --action_noise 0.3 --bptt_decay 0.95 --gamma 0.95`.
- **SuperTrack, selected** (`--config go2-supertrack`, `go2-supertrack-clip`): the hand task's systems options and its
  SuperTrack options, `--lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights`.
- **SuperTrack, systems options only:** the trainer's defaults with the systems options, for comparison.
- **Reference's targets, open loop:** the same rollouts with the reference's joint positions as the position targets and
  no policy.

```bash
benchmarks/go2_lifelike_tracking/run.sh <task> <seed> logs/go2_sm --config go2-supermimic       # all
benchmarks/go2_lifelike_tracking/run.sh <task> <seed> logs/go2_sm --config go2-supermimic-clip  # walk, run, play
```

## All 65 clips

| | success | return | feet (trunk frame) | feet (world) | trunk | joints | failed clips | failed clips, push 1 / 2 m/s | time to target | training | iterations/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference's targets, open loop | 0.052 | | | | 141 mm | | | | | | |
| SuperTrack, systems options only | 0.9821 ± 0.0008 | 0.9425 ± 0.0014 | 11.94 ± 0.02 mm | 19.25 ± 0.16 mm | 12.79 ± 0.15 mm | 2.30 ± 0.01° | 6.0 ± 1.0 | 24.1 ± 0.8 / 54.0 ± 0.3 | not reached | 11.0 min | 15.13 ± 0.03 |
| SuperTrack, selected | 0.9987 ± 0.0012 | 0.9767 ± 0.0014 | 7.26 ± 0.04 mm | 16.24 ± 0.13 mm | 12.81 ± 0.08 mm | 1.37 ± 0.01° | 4.3 ± 2.5 | 21.5 ± 0.4 / 54.0 ± 0.4 | 4.43 ± 0.58 min | 11.1 min | 14.91 ± 0.04 |
| **SuperMimic, selected** | **1.0000 ± 0.0000** | **0.9856 ± 0.0001** | **5.33 ± 0.04 mm** | **14.96 ± 0.02 mm** | **12.12 ± 0.04 mm** | **1.10 ± 0.01°** | **0.0 ± 0.0** | **5.5 ± 0.7** / 51.6 ± 0.8 | **1.31 ± 0.03 min** | 14.0 min | 11.81 ± 0.02 |

## One clip

| task | | return | feet (trunk frame) | trunk | joints | time to target | training | iterations/s |
|---|---|---|---|---|---|---|---|---|
| `walk` | reference's targets, open loop | success 0.0 | | 226 mm | | | | |
| | SuperTrack, systems options only | 0.9841 ± 0.0002 | 10.41 ± 0.08 mm | 4.95 ± 0.15 mm | 2.19 ± 0.00° | 1.06 ± 0.02 min | 6.6 min | 15.05 |
| | SuperTrack, selected | 0.9915 ± 0.0001 | 6.13 ± 0.05 mm | 11.40 ± 0.13 mm | 1.12 ± 0.01° | 1.08 ± 0.01 min | 6.7 min | 14.83 |
| | SuperMimic, selected | **0.9947 ± 0.0000** | **4.15 ± 0.03 mm** | 10.53 ± 0.03 mm | **0.92 ± 0.00°** | 1.32 ± 0.00 min | 8.4 min | 11.78 |
| `run` | reference's targets, open loop | success 0.0 | | 100 mm | | | | |
| | SuperTrack, systems options only | 0.9530 ± 0.0004 | 14.68 ± 0.16 mm | 21.45 ± 0.28 mm | 2.56 ± 0.03° | not reached | 7.3 min | 14.77 |
| | SuperTrack, selected | 0.9728 ± 0.0001 | 10.40 ± 0.04 mm | 15.63 ± 0.02 mm | 2.15 ± 0.01° | 1.28 ± 0.32 min | 6.8 min | 14.53 |
| | SuperMimic, selected | **0.9773 ± 0.0016** | **7.70 ± 0.17 mm** | 17.67 ± 0.26 mm | **1.66 ± 0.03°** | 2.52 ± 0.46 min | 8.9 min | 11.73 |
| `play` | reference's targets, open loop | success 0.0 | | 213 mm | | | | |
| | SuperTrack, systems options only | 0.9738 ± 0.0005 | 11.82 ± 0.07 mm | 10.47 ± 0.35 mm | 2.19 ± 0.01° | 2.16 ± 0.67 min | 7.2 min | 14.96 |
| | SuperTrack, selected | 0.9851 ± 0.0001 | 7.66 ± 0.03 mm | 12.22 ± 0.06 mm | 1.47 ± 0.00° | 1.08 ± 0.02 min | 6.7 min | 14.82 |
| | SuperMimic, selected | **0.9894 ± 0.0000** | **5.57 ± 0.03 mm** | 12.34 ± 0.11 mm | **1.13 ± 0.01°** | 1.61 ± 0.51 min | 8.7 min | 11.79 |

Every policy on one clip has a success rate of 1.000 (three seeds, 256 rollouts each). Under 1 m/s pushes the clip fails
in 1 to 16 % of SuperMimic's rollouts and in 57 to 86 % of SuperTrack's; under 2 m/s in 97 to 100 % of every
configuration's.

## Findings

- **SuperMimic is the most accurate and, on all clips, the fastest to the target.** On the 65 clips it fails none,
  its feet are 27 % closer than SuperTrack's, and it holds a success of 0.99 with the trunk within 20 mm after
  1.3 min against 4.4 min. On one clip both reach the target in 1 to 2.5 min; SuperTrack a little sooner (more
  iterations per second), SuperMimic more accurate.
- **The discount: 0.95 in place of 0.99** is what SuperMimic needed (the table below). An inference we did not test: at
  0.99 the critic's value at the window's end outweighs the imagined rewards in the policy's gradient.
- **SuperTrack: the hand task's options** raise the return on all clips from 0.943 to 0.977 and the success from 0.982 to
  0.999. Muon and HOI's other optimizer options on top of them (seed 0, all clips) did not help: a success of 0.989, a
  return of 0.972, the target not reached. The same options on top of the systems options only: a success of 0.995, a
  return of 0.965, the target after 7.4 min, still short of the selected configuration.
- **Pushes are the open problem.** Without pushes in training, about 6 of the 65 clips fail for SuperMimic at 1 m/s and
  about 22 for SuperTrack; at 2 m/s most clips fail for every configuration.
- **The reference is not physically consistent**: its joint positions as the targets keep the robot up in 5 % of the
  rollouts on all clips and in none on the three single clips; the policies find the contacts that the kinematic
  retarget does not provide.

## Beside the starting point

SuperMimic on all clips, seed 0, 4,000 iterations, 64 rollouts per clip (the selection runs, before the benchmark's):

| | discount 0.99 | 0.95 | 0.90 |
|---|---|---|---|
| the trainer's defaults (Adam) | success 0.908, return 0.706 | 0.9998, 0.962 | 0.998, 0.969 |
| HOI's optimizer options, `--lr 4e-4 --action_noise 0.3 --bptt_decay 0.95` | 0.968, 0.839 | 0.985, 0.968 | |

The discount accounts for most of the gain; HOI's options with 0.95 were kept, as they reach 1.000 and 0.986 at 10,000
iterations (the table above). SuperTrack: the hand task's options against the trainer's defaults are in the tables
above; Muon and HOI's other optimizer options did not help it (Findings).

## Notes

- Two of SuperMimic's first trainings on all clips (seeds 0 and 1) and one on `play` (seed 0) shared their GPU with
  another user's job; they were run again on idle GPUs and only the reruns are reported.
- Not measured: clips held out of training (zero-shot), and the two long clips (70 s and 79 s) split into shorter ones.
