# Animal motion tracking: results

MIMIC-MJX's rat, fruit fly and stick insect track their animals' recorded kinematics on MuJoCo-Warp, with SuperTrack and
SuperMimic ([task](../../docs/tasks/animals.md)): one configuration per algorithm, three training seeds each, at a fixed
budget.

## How the numbers were obtained

- **Clips:** one policy per body on clips 0-9 of its reference; held-out: clips 10-19, never trained on.
- **Runs:** SuperTrack with `hand-supertrack-fast`'s options (`--lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights`,
  Muon, the policy average, the learning-rate ramp, the hand task's systems options), SuperMimic with `hand-supermimic-fast`'s
  (`--lr 4e-4 --action_noise 0.3 --gamma 0.99 --bptt_decay 0.95` and the same). Both with the same changes:
  `--ref_qvel positions` (the reference's velocities from its positions; the files' are not theirs at the pinned revision,
  [the task](../../docs/tasks/animals.md)), `--pi_batch 2048 --fused_fk` (from the rat's SuperTrack iteration time,
  Development notes), `--wm_lr 2e-3` (LAFAN1's
  configurations'), the iterations below with the learning rate decayed over them (`--lr_decay_iters`), a checkpoint every
  250 (`--ckpt_every 250 --ckpt_keep`); `train_animal`'s defaults otherwise, 2,048 worlds. No other tuning on the bodies
  for either algorithm. `benchmarks/mimic_mjx_animal_tracking/run_formal.sh` runs them (`--config <body>-<algorithm>`).
- **Budget:** fixed and the same for both algorithms: 4,000 iterations (stick insect), 6,000 (fruit fly), 10,000 (rat);
  seeds 0, 1, 2. Over the last 1,000 iterations (five checkpoints) the stick insect's bodies' error changes by at most
  0.011 mm and the fly's with SuperMimic by at most 0.001 mm; the fly's with SuperTrack by less than 0.001 mm in seed 0,
  by 0.022 mm in seed 1 (0.083 mm at 5,250 iterations) and by 0.010 mm in seed 2 (Time to the target). The rat's mean
  episode length in training still rose over the last 2,000 iterations in every SuperMimic run (from 159-167 to 170-176
  steps) and in two of SuperTrack's three (seeds 1 and 2: from 199 and 200 to 205 and 204; seed 0: 211 and 210), so the
  rat's numbers are the budget's, not final.
- **Evaluation:** `evaluate_animal` on each run's last checkpoint: every clip played from its first frame and four more times
  from perturbed starts (0.02 rad on the joints), the policy without exploration noise, to the clip's end. Errors over the steps
  the episodes lived: the tracked bodies' and the end effectors' positions (mm; the fly's centimetres converted), the joints'
  RMS error. Ended: the share of episodes ended by a termination (MIMIC-MJX's) before their clip's end.
- **Target:** at most 2 % of the episodes ended with the bodies' error within 3 % of the body's size (the bodies' median
  largest distance from the root: stick insect 17.6 mm, fruit fly 2.62 mm, rat 157 mm). It is this benchmark's, a proxy:
  MIMIC-MJX reports its own metrics (below).
- **Timing:** each run alone on one A100 80GB SXM4. The clock starts at the first training iteration, after the simulator's
  setup and the warm-up collection; evaluations are not in it. The kernels' compilation was in the cache before these runs:
  on an empty cache a run takes 6 to 13 min longer (each configuration's first seed-0 run, in `a100_runs/cold_cache_seed0/`).
  Without a per-phase time table of each path: time to the target, not engineered.
- **Table:** mean ± sample SD over the seeds. Each run's metrics, time to the target and training summary:
  [`a100_runs/`](a100_runs/README.md).

## Results

| body | algorithm | ended ↓ | bodies ↓ | end effectors ↓ | joints ↓ | held-out: ended ↓ | held-out: bodies ↓ | training (min) | env steps |
|---|---|---|---|---|---|---|---|---|---|
| stick insect (17.6 mm) | SuperTrack | 0.0 ± 0.0 % | 0.35 ± 0.01 mm | 0.60 ± 0.02 mm | 8.4 ± 0.2° | 10.7 ± 1.2 % | 0.64 ± 0.01 mm | 6.4 ± 0.1 | 8.2 M |
| | SuperMimic | 0.0 ± 0.0 % | 0.33 ± 0.00 mm | 0.56 ± 0.00 mm | 4.2 ± 0.1° | 0.0 ± 0.0 % | 0.74 ± 0.03 mm | 7.6 ± 0.1 | 8.2 M |
| fruit fly (2.62 mm) | SuperTrack | 1.3 ± 2.3 % | 0.066 ± 0.006 mm | 0.123 ± 0.008 mm | 4.6 ± 0.6° | 9.3 ± 7.6 % | 0.091 ± 0.012 mm | 11.2 ± 0.0 | 12.3 M |
| | SuperMimic | 0.0 ± 0.0 % | 0.049 ± 0.002 mm | 0.086 ± 0.003 mm | 4.1 ± 0.4° | 0.0 ± 0.0 % | 0.064 ± 0.001 mm | 12.9 ± 0.0 | 12.3 M |
| rat (157 mm) | SuperTrack | 39 ± 12 % | 13.6 ± 1.8 mm | 12.5 ± 1.9 mm | 10.7 ± 0.2° | 64 ± 16 % | 24.9 ± 5.7 mm | 22.4 ± 0.1 | 20.5 M |
| | SuperMimic | 16 ± 7 % | 20.1 ± 0.8 mm | 15.8 ± 0.3 mm | 9.1 ± 0.3° | 40 ± 2 % | 27.4 ± 1.9 mm | 25.0 ± 0.1 | 20.5 M |

**In short:** with SuperTrack / SuperMimic, the stick insect's bodies are tracked to 0.35 / 0.33 mm and the fly's to
0.066 / 0.049 mm (2.0 / 1.9 % and 2.5 / 1.9 % of the body's size). On the training clips no episode ended in any run
except SuperTrack's fly run of seed 2 (2 of its 50 episodes, 4 %); on the held-out clips 10.7 / 0 % and 9.3 / 0 % ended.
The rat, torque-driven through 40 ms activation filters with 67 joints, is not solved at this budget by either: 39 / 16 %
of its episodes ended after 20.5 M environment steps, with the bodies within 13.6 / 20.1 mm (8.7 / 12.8 % of its size).
MIMIC-MJX's PPO reaches 95 % of its test reward after about 650 M steps (6 to 20 hours on one GPU, as published).

Not comparable as they stand: MIMIC-MJX reports the rat's median keypoint error against stac-mjx's keypoints (about 10 mm on
held-out clips); the bodies' error here is a mean over the body frames' origins, and the rat does not yet finish its clips.

## Time to the target

The target: at most 2 % of the episodes ended, with the bodies' error within 3 % of the body's size (stick insect 17.58 mm,
fly 2.623 mm, rat 156.6 mm: 0.53, 0.079 and 4.7 mm), set after earlier runs of these configurations. `time_to.py`
evaluates every kept checkpoint (every 250 iterations) as the last one is, and reports the wall time of the first
checkpoint from which every later one meets the target. The times are the runs' own wall clock, each run alone on its
GPU; the checkpoints are 250 iterations apart, so a time is its checkpoint's and an upper bound.

```bash
python benchmarks/mimic_mjx_animal_tracking/time_to.py logs/mimic_mjx_formal/{stick,fly}_*_s?
```

| body | algorithm | minutes to the target ↓ | iteration | seeds 0 / 1 / 2 (min) |
|---|---|---|---|---|
| stick insect | SuperTrack | 1.7 ± 0.1 | 1,000, every run | 1.8 / 1.6 / 1.6 |
| | SuperMimic | 2.1 ± 0.4 | 1,250 / 1,000 / 1,000 | 2.6 / 1.9 / 1.9 |
| fruit fly | SuperTrack | 8.4 ± 3.2 (2 of 3 runs) | 3,250 / 5,750 / not reached | 6.2 / 10.7 / not reached |
| | SuperMimic | 2.8 ± 0.0 | 1,250, every run | 2.8 / 2.8 / 2.9 |
| rat | both | not reached | | |

On the stick insect SuperTrack's runs hold the target from 1,000 iterations and SuperMimic's from 1,250 (seed 0) and
1,000 (seeds 1 and 2); no episode ends at any checkpoint from 750 (SuperTrack) and 500 (SuperMimic). On the fly,
SuperMimic's runs hold the target from 1,250 and end no episode at any checkpoint from 750. SuperTrack's fly runs meet
the target early and then miss it at some later checkpoints, at each of which 1 to 3 of the 50 episodes ended (2-6 %)
with the bodies' error at 0.069 to 0.083 mm:

- seed 0 first meets it at 1,750 (3.4 min), misses it at 2,500 and 3,000 and holds it from 3,250;
- seed 1 first meets it at 1,500 (2.9 min), misses it at 2,500, 2,750, 5,250 (the bodies' error, 0.083 mm, with 2 % of
  the episodes ended) and 5,500, and holds it from 5,750;
- seed 2 first meets it at 1,250 (2.4 min), misses it at 1,500, 2,250, 2,500, 3,000, 3,750 and at its last checkpoint,
  6,000 (2 of the 50 episodes ended, 4 %), so it has no time to the target.

## The rat with a longer budget

One extra run: `rat-supermimic`'s options with 40,000 iterations, the learning rate decayed over them (seed 0, 1.7 h,
alone on an A100 80GB SXM4). SuperTrack was not run at this budget, so this is not a comparison of the algorithms.

| | ended ↓ | bodies ↓ | end effectors ↓ | joints ↓ | held-out: ended ↓ | held-out: bodies ↓ |
|---|---|---|---|---|---|---|
| 10,000 iterations (seeds 0-2) | 16 ± 7 % | 20.1 ± 0.8 mm | 15.8 ± 0.3 mm | 9.1 ± 0.3° | 40 ± 2 % | 27.4 ± 1.9 mm |
| 40,000 iterations (seed 0) | 20 % | 13.2 mm | 11.6 mm | 9.1° | 12 % | 18.9 mm |

With four times the iterations the held-out clips' ended share goes from 40 to 12 % and their bodies' error from 27.4 to
18.9 mm; on the training clips the bodies' error goes from 20.1 to 13.2 mm and the ended share does not fall (20 %; 16 %
over the three seeds at 10,000 iterations, 20 % in seed 0). The training curve still rose at 40,000 iterations (episode
length 177 to 182 steps over the last 4,000). Not a configuration of its own.

## Renders

`tools/render_animal.py` renders a rollout recorded by `evaluate_animal`; `docs/media/animals/` holds each case's run with
the lowest bodies' error (stick insect SuperTrack seed 0 and SuperMimic seed 2, fly SuperTrack seed 1 and SuperMimic
seed 2, rat SuperTrack seed 0 and SuperMimic seed 2) from the reference's start (the fly at a quarter of real time), and
the grid of the six (`tools/make_grid.py`).

## Development notes

- **Iteration time** (rat, SuperTrack, 2,048 worlds, an A100 80GB SXM4 alone; `run_formal.sh`'s command, iterations 500 to
  1,000): 132 ms with `--pi_batch 2048 --fused_fk`, 306 ms with the default batch of 4,096 and the level-by-level kinematics.
  On an empty cache the rat's runs take 11 to 13 min longer (`a100_runs/cold_cache_seed0/`); the training times above are
  with the kernels cached.
