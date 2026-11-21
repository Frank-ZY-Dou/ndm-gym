# Animal motion tracking: results

MIMIC-MJX's rat, fruit fly and stick insect track their animals' recorded kinematics on MuJoCo-Warp, with SuperTrack and
SuperMimic ([task](../../docs/tasks/animals.md)): one configuration per algorithm, three training seeds each, at a fixed
budget.

## How the numbers were obtained

- **Clips:** one policy per body on clips 0-9 of its reference; held-out: clips 10-19, never trained on.
- **Runs:** SuperTrack with `hand-supertrack-fast`'s options (`--lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights`,
  Muon, the policy average, the learning-rate ramp, the hand task's systems options), SuperMimic with `hand-supermimic-fast`'s
  (`--lr 4e-4 --action_noise 0.3 --gamma 0.99 --bptt_decay 0.95` and the same). Both with the same changes:
  `--pi_batch 2048 --fused_fk` (from the rat's SuperTrack iteration time, Development notes), `--wm_lr 2e-3` (LAFAN1's
  configurations'), the iterations below with the learning rate decayed over them (`--lr_decay_iters`), a checkpoint every
  250 (`--ckpt_every 250 --ckpt_keep`); `train_animal`'s defaults otherwise, 2,048 worlds. No other tuning on the bodies
  for either algorithm. `benchmarks/mimic_mjx_animal_tracking/run_formal.sh` runs them (`--config <body>-<algorithm>`).
- **Budget:** fixed and the same for both algorithms: 4,000 iterations (stick insect), 6,000 (fruit fly), 10,000 (rat);
  seeds 0, 1, 2. Over the last 1,000 iterations (five checkpoints) the stick insect's bodies' error changes by at most
  0.007 mm and the fly's by at most 0.001 mm, except in SuperTrack's fly run of seed 0 (0.017 mm: 0.082 mm at 5,500
  iterations; Time to the target). The rat's runs of both algorithms were still improving at the end (the mean episode
  length in training rose over the last 2,000 iterations in every run: SuperTrack from 171-179 to 177-190 steps, SuperMimic
  from 141-155 to 152-165), so their numbers are the budget's, not final.
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
| stick insect (17.6 mm) | SuperTrack | 0.0 ± 0.0 % | 0.40 ± 0.01 mm | 0.66 ± 0.02 mm | 8.3 ± 0.2° | 2.7 ± 1.2 % | 0.75 ± 0.01 mm | 6.4 ± 0.1 | 8.2 M |
| | SuperMimic | 0.0 ± 0.0 % | 0.39 ± 0.00 mm | 0.59 ± 0.00 mm | 4.1 ± 0.0° | 0.0 ± 0.0 % | 0.84 ± 0.01 mm | 7.6 ± 0.1 | 8.2 M |
| fruit fly (2.62 mm) | SuperTrack | 0.0 ± 0.0 % | 0.064 ± 0.003 mm | 0.120 ± 0.007 mm | 4.6 ± 0.1° | 6.0 ± 2.0 % | 0.097 ± 0.009 mm | 11.1 ± 0.0 | 12.3 M |
| | SuperMimic | 0.0 ± 0.0 % | 0.046 ± 0.001 mm | 0.077 ± 0.002 mm | 3.6 ± 0.1° | 0.0 ± 0.0 % | 0.064 ± 0.002 mm | 12.9 ± 0.0 | 12.3 M |
| rat (157 mm) | SuperTrack | 69 ± 5 % | 35.2 ± 0.2 mm | 32.6 ± 1.0 mm | 11.9 ± 0.4° | 85 ± 1 % | 51.3 ± 1.9 mm | 22.2 ± 0.0 | 20.5 M |
| | SuperMimic | 21 ± 3 % | 22.7 ± 0.8 mm | 18.2 ± 1.3 mm | 9.8 ± 0.1° | 45 ± 9 % | 41.1 ± 2.0 mm | 24.8 ± 0.0 | 20.5 M |

**In short:** with SuperTrack / SuperMimic, the stick insect's bodies are tracked to 0.40 / 0.39 mm and the fly's to
0.064 / 0.046 mm (2.3 / 2.2 % and 2.5 / 1.7 % of the body's size). On the training clips no episode ended in any run; on
the held-out clips 2.7 / 0 % and 6.0 / 0 % ended. The rat,
torque-driven through 40 ms activation filters with 67 joints, is not solved at this budget by either: 69 / 21 % of its
episodes ended after 20.5 M environment steps. MIMIC-MJX's PPO reaches 95 % of its
test reward after about 650 M steps (6 to 20 hours on one GPU, as published).

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
| stick insect | SuperTrack | 2.0 ± 0.1 | 1,250, every run | 2.1 / 2.0 / 2.0 |
| | SuperMimic | 2.4 ± 0.1 | 1,250, every run | 2.5 / 2.5 / 2.3 |
| fruit fly | SuperTrack | 5.2 ± 4.7 | 5,750 / 1,250 / 1,250 | 10.7 / 2.5 / 2.4 |
| | SuperMimic | 3.0 ± 0.3 | 1,250 / 1,500 / 1,250 | 2.8 / 3.3 / 2.8 |
| rat | both | not reached | | |

On the stick insect every run of both algorithms holds the target from 1,250 iterations and ends no episode at any
checkpoint from 750. On the fly, SuperTrack's runs of seeds 1 and 2 hold the target from 1,250; the run of seed 0 first
meets it at 1,500 (2.9 min), misses it at 2,500, 3,250, 3,500 and 5,500 (2 to 4 of the 50 episodes ended, 4-8 %, with the
bodies' error at 0.073 to 0.084 mm) and holds it from 5,750. SuperMimic's runs hold it from 1,250 (seeds 0 and 2) and
1,500 (seed 1) and end no episode at any checkpoint from 750.

## The rat with a longer budget

One extra run: `rat-supermimic`'s options with 40,000 iterations, the learning rate decayed over them (seed 0, 1.7 h,
alone on an A100 80GB SXM4). SuperTrack was not run at this budget, so this is not a comparison of the algorithms.

| | ended ↓ | bodies ↓ | end effectors ↓ | joints ↓ | held-out: ended ↓ | held-out: bodies ↓ |
|---|---|---|---|---|---|---|
| 10,000 iterations (seeds 0-2) | 21 ± 3 % | 22.7 ± 0.8 mm | 18.2 ± 1.3 mm | 9.8 ± 0.1° | 45 ± 9 % | 41.1 ± 2.0 mm |
| 40,000 iterations (seed 0) | 20 % | 16.8 mm | 14.2 mm | 9.1° | 22 % | 30.7 mm |

With four times the iterations the held-out clips' ended share goes from 45 to 22 % and their bodies' error from 41.1 to
30.7 mm; on the training clips the bodies' error goes from 22.7 to 16.8 mm and the ended share stays at 20 to 21 %. The
training curve still rose at 40,000 iterations (episode length 193 to 198 steps over the last 4,000). Not a configuration
of its own.

## Renders

`tools/render_animal.py` renders a rollout recorded by `evaluate_animal`; `docs/media/animals/` holds each case's run with
the lowest bodies' error (stick insect SuperTrack seed 2 and SuperMimic seed 0, fly SuperTrack seed 1 and SuperMimic
seed 0, rat seed 0 each) from the reference's start (the fly at a quarter of real time), and the grid of the six
(`tools/make_grid.py`).

## Development notes

- **Iteration time** (rat, SuperTrack, 2,048 worlds, an A100 80GB SXM4 alone; `run_formal.sh`'s command, iterations 500 to
  1,000): 132 ms with `--pi_batch 2048 --fused_fk`, 306 ms with the default batch of 4,096 and the level-by-level kinematics.
  On an empty cache the rat's runs take 11 to 13 min longer (`a100_runs/cold_cache_seed0/`); the training times above are
  with the kernels cached.
