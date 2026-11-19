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
  1,000 (`--ckpt_every 1000 --ckpt_keep`); `train_animal`'s defaults otherwise, 2,048 worlds. No other tuning on the bodies
  for either algorithm. `benchmarks/mimic_mjx_animal_tracking/run_formal.sh` runs them (`--config <body>-<algorithm>`).
- **Budget:** fixed and the same for both algorithms: 4,000 iterations (stick insect), 6,000 (fruit fly), 10,000 (rat);
  seeds 0, 1, 2. The stick insect's bodies' error changes by at most 0.02 mm over the last two checkpoints and the fly's by
  at most 0.015 mm over the last three (Time to the target); the rat's runs of both algorithms were still improving at the end, so their numbers are the budget's, not final.
- **Evaluation:** `evaluate_animal` on each run's last checkpoint: every clip played from its first frame and four more times
  from perturbed starts (0.02 rad on the joints), the policy without exploration noise, to the clip's end. Errors over the steps
  the episodes lived: the tracked bodies' and the end effectors' positions (mm; the fly's centimetres converted), the joints'
  RMS error. Ended: the share of episodes ended by a termination (MIMIC-MJX's) before their clip's end.
- **Target:** at most 2 % of the episodes ended with the bodies' error within 3 % of the body's size (the bodies' median
  largest distance from the root: stick insect 17.6 mm, fruit fly 2.62 mm, rat 157 mm). It is this benchmark's, a proxy:
  MIMIC-MJX reports its own metrics (below).
- **Timing:** one RTX 4090, three runs at a time. The clock starts at the first training iteration, after the simulator's
  setup and the warm-up collection; the compilation is in it, evaluations are not. The times are upper bounds of a run
  alone, not an A100's or H100's, and without a per-phase time table of each path: time to the target, not engineered.
- **Table:** mean ± sample SD over the seeds.

## Results

| body | algorithm | ended ↓ | bodies ↓ | end effectors ↓ | joints ↓ | held-out: ended ↓ | held-out: bodies ↓ | training (min) | env steps |
|---|---|---|---|---|---|---|---|---|---|
| stick insect (17.6 mm) | SuperTrack | 0.0 ± 0.0 % | 0.41 ± 0.01 mm | 0.67 ± 0.02 mm | 8.6 ± 0.1° | 0.7 ± 1.2 % | 0.77 ± 0.01 mm | 17 ± 3 | 8.2 M |
| | SuperMimic | 0.0 ± 0.0 % | 0.39 ± 0.01 mm | 0.59 ± 0.00 mm | 4.1 ± 0.1° | 1.3 ± 2.3 % | 0.87 ± 0.03 mm | 19 ± 3 | 8.2 M |
| fruit fly (2.62 mm) | SuperTrack | 0.0 ± 0.0 % | 0.062 ± 0.001 mm | 0.116 ± 0.003 mm | 4.8 ± 0.4° | 6.7 ± 3.1 % | 0.092 ± 0.008 mm | 27 ± 4 | 12.3 M |
| | SuperMimic | 0.0 ± 0.0 % | 0.047 ± 0.002 mm | 0.078 ± 0.003 mm | 3.8 ± 0.2° | 0.0 ± 0.0 % | 0.066 ± 0.000 mm | 30 ± 4 | 12.3 M |
| rat (157 mm) | SuperTrack | 69 ± 3 % | 35.8 ± 1.4 mm | 33.1 ± 2.2 mm | 12.1 ± 0.3° | 83 ± 8 % | 52.8 ± 4.1 mm | 53 ± 6 | 20.5 M |
| | SuperMimic | 19 ± 1 % | 19.6 ± 1.7 mm | 16.8 ± 1.1 mm | 9.8 ± 0.1° | 51 ± 6 % | 43.2 ± 3.0 mm | 54 ± 10 | 20.5 M |

**In short:** with SuperTrack / SuperMimic, the stick insect's bodies are tracked to 0.41 / 0.39 mm and the fly's to
0.062 / 0.047 mm (2.3 / 2.2 % and 2.4 / 1.8 % of the body's size), no episode ended on the training clips; on the held-out
clips 0.7 / 1.3 % and 6.7 / 0 % ended. The rat, torque-driven through 40 ms activation filters with 67 joints, is not solved
at this budget by either: 69 / 19 % of its episodes ended after 20.5 M environment steps. MIMIC-MJX's PPO reaches 95 % of its
test reward after about 650 M steps (6 to 20 hours on one GPU, as published).

Not comparable as they stand: MIMIC-MJX reports the rat's median keypoint error against stac-mjx's keypoints (about 10 mm on
held-out clips); the bodies' error here is a mean over the body frames' origins, and the rat does not yet finish its clips.

## Time to the target

The target: at most 2 % of the episodes ended, with the bodies' error within 3 % of the body's size (stick insect 17.58 mm,
fly 2.623 mm, rat 156.6 mm: 0.53, 0.079 and 4.7 mm), set after the runs. `time_to.py` evaluates every kept checkpoint
(every 1,000 iterations) as the last one is, and reports the wall time of the first checkpoint from which every later one
meets the target. The times are the runs' own wall clock with three runs sharing the GPU, the first checkpoint's including
the compilation, and the checkpoints are 1,000 iterations apart, so they are upper bounds.

```bash
python benchmarks/mimic_mjx_animal_tracking/time_to.py logs/mimic_mjx_formal/{stick,fly}_*_s?
```

| body | algorithm | minutes to the target ↓ | iteration | seeds 0 / 1 / 2 (min) |
|---|---|---|---|---|
| stick insect | SuperTrack | 8.8 ± 2.4 | 2,000, every run | 11.5 / 7.4 / 7.3 |
| | SuperMimic | 10.4 ± 2.7 | 2,000, every run | 13.5 / 9.1 / 8.7 |
| fruit fly | SuperTrack | 23.4 ± 6.7 | 5,000-6,000 | 30.7 / 22.0 / 17.5 |
| | SuperMimic | 11.3 ± 2.5 | 2,000, every run | 14.2 / 10.2 / 9.6 |
| rat | both | not reached | | |

On the fly, SuperTrack's runs meet the error target from 2,000 iterations and end 0 to 3 of the 50 episodes at the later
checkpoints (0-6 %); SuperMimic's meet it from 2,000 and end none at any checkpoint. On the stick insect both end none at any
checkpoint and meet the error target from 2,000.

## The rat with a longer budget

One extra run: `rat-supermimic`'s options with 40,000 iterations, the learning rate decayed over them (seed 0, 2.2 h,
sharing the RTX 4090). SuperTrack was not run at this budget, so this is not a comparison of the algorithms.

| | ended ↓ | bodies ↓ | end effectors ↓ | joints ↓ | held-out: ended ↓ | held-out: bodies ↓ |
|---|---|---|---|---|---|---|
| 10,000 iterations (seeds 0-2) | 19 ± 1 % | 19.6 ± 1.7 mm | 16.8 ± 1.1 mm | 9.8 ± 0.1° | 51 ± 6 % | 43.2 ± 3.0 mm |
| 40,000 iterations (seed 0) | 16 % | 17.2 mm | 14.3 mm | 9.0° | 22 % | 27.5 mm |

With four times the iterations the held-out clips' ended share goes from 51 to 22 % and their bodies' error from 43.2 to
27.5 mm; on the training clips from 19 to 16 % and from 19.6 to 17.2 mm. The training curve still rose at 40,000 iterations
(episode length 190 to 194 steps over the last 4,000). Not a configuration of its own.

## Renders

`tools/render_animal.py` renders a rollout recorded by `evaluate_animal`; `docs/media/animals/` holds each case's run with
the lowest bodies' error (stick insect SuperTrack seed 2 and SuperMimic seed 1, fly seeds 2 and 0, rat seed 1 each) from
the reference's start (the fly at a quarter of real time), and the grid of the six (`tools/make_grid.py`).

## Development notes

- **PD servos for the rat:** gains from each actuator's effective inertia and force (natural frequency 1-8 Hz) keep the open
  loop stable but leave it as short-lived as zero torque (17 % of a clip against 16 %): through 40 ms of activation lag a stable
  stiffness is two orders of magnitude below what holds the rat up, and the policy has to learn its feedforward torques. Not
  kept.
- **Iteration time** (rat, SuperTrack, 2,048 worlds, the GPU alone): collection 20 ms, world-model updates 28 ms, policy update
  43 ms with `--pi_batch 2048 --fused_fk`, from 145 ms with the default batch of 4,096 and the level-by-level kinematics.
  `--fused_fk` compiles for about 5 minutes on the rat, inside the training time above.
