# Manipulation demonstration tracking: results

SuperTrack and SuperMimic, with the library's MLP world model of generalized coordinates, make one Franka Panda (two for
Transport) track robomimic's first proficient-human demonstration of Lift, Square and Transport on MuJoCo-Warp
([the task](../../docs/tasks/robomimic.md), [the rules](README.md)).

## Setup

- **Configurations** (`ndm_gym/configs.py`), the same task defaults (`ROBOMIMIC_DEFAULTS`) and run options for both:
  - **SuperTrack** (`--config robomimic-supertrack`): `hand-supertrack-accurate`'s options, `--lr 4e-4 --st_fail_w 2
    --st_task_weights` and the hand task's systems options, without its `--st_imag_noise 0.2`;
  - **SuperMimic** (`--config robomimic-supermimic`): `hand-supermimic-accurate`'s options, SuperMimic's defaults with the
    hand task's systems options;
  - **both:** `--task <task> --demo 0 --solver_iterations 20`, 2,048 worlds, 1,500 iterations with the learning rate on
    a schedule of 4,000 (`--lr_decay_iters 4000`), a checkpoint kept every 500; seeds 0, 1, 2. Neither was tuned on these
    tasks.
- **Budget:** fixed and the same for both algorithms, 1,500 iterations; the numbers are the last checkpoint's.
- **Evaluation:** `evaluate_robomimic` on the last checkpoint: 256 rollouts of the whole demonstration from perturbed starts
  (evaluation seed 0), without exploration noise. The object errors are means over the frames of the rollouts that do not
  terminate.
- **Success:** a rollout succeeds if it never terminates and meets robosuite's success test (`_check_success` of Lift,
  NutAssemblySquare and TwoArmTransport, written as a function of the state) at the demonstration's last frame. This is a
  tracking proxy, not robomimic's success rate of a learned policy (the test met at any step within a horizon).
- **Reference targets:** the same rollouts with the reference's joint positions as the servos' targets and no policy.
- **Hardware and clock:** each run alone on one A100 80GB SXM4. The clock starts at the first training iteration, after the
  simulator's setup and the warm-up collection, and includes the checkpoints' saving; evaluations are not in it. The
  kernels' compilation was in the cache before these runs: on an empty cache a run takes 4 to 15 min longer (each
  configuration's first seed-0 run, in `a100_runs/cold_cache_seed0/`). Time to the target, not engineered: there is no
  per-phase time table. Iterations/s is `train_log.csv`'s last `it_per_s`.
- Mean ± sample SD over the three training seeds. Each run's log, metrics and time to the target:
  [`a100_runs/`](a100_runs/README.md).

```bash
benchmarks/robomimic_manipulation_tracking/run_formal.sh
```

## Results

| task | algorithm | success ↑ | object position ↓ | object position at the last frame ↓ | episodes ended ↓ | reference targets: success | training (min) | iterations/s |
|---|---|---|---|---|---|---|---|---|
| Lift | SuperTrack | 100.0 ± 0.0 % | 0.63 ± 0.04 mm | 1.94 ± 0.52 mm | 0.0 % | 0.0 % | 3.2 ± 0.0 | 7.96 ± 0.03 |
| | SuperMimic | 100.0 ± 0.0 % | 0.79 ± 0.07 mm | 2.34 ± 0.56 mm | 0.0 % | 0.0 % | 3.4 ± 0.0 | 7.28 ± 0.06 |
| Square | SuperTrack | 100.0 ± 0.0 % | 1.78 ± 0.19 mm | 3.52 ± 1.8 mm | 0.0 % | 5.9 ± 1.4 % | 6.5 ± 0.1 | 3.86 ± 0.04 |
| | SuperMimic | 96.9 ± 5.4 % | 5.96 ± 2.7 mm | 7.89 ± 5.3 mm | 0.4 ± 0.7 % | 7.9 ± 1.0 % | 6.9 ± 0.1 | 3.61 ± 0.09 |
| Transport | SuperTrack | 66.5 ± 9.4 % | 5.11 ± 0.12 mm | 95.9 ± 40 mm | 27.7 ± 13 % | 0.0 % | 8.5 ± 0.0 | 2.99 ± 0.02 |
| | SuperMimic | 6.6 ± 4.4 % | 12.6 ± 3.6 mm | 329 ± 10 mm | 92.8 ± 4.9 % | 0.0 % | 9.3 ± 0.0 | 2.72 ± 0.02 |

Per seed (0 / 1 / 2), success and object position:

| task | SuperTrack | SuperMimic |
|---|---|---|
| Lift | 100 / 100 / 100 %; 0.60 / 0.67 / 0.61 mm | 100 / 100 / 100 %; 0.75 / 0.87 / 0.76 mm |
| Square | 100 / 100 / 100 %; 1.99 / 1.71 / 1.64 mm | 100 / 100 / 90.6 %; 4.01 / 4.82 / 9.05 mm |
| Transport | 77.3 / 61.7 / 60.5 %; 5.07 / 5.24 / 5.01 mm | 9.0 / 9.4 / 1.6 %; 15.82 / 8.69 / 13.32 mm |

On Lift both algorithms succeed in every rollout, and on Square SuperTrack in every rollout and SuperMimic in 96.9 %
(100 / 100 / 90.6 % over the seeds), where the reference's targets alone succeed in 0 and 5-9 %. On Transport (two arms,
a 36 s demonstration) SuperTrack succeeds in 66.5 % and SuperMimic in 6.6 % of the rollouts. The Transport runs of both
algorithms and Square's SuperMimic runs still change between their last checkpoints (below), so these are the fixed
budget's numbers, not final ones.

## Time to the target

The target is the hand task's: a success of at least 0.99 with the tracked objects within 10 mm. `time_to.py` evaluates
every kept checkpoint (every 500 iterations) as the last one is, without the reference's targets, and reports the wall time
of the first checkpoint from which every later one meets the target. The checkpoints are 500 iterations apart, so a time
is its checkpoint's and an upper bound (every Lift run meets the target at its first checkpoint).

```bash
python benchmarks/robomimic_manipulation_tracking/time_to.py logs/robomimic_formal/*_s?
```

| task | algorithm | minutes to the target ↓ | iteration | seeds 0 / 1 / 2 (min) |
|---|---|---|---|---|
| Lift | SuperTrack | 1.1 ± 0.0 | 500 / 500 / 500 | 1.1 / 1.1 / 1.0 |
| | SuperMimic | 1.1 ± 0.0 | 500 / 500 / 500 | 1.2 / 1.1 / 1.1 |
| Square | SuperTrack | 3.6 ± 1.2 | 1,000 / 500 / 1,000 | 4.4 / 2.2 / 4.3 |
| | SuperMimic | 6.9 ± 0.0, 2 of 3 runs | 1,500 / 1,500 / not reached | 6.9 / 6.8 / not reached |
| Transport | SuperTrack | not reached | | |
| | SuperMimic | not reached | | |

Each checkpoint's success (500 / 1,000 / 1,500 iterations), seeds 0, 1, 2:

| task | SuperTrack | SuperMimic |
|---|---|---|
| Lift | 100 / 100 / 100 % in every seed | 99.2 / 100 / 100 %; 100 / 100 / 100 %; 100 / 100 / 100 % |
| Square | 91.0 / 100 / 100 %; 100 / 100 / 100 %; 84.0 / 100 / 100 % | 100 / 97.7 / 100 %; 100 / 90.2 / 100 %; 100 / 1.6 / 90.6 % |
| Transport | 88.3 / 92.2 / 77.3 %; 99.2 / 85.2 / 61.7 %; 97.3 / 65.6 / 60.5 % | 0.4 / 0.0 / 9.0 %; 7.8 / 1.6 / 9.4 %; 0.0 / 14.8 / 1.6 % |

## Renders

`tools/render_robomimic.py` renders a rollout recorded by `evaluate_robomimic`. `docs/media/robomimic/` holds, per task and
algorithm, the seed with the highest success and then the lowest object error (Lift seed 0 for both, Square SuperTrack
seed 2 and SuperMimic seed 0, Transport SuperTrack seed 0 and SuperMimic seed 1), from the unperturbed start, Transport at
four times real time, and the grid of the six (`tools/make_grid.py`). SuperMimic's Transport rollout terminates after
29 s of the demonstration's 36 s, as its other two seeds' do (after 19 and 29 s); the other five play the whole
demonstration.
