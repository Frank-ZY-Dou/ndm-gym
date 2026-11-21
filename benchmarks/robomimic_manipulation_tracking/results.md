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
    a schedule of 4,000 (`--lr_decay_iters 4000`), a checkpoint kept every 100; seeds 0, 1, 2. Neither was tuned on these
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
  kernels' compilation was in the cache before these runs: on an empty cache a run takes 2.7 to 8.1 min longer (each
  configuration's first seed-0 run, in `a100_runs/cold_cache_seed0/`). Time to the target, not engineered: there is no
  per-phase time table. Iterations/s is the training log's last `it_per_s` (`train.json`).
- Mean ± sample SD over the three training seeds. Each run's metrics, time to the target and training summary, as JSON
  records: [`a100_runs/`](a100_runs/README.md).

```bash
benchmarks/robomimic_manipulation_tracking/run_formal.sh
```

## Results

| task | algorithm | success ↑ | object position ↓ | object position at the last frame ↓ | episodes ended ↓ | reference targets: success | training (min) | iterations/s |
|---|---|---|---|---|---|---|---|---|
| Lift | SuperTrack | 100.0 ± 0.0 % | 0.62 ± 0.03 mm | 2.35 ± 0.50 mm | 0.0 % | 0.0 % | 3.2 ± 0.1 | 7.94 ± 0.02 |
| | SuperMimic | 100.0 ± 0.0 % | 0.87 ± 0.16 mm | 2.64 ± 1.2 mm | 0.0 % | 0.0 % | 3.5 ± 0.0 | 7.18 ± 0.03 |
| Square | SuperTrack | 100.0 ± 0.0 % | 1.57 ± 0.09 mm | 3.55 ± 0.35 mm | 0.0 % | 6.0 ± 0.5 % | 6.6 ± 0.0 | 3.83 ± 0.02 |
| | SuperMimic | 84.9 ± 26.2 % | 5.34 ± 3.3 mm | 35.9 ± 53 mm | 10.7 ± 18.5 % | 5.7 ± 0.5 % | 6.9 ± 0.1 | 3.62 ± 0.05 |
| Transport | SuperTrack | 70.4 ± 16.8 % | 5.69 ± 0.18 mm | 90.3 ± 37 mm | 25.7 ± 12.9 % | 0.0 % | 8.6 ± 0.0 | 2.97 ± 0.01 |
| | SuperMimic | 29.7 ± 31.4 % | 10.6 ± 4.0 mm (2 of 3 seeds) | 247 ± 110 mm | 69.0 ± 32.5 % | 0.0 % | 9.4 ± 0.1 | 2.72 ± 0.02 |

Per seed (0 / 1 / 2), success and object position:

| task | SuperTrack | SuperMimic |
|---|---|---|
| Lift | 100 / 100 / 100 %; 0.64 / 0.59 / 0.62 mm | 100 / 100 / 100 %; 0.69 / 0.90 / 1.00 mm |
| Square | 100 / 100 / 100 %; 1.51 / 1.52 / 1.68 mm | 100 / 54.7 / 100 %; 4.03 / 9.09 / 2.88 mm |
| Transport | 88.7 / 55.5 / 67.2 %; 5.76 / 5.83 / 5.49 mm | 0 / 62.5 / 26.6 %; none / 7.71 / 13.42 mm |

On Lift both algorithms succeed in every rollout, and on Square SuperTrack in every rollout and SuperMimic in 84.9 %
(100 / 54.7 / 100 % over the seeds), where the reference's targets alone succeed in 0 and 5.5 to 6.2 %. On Transport
(two arms, a 36 s demonstration) SuperTrack succeeds in 70.4 % (88.7 / 55.5 / 67.2 %) and SuperMimic in
29.7 % (0 / 62.5 / 26.6 %) of the rollouts. In SuperMimic's Transport run of seed 0 every rollout terminates, so it has
no object error and that column's mean is over seeds 1 and 2. The Transport runs of both algorithms and Square's SuperMimic
run of seed 1 still change between their last checkpoints (below), so these are the fixed budget's numbers, not final ones.

## Time to the target

The target is the hand task's: a success of at least 0.99 with the tracked objects within 10 mm. `time_to.py` evaluates
every kept checkpoint (every 100 iterations) as the last one is, without the reference's targets, and reports the wall time
of the first checkpoint from which every later one meets the target. The checkpoints are 100 iterations apart, so a time
is its checkpoint's and an upper bound. A run fits the world model alone for its first 300 iterations (`--wm_only_iters`,
the trainer's default), so its checkpoints at 100, 200 and 300 hold a policy that has not been trained.

```bash
python benchmarks/robomimic_manipulation_tracking/time_to.py logs/robomimic_formal/*_s?
```

| task | algorithm | minutes to the target ↓ | iteration | seeds 0 / 1 / 2 (min) |
|---|---|---|---|---|
| Lift | SuperTrack | 0.9 ± 0.1 | 400 / 400 / 400 | 0.9 / 0.8 / 0.8 |
| | SuperMimic | 1.1 ± 0.3 | 400 / 600 / 400 | 1.0 / 1.4 / 0.9 |
| Square | SuperTrack | 3.7 ± 1.8 | 1,300 / 600 / 600 | 5.8 / 2.7 / 2.6 |
| | SuperMimic | 4.3 ± 0.9, 2 of 3 runs | 800 / not reached / 1,100 | 3.7 / not reached / 5.0 |
| Transport | SuperTrack | not reached | | |
| | SuperMimic | not reached | | |

Each checkpoint's success (%), by its iteration; in bold where the checkpoint meets the target:

| task | algorithm | seed | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1,000 | 1,100 | 1,200 | 1,300 | 1,400 | 1,500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lift | SuperTrack | 0 | 0 | 0 | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 1 | 0 | 0 | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 2 | 0 | 0 | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | SuperMimic | 0 | 0 | 0 | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 1 | 0 | 0 | 0 | 0 | 6.2 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 2 | 0 | 0 | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| Square | SuperTrack | 0 | 4.7 | 5.9 | 6.6 | 98.4 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | 93.0 | **100** | **100** | **100** |
| | | 1 | 2.0 | 10.2 | 4.3 | **100** | 91.0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 2 | 7.4 | 9.0 | 6.2 | 87.9 | 97.7 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | SuperMimic | 0 | 7.8 | 6.6 | 8.6 | 0 | 100 | **100** | 0 | **100** | **100** | **100** | **100** | **100** | **100** | **100** | **100** |
| | | 1 | 6.2 | 9.0 | 7.8 | 70.7 | 100 | **100** | **100** | 66.4 | **100** | 60.9 | 98.4 | 15.6 | **99.6** | 79.3 | 54.7 |
| | | 2 | 6.6 | 5.9 | 8.2 | 0 | 100 | **100** | **100** | **100** | **100** | 10.5 | **100** | **100** | **100** | **100** | **100** |
| Transport | SuperTrack | 0 | 0 | 0 | 0 | 59.0 | 82.0 | 96.5 | 72.7 | 83.6 | 34.4 | 48.0 | 60.9 | 55.9 | 39.5 | 43.0 | 88.7 |
| | | 1 | 0 | 0 | 0 | **100** | 95.3 | 7.4 | 83.6 | 62.9 | 58.6 | 69.1 | 54.7 | 20.7 | 48.4 | 61.3 | 55.5 |
| | | 2 | 0 | 0 | 0 | 73.0 | 94.9 | 69.5 | 93.4 | 25.4 | 59.8 | 78.5 | 40.2 | 60.5 | 62.1 | 73.4 | 67.2 |
| | SuperMimic | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3.5 | 0 | 0.8 | 0 |
| | | 1 | 0 | 0 | 0 | 0 | 1.2 | 0 | 0 | 0 | 0.4 | 27.7 | 55.1 | 64.1 | 84.0 | 21.9 | 62.5 |
| | | 2 | 0 | 0 | 0 | 0 | 0 | 0.8 | 12.9 | 0.4 | 23.0 | 2.3 | 31.2 | 30.9 | 34.0 | 27.3 | 26.6 |

On Lift every run of both algorithms meets the target at each checkpoint from 400 iterations, SuperMimic's run of seed 1
from 600. On Square, SuperTrack's runs first meet it at 500 / 400 / 600 iterations: the run of seed 0 misses it at 1,200
(93.0 %) and holds it from 1,300, the run of seed 1 misses it at 500 (91.0 %) and holds it from 600, the run of seed 2 holds
it from 600. SuperMimic's Square runs succeed in every rollout at 500 iterations with the nut 11.0 to 11.8 mm from its
demonstration, and first meet the target at 600: the run of seed 0 misses it at 700 (no rollout succeeds) and holds it
from 800, the run of seed 2 misses it at 1,000 (10.5 %) and holds it from 1,100, and the run of seed 1 meets it at 4 of its
15 checkpoints and not at the last (54.7 %). On Transport one checkpoint meets the target (SuperTrack's seed 1 at 400
iterations: 100 %, the objects within 8.3 mm); from 400 iterations SuperTrack's checkpoints have a success of 7.4 to
100 %, and SuperMimic's highest are 3.5 / 84.0 / 34.0 % (seeds 0 / 1 / 2).

## Renders

`tools/render_robomimic.py` renders a rollout recorded by `evaluate_robomimic`. `docs/media/robomimic/` holds, per task and
algorithm, the seed with the highest success and then the lowest object error (Lift SuperTrack seed 1 and SuperMimic
seed 0, Square SuperTrack seed 0 and SuperMimic seed 2, Transport SuperTrack seed 0 and SuperMimic seed 1), from the
unperturbed start, Transport at four times real time, and the grid of the six (`tools/make_grid.py`). The six play the
whole demonstration and pass the success test. Of the twelve runs not shown, the unperturbed rollout of five does not:
Square's SuperMimic run of seed 1 terminates after 4.1 s of 6.8 s; on Transport (36 s), SuperTrack's run of seed 1
terminates after 29.9 s and its run of seed 2 plays to the end without passing the test, and SuperMimic's runs of seeds 0
and 2 terminate after 22.4 and 29.3 s.
