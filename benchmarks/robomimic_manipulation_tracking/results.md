# Manipulation demonstration tracking: results

Contributed by [Jingdong Zhang](https://github.com/Evergreen0929) ([#10](https://github.com/Frank-ZY-Dou/ndm-gym/pull/10)).

SuperTrack and SuperMimic, with the library's MLP world model of generalized coordinates, make one Franka Panda (two for
Transport) track robomimic's first proficient-human demonstration of Lift, Square and Transport on MuJoCo-Warp
([the task](../../docs/tasks/robomimic.md), [the rules](README.md)).

## Setup

- **Reference:** demonstration 0 of each task (robomimic's v1.5 proficient-human files, revision `74fa018`), its last state
  held for 0.5 s: Lift 3.45 s, Square 6.85 s, Transport 36.2 s at 20 Hz.
- **Configurations** (`ndm_gym/configs.py`), each chosen by one rule among four ([below](#configurations)):
  - `robomimic-supertrack`: `--algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0 --st_task_weights` with the hand
    task's systems options;
  - `robomimic-supermimic`: `--algo supermimic` (learning rate 2e-4, action noise 0.1, discount 0.99) with the same
    systems options;
  - both: `--task <task> --demo 0 --solver_iterations 20`, 2,048 worlds, a cap of 6,000 iterations with the learning
    rate's cosine decay over it, a checkpoint kept every 250; training seeds 0, 1, 2.
- **Training:** one run per A100 40GB SXM4, three runs at a time on three GPUs of one machine, nothing else on them. Each
  kept checkpoint was evaluated as it appeared, on a fourth GPU, for the stop rule; the training ended at the stop.
- **Stop rule, the same for both algorithms:** each kept checkpoint on 256 perturbed starts (evaluation seed 1). With S(i)
  and E(i) the mean success and object error of the evaluations at i, i - 250 and i - 500, a run stops at the first i where
  E(i) > 0.97 E(i - 1,000) and either S(i) >= 0.99 or S(i) < S(i - 1,000) + 0.01; otherwise at 6,000. The checkpoint at
  the stop, not the best one, is the run's result.
- **Evaluation:** `ndm_gym.evaluate_robomimic` on the stop checkpoint: 256 rollouts of the whole demonstration from
  perturbed starts (0.01 rad on the arms' joints, evaluation seed 0, other starts than the stop rule's), without
  exploration noise. A rollout succeeds if it never terminates and meets robosuite's success test at the demonstration's
  last frame, read from the simulator as robosuite reads it. Errors are means over the frames of the rollouts that do not
  terminate. MuJoCo-Warp's contact order is not deterministic, so two evaluations of one checkpoint on the same starts
  differ: on Transport by about 4 points of success either way. Each number here is one evaluation.
- **Time to the target:** a success of at least 0.99 with the objects within 10 mm (the hand task's target); the training
  clock (from the first iteration, checkpoint writes included) of the first kept checkpoint from which every later one up
  to the stop meets it on the evaluation's starts. Not engineered: the hand task's systems options, no per-phase table.
- **Reference's targets, open loop:** the same rollouts with the demonstration's joint positions as the servos' targets
  and no policy.
- Mean ± sample SD over the three training seeds.

```bash
benchmarks/robomimic_manipulation_tracking/run_formal.sh
python benchmarks/robomimic_manipulation_tracking/time_to.py logs/robomimic_formal/*_s?
```

## Results

| | Lift | Square | Transport |
|---|---|---|---|
| success ↑, SuperTrack / SuperMimic | 100 / 100 % | 100 / 69.5 ± 52.8 % | 56.9 ± 16.3 / 54.9 ± 37.4 % |
| object position ↓, SuperTrack / SuperMimic | 0.41 ± 0.11 / 0.92 ± 0.18 mm | 1.45 ± 0.13 / 7.62 ± 7.06 mm | 5.42 ± 4.25 / 6.73 ± 2.10 mm |
| object position at the last frame ↓ | 0.9 ± 0.4 / 2.9 ± 0.8 mm | 3.6 ± 1.2 / 72.1 ± 118.2 mm | 70.8 ± 32.5 / 146.1 ± 131.9 mm |
| object rotation ↓ | 0.43 ± 0.14 / 0.80 ± 0.66° | 0.96 ± 0.27 / 6.94 ± 7.92° | 8.81 ± 10.45 / 7.27 ± 7.11° |
| grippers' keypoints ↓ | 1.53 ± 0.20 / 3.71 ± 0.30 mm | 2.12 ± 0.20 / 9.43 ± 4.86 mm | 2.87 ± 0.58 / 11.12 ± 0.43 mm |
| episodes ended ↓ | 0 / 0 % | 0 / 25.9 ± 44.9 % | 17.1 ± 8.8 / 41.9 ± 39.7 % |
| reference's targets, open loop: success | 0 % | 6.2 ± 1.4 % | 0 % |
| stop (iteration) | 3,083 ± 520 / 3,250 ± 901 | 3,750 ± 500 / 3,417 ± 804 | 4,167 ± 1,665 / 5,000 ± 901 |
| training to the stop (min) | 6.6 ± 1.1 / 7.6 ± 2.1 | 16.3 ± 2.2 / 15.8 ± 3.7 | 23.7 ± 9.5 / 31.1 ± 6.1 |
| iterations / s (301 to the stop) | 7.71 / 6.97 | 3.81 / 3.55 | 2.92 / 2.66 |

- **Lift and Square:** SuperTrack succeeds in every rollout of every seed; SuperMimic in every rollout of every seed on
  Lift and of seeds 1 and 2 on Square. SuperTrack's object error is the lower: 0.41 against 0.92 mm on Lift, 1.45 against
  7.62 mm on Square (3.93 and 3.16 mm in SuperMimic's seeds 1 and 2). The reference's joint positions alone succeed in 0
  and 6 % of the rollouts.
- **Transport:** success 56.9 ± 16.3 % for SuperTrack and 54.9 ± 37.4 % for SuperMimic, object error 5.42 and 6.73 mm,
  episodes ended 17.1 and 41.9 %; the seeds differ widely in both (below). Neither reaches the target.
- **The arms:** the grippers' keypoints stay closer to the demonstration's with SuperTrack on every task (1.53 / 2.12 /
  2.87 mm against 3.71 / 9.43 / 11.12 mm).

Every seed (0 / 1 / 2):

| task | | SuperTrack | SuperMimic |
|---|---|---|---|
| Lift | stop | 3,500 / 2,500 / 3,250 | 4,250 / 3,000 / 2,500 |
| | success; object | 100 / 100 / 100 %; 0.35 / 0.35 / 0.55 mm | 100 / 100 / 100 %; 0.75 / 1.11 / 0.92 mm |
| Square | stop | 4,250 / 3,250 / 3,750 | 2,500 / 4,000 / 3,750 |
| | success; object | 100 / 100 / 100 %; 1.41 / 1.33 / 1.59 mm | 8.6 / 100 / 100 %; 15.76 / 3.93 / 3.16 mm |
| | episodes ended | 0 / 0 / 0 % | 77.7 / 0 / 0 % |
| Transport | stop | 2,250 / 5,250 / 5,000 | 4,750 / 4,250 / 6,000 (the cap) |
| | success; object | 68.8 / 63.7 / 38.3 %; 10.32 / 2.91 / 3.02 mm | 12.9 / 67.6 / 84.4 %; 9.15 / 5.45 / 5.59 mm |
| | episodes ended | 8.6 / 26.2 / 16.4 % | 85.9 / 30.9 / 9.0 % |

SuperTrack's Transport seed 0 stopped early, at 2,250 (the rule's earliest stop is 1,750): over the last 1,000 iterations
the stop rule's evaluations saw its object error rise and its success not rise; it has the largest object error of the
three. SuperMimic's Transport seed 2 ran to the cap of 6,000; its Square seed 0 stopped at 2,500 with a success of 8.6 %.

## Time to the target

Per run, the training time at which the target holds from then on (above), with the object error at that checkpoint
beside it.

| task | algorithm | minutes to the target ↓ | iteration | object error there | seeds 0 / 1 / 2 (min) |
|---|---|---|---|---|---|
| Lift | SuperTrack | 1.04 ± 0.00 | 500 / 500 / 500 | 0.60 ± 0.15 mm | 1.04 / 1.04 / 1.04 |
| | SuperMimic | 1.32 ± 0.35 | 500 / 500 / 750 | 2.30 ± 0.99 mm | 1.11 / 1.13 / 1.72 |
| Square | SuperTrack | 2.14 ± 0.00 | 500 / 500 / 500 | 4.48 ± 1.73 mm | 2.14 / 2.14 / 2.13 |
| | SuperMimic | 6.88 ± 3.29 (2 of 3) | not reached / 2,000 / 1,000 | 4.53 ± 1.69 mm | not reached / 9.21 / 4.55 |
| Transport | SuperTrack | not reached | | | |
| | SuperMimic | not reached | | | |

On Lift and Square SuperTrack reaches the target at the second kept checkpoint (500 iterations) in every run, with the
lower object error on Lift (0.60 against 2.30 mm) and about the same on Square (4.48 against 4.53 mm).

## Configurations

Each algorithm's configuration is one of four, chosen by one rule before these runs: the hand task's accurate preset and
three single changes of the same kinds for both, each trained on the three tasks with the stop rule above (training seed 5)
and evaluated at its stop on 256 starts of evaluation seed 2; the highest mean success over the three tasks wins, and
within one point of it the lower mean object error. These runs ran on RTX 2080 Ti GPUs and serve the selection only:

| configuration | Lift: success, object, stop | Square | Transport | mean success | mean object |
|---|---|---|---|---|---|
| SuperTrack, the preset (imagination noise 0.2) | 100 %, 1.98 mm, 2,000 | 8.6 %, 20.44 mm, 2,500 | 92.2 %, 8.52 mm, 6,000 | 66.9 % | 10.31 mm |
| SuperTrack, imagination noise 0 (chosen) | 100 %, 0.41 mm, 2,250 | 100 %, 1.86 mm, 3,500 | 21.9 %, 4.24 mm, 3,000 | 74.0 % | 2.17 mm |
| SuperTrack, learning rate 2e-4 | 100 %, 0.94 mm, 2,000 | 0 %, -, 2,000 | 14.5 %, 20.53 mm, 2,500 | 38.2 % | - |
| SuperTrack, failure weight 5 | 0 %, 11.14 mm, 1,750 | 100 %, 3.13 mm, 5,750 | 95.3 %, 8.60 mm, 6,000 | 65.1 % | 7.63 mm |
| SuperMimic, the preset (chosen) | 100 %, 0.77 mm, 4,750 | 85.2 %, 8.59 mm, 3,000 | 30.1 %, 4.02 mm, 5,250 | 71.7 % | 4.46 mm |
| SuperMimic, action noise 0.3 | 100 %, 1.41 mm, 3,250 | 6.6 %, 30.94 mm, 1,750 | 65.6 %, 9.24 mm, 4,250 | 57.4 % | 13.86 mm |
| SuperMimic, learning rate 4e-4 | 100 %, 0.59 mm, 3,750 | 23.0 %, 19.84 mm, 4,500 | 25.8 %, 3.97 mm, 4,000 | 49.6 % | 8.13 mm |
| SuperMimic, discount 0.995 | 100 %, 1.64 mm, 3,500 | 98.4 %, 6.44 mm, 2,250 | 0 %, -, 6,000 | 66.1 % | - |

(-: every rollout terminated.) One seed per configuration: Transport and Square vary widely from seed to seed (the finals
above), so the choice is a coarse one.

## The contributed runs

The contributor's runs ([Jingdong Zhang](https://github.com/Evergreen0929)): seeds 0-2, each run alone on an A100 80GB
SXM4. They differ from the runs above in three ways:

- a fixed budget of 1,500 iterations (the learning rate on a schedule of 4,000, a checkpoint kept every 100), the last
  checkpoint evaluated;
- SuperTrack at imagination noise 0.1, the trainer's default;
- the earlier success tests: Square's without robosuite's release condition, Transport's by the objects' positions inside
  their bins instead of their contact with the bins' bases.

| | Lift | Square | Transport |
|---|---|---|---|
| success, SuperTrack / SuperMimic | 100 / 100 % | 100 / 84.9 % | 70.4 / 29.7 % |
| object position, SuperTrack / SuperMimic | 0.62 / 0.87 mm | 1.57 / 5.34 mm | 5.69 / 10.56 mm (SuperMimic: 2 of 3 seeds) |
| episodes ended, SuperTrack / SuperMimic | 0 / 0 % | 0 / 10.7 % | 25.7 / 69.0 % |
| to the target, SuperTrack / SuperMimic | 0.9 / 1.1 min | 3.7 / 4.3 min (SuperMimic: 2 of 3 runs) | not reached / not reached |

In the contributor's words: "The Transport runs of both algorithms and Square's SuperMimic run of seed 1 still change
between their last checkpoints (below), so these are the fixed budget's numbers, not final ones." Each checkpoint's success
(%), by its iteration, in bold where the checkpoint meets the target (the first 300 iterations fit the world model alone):

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

Every run's metrics, time to the target and training summary, as JSON records: [`a100_runs/`](a100_runs/README.md). The
grid of their renders, per case the seed with the highest success, then the lowest object error, from the unperturbed start
(Transport at four times real time; the six play the whole demonstration): [`a100_runs/media/grid.webp`](a100_runs/media/grid.webp).

## Renders

`tools/render_robomimic.py` renders a rollout recorded by `evaluate_robomimic`. `docs/media/robomimic/` holds, per task and
algorithm, the seed with the lowest object error (SuperTrack: Lift seed 0, Square seed 1, Transport seed 1; SuperMimic:
Lift seed 0, Square seed 2, Transport seed 1), each from the unperturbed start, which succeeds in all six, Transport at four
times real time, and the grid of the six (`tools/make_grid.py`).
