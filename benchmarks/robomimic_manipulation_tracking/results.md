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
- **Hardware and clock:** one RTX 4090 running three training runs at once, so the times are wall clock under that sharing
  and upper bounds of a run alone. The clock starts at the first training iteration, after the simulator's setup and the
  warm-up collection, and includes the compilation and the checkpoints' saving; evaluations are not in it. Time to the
  target, not engineered: there is no per-phase time table, and the times are not an A100's or H100's.
- Mean ± sample SD over the three training seeds.

```bash
benchmarks/robomimic_manipulation_tracking/run_formal.sh
```

## Results

| task | algorithm | success ↑ | object position ↓ | object position at the last frame ↓ | episodes ended ↓ | reference targets: success | training (min) | iterations/s |
|---|---|---|---|---|---|---|---|---|
| Lift | SuperTrack | 100.0 ± 0.0 % | 0.60 ± 0.10 mm | 1.70 ± 0.17 mm | 0.0 % | 0.0 % | 6.8 ± 1.4 | 4.2 ± 2.1 |
| | SuperMimic | 100.0 ± 0.0 % | 0.80 ± 0.21 mm | 2.22 ± 1.2 mm | 0.0 % | 0.0 % | 8.5 ± 2.2 | 2.9 ± 0.6 |
| Square | SuperTrack | 100.0 ± 0.0 % | 1.96 ± 0.53 mm | 4.03 ± 0.80 mm | 0.0 % | 7.0 ± 2.7 % | 12.8 ± 3.0 | 2.3 ± 1.2 |
| | SuperMimic | 100.0 ± 0.0 % | 4.08 ± 0.85 mm | 3.57 ± 0.13 mm | 0.0 % | 5.5 ± 2.4 % | 13.2 ± 2.8 | 1.9 ± 0.4 |
| Transport | SuperTrack | 89.8 ± 5.4 % | 7.13 ± 0.75 mm | 32.1 ± 12 mm | 5.7 ± 3.9 % | 0.0 % | 19.2 ± 6.3 | 1.4 ± 0.2 |
| | SuperMimic | 37.1 ± 27 % | 12.3 ± 1.7 mm | 308 ± 180 mm | 59.6 ± 32 % | 0.0 % | 22.0 ± 2.4 | 1.0 ± 0.0 |

Per seed (0 / 1 / 2), success and object position:

| task | SuperTrack | SuperMimic |
|---|---|---|
| Lift | 100 / 100 / 100 %; 0.48 / 0.66 / 0.66 mm | 100 / 100 / 100 %; 0.58 / 0.82 / 1.00 mm |
| Square | 100 / 100 / 100 %; 2.32 / 1.36 / 2.20 mm | 100 / 100 / 100 %; 5.03 / 3.42 / 3.77 mm |
| Transport | 88.7 / 85.2 / 95.7 %; 6.62 / 7.99 / 6.77 mm | 13.7 / 66.4 / 31.2 %; 13.37 / 10.36 / 13.29 mm |

On Lift and Square both algorithms succeed in every rollout, where the reference's targets alone succeed in 0 and 5-7 %.
On Transport (two arms, a 36 s demonstration) SuperTrack succeeds in 89.8 % and SuperMimic in 37.1 % of the rollouts. The
Transport runs still change between their last checkpoints (below), so these are the fixed budget's numbers, not final
ones.

## Time to the target

The target is the hand task's: a success of at least 0.99 with the tracked objects within 10 mm. `time_to.py` evaluates
every kept checkpoint (every 500 iterations) as the last one is, without the reference's targets, and reports the wall time
of the first checkpoint from which every later one meets the target. The checkpoints are 500 iterations apart and the
clock is the shared GPU's (above), so the times are upper bounds.

```bash
python benchmarks/robomimic_manipulation_tracking/time_to.py logs/robomimic_formal/*_s?
```

| task | algorithm | minutes to the target ↓ | iteration | seeds 0 / 1 / 2 (min) |
|---|---|---|---|---|
| Lift | SuperTrack | 2.5 ± 0.0 | 500 / 500 / 500 | 2.4 / 2.5 / 2.4 |
| | SuperMimic | 6.0 ± 4.8 | 1,500 / 1,000 / 500 | 11.0 / 5.4 / 1.4 |
| Square | SuperTrack | 4.9 ± 2.6 | 500 / 500 / 500 | 7.4 / 2.3 / 4.9 |
| | SuperMimic | 10.8 ± 4.9 | 1,000 / 1,500 / 1,000 | 7.8 / 16.4 / 8.3 |
| Transport | SuperTrack | not reached | | |
| | SuperMimic | not reached | | |

Each checkpoint's success (500 / 1,000 / 1,500 iterations), seeds 0, 1, 2:

| task | SuperTrack | SuperMimic |
|---|---|---|
| Lift | 100 / 100 / 100 % in every seed | 100 / 10.9 / 100 %; 28.5 / 100 / 100 %; 100 / 100 / 100 % |
| Square | 99.6 / 100 / 100 %; 100 / 100 / 100 %; 100 / 100 / 100 % | 30.9 / 100 / 100 %; 100 / 94.1 / 100 %; 41.4 / 100 / 100 % |
| Transport | 92.2 / 89.5 / 88.7 %; 89.1 / 83.6 / 85.2 %; 63.7 / 68.4 / 95.7 % | 15.6 / 0.0 / 13.7 %; 0.0 / 30.9 / 66.4 %; 0.0 / 2.3 / 31.2 % |

## Renders

`tools/render_robomimic.py` renders a rollout recorded by `evaluate_robomimic`. `docs/media/robomimic/` holds, per task and
algorithm, the seed with the highest success and then the lowest object error (Lift seed 0 for both, Square seed 1 for
both, Transport SuperTrack seed 2 and SuperMimic seed 1), from the unperturbed start, Transport at four times real time,
and the grid of the six (`tools/make_grid.py`).
