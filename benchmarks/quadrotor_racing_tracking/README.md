# Quadrotor trajectory tracking on Race Against the Machine

A racing quadrotor tracks flown trajectories of [Race Against the Machine](https://github.com/tii-racing/drone-racing-dataset)
(Bosello et al., RA-L 2024; the data CC BY 4.0) with SuperTrack and SuperMimic, on MuJoCo-Warp
([the task](../../docs/tasks/quadrotor.md), [results](results.md)).

| file | |
|---|---|
| `tasks.json` | the benchmark's tasks: one autonomous flight of each shape (`ellipse`, `lemniscate`, `race_track`), one piloted flight of each (`ellipse_piloted`, `lemniscate_piloted`, `race_track_piloted`), the 18 autonomous flights (`all`) and the 18 piloted ones, at the dataset's release |
| `run.sh` | train and evaluate one task and seed |
| `time_to.sh` | the training time at which a run first reaches the target, or from which it holds it for a given number of iterations (its kept checkpoints, evaluated in order) |
| `summarize.py` | the tables of `results.md` from the runs |
| `results.md` | setup, rules, results |

## Rules, fixed before the runs

The contributed study's ([results](results.md#the-contributed-study-on-an-h200)); the named configurations' selection is in
[the README](../../README.md#quadrotor-trajectory-tracking).

- **Tasks:** one flight of each shape, the first autonomous flight of the ellipse, of the lemniscate and of the race track;
  then one policy on all 18 autonomous flights; then the first piloted flight of each shape (added after the autonomous
  results, under the same rules and configurations, without a selection of their own).
- **Algorithms:** SuperTrack and SuperMimic, each from the hand task's selected configuration (its starting point here),
  then any change that improves on it, reported beside the starting point.
- **Selection:** on the ellipse with training seed 0: among the configurations whose 256 perturbed rollouts all succeed, the
  one with the lowest mean position error.
- **Seeds:** three training seeds (0, 1, 2) per reported configuration; mean and sample SD.
- **Evaluation:** `ndm_gym.evaluate_quadrotor` on the last checkpoint: rollouts of the whole flight from perturbed starts,
  without exploration noise and without resets. A rollout succeeds if it never terminates (the position within 1 m of the
  reference). Beside them, the reference's feedforward command flown open loop.
- **Target of the training time:** a success of at least 0.99 with a mean position error below 20 mm.
- **Length:** a run is expected to take minutes; a configuration that needs more than an hour is treated as a defect.
