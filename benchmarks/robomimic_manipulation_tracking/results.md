# Manipulation demonstration tracking: results

A Panda arm (two for Transport) tracks robomimic's first proficient-human demonstration of Lift, Square and Transport on
MuJoCo-Warp, with SuperTrack and SuperMimic. First round: the development configurations, three training seeds each.

## How the numbers were obtained

- **Runs:** `--config robomimic-supertrack-dev` (`--lr 4e-4 --st_fail_w 2 --st_task_weights`) and `--config robomimic-supermimic-dev`
  (SuperMimic's defaults), both with the hand task's systems options; `--task <task> --demo 0 --solver_iterations 20`,
  1,500 iterations in 2,048 worlds on the schedule of the development runs (`--lr_decay_iters 4000`), seeds 0, 1, 2.
- **Evaluation:** `evaluate_robomimic` on each run's last checkpoint: 256 rollouts of the whole demonstration from perturbed
  starts, without exploration noise. A rollout succeeds if it never terminates and meets robosuite's success test at the
  demonstration's last frame; the object errors are means over the frames of the rollouts that do not terminate.
- **Reference targets:** the same rollouts with the reference's joint positions as the servos' targets and no policy.
- **Hardware:** one RTX 4090 (WSL2), shared by three training queues at once. The training times are wall clock under that
  sharing, upper bounds of a run alone, and not engineered.
- **Table:** mean ± sample SD over the seeds.

## Results

| task | algorithm | success ↑ | object position ↓ | object position at the last frame ↓ | episodes ended ↓ | reference targets: success | training (min) | iterations/s |
|---|---|---|---|---|---|---|---|---|
| Lift | SuperTrack | 100.0 ± 0.0 % | 0.60 ± 0.10 mm | 1.70 ± 0.17 mm | 0.0 % | 0.0 % | 6.8 ± 1.4 | 4.2 ± 2.1 |
| | SuperMimic | 100.0 ± 0.0 % | 0.80 ± 0.21 mm | 2.22 ± 1.2 mm | 0.0 % | 0.0 % | 8.5 ± 2.2 | 2.9 ± 0.6 |
| Square | SuperTrack | 100.0 ± 0.0 % | 1.96 ± 0.53 mm | 4.03 ± 0.80 mm | 0.0 % | 7.0 ± 2.7 % | 12.8 ± 3.0 | 2.3 ± 1.2 |
| | SuperMimic | 100.0 ± 0.0 % | 4.08 ± 0.85 mm | 3.57 ± 0.13 mm | 0.0 % | 5.5 ± 2.4 % | 13.2 ± 2.8 | 1.9 ± 0.4 |
| Transport | SuperTrack | 89.8 ± 5.4 % | 7.13 ± 0.75 mm | 32.1 ± 12 mm | 5.7 ± 3.9 % | 0.0 % | 19.2 ± 6.3 | 1.4 ± 0.2 |
| | SuperMimic | 37.1 ± 27 % | 12.3 ± 1.7 mm | 308 ± 180 mm | 59.6 ± 32 % | 0.0 % | 22.0 ± 2.4 | 1.0 ± 0.0 |

**In short:** SuperTrack for every task. Both algorithms track Lift and Square at a success of 100 % where the reference's
targets alone succeed at 0 and 5-7 %; on Transport, two arms and a 36 s demonstration, SuperTrack succeeds in 90 % of the
perturbed rollouts and SuperMimic in 37 %, with large differences between its seeds.

Open: one policy on several demonstrations (Square's ten in development reach a success of 100 %; Lift's cubes differ in
size between demonstrations), the time to a target, and the speed options of the G1 path.
