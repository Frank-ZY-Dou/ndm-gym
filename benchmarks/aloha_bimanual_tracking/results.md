# Bimanual demonstration tracking: results

SuperTrack and SuperMimic, with the library's MLP world model of generalized coordinates, make ALOHA's two arms track one
demonstration of ACT's simulated Transfer Cube and Insertion on MuJoCo-Warp ([the task](../../docs/tasks/aloha.md),
[the rules](README.md)).

## Setup

- **Configurations** (`ndm_gym/configs.py`), the manipulation task's algorithm options and the same run options for both:
  - **SuperTrack** (`--config aloha-supertrack`): `robomimic-supertrack`'s options (`--lr 4e-4 --st_fail_w 2
    --st_task_weights` and the hand task's systems options);
  - **SuperMimic** (`--config aloha-supermimic`): `robomimic-supermimic`'s options (SuperMimic's defaults with the hand
    task's systems options);
  - **both:** `--task aloha_<task> --demo 0 --solver_iterations 20`, 2,048 worlds, episodes of up to 500 control steps
    (`--episode 500`, the first tuning round's choice for both), 12,000 iterations with the learning rate's cosine over
    them (the second round's choice for both), a checkpoint kept every 200; seeds 0, 1, 2.
- **Tuning:** two rounds, each of the same four variants for both algorithms on demonstration 1 of each task (below);
  no option was chosen on demonstration 0, which the reported runs track. The task's definition (the action, the success
  test) was fixed before the first round, after trial runs that included demonstration 0. The second round entered the
  rules after the runs at the first round's 6,000 iterations, where some run of every case had not settled
  ([below](#first-round-the-runs-at-6000-iterations)), and before its own runs; the rules allow no third.
- **Budget and stopping:** one rule for both. Every run trains 12,000 iterations (the cap) and its last checkpoint is
  the one evaluated. A run has settled if its last five kept checkpoints agree: their successes within 0.05 of one
  another and their objects' errors within 1 mm or a quarter of their mean, whichever is larger. Where runs of a case
  did not settle, its numbers are the cap's and no comparison is drawn from them.
- **Evaluation:** `evaluate_robomimic` on the last checkpoint: 256 rollouts of the whole demonstration from perturbed
  starts (evaluation seed 0), without exploration noise. The objects' and the joints' errors are means over the frames of
  the rollouts that do not terminate.
- **Success:** ACT's staged reward (`get_reward` of its `TransferCubeTask` and `InsertionTask`, 0 to 4) read off
  MuJoCo-Warp's contacts after every control step; a rollout succeeds if it never terminates and its highest reward is 4
  (the cube touches the left gripper's finger and not the table; the peg touches the pin). "Reward reached" is the share
  of the rollouts whose highest reward is at least 1, 2, 3 and 4. These rollouts track one demonstration from perturbed
  starts: the number is not ACT's success rate of an imitation policy from images.
- **Final state:** beside it, the last stage as a function of the state at the demonstration's last frame: the cube off
  the table at the left gripper's grasp point with the gripper closed on it; the peg inside the socket's opening with its
  end within 2 mm of the pin's.
- **Open loop:** the same rollouts without a policy, under the reference's joint positions as the position targets, and
  under the targets that the demonstration recorded (ACT's actions, which the policy does not get).
- **Hardware and clock:** each run alone on one A100 80GB SXM4. The clock starts at the first training iteration, after
  the simulator's setup and the warm-up collection, and includes the checkpoints' saving; evaluations are not in it. The
  kernels' compilation was in the cache before these runs (each run's job first trained 400 iterations with the same
  command on its own cache; that run is not a result): on an empty cache a run takes 4.2 to 5.3 min longer (each
  configuration's seed 0 run once more on an empty cache, in `a100_runs/cold_cache_seed0/`). Time to the target, not
  engineered: there is no per-phase time table. Iterations/s is the training log's last `it_per_s`.
- Mean ± sample SD over the three training seeds. Each run's metrics, every checkpoint's evaluation and the time to the
  target: [`a100_runs/`](a100_runs/README.md).

```bash
benchmarks/aloha_bimanual_tracking/run_formal.sh
```

## Results

The last checkpoint (12,000 iterations):

| task | algorithm | success ↑ | reward reached: 1 / 2 / 3 / 4 | final state ↑ | object position ↓ | object rotation ↓ | arms' joints ↓ | episodes ended ↓ | runs settled | training (min) | iterations/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Transfer Cube | SuperTrack | 100.0 ± 0.0 % | 100 / 100 / 100 / 100 % | 99.9 ± 0.2 % | 1.79 ± 0.80 mm | 1.54 ± 0.43° | 0.029 ± 0.006 rad | 0.0 ± 0.0 % | 3 of 3 | 19.8 ± 0.1 | 10.11 ± 0.04 |
| | SuperMimic | 92.8 ± 9.4 % | 99 / 99 / 94 / 94 % | 71.7 ± 29.5 % | 6.63 ± 0.31 mm | 5.83 ± 0.24° | 0.106 ± 0.010 rad | 7.2 ± 9.4 % | 0 of 3 | 22.8 ± 0.1 | 8.83 ± 0.11 |
| Insertion | SuperTrack | 0.0 ± 0.0 % | 100 / 100 / 100 / 0 % | 0.0 ± 0.0 % | 4.53 ± 0.54 mm | 3.72 ± 1.25° | 0.065 ± 0.020 rad | 0.0 ± 0.0 % | 1 of 3 | 23.2 ± 0.0 | 8.70 ± 0.04 |
| | SuperMimic | 0.1 ± 0.2 % | 91 / 84 / 54 / 0 % | 0.0 ± 0.0 % | 13.10 ± 2.52 mm | 15.94 ± 4.30° | 0.131 ± 0.013 rad | 16.9 ± 10.6 % | 2 of 3 | 26.2 ± 0.1 | 7.61 ± 0.05 |

Per seed (0 / 1 / 2):

| task | algorithm | success | final state | object position | episodes ended |
|---|---|---|---|---|---|
| Transfer Cube | SuperTrack | 100.0 / 100.0 / 100.0 % | 99.6 / 100.0 / 100.0 % | 2.69 / 1.14 / 1.56 mm | 0.0 / 0.0 / 0.0 % |
| | SuperMimic | 97.3 / 99.2 / 82.0 % | 97.3 / 39.5 / 78.5 % | 6.98 / 6.42 / 6.48 mm | 2.7 / 0.8 / 18.0 % |
| Insertion | SuperTrack | 0.0 / 0.0 / 0.0 % | 0.0 / 0.0 / 0.0 % | 5.14 / 4.12 / 4.34 mm | 0.0 / 0.0 / 0.0 % |
| | SuperMimic | 0.0 / 0.4 / 0.0 % | 0.0 / 0.0 / 0.0 % | 12.10 / 11.24 / 15.97 mm | 15.6 / 7.0 / 28.1 % |

Open loop, without a policy, from the same 256 starts (the range over the twelve runs' evaluations, which repeat these
rollouts):

| task | targets | success | reward reached: 1 / 2 / 3 / 4 | final state | object position | episodes ended |
|---|---|---|---|---|---|---|
| Transfer Cube | the reference's joint positions | 0.0 % | 14 to 21 / 12 to 20 / 0 / 0 % | 0.0 % | none (every rollout ends) | 100.0 % |
| | the demonstration's recorded targets | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 0.82 to 0.83 mm | 0.0 % |
| Insertion | the reference's joint positions | 0.0 % | 0 / 0 / 0 / 0 % | 0.0 % | none (every rollout ends) | 100.0 % |
| | the demonstration's recorded targets | 33.6 to 39.5 % | 100 / 100 / 100 / 34 to 39 % | 33.6 to 39.5 % | 1.82 to 1.90 mm | 0.0 % |

SuperTrack's three runs on Transfer Cube settled. In each of the other three cases some run did not, so their numbers
are the cap's, and no comparison between the algorithms is drawn on either task. Per case:

- **Transfer Cube, SuperTrack:** every rollout hands the cube over in the three seeds, with the cube within 2.69 / 1.14 /
  1.56 mm of the demonstration's, and none ends early. The three runs settled: each one's last five checkpoints have a
  success of at least 99.6 %, with the cube's errors within 0.54 mm of one another.
- **Transfer Cube, SuperMimic:** 97.3 / 99.2 / 82.0 % of the rollouts succeed, with the cube within 6.98 / 6.42 /
  6.48 mm and 2.7 / 0.8 / 18.0 % of the rollouts ending early. No run settled: the successes of the last five
  checkpoints span 9.8 / 5.1 / 6.6 points, where the test allows 5. The state test at the last frame gives 97.3 / 39.5 /
  78.5 %.
- **Insertion, SuperTrack:** no rollout ends early in any seed and every one reaches the reward of 3 (the peg touches
  the socket, both off the table); at the last checkpoint the peg touches the pin in none (0 / 0 / 0 %). The peg and the
  socket are within 5.14 / 4.12 / 4.34 mm of the demonstration's over the rollout and 11.1 / 10.3 / 11.4 mm at its last
  frame. Over a run the share that touches the pin is close to all or none at each checkpoint (below): 5 / 7 / 7 of each
  seed's 60 checkpoints meet the target, the last of them at 8,600 / 10,800 / 11,000 iterations. Seed 0 settled, at 0 %.
  Seed 1's last five have a success of 51 / 5 / 2 / 0 / 0 %. Seed 2's are at 0 % and its objects' errors lie 1.259 mm
  apart, where the test allows 1.257 mm (a quarter of their mean).
- **Insertion, SuperMimic:** at most 0.4 % of the rollouts succeed; 73 / 61 / 27 % reach the reward of 3 and 15.6 / 7.0 /
  28.1 % end early, with the objects within 12.10 / 11.24 / 15.97 mm. No checkpoint of the three runs meets the target.
  Seeds 0 and 1 settled, at a success of at most 0.4 %.
- **Open loop:** under the reference's joint positions every rollout of both tasks ends early. Under the
  demonstration's recorded targets every Transfer Cube rollout succeeds; on Insertion the peg touches the pin in 33.6 to
  39.5 % of the rollouts, with the objects 6.2 to 6.7 mm from the demonstration's at the last frame.

The same command run twice gives two different runs. Each configuration's seed 0 was run once more, on an empty kernel
cache: on Transfer Cube SuperTrack's two runs end at a success of 100 and 100 % (the cube within 1.34 and 2.69 mm) and
SuperMimic's at 100 and 97.3 % (4.77 and 6.98 mm); on Insertion SuperTrack's end at 0 and 0 % (the objects within 2.41
and 5.14 mm) and SuperMimic's at 2.0 and 0.0 %.

## Checkpoints and time to the target

The target is the manipulation task's: a success of at least 0.99 with the tracked objects within 10 mm. `time_to.py`
evaluates every kept checkpoint (every 200 iterations, 60 per run) as the last one is, without the open-loop rollouts,
and reports the wall time of the first checkpoint from which every later one meets the target, and whether the run
settled. A time is its checkpoint's and an upper bound.

```bash
python benchmarks/aloha_bimanual_tracking/time_to.py logs/aloha_formal/*_s?
```

| task | algorithm | minutes to the target ↓ | iteration (seeds 0 / 1 / 2) | minutes (seeds 0 / 1 / 2) | settled (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|
| Transfer Cube | SuperTrack | 12.6 ± 4.5 | 10,600 / 7,200 / 5,200 | 17.4 / 11.8 / 8.5 | yes / yes / yes |
| | SuperMimic | 21.9, 1 of 3 runs | not reached / 11,600 / not reached | not reached / 21.9 / not reached | no / no / no |
| Insertion | SuperTrack | not reached | | | yes / no / no |
| | SuperMimic | not reached | | | yes / yes / no |

How often a run's checkpoints meet the target, and the success and the objects' error at its last five (what the settled
test reads):

| task | algorithm | checkpoints at the target, of 60 (seeds 0 / 1 / 2) | the first one: iteration | the first one: minutes | success at 11,200 / 11,400 / 11,600 / 11,800 / 12,000 iterations; seeds 0, 1, 2 | objects' error at them (mm); seeds 0, 1, 2 |
|---|---|---|---|---|---|---|
| Transfer Cube | SuperTrack | 45 / 53 / 52 | 1,200 / 600 / 1,000 | 1.9 / 0.9 / 1.6 | 100 / 100 / 100 / 100 / 100 %; 100 / 100 / 100 / 100 / 100 %; 100 / 100 / 100 / 100 / 100 % | 2.1 / 2.4 / 2.3 / 2.3 / 2.7; 1.3 / 1.2 / 1.2 / 1.1 / 1.1; 1.4 / 1.3 / 1.6 / 1.4 / 1.6 |
| | SuperMimic | 2 / 4 / 10 | 6,600 / 7,200 / 4,000 | 12.5 / 13.5 / 7.5 | 93 / 90 / 100 / 95 / 97 %; 95 / 98 / 100 / 100 / 99 %; 78 / 82 / 83 / 77 / 82 % | 9.8 / 10.8 / 8.2 / 8.9 / 7.0; 13.2 / 4.9 / 4.6 / 4.9 / 6.4; 7.2 / 6.8 / 6.4 / 6.8 / 6.5 |
| Insertion | SuperTrack | 5 / 7 / 7 | 800 / 1,800 / 400 | 1.5 / 3.4 / 0.8 | 0 / 0 / 0 / 0 / 0 %; 51 / 5 / 2 / 0 / 0 %; 0 / 0 / 0 / 0 / 0 % | 4.1 / 4.2 / 4.4 / 4.8 / 5.1; 10.1 / 12.8 / 11.3 / 6.0 / 4.1; 5.2 / 5.6 / 5.2 / 4.8 / 4.3 |
| | SuperMimic | 0 / 0 / 0 | | | 0 / 0 / 0 / 0 / 0 %; 0 / 0 / 0 / 0 / 0 %; 0 / 0 / 0 / 0 / 0 % | 12.8 / 11.9 / 12.9 / 12.8 / 12.1; 10.9 / 9.8 / 9.4 / 10.9 / 11.2; 15.0 / 10.4 / 14.0 / 11.1 / 16.0 |

On Transfer Cube SuperTrack's runs first meet the target after 600 to 1,200 iterations (0.9 to 1.9 min) and meet it at
45 to 53 of their 60 checkpoints; the time to the target counts from the last checkpoint that misses it (10,400 / 7,000 /
5,000 iterations, at a success of 95.7 / 43.0 / 65.2 %). SuperMimic's runs first meet it after 4,000 to 7,200 iterations
(7.5 to 13.5 min) and meet it at 2 / 4 / 10 checkpoints; seed 1 holds it over its last three (from 11,600 iterations,
21.9 min). On Insertion SuperTrack's checkpoints go between a success of 0 and 100 %: 169 of the three runs' 180
checkpoints have a success below 5 % or above 95 %, while the objects' error is between 2.3 and 32.6 mm from 1,400
iterations on (median 5.2 mm). ACT's last stage is one contact, the peg's end on the pin.

## The last checkpoints evaluated again

Each run's last checkpoint evaluated three more times (256 perturbed rollouts, without the open-loop ones): with the same
evaluation seed, with evaluation seed 1, and with the scenes' own solver settings (100 iterations in place of the cap of
20 that the runs train and evaluate with). Success, seeds 0 / 1 / 2:

| task | algorithm | the run's evaluation | again, the same seed | evaluation seed 1 | the scenes' own solver |
|---|---|---|---|---|---|
| Transfer Cube | SuperTrack | 100.0 / 100.0 / 100.0 % | 100.0 / 100.0 / 100.0 % | 99.6 / 100.0 / 100.0 % | 100.0 / 100.0 / 100.0 % |
| | SuperMimic | 97.3 / 99.2 / 82.0 % | 95.7 / 99.2 / 82.4 % | 98.0 / 99.2 / 79.7 % | 97.3 / 99.6 / 84.4 % |
| Insertion | SuperTrack | 0.0 / 0.0 / 0.0 % | 0.0 / 0.0 / 0.0 % | 0.0 / 0.0 / 0.0 % | 0.0 / 0.0 / 0.0 % |
| | SuperMimic | 0.0 / 0.4 / 0.0 % | 0.0 / 0.4 / 0.0 % | 0.0 / 0.4 / 0.0 % | 0.0 / 0.0 / 0.0 % |

The three further evaluations of a checkpoint are within 2.3 points of the run's own (SuperMimic's seed 2 on Transfer
Cube: 82.0 % against 79.7 and 84.4 %). The records: `a100_runs/evaluated_again/`.

## Tuning round

The manipulation task's configurations are the starting point, and its budget of 1,500 iterations is for a 20 Hz task,
so both algorithms got one round of the same runs: demonstration 1 of each task, seed 0, 6,000 iterations with the
cosine over them, four variants. Per algorithm the variant with the highest success at its last checkpoint, averaged
over the two tasks, is taken; among those within 0.02 of it the one with the lowest objects' position error (averaged
over the tasks, over the rollouts that do not terminate), then the one with the fewest changes.

| variant | algorithm | Transfer Cube: success, reward 3 reached, object position, episodes ended | Insertion: success, reward 3 reached, object position, episodes ended | mean success | mean object position | training (min): Transfer Cube, Insertion |
|---|---|---|---|---|---|---|
| 1 (nothing else) | SuperTrack | 91.8 %, 92.2 %, 2.52 mm, 8.2 % | 0.0 %, 0.0 %, 2.24 mm, 0.0 % | 45.9 % | 2.38 mm | 12.6, 15.1 |
| | SuperMimic | 0.0 %, 0.0 %, none, 100.0 % | 0.0 %, 23.0 %, 15.43 mm, 14.8 % | 0.0 % | none | 14.6, 17.5 |
| 2 (`--horizon 40`) | SuperTrack | 100.0 %, 100.0 %, 1.98 mm, 0.0 % | 0.0 %, 3.1 %, none, 100.0 % | 50.0 % | none | 18.5, 20.5 |
| | SuperMimic | 16.8 %, 48.0 %, 11.93 mm, 83.2 % | 3.1 %, 52.7 %, 11.63 mm, 19.9 % | 10.0 % | 11.78 mm | 20.1, 24.8 |
| 3 (`--episode 500`) | SuperTrack | 100.0 %, 100.0 %, 2.30 mm, 0.0 % | 0.0 %, 0.0 %, 1.83 mm, 0.0 % | 50.0 % | 2.06 mm | 14.4, 16.3 |
| | SuperMimic | 99.2 %, 100.0 %, 9.66 mm, 0.8 % | 0.0 %, 0.8 %, 9.89 mm, 0.0 % | 49.6 % | 9.78 mm | 17.4, 17.7 |
| 4 (both) | SuperTrack | 100.0 %, 100.0 %, 2.53 mm, 0.0 % | 0.0 %, 0.0 %, 9.72 mm, 0.0 % | 50.0 % | 6.12 mm | 17.8, 21.5 |
| | SuperMimic | 91.4 %, 91.4 %, 8.10 mm, 8.6 % | 1.2 %, 66.4 %, 11.65 mm, 9.4 % | 46.3 % | 9.87 mm | 20.4, 23.1 |

- **SuperTrack:** variants 2, 3 and 4 have the highest mean success (50.0 %); variant 2 has no object error on Insertion
  (every rollout ends early), and of the other two variant 3 has the lower one (2.06 against 6.12 mm). Variant 3.
- **SuperMimic:** variant 3 has the highest mean success (49.6 %) and none is within 0.02 of it. Variant 3.

Each run alone on an A100 80GB SXM4, on an empty kernel cache (the compilation is in the minutes), a checkpoint kept every
500 iterations; the round's runs were evaluated without the state test at the last frame. Every run of the round:
`a100_runs/tuning_round/`.

## Second tuning round

At the first round's 6,000 iterations some run of every case had not settled
([below](#first-round-the-runs-at-6000-iterations)), so the rules got one more round, fixed before its runs:
demonstration 1 of each task, seed 0, the same runs for both algorithms from the first round's choice (`--episode 500`),
four variants with the learning rate's cosine over each one's iterations and a checkpoint kept every 200:

1. 12,000 iterations;
2. 12,000 iterations with the policy's average (`--pi_ema 0.995`, the value of the hand task's fast configurations: the
   checkpoints hold the average and the evaluation plays it);
3. 24,000 iterations;
4. 24,000 iterations with the policy's average.

One variant is taken for both algorithms, so that their budgets stay the same: the one with the most settled runs among
its four (two algorithms, two tasks); among those, the one with the highest success at its last checkpoint averaged over
the four; among those within 0.02 of it the one with the lowest objects' position error, then the one with fewer
iterations, then the one without the average.

| variant | algorithm | Transfer Cube: success, final state, object position, episodes ended; settled; checkpoints at the target | Insertion: the same | training (min): Transfer Cube, Insertion |
|---|---|---|---|---|
| 1 (12,000 iterations) | SuperTrack | 100.0 %, 100.0 %, 1.17 mm, 0.0 %; settled; 50 of 60 | 21.1 %, 100.0 %, 1.67 mm, 0.0 %; not settled; 5 of 60 | 23.1, 26.9 |
| | SuperMimic | 95.3 %, 75.0 %, 7.32 mm, 4.7 %; not settled; 2 of 60 | 0.0 %, 0.0 %, 15.04 mm, 54.3 %; settled; 0 of 60 | 26.1, 30.0 |
| 2 (12,000, the policy's average) | SuperTrack | 100.0 %, 100.0 %, 1.49 mm, 0.0 %; settled; 55 of 60 | 0.0 %, 0.0 %, 1.71 mm, 0.0 %; settled; 15 of 60 | 23.2, 27.1 |
| | SuperMimic | 62.9 %, 1.6 %, 6.54 mm, 37.1 %; not settled; 8 of 60 | 1.2 %, 0.8 %, 25.66 mm, 85.5 %; not settled; 0 of 60 | 26.2, 29.9 |
| 3 (24,000 iterations) | SuperTrack | 100.0 %, 97.7 %, 1.21 mm, 0.0 %; settled; 90 of 120 | 0.0 %, 0.0 %, 1.21 mm, 0.0 %; not settled; 10 of 120 | 43.1, 52.2 |
| | SuperMimic | 78.9 %, 36.7 %, 7.08 mm, 21.1 %; not settled; 13 of 120 | 0.0 %, 0.0 %, 8.04 mm, 0.0 %; settled; 0 of 120 | 48.3, 55.9 |
| 4 (24,000, the policy's average) | SuperTrack | 100.0 %, 99.2 %, 1.54 mm, 0.0 %; settled; 112 of 120 | 38.7 %, 38.7 %, 2.70 mm, 0.0 %; not settled; 8 of 120 | 43.0, 50.4 |
| | SuperMimic | 26.6 %, 10.9 %, 6.95 mm, 73.4 %; not settled; 14 of 120 | 0.0 %, 0.0 %, 6.98 mm, 5.9 %; settled; 0 of 120 | 48.7, 56.0 |

| variant | runs settled, of 4 | mean success | mean object position |
|---|---|---|---|
| 1 (12,000 iterations) | 2 | 54.1 % | 6.30 mm |
| 2 (12,000, the policy's average) | 2 | 41.0 % | 8.85 mm |
| 3 (24,000 iterations) | 2 | 44.7 % | 4.38 mm |
| 4 (24,000, the policy's average) | 2 | 41.3 % | 4.54 mm |

- **The choice:** every variant has two settled runs of four, so the success decides: variant 1 has the highest
  (54.1 %) and none is within 0.02 of it. Variant 1: 12,000 iterations, without the policy's average.
- **What settled:** SuperTrack's Transfer Cube run in the four variants, at a success of 100 %; one more run per
  variant, each at a success of 0 % on Insertion (SuperMimic's in variants 1, 3 and 4, SuperTrack's in variant 2).
  SuperMimic's Transfer Cube run settled in none (95.3 / 62.9 / 78.9 / 26.6 % at the last checkpoint).
- **Insertion:** SuperTrack's runs end at a success of 21.1 / 0 / 0 / 38.7 % with the objects within 1.21 to 2.70 mm,
  and meet the target at 5 and 15 of 60 and at 10 and 8 of 120 checkpoints; SuperMimic's end at 0 to 1.2 % and meet it
  at none.

Each run alone on an A100 80GB SXM4, on an empty kernel cache (the compilation is in the minutes). Every run of the
round: `a100_runs/tuning_round2/`.

## First round: the runs at 6,000 iterations

The benchmark's first runs trained 6,000 iterations (the first tuning round's budget), with the same configurations,
demonstration, seeds and evaluation otherwise. They stay here as they were. Their kernels' compilation was in the cache;
on an empty cache a run took 5 to 11 min longer. The records: `a100_runs/first_round_6000/`.

The two budgets side by side, success at the last checkpoint and the runs settled (seeds 0 / 1 / 2):

| task | algorithm | 6,000 iterations | 12,000 iterations |
|---|---|---|---|
| Transfer Cube | SuperTrack | 100.0 / 30.9 / 100.0 %; 2 of 3 | 100.0 / 100.0 / 100.0 %; 3 of 3 |
| | SuperMimic | 69.1 / 0.4 / 69.5 %; 0 of 3 | 97.3 / 99.2 / 82.0 %; 0 of 3 |
| Insertion | SuperTrack | 0.0 / 0.0 / 100.0 %; 1 of 3 | 0.0 / 0.0 / 0.0 %; 1 of 3 |
| | SuperMimic | 0.4 / 0.0 / 0.0 %; 0 of 3 | 0.0 / 0.4 / 0.0 %; 2 of 3 |

The last checkpoint (6,000 iterations):

| task | algorithm | success ↑ | reward reached: 1 / 2 / 3 / 4 | final state ↑ | object position ↓ | object rotation ↓ | arms' joints ↓ | episodes ended ↓ | runs settled | training (min) | iterations/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Transfer Cube | SuperTrack | 77.0 ± 39.9 % | 100 / 100 / 77 / 77 % | 68.0 ± 34.8 % | 1.89 ± 0.45 mm | 1.77 ± 0.19° | 0.029 ± 0.006 rad | 23.0 ± 39.9 % | 2 of 3 | 10.0 ± 0.1 | 10.19 ± 0.08 |
| | SuperMimic | 46.4 ± 39.8 % | 99 / 90 / 65 / 65 % | 13.2 ± 22.1 % | 15.83 ± 7.23 mm | 27.29 ± 30.36° | 0.115 ± 0.017 rad | 53.4 ± 39.4 % | 0 of 3 | 11.5 ± 0.1 | 8.77 ± 0.24 |
| Insertion | SuperTrack | 33.3 ± 57.7 % | 100 / 100 / 68 / 33 % | 33.9 ± 57.3 % | 1.79 ± 0.17 mm | 1.18 ± 0.41° | 0.020 ± 0.001 rad | 0.0 ± 0.0 % | 1 of 3 | 11.7 ± 0.1 | 8.55 ± 0.03 |
| | SuperMimic | 0.1 ± 0.2 % | 18 / 13 / 3 / 0 % | 0.1 ± 0.2 % | 21.50 ± 0.92 mm (2 of 3 seeds) | 16.47 ± 15.14° (2 of 3 seeds) | 0.109 ± 0.000 rad (2 of 3 seeds) | 89.3 ± 17.8 % | 0 of 3 | 13.0 ± 0.1 | 7.72 ± 0.02 |

Per seed (0 / 1 / 2):

| task | algorithm | success | final state | object position | episodes ended |
|---|---|---|---|---|---|
| Transfer Cube | SuperTrack | 100.0 / 30.9 / 100.0 % | 73.0 / 30.9 / 100.0 % | 2.22 / 2.07 / 1.37 mm | 0.0 / 69.1 / 0.0 % |
| | SuperMimic | 69.1 / 0.4 / 69.5 % | 38.7 / 0.0 / 0.8 % | 13.40 / 23.96 / 10.13 mm | 30.9 / 98.8 / 30.5 % |
| Insertion | SuperTrack | 0.0 / 0.0 / 100.0 % | 0.0 / 1.6 / 100.0 % | 1.92 / 1.60 / 1.84 mm | 0.0 / 0.0 / 0.0 % |
| | SuperMimic | 0.4 / 0.0 / 0.0 % | 0.4 / 0.0 / 0.0 % | 20.85 / 22.15 mm / none (every rollout ends) | 68.8 / 99.2 / 100.0 % |

Open loop, without a policy, from the same 256 starts (the range over the twelve runs' evaluations, which repeat these
rollouts):

| task | targets | success | reward reached: 1 / 2 / 3 / 4 | final state | object position | episodes ended |
|---|---|---|---|---|---|---|
| Transfer Cube | the reference's joint positions | 0.0 % | 17 to 24 / 16 to 23 / 0 / 0 % | 0.0 % | none (every rollout ends) | 100.0 % |
| | the demonstration's recorded targets | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 0.82 to 0.83 mm | 0.0 % |
| Insertion | the reference's joint positions | 0.0 % | 0 / 0 / 0 / 0 % | 0.0 % | none (every rollout ends) | 100.0 % |
| | the demonstration's recorded targets | 34.0 to 38.7 % | 100 / 100 / 100 / 34 to 39 % | 34.0 to 38.7 % | 1.83 to 1.89 mm | 0.0 % |

Every case has runs that did not settle, so these are the cap's numbers and no comparison between the algorithms is drawn
from them. Per case:

- **Transfer Cube, SuperTrack:** every rollout hands the cube over in seeds 0 and 2, with the cube within 2.22 and
  1.37 mm of the demonstration's, and 30.9 % in seed 1, where 69.1 % of the rollouts end early. Seed 1 met the target at
  every checkpoint from 800 to 5,000 iterations; its last five have a success of 34 / 0 / 0 / 97 / 31 %.
- **Transfer Cube, SuperMimic:** 69.1 / 0.4 / 69.5 % of the rollouts succeed, with the cube 10 to 24 mm from the
  demonstration's and 31 to 99 % of the rollouts ending early; no checkpoint of the three runs meets the target.
- **Insertion, SuperTrack:** no rollout ends early in any seed and the peg and the socket stay within 1.60 to 1.92 mm
  of the demonstration's, 1.56 to 2.83 mm at its last frame. The peg touches the pin in 0 / 0 / 100 % of the rollouts at
  the last checkpoint. Over a run the share is close to all or none at each checkpoint (below): 8 / 2 / 12 of each
  seed's 30 checkpoints meet the target, and seed 2 does at every one from 4,000 iterations. The state test at the last
  frame gives 0 / 1.6 / 100 %.
- **Insertion, SuperMimic:** 69 to 100 % of the rollouts end early and at most 0.4 % succeed.
- **Open loop:** under the reference's joint positions every rollout of both tasks ends early. Under the
  demonstration's recorded targets every Transfer Cube rollout succeeds; on Insertion the peg touches the pin in 34.0 to
  38.7 % of the rollouts, with the objects 6.3 to 6.6 mm from the demonstration's at the last frame.

The same command run twice gives two different runs. Each configuration's seed 0 was run first on an empty kernel cache
and then again: on Transfer Cube SuperTrack's two runs end at a success of 100 and 100 % (the cube within 1.08 and
2.22 mm) and SuperMimic's at 0.0 and 69.1 %; on Insertion SuperTrack's end at 0.0 and 0.0 % (the objects within 4.31 and
1.92 mm) and SuperMimic's at 0.0 and 0.4 %.

### Checkpoints and time to the target at 6,000 iterations

Every kept checkpoint (every 200 iterations, 30 per run):

| task | algorithm | minutes to the target ↓ | iteration (seeds 0 / 1 / 2) | minutes (seeds 0 / 1 / 2) | settled (seeds 0 / 1 / 2) |
|---|---|---|---|---|---|
| Transfer Cube | SuperTrack | 6.8 ± 3.0, 2 of 3 runs | 2,800 / not reached / 5,400 | 4.8 / not reached / 8.9 | yes / no / yes |
| | SuperMimic | not reached | | | no / no / no |
| Insertion | SuperTrack | 7.8, 1 of 3 runs | not reached / not reached / 4,000 | not reached / not reached / 7.8 | no / no / yes |
| | SuperMimic | not reached | | | no / no / no |

How often a run's checkpoints meet the target, and the success at its last five:

| task | algorithm | checkpoints at the target, of 30 (seeds 0 / 1 / 2) | the first one: iteration | the first one: minutes | success at 5,200 / 5,400 / 5,600 / 5,800 / 6,000 iterations; seeds 0, 1, 2 |
|---|---|---|---|---|---|
| Transfer Cube | SuperTrack | 27 / 22 / 24 | 600 / 800 / 800 | 1.1 / 1.5 / 1.3 | 100 / 100 / 100 / 100 / 100 %; 34 / 0 / 0 / 97 / 31 %; 99 / 100 / 100 / 100 / 100 % |
| | SuperMimic | 0 / 0 / 0 | | | 55 / 69 / 58 / 61 / 69 %; 0 / 0 / 0 / 0 / 0 %; 3 / 17 / 0 / 0 / 70 % |
| Insertion | SuperTrack | 8 / 2 / 12 | 1,400 / 2,200 / 1,400 | 2.9 / 4.3 / 2.7 | 0 / 34 / 77 / 3 / 0 %; 0 / 0 / 0 / 0 / 0 %; 100 / 100 / 100 / 100 / 100 % |
| | SuperMimic | 0 / 0 / 0 | | | 0 / 0 / 0 / 0 / 0 %; 0 / 0 / 0 / 0 / 0 %; 0 / 1 / 0 / 0 / 0 % |

On Transfer Cube SuperTrack's runs first meet the target after 600 to 800 iterations (1.1 to 1.5 min) and meet it at 22
to 27 of their 30 checkpoints; the time to the target counts from the last checkpoint that misses it (2,600 and 5,200
iterations in seeds 0 and 2, at a success of 94.9 and 98.8 %). On Insertion SuperTrack's checkpoints go between a success
of 0 and 100 % while the objects' error stays between 1.5 and 4.5 mm from 1,400 iterations on: 87 of the three runs' 90
checkpoints have a success below 5 % or above 95 %. ACT's last stage is one contact, the peg's end on the pin.

### The last checkpoints at 6,000 iterations evaluated again

Success, seeds 0 / 1 / 2:

| task | algorithm | the run's evaluation | again, the same seed | evaluation seed 1 | the scenes' own solver |
|---|---|---|---|---|---|
| Transfer Cube | SuperTrack | 100.0 / 30.9 / 100.0 % | 100.0 / 39.8 / 100.0 % | 100.0 / 40.2 / 100.0 % | 100.0 / 36.7 / 100.0 % |
| | SuperMimic | 69.1 / 0.4 / 69.5 % | 63.3 / 0.0 / 73.8 % | 64.1 / 0.0 / 66.0 % | 67.2 / 0.4 / 68.0 % |
| Insertion | SuperTrack | 0.0 / 0.0 / 100.0 % | 0.0 / 0.0 / 100.0 % | 0.0 / 0.0 / 100.0 % | 0.0 / 0.0 / 100.0 % |
| | SuperMimic | 0.4 / 0.0 / 0.0 % | 0.4 / 0.0 / 0.0 % | 0.8 / 0.0 / 0.0 % | 0.4 / 0.0 / 0.0 % |

A checkpoint whose rollouts all succeed, or none, does the same in the four evaluations, on Insertion too; the other
checkpoints move by up to 9 points, also between two evaluations with the same seed (30.9 and 39.8 %). The records:
`a100_runs/first_round_6000/evaluated_again/`.

## Renders

`tools/render_robomimic.py` renders a rollout recorded by `evaluate_robomimic`. `docs/media/aloha/` holds, per task and
algorithm, the seed with the highest success and then the lowest object error (seed 1 in the four cases), from the
unperturbed start, and the grid of the four (`tools/make_grid.py`). The four rollouts play the whole demonstration. Both
Transfer Cube rollouts hand the cube over; in both Insertion rollouts the peg meets the socket off the table (ACT's
reward of 3) and does not touch the pin.
