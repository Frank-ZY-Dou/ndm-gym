# Bimanual demonstration tracking: the A100 runs

The benchmark's 12 runs under its stopping rule (`benchmarks/aloha_bimanual_tracking/run_formal.sh`: trained to the cap
of 48,000 iterations, each stopped at the first checkpoint from 12,000 iterations on at which it has settled) with every
kept checkpoint's evaluation (`python benchmarks/aloha_bimanual_tracking/time_to.py logs/aloha_formal/*_s?`); each
configuration's seed 0 run once more on an empty kernel cache (`cold_cache_seed0/`); the 12 stopped checkpoints
evaluated again (`evaluated_again/`: with the same evaluation seed, with evaluation seed 1, with the scenes' own solver
settings; `metrics.json` each); the three tuning rounds' 16 runs each (`tuning_round/`, variants 1 to 4 as `v1` to `v4`,
and `tuning_round2/`, as `w1` to `w4`, on demonstration 1 of each task; `tuning_round3/`, as `x1` to `x4`, on Transfer
Cube's demonstration 1 and Insertion's demonstration 5); the runs at 12,000 iterations on demonstration 0 as they were
(`second_round_12000/`: the 12 runs, their `cold_cache_seed0/` and `evaluated_again/`); and the first round's runs at
6,000 iterations as they were (`first_round_6000/`, likewise).
[The results](../results.md) are computed from these files.

Per run, JSON records only (no logs, no checkpoints):

- `eval/metrics.json`: `evaluate_robomimic`'s evaluation of the checkpoint the run stops at, in the earlier rounds' and
  the tuning rounds' runs of the last checkpoint (the policy's 256 perturbed rollouts, the open-loop rollouts beside,
  the unperturbed rollout);
- `time_to.json`: every kept checkpoint's evaluation (the iteration, the training minutes at it, the success, the
  stages, the final state, the objects' error, the share ended early, whether it meets the target), the run's stop
  (`stop_iteration`, `stop_minutes`, whether it had settled there, and `after_stop`: the checkpoints after it and how
  many have a success within 0.05 of its) and the time to the target, counted up to the stop; in the earlier rounds'
  and the tuning rounds' runs the time to the target and whether the run had settled at its last checkpoint;
- `train.json`: the training log's row at the stop (`at_stop`) and at the cap (`at_cap`), in the earlier rounds' and the
  tuning rounds' runs its last row (the iteration, the minutes, the simulator's steps, iterations/s); the configuration
  and its options.

Notes:

- **GPU:** NVIDIA A100-SXM4-80GB; each run alone on its GPU.
- **Software:** Python 3.12.9, PyTorch 2.11.0 (CUDA 12.6 build), warp-lang 1.17.0 (CUDA 12 build), MuJoCo and MuJoCo-Warp
  3.15.0. The demonstrations: `tools/make_aloha_demos.py` with MuJoCo 2.3.7.
- **Demonstrations:** the 12 runs and their empty-cache runs track Transfer Cube's demonstration 0 and Insertion's
  demonstration 4; `second_round_12000/` and `first_round_6000/` track demonstration 0 of both tasks.
- **Compilation:** the 12 runs had the kernels' compilation in the cache (each run's job first trained 400 iterations
  with the same command on its own cache; so had the 12,000-iteration runs; the first round's 12 runs followed their
  configuration's empty-cache run on its cache). The runs in `cold_cache_seed0/` and the tuning rounds' started on an
  empty cache, and their minutes include the compilation.
- **Clock:** the training log's wall time from the first training iteration; evaluations are not in it.
- **A cross-check of the evaluation:** `second_round_12000/evaluated_again/same_seed_beside_the_later_runs/` has two of
  the 12,000-iteration runs' last checkpoints (SuperMimic on Transfer Cube, seeds 0 and 2) evaluated once more with the
  12 runs' code, beside them.
- **Checkpoints:** kept every 200 iterations: 240 per run (to the cap), 60 in the 12,000-iteration runs and the third
  tuning round's, 30 in the first round's runs, 60 or 120 in the second tuning round's. The first tuning round's were
  kept every 500 (12 per run); its evaluations precede the state test at the last frame (`final_state_rate`), which
  they do not have, and its time to the target and settled test are over its 500-iteration steps.

## Every run

The 12 runs under the stopping rule, their empty-cache runs and the third tuning round (a tuning run trains its budget
and its last checkpoint is the one evaluated):

| run | success | reward reached: 1 / 2 / 3 / 4 | final state | object position | episodes ended | stopped at iteration | settled | training (min) | to the target (min) | at iteration |
|---|---|---|---|---|---|---|---|---|---|---|
| transfer_cube_supertrack_s0 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.66 mm | 0.0 % | 12,000 | yes | 19.9 | 11.2 | 6,800 |
| transfer_cube_supertrack_s1 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.82 mm | 0.0 % | 12,000 | yes | 19.7 | 8.5 | 5,200 |
| transfer_cube_supertrack_s2 | 100.0 % | 100 / 100 / 100 / 100 % | 99.6 % | 1.15 mm | 0.0 % | 12,000 | yes | 19.8 | 10.1 | 6,200 |
| transfer_cube_supermimic_s0 | 98.8 % | 100 / 100 / 100 / 100 % | 60.5 % | 9.25 mm | 1.2 % | 16,600 | yes | 31.7 | 30.5 | 16,000 |
| transfer_cube_supermimic_s1 | 100.0 % | 100 / 100 / 100 / 100 % | 99.6 % | 4.48 mm | 0.0 % | 18,400 | yes | 34.8 | 33.6 | 17,800 |
| transfer_cube_supermimic_s2 | 3.1 % | 100 / 100 / 49 / 49 % | 2.7 % | 13.34 mm | 96.9 % | 48,000 | no | 90.9 | not reached |  |
| insertion_supertrack_s0 | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 4.96 mm | 0.0 % | 13,600 | yes | 26.3 | not reached |  |
| insertion_supertrack_s1 | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 4.11 mm | 0.0 % | 19,000 | yes | 37.2 | not reached |  |
| insertion_supertrack_s2 | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 7.19 mm | 0.0 % | 12,000 | yes | 23.1 | not reached |  |
| insertion_supermimic_s0 | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 7.84 mm | 0.0 % | 12,000 | yes | 26.0 | not reached |  |
| insertion_supermimic_s1 | 1.2 % | 97 / 92 / 39 / 1 % | 0.8 % | 13.75 mm | 28.1 % | 13,000 | yes | 28.3 | not reached |  |
| insertion_supermimic_s2 | 0.0 % | 100 / 100 / 0 / 0 % | 0.0 % | 9.49 mm | 0.0 % | 12,000 | yes | 26.3 | not reached |  |
| cold_cache_seed0/transfer_cube_supertrack_s0 | 100.0 % | 100 / 100 / 100 / 100 % | 82.0 % | 1.85 mm | 0.0 % | 12,200 | yes | 27.6 | 19.9 | 7,600 |
| cold_cache_seed0/transfer_cube_supermimic_s0 | 15.6 % | 100 / 100 / 84 / 84 % | 7.0 % | 13.39 mm | 84.4 % | 26,800 | yes | 55.4 | not reached |  |
| cold_cache_seed0/insertion_supertrack_s0 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.10 mm | 0.0 % | 12,000 | yes | 29.7 | 27.8 | 11,000 |
| cold_cache_seed0/insertion_supermimic_s0 | 0.0 % | 99 / 99 / 6 / 0 % | 0.0 % | 19.25 mm | 9.0 % | 14,600 | yes | 37.0 | not reached |  |
| tuning_round3/x1/transfer_cube_supertrack | 98.8 % | 100 / 100 / 99 / 99 % | 82.0 % | 2.06 mm | 1.2 % | 12,000 (its budget) | no | 23.6 | not reached |  |
| tuning_round3/x1/transfer_cube_supermimic | 80.1 % | 100 / 100 / 97 / 97 % | 53.1 % | 8.75 mm | 19.9 % | 12,000 (its budget) | no | 26.8 | not reached |  |
| tuning_round3/x1/insertion_supertrack | 0.0 % | 100 / 100 / 0 / 0 % | 0.0 % | 1.10 mm | 0.0 % | 12,000 (its budget) | yes | 27.7 | not reached |  |
| tuning_round3/x1/insertion_supermimic | 0.0 % | 98 / 95 / 57 / 0 % | 0.0 % | 8.60 mm | 14.8 % | 12,000 (its budget) | no | 30.8 | not reached |  |
| tuning_round3/x2/transfer_cube_supertrack | 2.0 % | 100 / 100 / 39 / 39 % | 0.0 % | 14.13 mm | 91.4 % | 12,000 (its budget) | no | 24.2 | not reached |  |
| tuning_round3/x2/transfer_cube_supermimic | 0.0 % | 100 / 9 / 7 / 7 % | 0.0 % | n/a | 100.0 % | 12,000 (its budget) | no | 27.1 | not reached |  |
| tuning_round3/x2/insertion_supertrack | 0.8 % | 100 / 100 / 38 / 1 % | 0.0 % | 12.05 mm | 59.8 % | 12,000 (its budget) | no | 27.7 | not reached |  |
| tuning_round3/x2/insertion_supermimic | 0.0 % | 98 / 97 / 2 / 0 % | 0.0 % | 14.49 mm | 4.3 % | 12,000 (its budget) | yes | 30.0 | not reached |  |
| tuning_round3/x3/transfer_cube_supertrack | 0.0 % | 100 / 100 / 1 / 1 % | 0.0 % | n/a | 100.0 % | 12,000 (its budget) | no | 24.7 | not reached |  |
| tuning_round3/x3/transfer_cube_supermimic | 0.0 % | 100 / 100 / 0 / 0 % | 0.0 % | n/a | 100.0 % | 12,000 (its budget) | no | 26.6 | not reached |  |
| tuning_round3/x3/insertion_supertrack | 0.4 % | 43 / 20 / 0 / 0 % | 0.4 % | 31.22 mm | 98.0 % | 12,000 (its budget) | no | 27.1 | not reached |  |
| tuning_round3/x3/insertion_supermimic | 0.0 % | 38 / 9 / 1 / 0 % | 0.0 % | 48.66 mm | 96.5 % | 12,000 (its budget) | yes | 30.3 | not reached |  |
| tuning_round3/x4/transfer_cube_supertrack | 0.0 % | 38 / 38 / 0 / 0 % | 0.0 % | n/a | 100.0 % | 12,000 (its budget) | no | 25.1 | not reached |  |
| tuning_round3/x4/transfer_cube_supermimic | 0.0 % | 100 / 100 / 0 / 0 % | 0.0 % | n/a | 100.0 % | 12,000 (its budget) | no | 27.4 | not reached |  |
| tuning_round3/x4/insertion_supertrack | 0.0 % | 100 / 100 / 0 / 0 % | 0.0 % | 20.45 mm | 0.0 % | 12,000 (its budget) | yes | 27.9 | not reached |  |
| tuning_round3/x4/insertion_supermimic | 0.0 % | 88 / 43 / 7 / 0 % | 0.4 % | 47.48 mm | 53.1 % | 12,000 (its budget) | no | 29.6 | not reached |  |

## The runs at 12,000 iterations

The 12 runs at 12,000 iterations on demonstration 0, their empty-cache runs and the second tuning round, as they were:

| run | success | reward reached: 1 / 2 / 3 / 4 | final state | object position | episodes ended | training (min) | to the target (min) | at iteration | settled |
|---|---|---|---|---|---|---|---|---|---|
| second_round_12000/transfer_cube_supertrack_s0 | 100.0 % | 100 / 100 / 100 / 100 % | 99.6 % | 2.69 mm | 0.0 % | 19.7 | 17.4 | 10,600 | yes |
| second_round_12000/transfer_cube_supertrack_s1 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.14 mm | 0.0 % | 19.8 | 11.8 | 7,200 | yes |
| second_round_12000/transfer_cube_supertrack_s2 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.56 mm | 0.0 % | 19.8 | 8.5 | 5,200 | yes |
| second_round_12000/transfer_cube_supermimic_s0 | 97.3 % | 100 / 100 / 99 / 99 % | 97.3 % | 6.98 mm | 2.7 % | 22.7 | not reached |  | no |
| second_round_12000/transfer_cube_supermimic_s1 | 99.2 % | 100 / 100 / 100 / 100 % | 39.5 % | 6.42 mm | 0.8 % | 22.7 | 21.9 | 11,600 | no |
| second_round_12000/transfer_cube_supermimic_s2 | 82.0 % | 98 / 97 / 84 / 84 % | 78.5 % | 6.48 mm | 18.0 % | 22.9 | not reached |  | no |
| second_round_12000/insertion_supertrack_s0 | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 5.14 mm | 0.0 % | 23.2 | not reached |  | yes |
| second_round_12000/insertion_supertrack_s1 | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 4.12 mm | 0.0 % | 23.2 | not reached |  | no |
| second_round_12000/insertion_supertrack_s2 | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 4.34 mm | 0.0 % | 23.1 | not reached |  | no |
| second_round_12000/insertion_supermimic_s0 | 0.0 % | 96 / 86 / 73 / 0 % | 0.0 % | 12.10 mm | 15.6 % | 26.2 | not reached |  | yes |
| second_round_12000/insertion_supermimic_s1 | 0.4 % | 95 / 94 / 61 / 0 % | 0.0 % | 11.24 mm | 7.0 % | 26.1 | not reached |  | yes |
| second_round_12000/insertion_supermimic_s2 | 0.0 % | 82 / 73 / 27 / 0 % | 0.0 % | 15.97 mm | 28.1 % | 26.3 | not reached |  | no |
| second_round_12000/cold_cache_seed0/transfer_cube_supertrack_s0 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.34 mm | 0.0 % | 25.0 | 20.0 | 9,000 | yes |
| second_round_12000/cold_cache_seed0/transfer_cube_supermimic_s0 | 100.0 % | 100 / 100 / 100 / 100 % | 89.1 % | 4.77 mm | 0.0 % | 26.9 | 23.1 | 10,000 | no |
| second_round_12000/cold_cache_seed0/insertion_supertrack_s0 | 0.0 % | 100 / 100 / 0 / 0 % | 0.0 % | 2.41 mm | 0.0 % | 28.1 | not reached |  | yes |
| second_round_12000/cold_cache_seed0/insertion_supermimic_s0 | 2.0 % | 82 / 64 / 47 / 2 % | 1.2 % | 13.48 mm | 41.8 % | 30.8 | not reached |  | no |
| tuning_round2/w1/transfer_cube_supertrack | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.17 mm | 0.0 % | 23.1 | 23.1 | 12,000 | yes |
| tuning_round2/w1/transfer_cube_supermimic | 95.3 % | 100 / 100 / 97 / 97 % | 75.0 % | 7.32 mm | 4.7 % | 26.1 | not reached |  | no |
| tuning_round2/w1/insertion_supertrack | 21.1 % | 100 / 100 / 21 / 21 % | 100.0 % | 1.67 mm | 0.0 % | 26.9 | not reached |  | no |
| tuning_round2/w1/insertion_supermimic | 0.0 % | 73 / 57 / 27 / 0 % | 0.0 % | 15.04 mm | 54.3 % | 30.0 | not reached |  | yes |
| tuning_round2/w2/transfer_cube_supertrack | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.49 mm | 0.0 % | 23.2 | 5.2 | 1,200 | yes |
| tuning_round2/w2/transfer_cube_supermimic | 62.9 % | 100 / 100 / 74 / 74 % | 1.6 % | 6.54 mm | 37.1 % | 26.2 | not reached |  | no |
| tuning_round2/w2/insertion_supertrack | 0.0 % | 100 / 100 / 0 / 0 % | 0.0 % | 1.71 mm | 0.0 % | 27.1 | not reached |  | yes |
| tuning_round2/w2/insertion_supermimic | 1.2 % | 24 / 17 / 12 / 1 % | 0.8 % | 25.66 mm | 85.5 % | 29.9 | not reached |  | no |
| tuning_round2/w3/transfer_cube_supertrack | 100.0 % | 100 / 100 / 100 / 100 % | 97.7 % | 1.21 mm | 0.0 % | 43.1 | 40.4 | 22,400 | yes |
| tuning_round2/w3/transfer_cube_supermimic | 78.9 % | 100 / 100 / 98 / 98 % | 36.7 % | 7.08 mm | 21.1 % | 48.3 | not reached |  | no |
| tuning_round2/w3/insertion_supertrack | 0.0 % | 100 / 100 / 0 / 0 % | 0.0 % | 1.21 mm | 0.0 % | 52.2 | not reached |  | no |
| tuning_round2/w3/insertion_supermimic | 0.0 % | 100 / 100 / 80 / 0 % | 0.0 % | 8.04 mm | 0.0 % | 55.9 | not reached |  | yes |
| tuning_round2/w4/transfer_cube_supertrack | 100.0 % | 100 / 100 / 100 / 100 % | 99.2 % | 1.54 mm | 0.0 % | 43.0 | 34.4 | 18,800 | yes |
| tuning_round2/w4/transfer_cube_supermimic | 26.6 % | 100 / 100 / 66 / 66 % | 10.9 % | 6.95 mm | 73.4 % | 48.7 | not reached |  | no |
| tuning_round2/w4/insertion_supertrack | 38.7 % | 100 / 100 / 95 / 39 % | 38.7 % | 2.70 mm | 0.0 % | 50.4 | not reached |  | no |
| tuning_round2/w4/insertion_supermimic | 0.0 % | 98 / 96 / 83 / 0 % | 0.0 % | 6.98 mm | 5.9 % | 56.0 | not reached |  | yes |

## The first round's runs

The 12 runs at 6,000 iterations, their empty-cache runs and the first tuning round, as they were:

| run | success | reward reached: 1 / 2 / 3 / 4 | final state | object position | episodes ended | training (min) | to the target (min) | at iteration | settled |
|---|---|---|---|---|---|---|---|---|---|
| first_round_6000/transfer_cube_supertrack_s0 | 100.0 % | 100 / 100 / 100 / 100 % | 73.0 % | 2.22 mm | 0.0 % | 10.0 | 4.8 | 2,800 | yes |
| first_round_6000/transfer_cube_supertrack_s1 | 30.9 % | 100 / 100 / 31 / 31 % | 30.9 % | 2.07 mm | 69.1 % | 10.2 | not reached |  | no |
| first_round_6000/transfer_cube_supertrack_s2 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.37 mm | 0.0 % | 9.9 | 8.9 | 5,400 | yes |
| first_round_6000/transfer_cube_supermimic_s0 | 69.1 % | 100 / 100 / 100 / 100 % | 38.7 % | 13.40 mm | 30.9 % | 11.5 | not reached |  | no |
| first_round_6000/transfer_cube_supermimic_s1 | 0.4 % | 98 / 70 / 0 / 0 % | 0.0 % | 23.96 mm | 98.8 % | 11.6 | not reached |  | no |
| first_round_6000/transfer_cube_supermimic_s2 | 69.5 % | 100 / 100 / 95 / 95 % | 0.8 % | 10.13 mm | 30.5 % | 11.3 | not reached |  | no |
| first_round_6000/insertion_supertrack_s0 | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 1.92 mm | 0.0 % | 11.8 | not reached |  | no |
| first_round_6000/insertion_supertrack_s1 | 0.0 % | 100 / 100 / 3 / 0 % | 1.6 % | 1.60 mm | 0.0 % | 11.7 | not reached |  | no |
| first_round_6000/insertion_supertrack_s2 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.84 mm | 0.0 % | 11.7 | 7.8 | 4,000 | yes |
| first_round_6000/insertion_supermimic_s0 | 0.4 % | 55 / 38 / 9 / 0 % | 0.4 % | 20.85 mm | 68.8 % | 13.1 | not reached |  | no |
| first_round_6000/insertion_supermimic_s1 | 0.0 % | 1 / 1 / 0 / 0 % | 0.0 % | 22.15 mm | 99.2 % | 12.9 | not reached |  | no |
| first_round_6000/insertion_supermimic_s2 | 0.0 % | 0 / 0 / 0 / 0 % | 0.0 % | n/a | 100.0 % | 13.0 | not reached |  | no |
| first_round_6000/cold_cache_seed0/transfer_cube_supertrack_s0 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.08 mm | 0.0 % | 14.7 | 6.1 | 800 | yes |
| first_round_6000/cold_cache_seed0/transfer_cube_supermimic_s0 | 0.0 % | 35 / 18 / 0 / 0 % | 0.0 % | n/a | 100.0 % | 16.1 | not reached |  | no |
| first_round_6000/cold_cache_seed0/insertion_supertrack_s0 | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 4.31 mm | 0.0 % | 23.0 | not reached |  | no |
| first_round_6000/cold_cache_seed0/insertion_supermimic_s0 | 0.0 % | 48 / 45 / 6 / 0 % | 0.0 % | 14.89 mm | 58.2 % | 18.5 | not reached |  | no |
| tuning_round/v1/transfer_cube_supertrack | 91.8 % | 100 / 100 / 92 / 92 % | not recorded | 2.52 mm | 8.2 % | 12.6 | not reached |  | no |
| tuning_round/v1/transfer_cube_supermimic | 0.0 % | 0 / 0 / 0 / 0 % | not recorded | n/a | 100.0 % | 14.6 | not reached |  | no |
| tuning_round/v1/insertion_supertrack | 0.0 % | 100 / 100 / 0 / 0 % | not recorded | 2.24 mm | 0.0 % | 15.1 | not reached |  | yes |
| tuning_round/v1/insertion_supermimic | 0.0 % | 96 / 93 / 23 / 0 % | not recorded | 15.43 mm | 14.8 % | 17.5 | not reached |  | yes |
| tuning_round/v2/transfer_cube_supertrack | 100.0 % | 100 / 100 / 100 / 100 % | not recorded | 1.98 mm | 0.0 % | 18.5 | 8.3 | 2,000 | yes |
| tuning_round/v2/transfer_cube_supermimic | 16.8 % | 100 / 64 / 48 / 48 % | not recorded | 11.93 mm | 83.2 % | 20.1 | not reached |  | no |
| tuning_round/v2/insertion_supertrack | 0.0 % | 4 / 3 / 3 / 0 % | not recorded | n/a | 100.0 % | 20.5 | not reached |  | no |
| tuning_round/v2/insertion_supermimic | 3.1 % | 94 / 92 / 53 / 3 % | not recorded | 11.63 mm | 19.9 % | 24.8 | not reached |  | yes |
| tuning_round/v3/transfer_cube_supertrack | 100.0 % | 100 / 100 / 100 / 100 % | not recorded | 2.30 mm | 0.0 % | 14.4 | 7.0 | 1,500 | yes |
| tuning_round/v3/transfer_cube_supermimic | 99.2 % | 100 / 100 / 100 / 100 % | not recorded | 9.66 mm | 0.8 % | 17.4 | 17.4 | 6,000 | no |
| tuning_round/v3/insertion_supertrack | 0.0 % | 100 / 100 / 0 / 0 % | not recorded | 1.83 mm | 0.0 % | 16.3 | not reached |  | no |
| tuning_round/v3/insertion_supermimic | 0.0 % | 100 / 100 / 1 / 0 % | not recorded | 9.89 mm | 0.0 % | 17.7 | not reached |  | no |
| tuning_round/v4/transfer_cube_supertrack | 100.0 % | 100 / 100 / 100 / 100 % | not recorded | 2.53 mm | 0.0 % | 17.8 | 7.7 | 2,000 | no |
| tuning_round/v4/transfer_cube_supermimic | 91.4 % | 100 / 99 / 91 / 91 % | not recorded | 8.10 mm | 8.6 % | 20.4 | not reached |  | no |
| tuning_round/v4/insertion_supertrack | 0.0 % | 100 / 100 / 0 / 0 % | not recorded | 9.72 mm | 0.0 % | 21.5 | not reached |  | no |
| tuning_round/v4/insertion_supermimic | 1.2 % | 98 / 98 / 66 / 1 % | not recorded | 11.65 mm | 9.4 % | 23.1 | not reached |  | no |
