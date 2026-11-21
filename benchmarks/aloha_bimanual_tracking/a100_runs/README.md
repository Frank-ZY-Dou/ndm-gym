# Bimanual demonstration tracking: the A100 runs

The benchmark's 12 runs (`benchmarks/aloha_bimanual_tracking/run_formal.sh`) with every kept checkpoint's evaluation
(`python benchmarks/aloha_bimanual_tracking/time_to.py logs/aloha_formal/*_s?`), each configuration's first seed-0 run
on an empty kernel cache (`cold_cache_seed0/`), the tuning round's 16 runs on demonstration 1 (`tuning_round/`,
variants 1 to 4 as `v1` to `v4`), and the 12 last checkpoints evaluated again (`evaluated_again/`: with the same
evaluation seed, with evaluation seed 1, with the scenes' own solver settings; `metrics.json` each).
[The results](../results.md) are computed from these files.

Per run, JSON records only (no logs, no checkpoints):

- `eval/metrics.json`: `evaluate_robomimic`'s evaluation of the last checkpoint (the policy's 256 perturbed rollouts, the
  open-loop rollouts beside, the unperturbed rollout);
- `time_to.json`: every kept checkpoint's evaluation (the iteration, the training minutes at it, the success, the
  stages, the final state, the objects' error, the share ended early, whether it meets the target), the time to the
  target and whether the run settled;
- `train.json`: the training log's last row (the iteration, the minutes, the simulator's steps, iterations/s), the
  configuration and its options.

Notes:

- **GPU:** NVIDIA A100-SXM4-80GB; each run alone on its GPU.
- **Software:** Python 3.12.9, PyTorch 2.11.0 (CUDA 12.6 build), warp-lang 1.17.0 (CUDA 12 build), MuJoCo and MuJoCo-Warp
  3.15.0. The demonstrations: `tools/make_aloha_demos.py` with MuJoCo 2.3.7.
- **Compilation:** the 12 runs had the kernels' compilation in the cache. The runs in `cold_cache_seed0/` and the tuning
  round's started on an empty cache, and their minutes include the compilation.
- **Clock:** the training log's wall time from the first training iteration; evaluations are not in it.
- **Checkpoints:** kept every 200 iterations (30 per run); the tuning round's every 500 (12 per run). The round's
  evaluations precede the state test at the last frame (`final_state_rate`), which they do not have, and its time to the
  target and settled test are over its 500-iteration steps.

## Every run

| run | success | reward reached: 1 / 2 / 3 / 4 | final state | object position | episodes ended | training (min) | to the target (min) | at iteration | settled |
|---|---|---|---|---|---|---|---|---|---|
| transfer_cube_supertrack_s0 | 100.0 % | 100 / 100 / 100 / 100 % | 73.0 % | 2.22 mm | 0.0 % | 10.0 | 4.8 | 2,800 | yes |
| transfer_cube_supertrack_s1 | 30.9 % | 100 / 100 / 31 / 31 % | 30.9 % | 2.07 mm | 69.1 % | 10.2 | not reached |  | no |
| transfer_cube_supertrack_s2 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.37 mm | 0.0 % | 9.9 | 8.9 | 5,400 | yes |
| transfer_cube_supermimic_s0 | 69.1 % | 100 / 100 / 100 / 100 % | 38.7 % | 13.40 mm | 30.9 % | 11.5 | not reached |  | no |
| transfer_cube_supermimic_s1 | 0.4 % | 98 / 70 / 0 / 0 % | 0.0 % | 23.96 mm | 98.8 % | 11.6 | not reached |  | no |
| transfer_cube_supermimic_s2 | 69.5 % | 100 / 100 / 95 / 95 % | 0.8 % | 10.13 mm | 30.5 % | 11.3 | not reached |  | no |
| insertion_supertrack_s0 | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 1.92 mm | 0.0 % | 11.8 | not reached |  | no |
| insertion_supertrack_s1 | 0.0 % | 100 / 100 / 3 / 0 % | 1.6 % | 1.60 mm | 0.0 % | 11.7 | not reached |  | no |
| insertion_supertrack_s2 | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.84 mm | 0.0 % | 11.7 | 7.8 | 4,000 | yes |
| insertion_supermimic_s0 | 0.4 % | 55 / 38 / 9 / 0 % | 0.4 % | 20.85 mm | 68.8 % | 13.1 | not reached |  | no |
| insertion_supermimic_s1 | 0.0 % | 1 / 1 / 0 / 0 % | 0.0 % | 22.15 mm | 99.2 % | 12.9 | not reached |  | no |
| insertion_supermimic_s2 | 0.0 % | 0 / 0 / 0 / 0 % | 0.0 % | n/a | 100.0 % | 13.0 | not reached |  | no |
| transfer_cube_supertrack_s0 (empty cache) | 100.0 % | 100 / 100 / 100 / 100 % | 100.0 % | 1.08 mm | 0.0 % | 14.7 | 6.1 | 800 | yes |
| transfer_cube_supermimic_s0 (empty cache) | 0.0 % | 35 / 18 / 0 / 0 % | 0.0 % | n/a | 100.0 % | 16.1 | not reached |  | no |
| insertion_supertrack_s0 (empty cache) | 0.0 % | 100 / 100 / 100 / 0 % | 0.0 % | 4.31 mm | 0.0 % | 23.0 | not reached |  | no |
| insertion_supermimic_s0 (empty cache) | 0.0 % | 48 / 45 / 6 / 0 % | 0.0 % | 14.89 mm | 58.2 % | 18.5 | not reached |  | no |
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
