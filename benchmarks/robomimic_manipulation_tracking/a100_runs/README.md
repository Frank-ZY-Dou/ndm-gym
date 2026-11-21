# Manipulation demonstration tracking: the A100 runs

The benchmark's 18 runs (`benchmarks/robomimic_manipulation_tracking/run_formal.sh`, unchanged, at d427ce0) and their
time to the target (`python benchmarks/robomimic_manipulation_tracking/time_to.py logs/robomimic_formal/*_s?`).

Per run: `train_log.csv`, `eval/metrics.json` and `time_to.json`. No checkpoints.

- **GPU:** NVIDIA A100-SXM4-80GB; each run alone on its GPU.
- **Software:** Python 3.12.9, PyTorch 2.11.0 (CUDA 12.6 build), warp-lang 1.17.0 (CUDA 12 build), MuJoCo and MuJoCo-Warp 3.15.0.
- **Compilation:** every run here had the kernels' compilation in the cache. `cold_cache_seed0/` holds each configuration's
  seed-0 run on an empty cache, 4 to 15 min longer.
- **Clock:** `train_log.csv`'s `wall_h`, from the first training iteration; a compilation during the run is in it,
  evaluations are not.
- Mean ± sample SD over seeds 0, 1, 2.

## Results

| task | algorithm | success ↑ | object position ↓ | object position at the last frame ↓ | episodes ended ↓ | reference targets: success | training (min) | 1,500 iterations / training time (/s) |
|---|---|---|---|---|---|---|---|---|
| Lift | SuperTrack | 100.0 ± 0.0 % | 0.63 ± 0.04 mm | 1.9 ± 0.5 mm | 0.0 ± 0.0 % | 0.0 ± 0.0 % | 3.2 ± 0.0 | 7.92 ± 0.06 |
|  | SuperMimic | 100.0 ± 0.0 % | 0.79 ± 0.07 mm | 2.3 ± 0.6 mm | 0.0 ± 0.0 % | 0.0 ± 0.0 % | 3.4 ± 0.0 | 7.27 ± 0.07 |
| Square | SuperTrack | 100.0 ± 0.0 % | 1.78 ± 0.19 mm | 3.5 ± 1.8 mm | 0.0 ± 0.0 % | 5.9 ± 1.4 % | 6.5 ± 0.1 | 3.83 ± 0.05 |
|  | SuperMimic | 96.9 ± 5.4 % | 5.96 ± 2.70 mm | 7.9 ± 5.3 mm | 0.4 ± 0.7 % | 7.9 ± 1.0 % | 6.9 ± 0.1 | 3.63 ± 0.04 |
| Transport | SuperTrack | 66.5 ± 9.4 % | 5.11 ± 0.12 mm | 95.9 ± 39.7 mm | 27.7 ± 12.5 % | 0.0 ± 0.0 % | 8.5 ± 0.0 | 2.94 ± 0.02 |
|  | SuperMimic | 6.6 ± 4.4 % | 12.61 ± 3.62 mm | 328.8 ± 10.2 mm | 92.8 ± 4.9 % | 0.0 ± 0.0 % | 9.3 ± 0.0 | 2.69 ± 0.01 |

## Every seed

| run | success | object position | episodes ended | training (min) | to the target (min) | at iteration | success at 500 / 1,000 / 1,500 |
|---|---|---|---|---|---|---|---|
| lift_supertrack_s0 | 100.0 % | 0.60 mm | 0.0 % | 3.2 | 1.1 | 500 | 100.0 / 100.0 / 100.0 % |
| lift_supertrack_s1 | 100.0 % | 0.67 mm | 0.0 % | 3.2 | 1.1 | 500 | 100.0 / 100.0 / 100.0 % |
| lift_supertrack_s2 | 100.0 % | 0.61 mm | 0.0 % | 3.1 | 1.0 | 500 | 100.0 / 100.0 / 100.0 % |
| lift_supermimic_s0 | 100.0 % | 0.75 mm | 0.0 % | 3.5 | 1.2 | 500 | 99.2 / 100.0 / 100.0 % |
| lift_supermimic_s1 | 100.0 % | 0.87 mm | 0.0 % | 3.4 | 1.1 | 500 | 100.0 / 100.0 / 100.0 % |
| lift_supermimic_s2 | 100.0 % | 0.76 mm | 0.0 % | 3.4 | 1.1 | 500 | 100.0 / 100.0 / 100.0 % |
| square_supertrack_s0 | 100.0 % | 1.99 mm | 0.0 % | 6.6 | 4.4 | 1,000 | 91.0 / 100.0 / 100.0 % |
| square_supertrack_s1 | 100.0 % | 1.71 mm | 0.0 % | 6.5 | 2.2 | 500 | 100.0 / 100.0 / 100.0 % |
| square_supertrack_s2 | 100.0 % | 1.64 mm | 0.0 % | 6.4 | 4.3 | 1,000 | 84.0 / 100.0 / 100.0 % |
| square_supermimic_s0 | 100.0 % | 4.01 mm | 0.0 % | 6.9 | 6.9 | 1,500 | 100.0 / 97.7 / 100.0 % |
| square_supermimic_s1 | 100.0 % | 4.82 mm | 0.0 % | 6.8 | 6.8 | 1,500 | 100.0 / 90.2 / 100.0 % |
| square_supermimic_s2 | 90.6 % | 9.05 mm | 1.2 % | 7.0 | not reached |  | 100.0 / 1.6 / 90.6 % |
| transport_supertrack_s0 | 77.3 % | 5.07 mm | 13.3 % | 8.6 | not reached |  | 88.3 / 92.2 / 77.3 % |
| transport_supertrack_s1 | 61.7 % | 5.24 mm | 34.8 % | 8.5 | not reached |  | 99.2 / 85.2 / 61.7 % |
| transport_supertrack_s2 | 60.5 % | 5.01 mm | 35.2 % | 8.5 | not reached |  | 97.3 / 65.6 / 60.5 % |
| transport_supermimic_s0 | 9.0 % | 15.82 mm | 89.8 % | 9.3 | not reached |  | 0.4 / 0.0 / 9.0 % |
| transport_supermimic_s1 | 9.4 % | 8.69 mm | 90.2 % | 9.3 | not reached |  | 7.8 / 1.6 / 9.4 % |
| transport_supermimic_s2 | 1.6 % | 13.32 mm | 98.4 % | 9.3 | not reached |  | 0.0 / 14.8 / 1.6 % |
| lift_supertrack_s0 (cold cache) | 100.0 % | 0.53 mm | 0.0 % | 18.4 | 16.3 | 500 | 100.0 / 100.0 / 100.0 % |
| lift_supermimic_s0 (cold cache) | 100.0 % | 0.99 mm | 0.0 % | 12.0 | 9.7 | 500 | 100.0 / 100.0 / 100.0 % |
| square_supertrack_s0 (cold cache) | 100.0 % | 1.65 mm | 0.0 % | 11.2 | 6.9 | 500 | 100.0 / 100.0 / 100.0 % |
| square_supermimic_s0 (cold cache) | 100.0 % | 4.87 mm | 0.0 % | 10.8 | 8.5 | 1,000 | 100.0 / 100.0 / 100.0 % |
| transport_supertrack_s0 (cold cache) | 35.2 % | 8.82 mm | 59.0 % | 19.6 | not reached |  | 61.3 / 48.8 / 35.2 % |
| transport_supermimic_s0 (cold cache) | 1.2 % | 12.10 mm | 98.8 % | 15.3 | not reached |  | 0.0 / 1.2 / 1.2 % |
