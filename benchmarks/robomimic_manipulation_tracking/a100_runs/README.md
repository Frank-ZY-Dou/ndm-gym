# Manipulation demonstration tracking: the A100 runs

The benchmark's 18 runs (`benchmarks/robomimic_manipulation_tracking/run_formal.sh`, unchanged, at bf7b1d4) and their
time to the target (`python benchmarks/robomimic_manipulation_tracking/time_to.py logs/robomimic_formal/*_s?`).

Per run, JSON records: `eval/metrics.json`, `time_to.json` (every kept checkpoint's evaluation, one every 100 iterations)
and `train.json` (the training log's last row: the iteration, the minutes, the environment steps and the iterations per
second; `ep_len`, the mean episode length in training at 80, 90 and 100 % of the run). No logs, no checkpoints.

The grid of these runs' renders (each case's seed with the highest success, then the lowest object error, from the
unperturbed start; Transport at four times real time): [media/grid.webp](media/grid.webp).

- **GPU:** NVIDIA A100-SXM4-80GB; each run alone on its GPU.
- **Software:** Python 3.12.9, PyTorch 2.11.0 (CUDA 12.6 build), warp-lang 1.17.0 (CUDA 12 build), MuJoCo and MuJoCo-Warp 3.15.0.
- **Compilation:** every run here had the kernels' compilation in the cache. `cold_cache_seed0/` holds each configuration's
  seed-0 run on an empty cache, 2.7 to 8.1 min longer.
- **Clock:** `train.json`'s `minutes`, from the first training iteration; a compilation during the run is in it,
  evaluations are not.
- Mean ± sample SD over seeds 0, 1, 2. An object position is the mean over the rollouts that do not terminate; a run in
  which every rollout terminates has none.

## Results

| task | algorithm | success ↑ | object position ↓ | object position at the last frame ↓ | episodes ended ↓ | reference targets: success | training (min) | 1,500 iterations / training time (/s) |
|---|---|---|---|---|---|---|---|---|
| Lift | SuperTrack | 100.0 ± 0.0 % | 0.62 ± 0.03 mm | 2.4 ± 0.5 mm | 0.0 ± 0.0 % | 0.0 ± 0.0 % | 3.2 ± 0.1 | 7.85 ± 0.14 |
|  | SuperMimic | 100.0 ± 0.0 % | 0.87 ± 0.16 mm | 2.6 ± 1.2 mm | 0.0 ± 0.0 % | 0.0 ± 0.0 % | 3.5 ± 0.0 | 7.19 ± 0.09 |
| Square | SuperTrack | 100.0 ± 0.0 % | 1.57 ± 0.09 mm | 3.6 ± 0.4 mm | 0.0 ± 0.0 % | 6.0 ± 0.5 % | 6.6 ± 0.0 | 3.80 ± 0.02 |
|  | SuperMimic | 84.9 ± 26.2 % | 5.34 ± 3.30 mm | 35.9 ± 52.6 mm | 10.7 ± 18.5 % | 5.7 ± 0.5 % | 6.9 ± 0.1 | 3.64 ± 0.03 |
| Transport | SuperTrack | 70.4 ± 16.8 % | 5.69 ± 0.18 mm | 90.3 ± 37.3 mm | 25.7 ± 12.9 % | 0.0 ± 0.0 % | 8.6 ± 0.0 | 2.90 ± 0.01 |
|  | SuperMimic | 29.7 ± 31.4 % | 10.56 ± 4.04 mm (2 of 3 seeds) | 247.4 ± 109.6 mm | 69.0 ± 32.5 % | 0.0 ± 0.0 % | 9.4 ± 0.1 | 2.67 ± 0.02 |

## Every seed

| run | success | object position | episodes ended | training (min) | to the target (min) | at iteration | checkpoints at the target (of 15) |
|---|---|---|---|---|---|---|---|
| lift_supertrack_s0 | 100.0 % | 0.64 mm | 0.0 % | 3.2 | 0.9 | 400 | 12 |
| lift_supertrack_s1 | 100.0 % | 0.59 mm | 0.0 % | 3.1 | 0.8 | 400 | 12 |
| lift_supertrack_s2 | 100.0 % | 0.62 mm | 0.0 % | 3.2 | 0.8 | 400 | 12 |
| lift_supermimic_s0 | 100.0 % | 0.69 mm | 0.0 % | 3.5 | 1.0 | 400 | 12 |
| lift_supermimic_s1 | 100.0 % | 0.90 mm | 0.0 % | 3.5 | 1.4 | 600 | 10 |
| lift_supermimic_s2 | 100.0 % | 1.00 mm | 0.0 % | 3.4 | 0.9 | 400 | 12 |
| square_supertrack_s0 | 100.0 % | 1.51 mm | 0.0 % | 6.6 | 5.8 | 1,300 | 10 |
| square_supertrack_s1 | 100.0 % | 1.52 mm | 0.0 % | 6.6 | 2.7 | 600 | 11 |
| square_supertrack_s2 | 100.0 % | 1.68 mm | 0.0 % | 6.5 | 2.6 | 600 | 10 |
| square_supermimic_s0 | 100.0 % | 4.03 mm | 0.0 % | 6.9 | 3.7 | 800 | 9 |
| square_supermimic_s1 | 54.7 % | 9.09 mm | 32.0 % | 6.9 | not reached |  | 4 |
| square_supermimic_s2 | 100.0 % | 2.88 mm | 0.0 % | 6.8 | 5.0 | 1,100 | 9 |
| transport_supertrack_s0 | 88.7 % | 5.76 mm | 10.9 % | 8.6 | not reached |  | 0 |
| transport_supertrack_s1 | 55.5 % | 5.83 mm | 34.8 % | 8.6 | not reached |  | 1 |
| transport_supertrack_s2 | 67.2 % | 5.49 mm | 31.2 % | 8.7 | not reached |  | 0 |
| transport_supermimic_s0 | 0.0 % | none (every rollout ended) | 100.0 % | 9.3 | not reached |  | 0 |
| transport_supermimic_s1 | 62.5 % | 7.71 mm | 35.2 % | 9.4 | not reached |  | 0 |
| transport_supermimic_s2 | 26.6 % | 13.42 mm | 71.9 % | 9.4 | not reached |  | 0 |
| lift_supertrack_s0 (cold cache) | 100.0 % | 0.53 mm | 0.0 % | 6.0 | 3.6 | 400 | 12 |
| lift_supermimic_s0 (cold cache) | 100.0 % | 0.68 mm | 0.0 % | 7.3 | 5.4 | 700 | 11 |
| square_supertrack_s0 (cold cache) | 100.0 % | 1.53 mm | 0.0 % | 11.0 | 6.1 | 400 | 12 |
| square_supermimic_s0 (cold cache) | 100.0 % | 4.96 mm | 0.0 % | 10.7 | 6.6 | 600 | 11 |
| transport_supertrack_s0 (cold cache) | 93.0 % | 4.22 mm | 6.6 % | 16.2 | not reached |  | 0 |
| transport_supermimic_s0 (cold cache) | 0.0 % | none (every rollout ended) | 100.0 % | 17.4 | not reached |  | 0 |
