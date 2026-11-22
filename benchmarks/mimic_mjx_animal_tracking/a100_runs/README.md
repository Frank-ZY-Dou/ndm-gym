# Animal motion tracking: the A100 runs

The benchmark's 18 runs (`benchmarks/mimic_mjx_animal_tracking/run_formal.sh supertrack` and `supermimic`, unchanged,
at eea1dd2: with `--ref_qvel positions`) and their time to the target (`python benchmarks/mimic_mjx_animal_tracking/time_to.py logs/mimic_mjx_formal/{stick,fly}_*_s?`).

Per run, JSON records: `eval_train/metrics.json` (clips 0-9), `eval_heldout/metrics.json` (clips 10-19), `train.json` (the
training log's last row: the iteration, the minutes, the environment steps and the iterations per second; `ep_len`, the mean
episode length in training at 80, 90 and 100 % of the run, where the log has a row at that iteration) and, for the stick
insect and the fruit fly, `time_to.json` (every kept checkpoint's evaluation). No logs, no checkpoints.

- **GPU:** NVIDIA A100-SXM4-80GB; each run alone on its GPU.
- **Software:** Python 3.12.9, PyTorch 2.11.0 (CUDA 12.6 build), warp-lang 1.17.0 (CUDA 12 build), MuJoCo and MuJoCo-Warp 3.15.0.
- **Compilation:** every run here had the kernels' compilation in the cache. `cold_cache_seed0/` holds each configuration's
  seed-0 run on an empty cache, 6 to 13 min longer.
- **Clock:** `train.json`'s `minutes`, from the first training iteration; a compilation during the run is in it,
  evaluations are not.
- Mean ± sample SD over seeds 0, 1, 2.

## Results

| body | algorithm | ended ↓ | bodies ↓ | end effectors ↓ | joints ↓ | held-out: ended ↓ | held-out: bodies ↓ | training (min) |
|---|---|---|---|---|---|---|---|---|
| stick insect | SuperTrack | 0.0 ± 0.0 % | 0.35 ± 0.01 mm | 0.60 ± 0.02 mm | 8.4 ± 0.2° | 10.7 ± 1.2 % | 0.64 ± 0.01 mm | 6.4 ± 0.1 |
|  | SuperMimic | 0.0 ± 0.0 % | 0.33 ± 0.00 mm | 0.56 ± 0.00 mm | 4.2 ± 0.1° | 0.0 ± 0.0 % | 0.74 ± 0.03 mm | 7.6 ± 0.1 |
| fruit fly | SuperTrack | 1.3 ± 2.3 % | 0.066 ± 0.006 mm | 0.123 ± 0.008 mm | 4.6 ± 0.6° | 9.3 ± 7.6 % | 0.091 ± 0.012 mm | 11.2 ± 0.0 |
|  | SuperMimic | 0.0 ± 0.0 % | 0.049 ± 0.002 mm | 0.086 ± 0.003 mm | 4.1 ± 0.4° | 0.0 ± 0.0 % | 0.064 ± 0.001 mm | 12.9 ± 0.0 |
| rat | SuperTrack | 38.7 ± 11.7 % | 13.6 ± 1.8 mm | 12.5 ± 1.9 mm | 10.7 ± 0.2° | 64.0 ± 15.9 % | 24.9 ± 5.7 mm | 22.4 ± 0.1 |
|  | SuperMimic | 16.0 ± 6.9 % | 20.1 ± 0.8 mm | 15.8 ± 0.3 mm | 9.1 ± 0.3° | 40.0 ± 2.0 % | 27.4 ± 1.9 mm | 25.0 ± 0.1 |

## Every seed

| run | ended | bodies | held-out: ended | held-out: bodies | training (min) | to the target (min) | at iteration |
|---|---|---|---|---|---|---|---|
| stick_supertrack_s0 | 0 % | 0.34 mm | 10 % | 0.63 mm | 6.6 | 1.8 | 1,000 |
| stick_supertrack_s1 | 0 % | 0.36 mm | 10 % | 0.64 mm | 6.4 | 1.6 | 1,000 |
| stick_supertrack_s2 | 0 % | 0.35 mm | 12 % | 0.64 mm | 6.4 | 1.6 | 1,000 |
| stick_supermimic_s0 | 0 % | 0.33 mm | 0 % | 0.74 mm | 7.7 | 2.6 | 1,250 |
| stick_supermimic_s1 | 0 % | 0.33 mm | 0 % | 0.76 mm | 7.6 | 1.9 | 1,000 |
| stick_supermimic_s2 | 0 % | 0.32 mm | 0 % | 0.70 mm | 7.5 | 1.9 | 1,000 |
| fly_supertrack_s0 | 0 % | 0.063 mm | 18 % | 0.105 mm | 11.2 | 6.2 | 3,250 |
| fly_supertrack_s1 | 0 % | 0.062 mm | 6 % | 0.084 mm | 11.1 | 10.7 | 5,750 |
| fly_supertrack_s2 | 4 % | 0.073 mm | 4 % | 0.085 mm | 11.1 | not reached |  |
| fly_supermimic_s0 | 0 % | 0.051 mm | 0 % | 0.065 mm | 12.9 | 2.8 | 1,250 |
| fly_supermimic_s1 | 0 % | 0.050 mm | 0 % | 0.064 mm | 12.9 | 2.8 | 1,250 |
| fly_supermimic_s2 | 0 % | 0.048 mm | 0 % | 0.063 mm | 13.0 | 2.9 | 1,250 |
| rat_supertrack_s0 | 30 % | 12.4 mm | 52 % | 22.1 mm | 22.4 |  |  |
| rat_supertrack_s1 | 34 % | 12.7 mm | 58 % | 21.0 mm | 22.3 |  |  |
| rat_supertrack_s2 | 52 % | 15.6 mm | 82 % | 31.4 mm | 22.3 |  |  |
| rat_supermimic_s0 | 20 % | 21.0 mm | 40 % | 27.4 mm | 25.2 |  |  |
| rat_supermimic_s1 | 8 % | 19.6 mm | 38 % | 25.5 mm | 24.9 |  |  |
| rat_supermimic_s2 | 20 % | 19.5 mm | 42 % | 29.3 mm | 24.9 |  |  |
| stick_supertrack_s0 (cold cache) | 0 % | 0.34 mm | 10 % | 0.61 mm | 12.5 | 7.7 | 1,000 |
| stick_supermimic_s0 (cold cache) | 0 % | 0.33 mm | 0 % | 0.71 mm | 15.2 | 10.0 | 1,250 |
| fly_supertrack_s0 (cold cache) | 0 % | 0.058 mm | 6 % | 0.077 mm | 16.8 | 13.6 | 4,250 |
| fly_supermimic_s0 (cold cache) | 0 % | 0.050 mm | 0 % | 0.064 mm | 20.9 | 10.8 | 1,250 |
| rat_supertrack_s0 (cold cache) | 32 % | 13.3 mm | 64 % | 23.9 mm | 33.1 |  |  |
| rat_supermimic_s0 (cold cache) | 20 % | 18.9 mm | 38 % | 28.6 mm | 38.6 |  |  |

The rat's runs have no `time_to.json`: the documented command covers the stick insect and the fruit fly.

## Extra runs

`extra/`, each alone on an A100-SXM4-80GB:

- `rat_supermimic_40k_s0/`: `run_formal.sh`'s rat SuperMimic command of seed 0 with `--iterations 40000 --lr_decay_iters 40000`
  and a checkpoint every 1,000 iterations, on a warm cache, and its two evaluations (results.md, "The rat with a longer
  budget"); `train.json` as above.
- `ittime_rat_supertrack_fused/`, `ittime_rat_supertrack_default/`: `run_formal.sh`'s rat SuperTrack command of seed 0 for
  1,000 iterations (`--iterations 1000 --lr_decay_iters 10000 --log_every 50`, no checkpoints), with its
  `--pi_batch 2048 --fused_fk` and without them; each compiled its kernels in its first 500 iterations. `train.json`:
  the mean iterations per second of iterations 500 to 1,000 and the iteration time from it (results.md, "Development notes").
