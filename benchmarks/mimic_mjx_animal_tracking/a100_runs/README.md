# Animal motion tracking: the A100 runs

The benchmark's 18 runs (`benchmarks/mimic_mjx_animal_tracking/run_formal.sh supertrack` and `supermimic`, unchanged,
at 08f6d19) and their time to the target (`python benchmarks/mimic_mjx_animal_tracking/time_to.py logs/mimic_mjx_formal/{stick,fly}_*_s?`).

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
| stick insect | SuperTrack | 0.0 ± 0.0 % | 0.40 ± 0.01 mm | 0.66 ± 0.02 mm | 8.3 ± 0.2° | 2.7 ± 1.2 % | 0.75 ± 0.01 mm | 6.4 ± 0.1 |
|  | SuperMimic | 0.0 ± 0.0 % | 0.39 ± 0.00 mm | 0.59 ± 0.00 mm | 4.1 ± 0.0° | 0.0 ± 0.0 % | 0.84 ± 0.01 mm | 7.6 ± 0.1 |
| fruit fly | SuperTrack | 0.0 ± 0.0 % | 0.064 ± 0.003 mm | 0.120 ± 0.007 mm | 4.6 ± 0.1° | 6.0 ± 2.0 % | 0.097 ± 0.009 mm | 11.1 ± 0.0 |
|  | SuperMimic | 0.0 ± 0.0 % | 0.046 ± 0.001 mm | 0.077 ± 0.002 mm | 3.6 ± 0.1° | 0.0 ± 0.0 % | 0.064 ± 0.002 mm | 12.9 ± 0.0 |
| rat | SuperTrack | 68.7 ± 4.6 % | 35.2 ± 0.2 mm | 32.6 ± 1.0 mm | 11.9 ± 0.4° | 85.3 ± 1.2 % | 51.3 ± 1.9 mm | 22.2 ± 0.0 |
|  | SuperMimic | 20.7 ± 3.1 % | 22.7 ± 0.8 mm | 18.2 ± 1.3 mm | 9.8 ± 0.1° | 44.7 ± 9.0 % | 41.1 ± 2.0 mm | 24.8 ± 0.0 |

## Every seed

| run | ended | bodies | held-out: ended | held-out: bodies | training (min) | to the target (min) | at iteration |
|---|---|---|---|---|---|---|---|
| stick_supertrack_s0 | 0 % | 0.40 mm | 2 % | 0.74 mm | 6.5 | 2.1 | 1,250 |
| stick_supertrack_s1 | 0 % | 0.42 mm | 4 % | 0.76 mm | 6.3 | 2.0 | 1,250 |
| stick_supertrack_s2 | 0 % | 0.39 mm | 2 % | 0.75 mm | 6.3 | 2.0 | 1,250 |
| stick_supermimic_s0 | 0 % | 0.39 mm | 0 % | 0.83 mm | 7.6 | 2.5 | 1,250 |
| stick_supermimic_s1 | 0 % | 0.39 mm | 0 % | 0.85 mm | 7.6 | 2.5 | 1,250 |
| stick_supermimic_s2 | 0 % | 0.39 mm | 0 % | 0.84 mm | 7.5 | 2.3 | 1,250 |
| fly_supertrack_s0 | 0 % | 0.065 mm | 8 % | 0.091 mm | 11.1 | 10.7 | 5,750 |
| fly_supertrack_s1 | 0 % | 0.061 mm | 4 % | 0.107 mm | 11.1 | 2.5 | 1,250 |
| fly_supertrack_s2 | 0 % | 0.066 mm | 6 % | 0.091 mm | 11.1 | 2.4 | 1,250 |
| fly_supermimic_s0 | 0 % | 0.044 mm | 0 % | 0.062 mm | 12.9 | 2.8 | 1,250 |
| fly_supermimic_s1 | 0 % | 0.046 mm | 0 % | 0.062 mm | 13.0 | 3.3 | 1,500 |
| fly_supermimic_s2 | 0 % | 0.046 mm | 0 % | 0.066 mm | 12.9 | 2.8 | 1,250 |
| rat_supertrack_s0 | 66 % | 35.0 mm | 84 % | 49.4 mm | 22.2 |  |  |
| rat_supertrack_s1 | 66 % | 35.3 mm | 86 % | 53.2 mm | 22.2 |  |  |
| rat_supertrack_s2 | 74 % | 35.4 mm | 86 % | 51.4 mm | 22.2 |  |  |
| rat_supermimic_s0 | 20 % | 21.9 mm | 44 % | 39.9 mm | 24.8 |  |  |
| rat_supermimic_s1 | 24 % | 22.7 mm | 54 % | 43.4 mm | 24.8 |  |  |
| rat_supermimic_s2 | 18 % | 23.6 mm | 36 % | 39.8 mm | 24.8 |  |  |
| stick_supertrack_s0 (cold cache) | 0 % | 0.41 mm | 4 % | 0.82 mm | 12.4 | 8.0 | 1,250 |
| stick_supermimic_s0 (cold cache) | 0 % | 0.39 mm | 6 % | 1.05 mm | 14.9 | 9.7 | 1,250 |
| fly_supertrack_s0 (cold cache) | 2 % | 0.059 mm | 4 % | 0.075 mm | 17.1 | 13.0 | 3,750 |
| fly_supermimic_s0 (cold cache) | 0 % | 0.050 mm | 0 % | 0.067 mm | 20.9 | 10.8 | 1,250 |
| rat_supertrack_s0 (cold cache) | 70 % | 37.2 mm | 80 % | 47.0 mm | 33.0 |  |  |
| rat_supermimic_s0 (cold cache) | 22 % | 24.8 mm | 36 % | 40.4 mm | 37.5 |  |  |

The rat's runs have no `time_to.json`: the documented command covers the stick insect and the fruit fly.

## Extra runs

`extra/`, each alone on an A100-SXM4-80GB, on the warm cache:

- `rat_supermimic_40k_s0/`: `run_formal.sh`'s rat SuperMimic command of seed 0 with `--iterations 40000 --lr_decay_iters 40000`
  and a checkpoint every 1,000 iterations, and its two evaluations (results.md, "The rat with a longer budget"); `train.json`
  as above.
- `ittime_rat_supertrack_fused/`, `ittime_rat_supertrack_default/`: `run_formal.sh`'s rat SuperTrack command of seed 0 for
  1,000 iterations (`--iterations 1000 --lr_decay_iters 10000 --log_every 50`, no checkpoints), with its
  `--pi_batch 2048 --fused_fk` and without them; the default one compiled its kernels in its first iterations. `train.json`:
  the mean iterations per second of iterations 500 to 1,000 and the iteration time from it (results.md, "Development notes").
