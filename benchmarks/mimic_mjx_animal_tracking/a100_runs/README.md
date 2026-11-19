# Animal motion tracking: the A100 runs

The benchmark's 18 runs (`benchmarks/mimic_mjx_animal_tracking/run_formal.sh supertrack` and `supermimic`, unchanged,
at f51b8e5) and their time to the target (`python benchmarks/mimic_mjx_animal_tracking/time_to.py logs/mimic_mjx_formal/{stick,fly}_*_s?`).

Per run: `train_log.csv`, `eval_train/metrics.json` (clips 0-9), `eval_heldout/metrics.json` (clips 10-19) and, for the
stick insect and the fruit fly, `time_to.json`. No checkpoints.

- **GPU:** NVIDIA A100-SXM4-80GB; each run alone on its GPU (its own job). The two algorithms' queues of the animal task and
  the manipulation task's queue ran at the same time on different nodes; within a queue the runs ran one after another.
- **Software:** Python 3.12.9, PyTorch 2.11.0 (CUDA 12.6 build), warp-lang 1.17.0 (CUDA 12 build), MuJoCo and MuJoCo-Warp 3.15.0.
- **Seed 0:** each configuration's seed 0 ran first, on a cold compilation cache, so its clock held a one-time compilation
  of several minutes. The seed-0 runs here were run again on the warm cache, as seeds 1 and 2 were; the first seed-0 runs
  are in `cold_cache_seed0/`.
- **Clock:** `train_log.csv`'s `wall_h`, from the first training iteration; the compilation is in it, evaluations are not.
- Mean ± sample SD over seeds 0, 1, 2.

## Results

| body | algorithm | ended ↓ | bodies ↓ | end effectors ↓ | joints ↓ | held-out: ended ↓ | held-out: bodies ↓ | training (min) |
|---|---|---|---|---|---|---|---|---|
| stick insect | SuperTrack | 0.0 ± 0.0 % | 0.40 ± 0.01 mm | 0.66 ± 0.01 mm | 8.2 ± 0.1° | 3.3 ± 1.2 % | 0.75 ± 0.01 mm | 6.4 ± 0.1 |
|  | SuperMimic | 0.0 ± 0.0 % | 0.39 ± 0.00 mm | 0.59 ± 0.00 mm | 4.2 ± 0.1° | 2.0 ± 3.5 % | 0.87 ± 0.06 mm | 7.6 ± 0.1 |
| fruit fly | SuperTrack | 1.3 ± 2.3 % | 0.068 ± 0.012 mm | 0.125 ± 0.016 mm | 4.6 ± 0.1° | 6.7 ± 3.1 % | 0.096 ± 0.017 mm | 11.1 ± 0.0 |
|  | SuperMimic | 0.0 ± 0.0 % | 0.047 ± 0.002 mm | 0.079 ± 0.003 mm | 3.7 ± 0.0° | 0.0 ± 0.0 % | 0.065 ± 0.002 mm | 12.9 ± 0.0 |
| rat | SuperTrack | 70.7 ± 4.2 % | 34.7 ± 0.7 mm | 32.6 ± 0.6 mm | 12.0 ± 0.3° | 84.0 ± 5.3 % | 52.4 ± 0.5 mm | 22.1 ± 0.0 |
|  | SuperMimic | 19.3 ± 1.2 % | 21.3 ± 2.1 mm | 17.1 ± 0.4 mm | 9.8 ± 0.1° | 42.0 ± 12.5 % | 39.6 ± 6.2 mm | 24.9 ± 0.1 |

## Every seed

| run | ended | bodies | held-out: ended | held-out: bodies | training (min) | to the target (min) | at iteration |
|---|---|---|---|---|---|---|---|
| stick_supertrack_s0 | 0 % | 0.39 mm | 4 % | 0.73 mm | 6.5 | 3.3 | 2,000 |
| stick_supertrack_s1 | 0 % | 0.40 mm | 2 % | 0.75 mm | 6.5 | 1.7 | 1,000 |
| stick_supertrack_s2 | 0 % | 0.41 mm | 4 % | 0.76 mm | 6.3 | 3.2 | 2,000 |
| stick_supermimic_s0 | 0 % | 0.39 mm | 0 % | 0.84 mm | 7.6 | 3.8 | 2,000 |
| stick_supermimic_s1 | 0 % | 0.39 mm | 0 % | 0.82 mm | 7.7 | 3.9 | 2,000 |
| stick_supermimic_s2 | 0 % | 0.38 mm | 6 % | 0.93 mm | 7.5 | 3.7 | 2,000 |
| fly_supertrack_s0 | 0 % | 0.061 mm | 6 % | 0.091 mm | 11.1 | 3.8 | 2,000 |
| fly_supertrack_s1 | 4 % | 0.082 mm | 10 % | 0.115 mm | 11.1 | not reached |  |
| fly_supertrack_s2 | 0 % | 0.061 mm | 4 % | 0.081 mm | 11.1 | 9.2 | 5,000 |
| fly_supermimic_s0 | 0 % | 0.045 mm | 0 % | 0.063 mm | 12.8 | 4.3 | 2,000 |
| fly_supermimic_s1 | 0 % | 0.049 mm | 0 % | 0.067 mm | 12.9 | 4.4 | 2,000 |
| fly_supermimic_s2 | 0 % | 0.047 mm | 0 % | 0.065 mm | 12.8 | 4.3 | 2,000 |
| rat_supertrack_s0 | 74 % | 35.2 mm | 86 % | 52.8 mm | 22.2 |  |  |
| rat_supertrack_s1 | 66 % | 34.8 mm | 78 % | 52.5 mm | 22.1 |  |  |
| rat_supertrack_s2 | 72 % | 33.9 mm | 88 % | 51.9 mm | 22.1 |  |  |
| rat_supermimic_s0 | 20 % | 21.9 mm | 28 % | 32.9 mm | 24.9 |  |  |
| rat_supermimic_s1 | 20 % | 18.9 mm | 52 % | 40.8 mm | 24.8 |  |  |
| rat_supermimic_s2 | 18 % | 23.0 mm | 46 % | 45.1 mm | 24.8 |  |  |
| stick_supertrack_s0 (cold cache) | 0 % | 0.40 mm | 0 % | 0.74 mm | 32.2 | 29.0 | 2,000 |
| stick_supermimic_s0 (cold cache) | 0 % | 0.39 mm | 0 % | 0.86 mm | 30.9 | 27.1 | 2,000 |
| fly_supertrack_s0 (cold cache) | 0 % | 0.062 mm | 10 % | 0.118 mm | 19.1 | 17.3 | 5,000 |
| fly_supermimic_s0 (cold cache) | 0 % | 0.045 mm | 0 % | 0.064 mm | 26.2 | 17.8 | 2,000 |
| rat_supertrack_s0 (cold cache) | 68 % | 33.7 mm | 78 % | 50.0 mm | 38.7 |  |  |
| rat_supermimic_s0 (cold cache) | 20 % | 20.4 mm | 52 % | 43.8 mm | 41.8 |  |  |

The rat's runs have no `time_to.json`: the documented command covers the stick insect and the fruit fly.
