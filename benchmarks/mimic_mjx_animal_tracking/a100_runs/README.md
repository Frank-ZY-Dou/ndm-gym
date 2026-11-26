# Animal motion tracking: the contributed A100 runs

Contributed by [Jingdong Zhang](https://github.com/Evergreen0929) ([#8](https://github.com/Frank-ZY-Dou/ndm-gym/pull/8)):
the stick insect's six runs of `benchmarks/mimic_mjx_animal_tracking/run_formal.sh supertrack` and `supermimic`, unchanged,
at 8a35d7a in this history (with `--ref_qvel positions`), and their time to the target (`time_to.py` there). The rat's and
the fruit fly's records are in this history too (7029b6a); their results here are in progress. How these runs differ from the results':
[results.md](../results.md#the-contributed-runs).

Per run, JSON records: `eval_train/metrics.json` (clips 0-9), `eval_heldout/metrics.json` (clips 10-19), `train.json` (the
training log's last row: the iteration, the minutes, the environment steps and the iterations per second; `ep_len`, the mean
episode length in training at 80, 90 and 100 % of the run, where the log has a row at that iteration) and `time_to.json`
(every kept checkpoint's evaluation). No logs, no checkpoints.

- **GPU:** NVIDIA A100-SXM4-80GB; each run alone on its GPU.
- **Software:** Python 3.12.9, PyTorch 2.11.0 (CUDA 12.6 build), warp-lang 1.17.0 (CUDA 12 build), MuJoCo and MuJoCo-Warp 3.15.0.
- **Compilation:** every run here had the kernels' compilation in the cache. `cold_cache_seed0/` holds each configuration's
  seed-0 run on an empty cache, 5.9 and 7.5 min longer.
- **Target:** the time to the target is the contributor's: at most 2 % of the 50 episodes ended with the bodies within 3 %
  of the body's size, on each kept checkpoint (`time_to.py`).
- **Clock:** `train.json`'s `minutes`, from the first training iteration; a compilation during the run is in it,
  evaluations are not.
- Mean ± sample SD over seeds 0, 1, 2.

## Results

| body | algorithm | ended ↓ | bodies ↓ | end effectors ↓ | joints ↓ | held-out: ended ↓ | held-out: bodies ↓ | training (min) |
|---|---|---|---|---|---|---|---|---|
| stick insect | SuperTrack | 0.0 ± 0.0 % | 0.35 ± 0.01 mm | 0.60 ± 0.02 mm | 8.4 ± 0.2° | 10.7 ± 1.2 % | 0.64 ± 0.01 mm | 6.4 ± 0.1 |
|  | SuperMimic | 0.0 ± 0.0 % | 0.33 ± 0.00 mm | 0.56 ± 0.00 mm | 4.2 ± 0.1° | 0.0 ± 0.0 % | 0.74 ± 0.03 mm | 7.6 ± 0.1 |

## Every seed

| run | ended | bodies | held-out: ended | held-out: bodies | training (min) | to the target (min) | at iteration |
|---|---|---|---|---|---|---|---|
| stick_supertrack_s0 | 0 % | 0.34 mm | 10 % | 0.63 mm | 6.6 | 1.8 | 1,000 |
| stick_supertrack_s1 | 0 % | 0.36 mm | 10 % | 0.64 mm | 6.4 | 1.6 | 1,000 |
| stick_supertrack_s2 | 0 % | 0.35 mm | 12 % | 0.64 mm | 6.4 | 1.6 | 1,000 |
| stick_supermimic_s0 | 0 % | 0.33 mm | 0 % | 0.74 mm | 7.7 | 2.6 | 1,250 |
| stick_supermimic_s1 | 0 % | 0.33 mm | 0 % | 0.76 mm | 7.6 | 1.9 | 1,000 |
| stick_supermimic_s2 | 0 % | 0.32 mm | 0 % | 0.70 mm | 7.5 | 1.9 | 1,000 |
| stick_supertrack_s0 (cold cache) | 0 % | 0.34 mm | 10 % | 0.61 mm | 12.5 | 7.7 | 1,000 |
| stick_supermimic_s0 (cold cache) | 0 % | 0.33 mm | 0 % | 0.71 mm | 15.2 | 10.0 | 1,250 |
