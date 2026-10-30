# Motion tracking on G1 / LAFAN1

A benchmark covering SuperTrack and SuperMimic, with an ablation study, run with NDM Gym: the same simulator, data,
observations, networks, schedule and evaluation for both algorithms, so that only the algorithm differs.

| | |
|---|---|
| [`results.md`](results.md) | how the numbers were obtained, the H100 supplement, SuperMimic's late drop and its noise check, the RigidFormer checks, the policy-gradient diagnostic |
| [`ablations.md`](ablations.md) | which factor contributes what, and training efficiency (tables from `tools/ablations.py`) |
| [`splits/`](splits/) | clip lists for `ndm_gym.train --clips`: LAFAN1's own split, subject 5 held out (RigidFormer check 3) |
| [`grad_diag.sh`](grad_diag.sh) | the policy-gradient diagnostic of finished runs (`tools/grad_stats.py`) |

## Reproducing

The selected configurations' flags are in the [README](../../README.md#-getting-started); each ablation variant's change
is in [`ablations.md`](ablations.md). The A100 40GB runs were launched with `scripts/run_train.sh` (memory cap, CPU
pinning to the GPU's NUMA node, GPU lock, evaluation watcher) and the offline world-model fits with
`scripts/run_wmbench.sh`. The runs on other GPU types ran `ndm-train`, then `ndm-eval` on every checkpoint.
