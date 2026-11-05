# Motion tracking on G1 / LAFAN1

A benchmark and ablation study covering SuperTrack and SuperMimic in NDM Gym, with shared simulator, data, observations,
networks, schedule and evaluation.

| | |
|---|---|
| [`results.md`](results.md) | how the numbers were obtained, the training speed, the H100 supplement, SuperMimic's late drop and its noise check, the RigidFormer checks and final configurations, the policy-gradient diagnostic |
| [`ablations.md`](ablations.md) | which factor contributes what, and training efficiency (tables from `tools/ablations.py`) |
| [`splits/`](splits/) | clip lists for `ndm_gym.train --clips`: LAFAN1's own split, subject 5 held out (RigidFormer check 3) |
| [`grad_diag.sh`](grad_diag.sh) | the policy-gradient diagnostic of finished runs (`tools/grad_stats.py`) |

## Reproducing

Commands: [selected configurations](../../README.md#lafan1-motion-tracking),
[fastest configurations](../../README.md#training-speed), [ablation changes](ablations.md).
A100 40GB runs used `scripts/run_train.sh` (memory cap, CPU
pinning to the GPU's NUMA node, GPU lock, evaluation watcher) and the offline world-model fits with
`scripts/run_wmbench.sh`. Other GPU types used `ndm-train`, then `ndm-eval` on every checkpoint.
