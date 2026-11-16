# NablaMimic's own code on LAFAN1

[NablaMimic](https://github.com/sailab-ai/NablaMimic) (W. Lu, Y. Chen; [arXiv:2610.41654](https://arxiv.org/abs/2610.41654); Apache-2.0) trains on
the study's 40 clips through its own repository, entry point, simulator ([NablaEngine](https://github.com/sailab-ai/NablaEngine), Apache-2.0) and
evaluation, at the pinned commit `bfbb2a6`, not copied here and not modified.
[`docs/algorithms/nablamimic.md`](../../../docs/algorithms/nablamimic.md) says how it trains, what its evaluation measures beside this library's,
and how much GPU memory its configurations need; the results are in [`results.md`](../results.md#nablamimics-own-code-timed).

| | |
|---|---|
| `run.sh` | one run: the checks, the GPU lock of `scripts/run_train.sh`, its `python -m nablamimic.train`, then its `scripts/evaluate_run.sh` |
| `remat_config.py` | its `configs/ours.args` without `--no-remat` (rematerialized windows, which take less GPU memory), written from the checkout |
| `nm_table.py` | its `config.json`, `log.jsonl` and `eval_nominal.json` as this study's columns: time, updates, its return, clips with a fall |

```bash
# its checkout and environment, as its README says (Python 3.12, a CUDA 12 nvcc for NablaEngine's one-time build)
git clone https://github.com/sailab-ai/NablaMimic && cd NablaMimic && git checkout bfbb2a6773b4776c17402a34a93ec7589d9c7c18
bash scripts/setup_env.sh && source .venv/bin/activate && pip install huggingface_hub && bash scripts/download_lafan1.sh --cache

NABLAMIMIC=<checkout> GPU=0 CPUS=0-7 ./run.sh nm_s0 published 0 --hours 1      # its published configuration, about 50 GB
NABLAMIMIC=<checkout> GPU=0 CPUS=0-7 ./run.sh nm_remat_s0 remat 0 --hours 1    # the same with rematerialized windows
NABLAMIMIC=<checkout> GPU=0 CPUS=0-7 ./run.sh nm_remat250_s0 remat 0 --hours 1 --replay-steps 250   # what trains on 40 GB
python nm_table.py <checkout>/runs/nm_s0 <checkout>/runs/nm_remat_s0 <checkout>/runs/nm_remat250_s0
```

`run.sh`'s header lists its environment variables, among them `LOCKS`, the directory of the GPU lock files it shares with `scripts/run_train.sh`.
Options after the seed go to `nablamimic.train` behind its `--config` and override the file. `run.sh` records the training's PID; a run is stopped
by that PID. A run that was stopped continues by hand with its `--resume latest`, and `nm_table.py` marks such a run, since it is not one timed
run. `tests/test_nablamimic_table.py` tests `nm_table.py` on run directories in the format of its files.

Neither its trainer nor its evaluator runs without a GPU: NablaEngine is CUDA only, and its MuJoCo Warp path asks for `cuda:0`
(`src/nablamimic/warp_physics.py:196`). `GPU=cpu` tests `run.sh` up to that point.
