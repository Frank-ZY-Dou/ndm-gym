#!/bin/bash
# The RigidFormer inference paths of results.md on one GPU that runs nothing else: first the kernels' checks on the
# device, then, per task, the physics rows and the NDM paths at every N. Rows are appended to <out.csv>; a configuration
# that fails (e.g. out of memory at a large N) prints its error and leaves no row.
# Usage: CUDA_VISIBLE_DEVICES=<gpu> [PY=python] benchmarks/ndm_speed/run_speed_rf.sh <out.csv>
set -uo pipefail
OUT=$1
PY=${PY:-python}
cd "$(dirname "$0")/../.." || exit 1
THREE="move_suitcase open_door-feet truman"
run() { "$PY" -m ndm_gym.wmbench.speed_hoi --out "$OUT" "$@" 2>&1 | grep -E "speed_hoi\]|Traceback|Error"; }
"$PY" -m pytest tests/test_rf_kernel.py -q -p no:cacheprovider 2>&1 | tail -3
RF=(--mode ndm --backbone rigidformer --kind accel --compile reduce-overhead)
for t in $THREE; do
  for n in 256 1024 4096 16384; do run --task "$t" --mode physics --num_envs $n; done
  for n in 256 1024 4096 16384 65536; do
    run --task "$t" "${RF[@]}" --num_envs $n
    run --task "$t" "${RF[@]}" --inference_copy bf16 --rf_attention token --fused --num_envs $n
    run --task "$t" "${RF[@]}" --inference_copy amp --rf_attention token --fused --rf_kernels blocks --num_envs $n
    run --task "$t" --mode ndm --backbone mlp --kind accel --compile reduce-overhead --inference_copy bf16 --fused --num_envs $n
  done
done
echo SPEED_RF_DONE
