#!/bin/bash
# The accuracy of the RigidFormer inference paths (results.md): each of the three tasks' RigidFormer accel fit (run_fits.sh)
# scored on its held-out dataset through the eager model and through the timed paths, with fit_hoi's windows and metrics.
# FP32=1 scores only the eager model without autocast (bf16 autocast's own difference).
# Usage: CUDA_VISIBLE_DEVICES=<gpu> [PY=python] [FP32=1] benchmarks/ndm_speed/run_score_rf.sh <data_dir> <results_dir> <out.csv>
set -uo pipefail
D=$1
R=$2
OUT=$3
PY=${PY:-python}
cd "$(dirname "$0")/../.." || exit 1
score() { "$PY" -m ndm_gym.wmbench.score_hoi --model "$R/${t}_rigidformer_accel_wm.pt" --test held="$D/${t}_s1.pt" --out "$OUT" "$@" 2>&1 \
  | grep -E "score_hoi\]|Traceback|Error"; }
for t in move_suitcase open_door-feet truman; do
  if [ "${FP32:-0}" = 1 ]; then
    score --amp none
    continue
  fi
  score
  score --compile reduce-overhead
  score --compile reduce-overhead --inference_copy bf16 --rf_attention token --fused
  score --compile reduce-overhead --inference_copy amp --rf_attention token --fused --rf_kernels blocks
done
echo SCORE_RF_DONE
