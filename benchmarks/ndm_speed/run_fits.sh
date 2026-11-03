#!/bin/bash
# The accuracy beside the speeds (results.md): per task, datasets with seeds 0 (training) and 1 (held out), then
# the offline fits of both backbones in both parameterizations, eager (fit_hoi --eager; fit_hoi compiles its loss by
# default). Skips finished outputs, so a rerun resumes.
# Usage: CUDA_VISIBLE_DEVICES=<gpu> [PY=python] benchmarks/ndm_speed/run_fits.sh <data_dir> <results_dir> [task ...]
set -uo pipefail
D=$1
R=$2
shift 2
PY=${PY:-python}
cd "$(dirname "$0")/../.." || exit 1
mkdir -p "$D" "$R"
TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(move_suitcase open_door-feet truman)
for t in "${TASKS[@]}"; do
  for s in 0 1; do
    [ -f "$D/${t}_s$s.pt" ] || "$PY" -m ndm_gym.wmbench.collect_hoi --task "$t" --num_envs 4096 --steps 400 --seed $s --out "$D/${t}_s$s.pt" 2>&1 \
      | grep -E "^saved|Traceback|Error"
  done
  for bb in mlp rigidformer; do
    for k in accel increment; do
      [ -f "$R/${t}_${bb}_${k}_wm.pt" ] || "$PY" -m ndm_gym.wmbench.fit_hoi --task "$t" --kind $k --backbone $bb --train "$D/${t}_s0.pt" \
        --test held="$D/${t}_s1.pt" --steps 20000 --eager --out "$R/${t}_${bb}_${k}.csv" 2>&1 | grep -E "wmbench_hoi\] step 20000|Traceback|Error"
    done
  done
done
echo FITS_DONE
