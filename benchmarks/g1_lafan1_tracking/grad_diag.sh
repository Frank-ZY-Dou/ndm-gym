#!/bin/bash
# Policy-gradient diagnostic for the RigidFormer tuning (outside the study's runs and selection): for each run directory,
# record a replay dataset with the run's own policy (ndm_gym.wmbench.collect, the run's action noise), write the
# reference direction with the run's own setting, then the statistics of every trick against spikes (tools/grad_stats.py;
# RigidFormer runs also get --rf_rope_detach). Finished outputs are skipped, so a rerun resumes.
# Usage: CUDA_VISIBLE_DEVICES=<gpu> [PY=python] [UPDATES=1000] [REF_UPDATES=500] [ENVS=4096] [STEPS=400] \
#        benchmarks/g1_lafan1_tracking/grad_diag.sh <out_dir> <run_dir> [<run_dir> ...]
set -euo pipefail
OUT=$1
shift
PY=${PY:-python}
UPDATES=${UPDATES:-1000}
REF_UPDATES=${REF_UPDATES:-500}
ENVS=${ENVS:-4096}
STEPS=${STEPS:-400}
cd "$(dirname "$0")/../.."
arg() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1/args.json" "$2"; }
for RUN_DIR in "$@"; do
  d="$OUT/$(basename "$RUN_DIR")"
  mkdir -p "$d"
  echo "[$(date -u +%H:%M:%S)] $RUN_DIR -> $d"
  [ -f "$d/data.pt" ] || "$PY" -m ndm_gym.wmbench.collect --ckpt "$RUN_DIR/last.pt" --out "$d/data.pt" --num_envs "$ENVS" --steps "$STEPS" \
    --action_noise "$(arg "$RUN_DIR" action_noise)" --seed 0
  [ -f "$d/base.json" ] || "$PY" tools/grad_stats.py --ckpt "$RUN_DIR/last.pt" --data "$d/data.pt" --save_ref "$d/ref.pt" --out "$d/base.json" \
    --updates "$UPDATES" --ref_updates "$REF_UPDATES"
  settings="clip3:bptt_clip_k=3 clip10:bptt_clip_k=10 fp32:wm_bptt_fp32=1"
  if [ "$(arg "$RUN_DIR" wm_backbone)" = rigidformer ]; then
    settings="rope_detach:rf_rope_detach=1 $settings"
  fi
  for s in $settings; do
    [ -f "$d/${s%%:*}.json" ] || "$PY" tools/grad_stats.py --ckpt "$RUN_DIR/last.pt" --data "$d/data.pt" --ref "$d/ref.pt" --set "${s#*:}" \
      --out "$d/${s%%:*}.json" --updates "$UPDATES"
  done
done
echo "[$(date -u +%H:%M:%S)] done"
