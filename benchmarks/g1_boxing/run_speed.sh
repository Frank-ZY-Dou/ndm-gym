#!/bin/bash
# The two-G1 boxing example's speed table (results.md, Speed), on one GPU that runs nothing else:
#   (1) simulation throughput, ndm_gym/wmbench/speed_pair.py: the physics of the two-robot scene and the open-loop rollouts
#       of ST-MLP's MLP and ST-RF's RigidFormer, N in {256, 1024, 4096} pairs (a size that does not fit is recorded and skipped);
#   (2) training speed: TRAIN_ITERS iterations (default 1500; the first 500 fit the world model only; 0 skips it) of each
#       configuration on the clip, from its train_log.csv's it_per_s.
# Usage: run_speed.sh <repo tree> <pair .npz> <out dir>   (the GPU: CUDA_VISIBLE_DEVICES; the interpreter: PY, default python)
set -uo pipefail
REPO=$(cd "$1" && pwd)
PAIR=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
OUT=$3
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
PY=${PY:-python}
ITERS=${TRAIN_ITERS:-1500}
cd "$REPO" || exit 1
SPEED="--wm_cudagraphs --pi_cudagraphs --fused_adam --elementwise_3x3 --combo_kernels --compile_eager"
MLP="--algo supertrack --st_fail_w 5 --st_balance_w 4"
RF="--algo supertrack --st_fail_w 5 --st_balance_w 2 --wm_backbone rigidformer --wm_lr 4e-4 --wm_batches 5"
RF="$RF --rf_fused_train --rf_bptt_keep_blocks --rf_fit_keep_ops"
nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv > "$OUT/gpu.txt"
for N in 256 1024 4096; do
  "$PY" -m ndm_gym.wmbench.speed_pair --pair "$PAIR" --mode physics --num_envs "$N" --out "$OUT/speed_pair.csv" > "$OUT/physics_$N.log" 2>&1 \
    || echo "physics N=$N failed (rc $?), see physics_$N.log" >> "$OUT/skipped.txt"
  for cfg in mlp rf; do
    args=$MLP; [ $cfg = rf ] && args=$RF
    "$PY" -m ndm_gym.wmbench.speed_pair --pair "$PAIR" --mode ndm --num_envs "$N" --wm_args "$args $SPEED" --out "$OUT/speed_pair.csv" \
      > "$OUT/ndm_${cfg}_$N.log" 2>&1 || echo "ndm $cfg N=$N failed (rc $?), see ndm_${cfg}_$N.log" >> "$OUT/skipped.txt"
  done
done
for cfg in mlp rf; do
  [ "$ITERS" -gt 0 ] || break
  args=$MLP; [ $cfg = rf ] && args=$RF
  rm -rf "$OUT/train_$cfg"
  # shellcheck disable=SC2086
  "$PY" -m ndm_gym.train $args $SPEED --motions interhuman --paired --motion_dir "$PAIR" --iterations "$ITERS" --seed 0 \
    --log_dir "$OUT/train_$cfg" > "$OUT/train_$cfg.log" 2>&1 || echo "train $cfg failed (rc $?)" >> "$OUT/skipped.txt"
done
echo "done $(date -u +%FT%T)" >> "$OUT/gpu.txt"
