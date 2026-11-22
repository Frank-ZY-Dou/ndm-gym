#!/usr/bin/env bash
# The benchmark's runs: Transfer Cube and Insertion (demonstration 0), SuperTrack and SuperMimic at their configurations
# (aloha-supertrack, aloha-supermimic), seeds 0-2; then each run's last checkpoint evaluated (256 perturbed rollouts, the
# reference's joint positions and the demonstration's recorded targets open loop beside).
#   benchmarks/aloha_bimanual_tracking/run_formal.sh   (OUT=..., DATA=... to override; TASKS, ALGOS, SEEDS: a part of the runs)
set -euo pipefail  # a run that fails stops the queue
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$REPO/logs/aloha_formal}
DATA=${DATA:-$REPO/data/aloha}
mkdir -p "$OUT"
cd "$REPO"
for task in ${TASKS:-transfer_cube insertion}; do
  for algo in ${ALGOS:-supertrack supermimic}; do
    for seed in ${SEEDS:-0 1 2}; do
      name=${task}_${algo}_s${seed}
      [ -f "$OUT/$name/eval/metrics.json" ] && continue
      rm -rf "$OUT/$name"
      python -u -m ndm_gym.train_robomimic --config aloha-$algo --task aloha_$task --demo 0 --seed $seed --data_dir "$DATA" \
        --log_dir "$OUT/$name" > "$OUT/$name.log" 2>&1
      python -u -m ndm_gym.evaluate_robomimic --ckpt "$OUT/$name/last.pt" --out "$OUT/$name/eval" > "$OUT/${name}_eval.log" 2>&1
    done
  done
done
