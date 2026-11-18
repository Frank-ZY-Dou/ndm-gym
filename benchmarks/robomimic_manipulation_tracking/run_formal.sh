#!/usr/bin/env bash
# The formal runs: Lift, Square and Transport (demonstration 0), SuperTrack and SuperMimic at their configurations
# (robomimic-supertrack, robomimic-supermimic: 1,500 iterations on the schedule of 4,000, the solver at 20 iterations, a
# checkpoint kept every 500), seeds 0-2; then each run's last checkpoint evaluated (256 perturbed rollouts, the reference's
# targets beside).
#   benchmarks/robomimic_manipulation_tracking/run_formal.sh   (OUT=..., DATA=... to override)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$REPO/logs/robomimic_formal}
DATA=${DATA:-$REPO/data/robomimic}
mkdir -p "$OUT"
cd "$REPO"
for task in lift square transport; do
  for algo in supertrack supermimic; do
    for seed in 0 1 2; do
      name=${task}_${algo}_s${seed}
      [ -f "$OUT/$name/eval/metrics.json" ] && continue
      rm -rf "$OUT/$name"
      python -u -m ndm_gym.train_robomimic --config robomimic-$algo --task $task --demo 0 --seed $seed --data_dir "$DATA" \
        --log_dir "$OUT/$name" > "$OUT/$name.log" 2>&1
      python -u -m ndm_gym.evaluate_robomimic --ckpt "$OUT/$name/last.pt" --out "$OUT/$name/eval" > "$OUT/${name}_eval.log" 2>&1
    done
  done
done
