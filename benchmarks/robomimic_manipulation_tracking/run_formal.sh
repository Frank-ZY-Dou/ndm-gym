#!/usr/bin/env bash
# The formal runs: Lift, Square and Transport (demonstration 0), SuperTrack and SuperMimic at their configurations
# (robomimic-supertrack, robomimic-supermimic: a cap of 6,000 iterations, the solver at 20 iterations, a checkpoint kept every
# 250), seeds 10-12; each run's stop checkpoint by the stop rule (stop_rule.py), evaluated on the 256 perturbed starts of
# evaluation seed 5 with the reference's targets beside (<run>/eval, the position targets saved for motion.py). A failed step
# ends the script.
#   benchmarks/robomimic_manipulation_tracking/run_formal.sh   (OUT=..., DATA=... to override)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$REPO/logs/robomimic_formal}
DATA=${DATA:-$REPO/data/robomimic}
mkdir -p "$OUT"
cd "$REPO"
for task in lift square transport; do
  for seed in 10 11 12; do
    for algo in supertrack supermimic; do
      name=${task}_${algo}_s${seed}
      [ -f "$OUT/$name/eval/metrics.json" ] && continue
      rm -rf "$OUT/$name"
      python -u -m ndm_gym.train_robomimic --config robomimic-$algo --task $task --demo 0 --seed $seed --data_dir "$DATA" \
        --log_dir "$OUT/$name" > "$OUT/$name.log" 2>&1
      python benchmarks/robomimic_manipulation_tracking/stop_rule.py "$OUT/$name"
      ckpt=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$OUT/$name/stop.json" stop_ckpt)
      python -u -m ndm_gym.evaluate_robomimic --ckpt "$ckpt" --out "$OUT/$name/eval" --seed 5 --save_targets --data_dir "$DATA" \
        > "$OUT/${name}_eval.log" 2>&1
    done
  done
done
