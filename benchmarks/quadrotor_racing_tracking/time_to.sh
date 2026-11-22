#!/bin/bash
# The training time at which a run reaches the benchmark's target: its kept checkpoints (train_quadrotor --ckpt_keep), in
# order, each evaluated on its flights from 64 perturbed starts; the target is a success of at least 0.99 with a mean
# position error below 20 mm. With a hold of k iterations, the time is the first checkpoint's from which every checkpoint up
# to k iterations later meets it too (0, the default: the first checkpoint that meets it). Writes <run>/time_to.json: the
# checkpoint's iteration and training time, or null within the largest iteration.
# Usage: benchmarks/quadrotor_racing_tracking/time_to.sh <run directory> [largest iteration, default 3000] [hold, default 0]
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=$1 MAX=${2:-3000} HOLD=${3:-0}
PY=${PY:-python}
FROM="" FROM_IT=0  # the first checkpoint of the latest run of checkpoints that meet the target
for CK in "$RUN"/ckpt_*.pt; do
    IT=$((10#$(basename "$CK" .pt | cut -d_ -f2)))
    [ "$IT" -le "$MAX" ] || break
    $PY -m ndm_gym.evaluate_quadrotor --ckpt "$CK" --out "$RUN/evaluation_$IT" --num_envs 64 --no_reference > /dev/null
    if POINT=$($PY - "$RUN/evaluation_$IT/metrics.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
p = m["policy"]
print(json.dumps({"iteration": m["iteration"], "minutes": m["wall_time_s"] / 60, "success_rate": p["success_rate"], "pos_mm": p["pos_mm"]}))
sys.exit(0 if p["success_rate"] >= 0.99 and p["pos_mm"] < 20.0 else 1)
PYEOF
    ); then
        [ -n "$FROM" ] || { FROM=$POINT; FROM_IT=$IT; }
        if [ $((IT - FROM_IT)) -ge "$HOLD" ]; then
            echo "$FROM" | tee "$RUN/time_to.json"; exit 0
        fi
    else
        FROM=""
    fi
done
echo null > "$RUN/time_to.json"; echo "not within $MAX iterations"
