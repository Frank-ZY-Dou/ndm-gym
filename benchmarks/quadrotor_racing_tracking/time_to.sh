#!/bin/bash
# The training time at which a run first reaches the benchmark's target: its kept checkpoints (train_quadrotor --ckpt_keep), in
# order, each evaluated on its flights from 64 perturbed starts, until one has a success of at least 0.99 and a mean position
# error below 20 mm. Writes <run>/time_to.json: the checkpoint's iteration and training time, or null within --max iterations.
# Usage: benchmarks/quadrotor_racing_tracking/time_to.sh <run directory> [largest iteration, default 3000]
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=$1 MAX=${2:-3000}
PY=${PY:-python}
for CK in "$RUN"/ckpt_*.pt; do
    IT=$((10#$(basename "$CK" .pt | cut -d_ -f2)))
    [ "$IT" -le "$MAX" ] || break
    $PY -m ndm_gym.evaluate_quadrotor --ckpt "$CK" --out "$RUN/evaluation_$IT" --num_envs 64 --no_reference > /dev/null
    if $PY - "$RUN/evaluation_$IT/metrics.json" "$RUN/time_to.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
p = m["policy"]
ok = p["success_rate"] >= 0.99 and p["pos_mm"] < 20.0
if ok:
    json.dump({"iteration": m["iteration"], "minutes": m["wall_time_s"] / 60, "success_rate": p["success_rate"], "pos_mm": p["pos_mm"]}, open(sys.argv[2], "w"))
sys.exit(0 if ok else 1)
PYEOF
    then
        cat "$RUN/time_to.json"; echo; exit 0
    fi
done
echo null > "$RUN/time_to.json"; echo "not within $MAX iterations"
