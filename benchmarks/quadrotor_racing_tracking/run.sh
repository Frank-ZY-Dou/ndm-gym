#!/bin/bash
# Train and evaluate one task and seed of the quadrotor tracking benchmark (tasks.json).
# Usage: benchmarks/quadrotor_racing_tracking/run.sh <task id> <seed> [log dir] [extra ndm_gym.train_quadrotor arguments]
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK=$1 SEED=$2 LOGS=${3:-logs/quadrotor_racing_tracking}
PY=${PY:-python}
FLIGHTS=$($PY - "$TASK" <<'PYEOF'
import json, sys
suite = json.load(open("benchmarks/quadrotor_racing_tracking/tasks.json"))
print(" ".join(suite["all_flights"] if sys.argv[1] == "all" else next(t for t in suite["tasks"] if t["id"] == sys.argv[1])["flights"]))
PYEOF
)
RUN=$LOGS/$TASK/seed_$SEED
# shellcheck disable=SC2086
$PY -m ndm_gym.train_quadrotor --flight $FLIGHTS --seed "$SEED" --log_dir "$RUN" "${@:4}"
$PY -m ndm_gym.evaluate_quadrotor --ckpt "$RUN/last.pt" --out "$RUN/evaluation"
