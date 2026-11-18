#!/bin/bash
# Train and evaluate one task and seed of the Go2 tracking benchmark (tasks.json; "all": every clip), then time the run
# (time_to.sh). The evaluation of the last checkpoint includes the trunk pushes of 1 and 2 m/s.
# Usage: benchmarks/go2_lifelike_tracking/run.sh <task id> <seed> [log dir] [extra ndm_gym.train_quadruped arguments]
#   e.g. run.sh all 0 logs/go2_sm --config go2-supermimic;  run.sh walk 0 logs/go2_sm --config go2-supermimic-clip
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK=$1 SEED=$2 LOGS=${3:-logs/go2_lifelike_tracking}
PY=${PY:-python}
CLIPS=$($PY - "$TASK" <<'PYEOF'
import json, sys
suite = json.load(open("benchmarks/go2_lifelike_tracking/tasks.json"))
print(",".join(next(t for t in suite["tasks"] if t["id"] == sys.argv[1])["clips"]))
PYEOF
)
RUN=$LOGS/$TASK/seed_$SEED
# shellcheck disable=SC2086
$PY -m ndm_gym.train_quadruped ${CLIPS:+--clips $CLIPS} --seed "$SEED" --log_dir "$RUN" "${@:4}"
$PY -m ndm_gym.evaluate_quadruped --ckpt "$RUN/last.pt" --out "$RUN/evaluation" --push 1 2 --save_rollout
benchmarks/go2_lifelike_tracking/time_to.sh "$RUN"
