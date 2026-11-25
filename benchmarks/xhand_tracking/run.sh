#!/bin/bash
# Train, evaluate and render one task and seed of a hand tracking task list (TASKS; default tasks.json), with the hand
# of ROBOT (one of the task list's robots; default xhand).
# With ALGO (supertrack or supermimic), the run takes that algorithm's arguments for the hand and the task from SETTINGS
# (default settings.json): the algorithm's default, then the hand's own for the task where it has them.
# The run's folder is <log dir>/<task id>/seed_<seed>, the log dir by default logs/xhand_tracking/<robot>_<algorithm> (ALGO's, or
# default: train_hand's defaults and the extra arguments).
# Usage: [ROBOT=<robot>] [TASKS=<task list>] [ALGO=<algorithm>] benchmarks/xhand_tracking/run.sh <task id> <seed> [data dir] [log dir] [extra ndm_gym.train_hand arguments]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python} ROBOT=${ROBOT:-xhand} TASKS=${TASKS:-benchmarks/xhand_tracking/tasks.json}
TASK=$1 SEED=$2 DATA=${3:-data/spider} LOGS=${4:-logs/xhand_tracking/${ROBOT}_${ALGO:-default}}
CLIP=$DATA/$($PY - "$TASK" "$ROBOT" "$TASKS" <<'PYEOF'
import json, sys
suite = json.load(open(sys.argv[3]))
if sys.argv[2] not in suite["robots"]:
    sys.exit(f"ROBOT must be one of {suite['robots']}")
t = next(t for t in suite["tasks"] if t["id"] == sys.argv[1])
print(f"processed/{t['dataset']}/{sys.argv[2]}/{suite['hand']}/{t['task']}/{t['trial']}")
PYEOF
)
ARGUMENTS=""
if [ -n "${ALGO:-}" ]; then
    ARGUMENTS=$($PY - "$ALGO" "$ROBOT" "$TASK" "${SETTINGS:-benchmarks/xhand_tracking/settings.json}" <<'PYEOF'
import json, sys
algo, robot, task = sys.argv[1:4]
settings = json.load(open(sys.argv[4]))
if algo not in settings:
    sys.exit(f"ALGO must be one of {list(settings)}")
print(settings[algo]["default"], settings[algo].get(robot, {}).get(task, ""))
PYEOF
)
fi
RUN=$LOGS/$TASK/seed_$SEED
# shellcheck disable=SC2086
$PY -m ndm_gym.train_hand --clip "$CLIP" --seed "$SEED" --log_dir "$RUN" $ARGUMENTS "${@:5}"
$PY -m ndm_gym.evaluate_hand --ckpt "$RUN/last.pt" --out "$RUN/evaluation" --seed 0   # the results' evaluation seed, whatever the training seed
for source in policy reference; do
    $PY tools/render_hand.py --clip "$CLIP" --rollout "$RUN/evaluation/rollout.npz" --source $source --out "$RUN/evaluation/$source.mp4"
done
