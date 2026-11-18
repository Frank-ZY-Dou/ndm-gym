#!/bin/bash
# Train, evaluate and render one task and seed of a hand tracking task list (TASKS; default tasks.json), with the hand
# of ROBOT (one of the task list's robots; default xhand).
# Usage: [ROBOT=<robot>] [TASKS=<task list>] benchmarks/xhand_tracking/run.sh <task id> <seed> [data dir] [log dir] [extra ndm_gym.train_hand arguments]
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK=$1 SEED=$2 DATA=${3:-data/spider} LOGS=${4:-logs/xhand_tracking}
PY=${PY:-python} ROBOT=${ROBOT:-xhand} TASKS=${TASKS:-benchmarks/xhand_tracking/tasks.json}
CLIP=$DATA/$($PY - "$TASK" "$ROBOT" "$TASKS" <<'PYEOF'
import json, sys
suite = json.load(open(sys.argv[3]))
if sys.argv[2] not in suite["robots"]:
    sys.exit(f"ROBOT must be one of {suite['robots']}")
t = next(t for t in suite["tasks"] if t["id"] == sys.argv[1])
print(f"processed/{t['dataset']}/{sys.argv[2]}/{suite['hand']}/{t['task']}/{t['trial']}")
PYEOF
)
RUN=$LOGS/$TASK/seed_$SEED
$PY -m ndm_gym.train_hand --clip "$CLIP" --seed "$SEED" --log_dir "$RUN" "${@:5}"
$PY -m ndm_gym.evaluate_hand --ckpt "$RUN/last.pt" --out "$RUN/evaluation" --seed 0   # the results' evaluation seed, whatever the training seed
for source in policy reference; do
    $PY tools/render_hand.py --clip "$CLIP" --rollout "$RUN/evaluation/rollout.npz" --source $source --out "$RUN/evaluation/$source.mp4"
done
