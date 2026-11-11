#!/bin/bash
# Train, evaluate and render one task and seed of the XHand tracking benchmark (tasks.json).
# Usage: benchmarks/xhand_tracking/run.sh <task id> <seed> [data dir] [log dir] [extra ndm_gym.train_hand arguments]
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK=$1 SEED=$2 DATA=${3:-data/spider} LOGS=${4:-logs/xhand_tracking}
PY=${PY:-python}
CLIP=$DATA/$($PY - "$TASK" <<'PYEOF'
import json, sys
t = next(t for t in json.load(open("benchmarks/xhand_tracking/tasks.json"))["tasks"] if t["id"] == sys.argv[1])
print(f"processed/{t['dataset']}/xhand/right/{t['task']}/{t['trial']}")
PYEOF
)
RUN=$LOGS/$TASK/seed_$SEED
$PY -m ndm_gym.train_hand --clip "$CLIP" --seed "$SEED" --log_dir "$RUN" "${@:5}"
$PY -m ndm_gym.evaluate_hand --ckpt "$RUN/last.pt" --out "$RUN/evaluation" --seed "$SEED"
for source in policy reference; do
    $PY tools/render_hand.py --clip "$CLIP" --rollout "$RUN/evaluation/rollout.npz" --source $source --out "$RUN/evaluation/$source.mp4"
done
