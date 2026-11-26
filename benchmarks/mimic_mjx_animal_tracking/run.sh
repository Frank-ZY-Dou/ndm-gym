#!/bin/bash
# One run of the animal tracking benchmark (README.md): the named configuration's training, ended by the benchmark's stop rule
# (watch.py: the kept checkpoints' quick evaluations, here on the same GPU), then the stop checkpoint's evaluations on the
# training and the held-out clips and the time to the target (final.py).
# Usage: benchmarks/mimic_mjx_animal_tracking/run.sh <stick|fly|rat> <supertrack|supermimic> <seed> [log dir] [data dir]
#   e.g. run.sh stick supertrack 0 logs/animals data/mimic_mjx
set -euo pipefail
cd "$(dirname "$0")/../.."
ANIMAL=$1 ALGO=$2 SEED=$3 LOGS=${4:-logs/mimic_mjx_animal_tracking} DATA=${5:-data/mimic_mjx}
PY=${PY:-python}
B=benchmarks/mimic_mjx_animal_tracking
RUN=$LOGS/${ANIMAL}_${ALGO}_s$SEED
if [ -e "$RUN" ]; then echo "$RUN exists" >&2; exit 1; fi
mkdir -p "$RUN"
setsid "$PY" -m ndm_gym.train_animal --config "$ANIMAL-$ALGO" --seed "$SEED" --data_dir "$DATA" --log_dir "$RUN" > "$RUN/train.out" 2>&1 &
TRAIN=$!
"$PY" "$B/watch.py" "$RUN" --data_dir "$DATA" > "$RUN/watch.out" 2>&1 &
WATCH=$!
trap 'if [ -n "$TRAIN" ]; then kill -TERM -- "-$TRAIN" 2> /dev/null || true; fi; kill "$WATCH" 2> /dev/null || true' EXIT
while [ ! -e "$RUN/stop.json" ]; do
  if ! kill -0 "$TRAIN" 2> /dev/null; then  # the training ended (at the cap): the watcher reads the rest
    wait "$TRAIN" || { echo "the training failed ($RUN/train.out)" >&2; exit 1; }
    TRAIN=""
    touch "$RUN/train.done"
    break
  fi
  kill -0 "$WATCH" 2> /dev/null || { echo "the watcher ended before the training ($RUN/watch.out)" >&2; exit 1; }
  sleep 10
done
if [ -n "$TRAIN" ]; then  # the rule is met: the training ends
  kill -TERM -- "-$TRAIN" 2> /dev/null || true
  wait "$TRAIN" 2> /dev/null || true
  TRAIN=""
fi
wait "$WATCH" || { echo "the watcher failed ($RUN/watch.out)" >&2; exit 1; }
read -r STOP REASON < <("$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d["stop"], d["reason"])' "$RUN/stop.json")
"$PY" "$B/final.py" "$RUN" --stop "$STOP" --reason "$REASON" --data_dir "$DATA" --record
