#!/bin/bash
# Train and evaluate one task and seed of the H1 tracking benchmark. evaluation/: the last checkpoint, 64 rollouts per clip,
# with the pelvis pushes of 1 and 2 m/s and the open-loop reference targets (metrics.json). checkpoints/: every kept
# checkpoint (train_humanoid --keep_ckpts), 4 rollouts per clip, in one pass (checkpoints.json).
# <task>: "selected": the configuration's clips (LAFAN1 without its six fallAndGetUp clips, 34); "all": all 40 clips;
#   "train": the selected clips of LAFAN1's training subjects 1-4 (28), then also the last checkpoint on subject 5's
#   (6, held out; heldout/); otherwise clip names or globs, comma-separated.
# Usage: benchmarks/h1_lafan1_tracking/run.sh <task> <seed> <log dir> [extra ndm_gym.train_humanoid arguments]
#   e.g. run.sh selected 0 logs/h1_sm --config h1-supermimic;  run.sh walk1_subject1 0 logs/h1_st --config h1-supertrack
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK=$1 SEED=$2 LOGS=$3
PY=${PY:-python}
KINDS="dance fight jumps run sprint walk"
case "$TASK" in
  selected) CLIPS="" ;;
  all) CLIPS="*" ;;
  train) CLIPS=$(for k in $KINDS; do printf '%s*subject[1-4],' "$k"; done) ;;
  *) CLIPS=$TASK ;;
esac
RUN=$LOGS/$TASK/seed_$SEED
set -f  # the clips' globs are the trainer's, not the shell's
# shellcheck disable=SC2086
$PY -m ndm_gym.train_humanoid "${@:4}" ${CLIPS:+--clips ${CLIPS%,}} --seed "$SEED" --log_dir "$RUN"
$PY -m ndm_gym.evaluate_humanoid --ckpt "$RUN/last.pt" --out "$RUN/evaluation" --num_envs 64 --push 1 2
$PY -m ndm_gym.evaluate_humanoid --run "$RUN" --out "$RUN/checkpoints" --num_envs 4
if [ "$TASK" = train ]; then
  HELD="dance*subject5,fight*subject5,jumps*subject5,run*subject5,walk*subject5"  # subject 5 has no sprint clip
  $PY -m ndm_gym.evaluate_humanoid --ckpt "$RUN/last.pt" --clips "$HELD" --out "$RUN/heldout" --num_envs 64
fi
