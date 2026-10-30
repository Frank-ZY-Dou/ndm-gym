#!/bin/bash
# Launch one training run and its checkpoint-eval watcher on one GPU of a multi-GPU Linux machine.
# Usage: GPU=<train gpu> EVAL_GPU=<eval gpu> [CPUS=<cpu list>] [EVAL_CPUS=<cpu list>] [MEM=24G] [PY=<python>] \
#        scripts/run_train.sh <run_name> [ndm_gym.train args...]
# Each process gets its own systemd user scope (memory cap, no swap) and is pinned to CPUS / EVAL_CPUS, by default 8
# physical cores (16 threads) local to its GPU's NUMA node. Training holds an exclusive flock on
# ~/ndm_gym/locks/gpu<N>.lock and refuses a GPU that is already in use; the eval GPU is shared by all watchers. Stop a
# run only by its unit names:
#   systemctl --user stop ndm-<run_name>-train.scope ndm-<run_name>-eval.scope
set -euo pipefail
RUN=$1
shift
: "${GPU:?set GPU=<index>}"
: "${EVAL_GPU:?set EVAL_GPU=<index> (the GPU of the checkpoint-eval watcher)}"
MEM=${MEM:-24G}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-$HOME/ndm_gym/venv/bin/python}
LOCKS=$HOME/ndm_gym/locks
# CPU list for a process on GPU $1: $2 when given, else 8 physical cores per GPU and their HT siblings (+64), as on the
# benchmark machine (2x32 cores)
cpus() { if [ -n "${2:-}" ]; then echo "$2"; else echo "$((8 * $1))-$((8 * $1 + 7)),$((8 * $1 + 64))-$((8 * $1 + 71))"; fi; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$used" -gt 1000 ]; then
  echo "GPU $GPU already has ${used} MiB in use; not starting" >&2
  exit 1
fi
mkdir -p "$REPO/logs/$RUN" "$LOCKS"
# fingerprint of the code this run starts with, to compare runs
(cd "$REPO" && find ndm_gym scripts -type f \( -name '*.py' -o -name '*.sh' \) | sort | xargs sha256sum) > "$REPO/logs/$RUN/code.sha256"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export WARP_CACHE_PATH=$HOME/ndm_gym/cache/warp TORCHINDUCTOR_CACHE_DIR=$HOME/ndm_gym/cache/inductor MPLCONFIGDIR=/tmp/mpl_ndm_gym
cd "$REPO"
# shellcheck disable=SC2016
CUDA_VISIBLE_DEVICES=$GPU nohup systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 --unit="ndm-$RUN-train" -- \
  bash -c 'exec 9>"$0"; flock -n 9 || { echo "GPU lock $0 is held; not starting"; exit 75; }; exec "$@"' "$LOCKS/gpu$GPU.lock" \
  taskset -c "$(cpus "$GPU" "${CPUS:-}")" "$PY" -m ndm_gym.train --log_dir "logs/$RUN" "$@" >> "logs/$RUN/stdout.log" 2>&1 &
CUDA_VISIBLE_DEVICES=$EVAL_GPU nohup systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 --unit="ndm-$RUN-eval" -- \
  taskset -c "$(cpus "$EVAL_GPU" "${EVAL_CPUS:-}")" "$PY" -m ndm_gym.evaluate --run_dir "logs/$RUN" --watch >> "logs/$RUN/eval_stdout.log" 2>&1 &
echo "launched ndm-$RUN-{train,eval}: train GPU $GPU (cpus $(cpus "$GPU" "${CPUS:-}")), eval GPU $EVAL_GPU, logs $REPO/logs/$RUN"
