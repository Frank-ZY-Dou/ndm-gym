#!/bin/bash
# Launch one offline world-model fit (ndm_gym.wmbench.fit) on one GPU.
# Usage: GPU=<gpu> [CPUS=<cpu list>] [MEM=24G] [SHARED=1] [PY=<python>] scripts/run_wmbench.sh <name> [ndm_gym.wmbench.fit args...]
# Same isolation as run_train.sh: own systemd user scope (memory cap, no swap), pinned to CPUS (by default 8 NUMA-local
# physical cores), an exclusive flock on the GPU, refuses a GPU already in use unless SHARED=1 (for the eval GPU, whose
# watchers hold no lock; fits on it do not compete with timed training runs). Output: results/wmbench/<name>.csv
# (+ _meta.json, .log). Stop only by unit name: systemctl --user stop ndm-wmb-<name>.scope
set -euo pipefail
NAME=$1
shift
: "${GPU:?set GPU=<index>}"
MEM=${MEM:-24G}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-$HOME/ndm_gym/venv/bin/python}
LOCKS=$HOME/ndm_gym/locks
# CPU list for a process on GPU $1: $2 when given, else 8 physical cores per GPU and their HT siblings (+64), as on the
# benchmark machine (2x32 cores)
cpus() { if [ -n "${2:-}" ]; then echo "$2"; else echo "$((8 * $1))-$((8 * $1 + 7)),$((8 * $1 + 64))-$((8 * $1 + 71))"; fi; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$used" -gt 1000 ] && [ "${SHARED:-0}" != 1 ]; then
  echo "GPU $GPU already has ${used} MiB in use; not starting" >&2
  exit 1
fi
mkdir -p "$REPO/results/wmbench" "$LOCKS"
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export WARP_CACHE_PATH=$HOME/ndm_gym/cache/warp TORCHINDUCTOR_CACHE_DIR=$HOME/ndm_gym/cache/inductor MPLCONFIGDIR=/tmp/mpl_ndm_gym
cd "$REPO"
# shellcheck disable=SC2016
CUDA_VISIBLE_DEVICES=$GPU nohup systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 --unit="ndm-wmb-$NAME" -- \
  bash -c 'exec 9>"$0"; flock -n 9 || { echo "GPU lock $0 is held; not starting"; exit 75; }; exec "$@"' "$LOCKS/gpu$GPU.lock" \
  taskset -c "$(cpus "$GPU" "${CPUS:-}")" "$PY" -m ndm_gym.wmbench.fit --out "results/wmbench/$NAME.csv" "$@" > "results/wmbench/$NAME.log" 2>&1 &
echo "launched ndm-wmb-$NAME on GPU $GPU (cpus $(cpus "$GPU" "${CPUS:-}")), log results/wmbench/$NAME.log"
