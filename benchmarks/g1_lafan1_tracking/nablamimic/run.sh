#!/bin/bash
# One run of NablaMimic's own code, unmodified, and its own evaluation (docs/algorithms/nablamimic.md): its checkout at the pinned commit, in
# the environment of its scripts/setup_env.sh, started by its entry point and scored by its scripts/evaluate_run.sh:
#   python -m nablamimic.train --config <file> --out <run directory> --seed <seed> [its options]
#   bash scripts/evaluate_run.sh <run directory>
# This script checks the checkout, takes the GPU, records what it starts and waits; nm_table.py reads the run's files.
#
# Usage: NABLAMIMIC=<its checkout> GPU=<index> CPUS=<cpu list> run.sh <run> <published|remat> <seed> [its options]
#   published     its configs/ours.args (about 50 GB of GPU memory, its README)
#   remat         the same without --no-remat: the windows rematerialized in the backward pass, for smaller GPUs. A configuration
#                 file of NablaMimic cannot remove an included option, so remat_config.py writes the file out from its own
#   its options   of nablamimic.train; they follow its --config and override the file: --hours 1, --replay-steps 500, ...
# Environment:
#   ENV_SH             a file sourced first (caches, NVCC, DK_CACHE_DIR, ...)
#   PY, PIN, OUT       its interpreter (<checkout>/.venv/bin/python), the commit (bfbb2a6), the runs' directory (<checkout>/runs)
#   QUIET=0            start although other GPU processes run on the node (a timed run needs the node to itself: refused otherwise)
#   EVAL=0             no evaluation after the training
#   NICE               niceness of both (0); TIMEOUT_MIN, EVAL_TIMEOUT_MIN: each is ended after that many minutes (240), with an alarm
#   LOCKS              the directory of the GPU lock files, gpu<index>.lock (default ~/ndm_gym/locks, where scripts/run_train.sh takes its
#                      locks, so that the two never start on one GPU); the lock is held for the training and the evaluation
#   GPU=cpu            a test of this script without a GPU (CUDA hidden, no lock); its trainer stops at its physics' first call there
# Written beside the run directory OUT/<run>: <run>.launch.txt (the checks, the command, the PIDs and exit codes), <run>.config.args
# and .config.log (remat), <run>.freeze.txt, <run>.train.log, <run>.eval.log; OUT/ALARM gets a line per failure. A run is made
# once. Stop one only by the PID in its record: kill <train pid>.
set -uo pipefail
USAGE="run.sh <run> <published|remat> <seed> [options of nablamimic.train]"
RUN=${1:?$USAGE}
VARIANT=${2:?$USAGE}
SEED=${3:?$USAGE}
shift 3
: "${NABLAMIMIC:?set NABLAMIMIC=<its checkout>}" "${GPU:?set GPU=<index> (cpu: a test without a GPU)}" "${CPUS:?set CPUS=<cpu list>}"
HERE=$(dirname "$(realpath "$0")")
LOCKS=${LOCKS:-$HOME/ndm_gym/locks} # read before ENV_SH, which may move HOME
PIN=${PIN:-bfbb2a6773b4776c17402a34a93ec7589d9c7c18}
NM=$(realpath "$NABLAMIMIC")
OUT=$(realpath -m "${OUT:-$NM/runs}")
PY=${PY:-$NM/.venv/bin/python}
NICE=${NICE:-0}
REC=$OUT/$RUN.launch.txt
fail() { # a line on stderr and in OUT/ALARM, then exit
  echo "$(date -u +%FT%TZ) $RUN: $1" >&2
  mkdir -p "$OUT" && echo "$(date -u +%FT%TZ) $RUN: $1" >> "$OUT/ALARM"
  exit "${2:-1}"
}
[ ! -e "$OUT/$RUN" ] && [ ! -e "$REC" ] || fail "$OUT/$RUN exists: a run is made once"
case "$VARIANT" in published | remat) ;; *) fail "the configuration is published or remat, not $VARIANT" ;; esac
if [ -n "${ENV_SH:-}" ]; then
  # shellcheck disable=SC1090
  source "$ENV_SH" || fail "cannot source $ENV_SH"
fi

# ---- the checkout: the pinned commit, nothing changed, the revisions and data its setup scripts fetch
[ -f "$NM/src/nablamimic/train.py" ] && [ -f "$NM/scripts/evaluate_run.sh" ] || fail "$NM is not a NablaMimic checkout"
[ -x "$PY" ] || fail "no interpreter $PY (its scripts/setup_env.sh makes <checkout>/.venv)"
commit=$(git -C "$NM" rev-parse HEAD) || fail "$NM is not a git checkout"
[ "$commit" = "$PIN" ] || fail "the checkout is at $commit, not at the pinned $PIN"
[ -z "$(git -C "$NM" status --porcelain --untracked-files=no)" ] || fail "tracked files of the checkout are modified"
[ -z "$(git -C "$NM" status --porcelain -- src scripts patches)" ] || fail "the checkout has files of its own under src, scripts or patches"
want=$(grep -o 'NablaEngine\.git@[0-9a-f]\{40\}' "$NM/scripts/setup_env.sh" | head -n 1 | cut -d@ -f2)
engine=$("$PY" -I -c 'import importlib.metadata as m, json
print(json.loads(m.distribution("nablaengine").read_text("direct_url.json"))["vcs_info"]["commit_id"])') ||
  fail "NablaEngine is not installed from a git revision in $PY's environment"
[ -n "$want" ] && [ "$engine" = "$want" ] || fail "the installed NablaEngine is $engine, its scripts/setup_env.sh pins ${want:-nothing}"
tree=$(cd "$NM/third_party/mujoco_warp" && find mujoco_warp -type f -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
[ "$tree" = "$(grep -v '^#' "$NM/patches/mujoco_warp/tree.sha256" | head -n 1)" ] ||
  fail "third_party/mujoco_warp is not its patched tree (scripts/setup_mujoco_warp.sh)"
MENAGERIE=${MENAGERIE_DIR:-$NM/third_party/mujoco_menagerie}
(cd "$MENAGERIE" && grep -v '^#' "$NM/scripts/menagerie_unitree_g1.sha256" | sha256sum -c --quiet -) ||
  fail "$MENAGERIE does not match scripts/menagerie_unitree_g1.sha256"
LAFAN=${LAFAN1_G1_DIR:-$NM/data/lafan1_g1}
[ "$(find "$LAFAN" -maxdepth 1 -name '*.csv' | wc -l)" = 40 ] || fail "$LAFAN does not hold exactly 40 CSVs (scripts/download_lafan1.sh)"
(cd "$LAFAN" && grep -v '^#' "$NM/scripts/lafan1_g1.sha256" | sha256sum -c --quiet -) || fail "$LAFAN does not match scripts/lafan1_g1.sha256"

# ---- the GPU: the lock of scripts/run_train.sh, free, the node quiet
export CUDA_DEVICE_ORDER=PCI_BUS_ID # CUDA's device index is nvidia-smi's
if [ "$GPU" = cpu ]; then
  export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
  gpu="cpu (a test without a GPU: no lock)"
else
  unset PYTHONPATH PYTHONSTARTUP # nothing but its environment's own packages
  mkdir -p "$LOCKS"
  exec 9> "$LOCKS/gpu$GPU.lock"
  flock -n 9 || fail "GPU lock $LOCKS/gpu$GPU.lock is held" 75
  q=$(timeout 90 nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits -i "$GPU") || fail "nvidia-smi failed or hung"
  [ "${q##*, }" -le 1000 ] || fail "GPU $GPU already has ${q##*, } MiB in use"
  if [ "${QUIET:-1}" = 1 ]; then
    busy=$(timeout 90 nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader) || fail "nvidia-smi failed or hung"
    [ -z "$busy" ] || fail "the node is not quiet: $(echo "$busy" | tr '\n' ';') (QUIET=0 starts anyway; such a run is not a timed one)"
  fi
  gpu="$GPU: $q (name, MiB in all, MiB in use); node quiet: $([ "${QUIET:-1}" = 1 ] && echo checked || echo not checked)"
  export CUDA_VISIBLE_DEVICES=$GPU
fi

# ---- the configuration and the record
mkdir -p "$OUT"
cd "$NM" || fail "cannot enter $NM"
CFG=$NM/configs/ours.args
if [ "$VARIANT" = remat ]; then
  CFG=$OUT/$RUN.config.args
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PY" -I "$HERE/remat_config.py" "$CFG" > "$OUT/$RUN.config.log" 2>&1 ||
    fail "remat_config.py failed ($OUT/$RUN.config.log)"
fi
cmd=("$PY" -m nablamimic.train --config "$CFG" --out "$OUT/$RUN" --seed "$SEED" "$@")
{
  echo "date_utc $(date -u +%FT%TZ)"
  echo "run $RUN"
  echo "variant $VARIANT"
  echo "seed $SEED"
  echo "pin $PIN"
  echo "commit $commit"
  echo "checkout $NM: tracked files unmodified, nothing of its own under src, scripts, patches"
  echo "engine $engine, the revision its scripts/setup_env.sh pins"
  echo "mujoco_warp $tree, its patches/mujoco_warp/tree.sha256"
  echo "menagerie $MENAGERIE matches its scripts/menagerie_unitree_g1.sha256"
  echo "lafan1 $LAFAN: 40 CSVs match its scripts/lafan1_g1.sha256"
  echo "config $CFG sha256 $(sha256sum "$CFG" | cut -d' ' -f1)"
  echo "gpu $gpu"
  echo "cpus $CPUS"
  echo "command $(printf '%q ' "${cmd[@]}")"
} > "$REC"
"$PY" -I -m pip freeze > "$OUT/$RUN.freeze.txt" 2>&1 || fail "pip freeze failed ($OUT/$RUN.freeze.txt)"

# ---- its training, then its evaluation; each one's PID is in the record, and ending that process ends it
export XLA_PYTHON_CLIENT_PREALLOCATE=false # its README
t0=$(date +%s)
nice -n "$NICE" timeout -k 60 "${TIMEOUT_MIN:-240}m" taskset -c "$CPUS" "${cmd[@]}" > "$OUT/$RUN.train.log" 2>&1 < /dev/null &
echo "train pid $!" >> "$REC"
wait $!
rc=$?
echo "train exit $rc after $(($(date +%s) - t0)) s, ended $(date -u +%FT%TZ)" >> "$REC"
[ $rc = 0 ] || fail "its training exited with $rc ($OUT/$RUN.train.log)" $rc
if [ "${EVAL:-1}" = 1 ]; then
  t0=$(date +%s)
  PYTHON=$PY nice -n "$NICE" timeout -k 60 "${EVAL_TIMEOUT_MIN:-240}m" taskset -c "$CPUS" bash scripts/evaluate_run.sh "$OUT/$RUN" \
    > "$OUT/$RUN.eval.log" 2>&1 < /dev/null &
  echo "evaluation pid $!" >> "$REC"
  wait $!
  rc=$?
  echo "evaluation exit $rc after $(($(date +%s) - t0)) s, ended $(date -u +%FT%TZ)" >> "$REC"
  [ $rc = 0 ] || fail "its evaluation exited with $rc ($OUT/$RUN.eval.log)" $rc
fi
echo "$(date -u +%FT%TZ) $RUN: done ($OUT/$RUN)"
