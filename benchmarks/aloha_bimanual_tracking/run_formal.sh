#!/usr/bin/env bash
# The benchmark's runs: Transfer Cube (demonstration 0) and Insertion (demonstration 4), SuperTrack and SuperMimic at
# their configurations (aloha-supertrack, aloha-supermimic), seeds 0-2. A run trains to the cap (the configuration's
# 12,000 iterations, then on at the learning rate they end on) with every checkpoint kept; time_to.py evaluates the
# checkpoints and finds the one the run stops at (the first from 12,000 iterations on at which it has settled:
# README.md); that checkpoint is then evaluated (256 perturbed rollouts, the reference's joint positions and the
# demonstration's recorded targets open loop beside).
#   benchmarks/aloha_bimanual_tracking/run_formal.sh
# OUT=..., DATA=... to override; TASKS, ALGOS, SEEDS: a part of the runs; TRAIN_ONLY=1: the training alone (the
# checkpoints' evaluations in a later call, or before it on several GPUs with time_to.py --part); CAP=...: a test.
set -euo pipefail  # a run that fails stops the queue
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$REPO/logs/aloha_formal}
DATA=${DATA:-$REPO/data/aloha}
CAP=${CAP:-48000}
declare -A DEMO=([transfer_cube]=0 [insertion]=4)  # the first whose recorded targets reach ACT's reward of 4 on MuJoCo-Warp
mkdir -p "$OUT"
cd "$REPO"
for task in ${TASKS:-transfer_cube insertion}; do
  for algo in ${ALGOS:-supertrack supermimic}; do
    for seed in ${SEEDS:-0 1 2}; do
      name=${task}_${algo}_s${seed}
      run=$OUT/$name
      [ -f "$run/eval/metrics.json" ] && continue
      if [ ! -f "$run/ckpt_$(printf %06d "$CAP").pt" ]; then
        rm -rf "$run"
        python -u -m ndm_gym.train_robomimic --config aloha-$algo --task aloha_$task --demo ${DEMO[$task]} --seed $seed \
          --data_dir "$DATA" --iterations "$CAP" --log_dir "$run" > "$run.log" 2>&1
      fi
      [ -n "${TRAIN_ONLY:-}" ] && continue
      python -u benchmarks/aloha_bimanual_tracking/time_to.py "$run" > "${run}_time_to.log" 2>&1
      stop=$(python -c "import json, sys; print('%06d' % json.load(open(sys.argv[1]))['stop_iteration'])" "$run/time_to.json")
      python -u -m ndm_gym.evaluate_robomimic --ckpt "$run/ckpt_$stop.pt" --out "$run/eval" > "${run}_eval.log" 2>&1
    done
  done
done
