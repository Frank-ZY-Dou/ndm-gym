#!/bin/bash
# The throughput measurements of results.md (physics, MLP, RigidFormer) on one GPU that runs nothing else. Rows are appended to
# <out.csv>; a configuration that fails (e.g. out of memory at a large N) prints its error and leaves no row.
# Usage: CUDA_VISIBLE_DEVICES=<gpu> [PY=python] benchmarks/ndm_speed/run_speed.sh <out.csv>
set -uo pipefail
OUT=$1
PY=${PY:-python}
cd "$(dirname "$0")/../.." || exit 1
THREE="move_suitcase open_door-feet truman"
ALL=$("$PY" -c "from ndm_gym.tasks.hdmi_tasks import TASKS; print(' '.join(TASKS))")
run() { "$PY" -m ndm_gym.wmbench.speed_hoi --out "$OUT" "$@" 2>&1 | grep -E "speed_hoi\]|Traceback|Error"; }
for t in $ALL; do run --task "$t" --mode physics --num_envs 4096; done
for t in $THREE; do
  for n in 256 1024 16384; do run --task "$t" --mode physics --num_envs $n; done
  for bb in mlp rigidformer; do
    for n in 256 1024 4096 16384 65536; do run --task "$t" --mode ndm --backbone $bb --compile reduce-overhead --num_envs $n; done
  done
  for n in 256 1024 4096; do run --task "$t" --mode env --num_envs $n; done
done
echo SPEED_DONE
