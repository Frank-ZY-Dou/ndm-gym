#!/bin/bash
# The training time at which a run reaches the benchmark's target for good: every kept checkpoint (train_quadruped
# --keep_ckpts) evaluated on the run's clips from perturbed starts (at most 2,048 worlds in all), the policies batched in one
# simulator; the wall time of the first checkpoint from which every later one has a success of at least 0.99 and a mean
# trunk error below 20 mm. Writes <run>/evaluation/checkpoints.json (minutes_to_success: null if never).
# Usage: benchmarks/go2_lifelike_tracking/time_to.sh <run directory>
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=$1
PY=${PY:-python}
$PY -m ndm_gym.evaluate_quadruped --ckpts "$RUN" --out "$RUN/evaluation" --success 0.99 --success_trunk_mm 20 | tail -1
