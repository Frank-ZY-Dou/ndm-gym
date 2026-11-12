#!/usr/bin/env bash
# NablaEngine's own cross-simulator timing kit (scripts/paper_bench, unmodified) on ONE GPU: the reproduction of the
# table in docs/results/paper_h100/paper_bench.md.
#
#   bash run_paper_bench.sh GPU OUT_DIR [MJC_THREADS=16]
#
# Everything it needs is set up (scripts/paper_bench/setup.sh ran: venv_mjw, venv_pr, the two pull-request trees and
# Menagerie under nablaengine/NablaEngine/runs/paper_bench_env; the engine library for sm_80 is in cache/nablaengine).
# One job, at most 55 minutes (the kit's own budget, PAPER_BENCH_BUDGET_MIN=53). The MuJoCo C part uses MJC_THREADS
# CPU threads (the kit's default would be every core of the machine; the published H100 run used 13, the RTX PRO
# 6000 run 16). The Genesis part runs only if the kit's Genesis 1.4.5 environment exists
# (nablaengine/NablaEngine/runs/bench/envs/genesis/venv, README.md); otherwise the kit records it as skipped.
set -uo pipefail
ROOT=${DIFFSIM_ROOT:?set DIFFSIM_ROOT to the folder with the simulators environments (env.sh) and results}
GPU="${1:?usage: run_paper_bench.sh GPU OUT_DIR [MJC_THREADS]}"
OUT="$(realpath -m "${2:?usage: run_paper_bench.sh GPU OUT_DIR [MJC_THREADS]}")"
source "$ROOT/env.sh"
cd "$ROOT/nablaengine/NablaEngine" && source .dk_env/activate.sh
export CUDA_VISIBLE_DEVICES="$GPU"
export PAPER_BENCH_DEV_PYTHON="$(command -v python)"
export PAPER_BENCH_SKIP_SETUP=1
export PAPER_BENCH_MJC_THREADS="${3:-16}"
scripts/paper_bench/run.sh "$OUT"
echo "exit $?: $OUT/paper_bench.md"
