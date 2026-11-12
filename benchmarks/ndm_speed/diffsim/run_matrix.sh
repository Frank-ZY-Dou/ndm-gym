#!/usr/bin/env bash
# The benchmark matrix on ONE GPU: every simulator x mode x world count, one process per measurement, all lines
# appended to one CSV.
#
#   bash run_matrix.sh GPU OUT.csv ["nablaengine mjwarp mjx"] ["1024 4096 16384"] ["as_published tuned"]
#
# GPU is the nvidia-smi index (env.sh sets CUDA_DEVICE_ORDER=PCI_BUS_ID). The first run of each simulator compiles
# (NablaEngine: seconds, its library is prebuilt; MuJoCo Warp: its kernels, a few minutes, then cached; MJX: every
# program, minutes each, never cached). Environments and versions: README.md.
set -uo pipefail
ROOT=${DIFFSIM_ROOT:?set DIFFSIM_ROOT to the folder with the simulators environments (env.sh) and results}
GPU="${1:?usage: run_matrix.sh GPU OUT.csv [sims] [nworlds] [modes]}"
OUT="$(realpath -m "${2:?usage: run_matrix.sh GPU OUT.csv [sims] [nworlds] [modes]}")"
SIMS="${3:-nablaengine mjwarp mjx}"
NWORLDS="${4:-1024 4096 16384}"
MODES="${5:-as_published tuned}"
mkdir -p "$(dirname "$OUT")"
LOG="${OUT%.csv}.log"
for sim in $SIMS; do for mode in $MODES; do for n in $NWORLDS; do
  echo "=== $sim $mode nworld $n $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
  case "$sim" in
    nablaengine) ( source "$ROOT/env.sh"; cd "$ROOT/nablaengine/NablaEngine" && source .dk_env/activate.sh && cd "$ROOT/bench" &&
                   CUDA_VISIBLE_DEVICES="$GPU" python bench_nablaengine.py --$mode --nworld "$n" --measure both --csv "$OUT" ) ;;
    mjwarp)      ( source "$ROOT/mjwarp_adjoint/activate.sh"; cd "$ROOT/bench" &&
                   CUDA_VISIBLE_DEVICES="$GPU" python bench_mjwarp_adjoint.py --$mode --nworld "$n" --measure both --csv "$OUT" ) ;;
    mjx)         ( source "$ROOT/mjx/activate.sh"; cd "$ROOT/bench" &&
                   CUDA_VISIBLE_DEVICES="$GPU" python bench_mjx.py --$mode --nworld "$n" --measure both --csv "$OUT" ) ;;
    *) echo "unknown simulator $sim (nablaengine, mjwarp, mjx)"; exit 2 ;;
  esac 2>&1 | grep -v '^CSV,' | tee -a "$LOG"
  echo "    exit ${PIPESTATUS[0]}" | tee -a "$LOG"
done; done; done
echo "done: $OUT ($(($(wc -l < "$OUT") - 1)) measurements)"
