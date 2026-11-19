#!/usr/bin/env bash
# Named Ode to Joy runs. Additional trainer flags follow the output directory.
set -euo pipefail
if (( $# < 2 )); then
    echo "Usage: $0 {supertrack|supermimic|scripted|replay} OUT [options]" >&2
    exit 2
fi
method=$1
out=$2
shift 2
bench=$(cd -- "$(dirname -- "$0")" && pwd)
assets=${WUJI_PIANO_ASSETS:-"$bench/../../data/wuji_piano"}
flags=(--hand-dir "$assets/wuji-description-c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95/hand2/hand2_beta2/body/mjcf" --keyboard "$assets/keyboard/piano.xml" --out "$out")
case "$method" in
    supertrack|supermimic) exec "${PYTHON:-python}" "$bench/train_ode_to_joy.py" "${flags[@]}" --algo "$method" "$@" ;;
    scripted) exec "${PYTHON:-python}" "$bench/scripted_ode.py" "${flags[@]}" "$@" ;;
    replay) ck=${1:?supply the checkpoint path}; shift; exec "${PYTHON:-python}" "$bench/train_ode_to_joy.py" "${flags[@]}" --replay "$ck" "$@" ;;
    *) echo "Unknown run: $method" >&2; exit 2 ;;
esac
