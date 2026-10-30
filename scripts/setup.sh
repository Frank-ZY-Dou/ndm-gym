#!/bin/bash
# Setup on a Linux GPU server with an NVIDIA driver for CUDA 12.6 or newer.
# 1. Isaac Lab 3 @ 46d6cc3 with its locked dependencies (the kit-less image's `uv sync`), into $WS/venv.
# 2. Swap torch/torchvision/warp for the CUDA 12 builds of the same versions: the lock pins CUDA 13 builds,
#    which need driver >= 580. Skip this step on a driver >= 580.
# 3. LAFAN1 G1 motions + G1 description into <repo>/data, then MJCF/USD conversion.
# 4. ndm_gym itself, editable, with the test extras (the venv is built by uv and has no pip).
# Usage: [WS=$HOME/ndm_gym] scripts/setup.sh; then `source $WS/venv/bin/activate` puts ndm-train and ndm-eval on PATH.
set -euo pipefail
WS=${WS:-$HOME/ndm_gym}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
UV_VERSION=0.12.9
ISAACLAB_COMMIT=46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
export UV_HTTP_RETRIES=12 UV_HTTP_TIMEOUT=300

mkdir -p "$WS/tools/uv"
curl -sSfL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" | tar xz -C "$WS/tools/uv" --strip-components=1
UV=$WS/tools/uv/uv
[ -d "$WS/IsaacLab" ] || git clone https://github.com/isaac-sim/IsaacLab.git "$WS/IsaacLab"
git -C "$WS/IsaacLab" checkout "$ISAACLAB_COMMIT"
(cd "$WS/IsaacLab" && UV_PROJECT_ENVIRONMENT="$WS/venv" "$UV" sync --frozen --inexact --extra all --extra importers)

PY=$WS/venv/bin/python
"$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu126 "torch==2.11.0+cu126" "torchvision==0.26.0+cu126"
"$UV" pip install --python "$PY" --no-deps \
  "warp-lang @ https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl"

(cd "$REPO" && PY="$PY" tools/download_data.sh)
(cd "$REPO" && "$UV" pip install --python "$PY" -e ".[test]")
echo "done: python $PY, data in $REPO/data; run 'source $WS/venv/bin/activate'"
