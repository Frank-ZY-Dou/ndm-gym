#!/bin/bash
# Download the H1-retargeted LAFAN1 motions and the H1 description at the revision tools/download_data.sh pins, check every
# file against tools/download_h1.sha256, then convert the description to MJCF (tools/urdf_to_mjcf.py --robot h1).
# Usage: [PY=<python>] tools/download_h1.sh
# LAFAN1 (Ubisoft La Forge) and its retargets are CC BY-NC-ND 4.0; the Unitree H1 description (unitree_ros,
# robots/h1_description) is BSD-3-Clause. Neither is redistributed here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0" >&2; exit 2; }
cd "$(dirname "$0")/.."
REV=ce1572906efe6157840e8474d5a0d7aa87481e74
HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/$REV
API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/$REV
list() { curl -sSf "$API/$1" | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin) if x['type'] == 'file']"; }
count() { printf '%s\n' "$1" | grep -c . || true; }
clips=$(list h1)
meshes=$(list robot_description/h1/meshes)
[ "$(count "$clips")" -eq 40 ] || { echo "expected 40 clips at revision $REV, the listing has $(count "$clips")" >&2; exit 1; }
[ "$(count "$meshes")" -eq 21 ] || { echo "expected 21 meshes at revision $REV, the listing has $(count "$meshes")" >&2; exit 1; }
mkdir -p data/lafan1_h1 data/h1_description/meshes
for f in $clips; do curl -sSfL -o "data/lafan1_h1/$(basename "$f")" "$HF/$f"; done
curl -sSfL -o data/h1_description/h1.urdf "$HF/robot_description/h1/h1.urdf"
for f in $meshes; do curl -sSfL -o "data/h1_description/meshes/$(basename "$f")" "$HF/$f"; done
sha256sum --quiet --strict -c tools/download_h1.sha256
echo "motions: 40 clips, H1 description: URDF and 21 meshes (checked)"
"${PY:-python}" tools/urdf_to_mjcf.py --robot h1
