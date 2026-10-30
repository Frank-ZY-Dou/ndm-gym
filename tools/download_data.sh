#!/bin/bash
# Download the G1-retargeted LAFAN1 motions and the G1 description at a pinned revision, check every file against
# tools/download_data.sha256, then build the MJCF/USD assets.
# LAFAN1 (Ubisoft La Forge) and its G1 retargets are CC BY-NC-ND 4.0; the Unitree G1 description is BSD-3-Clause.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=ce1572906efe6157840e8474d5a0d7aa87481e74
HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/$REV
API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/$REV
list() { curl -sSf "$API/$1" | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin) if x['type'] == 'file']"; }
count() { printf '%s\n' "$1" | grep -c . || true; }
clips=$(list g1)
meshes=$(list robot_description/g1/meshes)
[ "$(count "$clips")" -eq 40 ] || { echo "expected 40 clips at revision $REV, the listing has $(count "$clips")" >&2; exit 1; }
[ "$(count "$meshes")" -eq 38 ] || { echo "expected 38 meshes at revision $REV, the listing has $(count "$meshes")" >&2; exit 1; }
mkdir -p data/lafan1_g1 data/g1_description/meshes
for f in $clips; do curl -sSfL -o "data/lafan1_g1/$(basename "$f")" "$HF/$f"; done
curl -sSfL -o data/g1_description/g1_29dof_rev_1_0.urdf "$HF/robot_description/g1/g1_29dof_rev_1_0.urdf"
for f in $meshes; do curl -sSfL -o "data/g1_description/meshes/$(basename "$f")" "$HF/$f"; done
sha256sum --quiet --strict -c tools/download_data.sha256
echo "motions: 40 clips, G1 description: URDF and 38 meshes (checked)"
# asset conversion needs mujoco + the kit-less USD importers of the Isaac Lab environment (PY, default: python)
"${PY:-python}" tools/urdf_to_mjcf.py
"${PY:-python}" tools/mjcf_to_usd.py
# the pristine upstream checkout used by the equivalence tests reads the same assets from its own data/
if [ -d third_party/supermimic/supermimic ] && [ ! -e third_party/supermimic/data ]; then
  ln -s ../../data third_party/supermimic/data
fi
