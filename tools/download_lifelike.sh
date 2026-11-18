#!/bin/bash
# Download, at pinned commits, what the quadruped tracking task reads: Tencent's Labrador motion capture (33 BVH files of
# "Lifelike agility and play in quadrupedal robots", Nature Machine Intelligence 2024; MIT, its README: for research
# purposes only) into data/lifelike, and MuJoCo Menagerie's Unitree Go2 (BSD-3-Clause) into data/menagerie/unitree_go2.
# Neither is redistributed. Then: tools/convert_lifelike.py retargets the motion capture to the Go2.
# Usage: tools/download_lifelike.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LIFELIKE=cc9eaa9e79b35f1dd03983f28b669365098e30a5
MENAGERIE=370c43965323c0944b433b202aac5058b5f907a9

fetch() {  # fetch <repository> <commit> <path in it> <destination>
  local tmp
  tmp=$(mktemp -d)
  git clone -q --filter=blob:none --no-checkout "https://github.com/$1.git" "$tmp"
  git -C "$tmp" sparse-checkout set --no-cone "$3"
  git -C "$tmp" checkout -q "$2"
  mkdir -p "$(dirname "$4")"
  rm -rf "$4"
  mv "$tmp/$3" "$4"
  rm -rf "$tmp"
}

fetch Tencent-RoboticsX/lifelike-agility-and-play $LIFELIKE data/raw_mocap_data data/lifelike/raw_mocap_data
fetch google-deepmind/mujoco_menagerie $MENAGERIE unitree_go2 data/menagerie/unitree_go2
n=$(find data/lifelike/raw_mocap_data -name '*.bvh' | wc -l)
[ "$n" -eq 33 ] || { echo "expected 33 BVH files at commit $LIFELIKE, found $n" >&2; exit 1; }
[ -f data/menagerie/unitree_go2/go2.xml ] || { echo "no go2.xml at commit $MENAGERIE" >&2; exit 1; }
echo "motion capture: 33 BVH files in data/lifelike/raw_mocap_data; Go2: data/menagerie/unitree_go2"
