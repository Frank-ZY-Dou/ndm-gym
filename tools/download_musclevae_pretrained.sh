#!/bin/bash
# Download MuscleVAE's pretrained low-level (tracking) model, the Google Drive folder linked from its README
# (wangchek/MuscleVAE_ @ 54c001c): its config.yml and trained.data into data/musclevae/pretrained/low_level/, each checked
# against its SHA-256. With --dataset, also their processed motion dataset (walk_run_jump.pickle, the README's file
# link) into data/musclevae/pretrained/motion_dataset/, for checking the policy's target observations
# (tools/replay_musclevae.py --their_dataset). For validation only: the repository has no license, so none of this is
# committed or redistributed (data/ is git-ignored). trained.data is a pickled torch checkpoint: ndm_gym opens it with
# torch.load(weights_only=True) (ndm_gym.policies.musclevae_pretrained); the dataset pickle is never unpickled.
# Plain HTTPS (curl).
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET=0
for arg in "$@"; do
  case "$arg" in
    --dataset) DATASET=1 ;;
    *) echo "usage: $0 [--dataset]" >&2; exit 2 ;;
  esac
done
OUT=data/musclevae/pretrained
# path, Google Drive file id, SHA-256
FILES=(
  "low_level/config.yml 1AnkTj440axvIhsnvQeW6I0fFaikfhbBy b14da5a2329aa852e7b9047c104f0f4a3dff984d6e14097ba89966dab6bc6559"
  "low_level/trained.data 10CZ7TWBdg9xeAjzSB38mboN-RN4LT5Rg 29537eb79bd67845972dd072c9f1191587023163c22bcad38a8e90d95f777816"
)
if [ "$DATASET" = 1 ]; then
  FILES+=("motion_dataset/walk_run_jump.pickle 1b0iNcuDOpwi62Mu7Xn9j86obEvgs3QKb 0f0a5ea0c71e3907df2e2056bdbd4232df143396eb6e72ba2d0ee114446a20b7")
fi

for entry in "${FILES[@]}"; do
  read -r path id sha <<< "$entry"
  dest="$OUT/$path"
  if [ -e "$dest" ]; then
    have=$(sha256sum "$dest" | cut -d' ' -f1)
    [ "$have" = "$sha" ] || { echo "$dest exists with SHA-256 $have, expected $sha; remove it and run this script again" >&2; exit 1; }
    echo "$dest: present, SHA-256 checked"
    continue
  fi
  mkdir -p "$(dirname "$dest")"
  tmp=$(mktemp "$dest.part.XXXXXX")
  trap 'rm -f "$tmp"' EXIT
  curl -sS -L --fail -o "$tmp" "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=t"
  have=$(sha256sum "$tmp" | cut -d' ' -f1)
  [ "$have" = "$sha" ] || { echo "$path: downloaded SHA-256 $have, expected $sha" >&2; exit 1; }
  mv "$tmp" "$dest"
  trap - EXIT
  echo "$dest: $(du -h --apparent-size "$dest" | cut -f1), SHA-256 checked"
done
