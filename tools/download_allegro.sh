#!/bin/bash
# Menagerie Allegro v3 (BSD-2-Clause), external assets only.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/menagerie
REV=98a70c569c4c1b6da467a20012f766ee1f8dc521
if [ -e "$DEST" ]; then
  echo "Destination $DEST already exists; refusing to overwrite it." >&2
  exit 1
fi
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
git -C "$DEST" sparse-checkout set wonik_allegro
git -C "$DEST" checkout --detach "$REV"
