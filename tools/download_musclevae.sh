#!/bin/bash
# Download MuscleVAE (https://github.com/wangchek/MuscleVAE_) at the pinned commit into data/musclevae/src/: a shallow
# fetch of that one commit, checked against its hash. The repository has no license, so nothing of it is committed or
# redistributed (data/ is git-ignored); its pretrained models (Google Drive folders linked from its README) are not
# fetched here. Then convert the character, the muscles and the reference motions: python tools/convert_musclevae.py
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=https://github.com/wangchek/MuscleVAE_.git
COMMIT=54c001c913b156c0bba29e593a4a291e02d1f9ce
OUT=data/musclevae/src
GIT=(git -c advice.detachedHead=false)

if [ -e "$OUT" ]; then
  head=$("${GIT[@]}" -C "$OUT" rev-parse HEAD 2>/dev/null || true)
  if [ "$head" != "$COMMIT" ] || [ -n "$("${GIT[@]}" -C "$OUT" status --porcelain)" ]; then
    echo "$OUT exists but is not a clean checkout of $COMMIT; remove it and run this script again" >&2
    exit 1
  fi
  echo "MuscleVAE @ ${COMMIT:0:7} already in $OUT"
  exit 0
fi
mkdir -p "$(dirname "$OUT")"
TMP=$(mktemp -d "$OUT.part.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
"${GIT[@]}" init -q "$TMP"
"${GIT[@]}" -C "$TMP" fetch -q --depth 1 "$REPO" "$COMMIT"
"${GIT[@]}" -C "$TMP" checkout -q FETCH_HEAD
head=$("${GIT[@]}" -C "$TMP" rev-parse HEAD)
[ "$head" = "$COMMIT" ] || { echo "fetched $head, expected $COMMIT" >&2; exit 1; }
mv "$TMP" "$OUT"
trap - EXIT
echo "MuscleVAE @ ${COMMIT:0:7}: $("${GIT[@]}" -C "$OUT" ls-files | wc -l) files in $OUT ($(du -sh --apparent-size "$OUT" | cut -f1))"
