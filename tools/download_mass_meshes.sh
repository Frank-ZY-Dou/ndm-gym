#!/bin/bash
# Download the bone meshes of MASS (Lee et al., SIGGRAPH 2019; https://github.com/lsw9021/MASS) at the pinned commit:
# its 23 skeleton OBJs (centimetres, in MASS's T-pose world frame) and its license (Apache-2.0, which covers the meshes).
# MuscleVAE's character is MASS's skeleton and its videos show these meshes; tools/render_musclevae.py draws them on the
# character's bodies. Files keep MASS's repository layout under data/mass/ (git-ignored: used locally, not committed or
# redistributed), each is checked against its SHA-256, and any missing or different file stops the script.
set -euo pipefail
cd "$(dirname "$0")/.."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY  # direct download: the machine's proxy must not be used
COMMIT=e44c2ef184cb55d45afd16b000414c84092ae18e
RAW=https://raw.githubusercontent.com/lsw9021/MASS/$COMMIT
OUT=data/mass
FILES="
dfaa4ef516f7133b600559ed519a7b77639c3889b8819efc1b4b87f201c9497d LICENSE
1650958ecd218cf2194bbaa1bfc2e929358ced115040cc1da47a64e2492d00f7 data/OBJ/L_Femur.obj
c1a5ad424a6ecce9972749a0741d93f812a77fd8141d2f9e610f3ef262601a73 data/OBJ/L_FootPinky.obj
c17d44c9153357bc2d036a009efa605dbe55decab0b5dd1c1462a63ffa0039fc data/OBJ/L_FootThumb.obj
a0644170c2d30426d6e122d8d6c8d20b0c3703aa3bec39640b3bd4881ac1d7dc data/OBJ/L_Hand.obj
1505cb07aa5b8e992c5f010734c4a2a1a7bebfcd5afb76c5c1d09aad7417d149 data/OBJ/L_Humerus.obj
70c2615c3b93db8b9886771bf06175332849dace8b7fe05d9c5c2039fb531e4b data/OBJ/L_Radius.obj
975f7cf5258383e20d5714bf3535b041c712a093bb15eff833ff221ceb1dbc23 data/OBJ/L_Shoulder.obj
6df05bd6f445dcc0cb91f147176f8dd1ca6483571732a65597e0edc2a29bf97c data/OBJ/L_Talus.obj
7b4c6d58c4f590d4cb4fef34f0d200bbf36cfcedda58383a3a86e791b714b86a data/OBJ/L_Tibia.obj
83e70c296e38f86357474dbdfa646117f88b993e1b9672376d2a7c783a09e058 data/OBJ/Neck.obj
76b4e418e3d80d3eb0b6effd258d4bdba07cbe43697d697960d087553878a7a1 data/OBJ/Pelvis.obj
50bc19d9d0151d5ccafc33424ea71d7fbf78c3e5d3c7d3670b166b5a4e9d7668 data/OBJ/R_Femur.obj
e9b31f0bd5319908e4b1824bc1fe08ea3c3f926918dbeece84e0f6fe0c0d87f3 data/OBJ/R_FootPinky.obj
f0b0fec3537b6663aaec78b70df6ac01c81aabb82ee6473aa8e7ce2b914c72c6 data/OBJ/R_FootThumb.obj
abb8681cd779d68a337d20b927879d1d81961ede458431ef8ce7f052c3978295 data/OBJ/R_Hand.obj
fd0be4f8cbc32b11f3c6791938969a3d9bc507c5b138f1b672a32fd3c2ef1ad4 data/OBJ/R_Humerus.obj
8a4f5e7dea5ae60afdbb779c493f03fc523143b61694d272bcca45579c43b316 data/OBJ/R_Radius.obj
7891bebd984fc35bd69ef7d4ba7dc64cab4c84aed258a1fe65e8376c08a022c5 data/OBJ/R_Shoulder.obj
ff8620b5af45a38f0103fcfe4319cfc7ce5c2640c23e5e2c980822640293c97c data/OBJ/R_Talus.obj
6178250feae1323c2ab520fb5e49e65d3a8c9bf9a833355d029f227b39a9e393 data/OBJ/R_Tibia.obj
cdeb0c52a92a20a4c47dae7e2e853504d57ba93aa8800731c6dc6afe613b16f2 data/OBJ/Skull.obj
5dfc4f704a96f6bae490fc44e0c47824732bc33659460a931369402edbbcf9fc data/OBJ/Spine.obj
e3af6b8da75fd90caa9f15d5436ca9c1da1b54532e90a2c2d18cf33349efb86d data/OBJ/Torso.obj
"
n=0
while read -r sum path; do
  [ -n "$path" ] || continue
  mkdir -p "$OUT/$(dirname "$path")"
  curl -sSfL --retry 3 -o "$OUT/$path.part" "$RAW/$path" || { rm -f "$OUT/$path.part"; echo "download failed: $path" >&2; exit 1; }
  echo "$sum  $OUT/$path.part" | sha256sum --check --quiet || { rm -f "$OUT/$path.part"; echo "checksum mismatch: $path" >&2; exit 1; }
  mv "$OUT/$path.part" "$OUT/$path"
  n=$((n + 1))
done <<< "$FILES"
[ "$n" -eq "$(grep -c . <<< "$FILES")" ] || { echo "only $n files downloaded" >&2; exit 1; }
echo "$COMMIT" > "$OUT/COMMIT"
echo "MASS @ ${COMMIT:0:7}: $n files in $OUT ($(du -sh "$OUT" | cut -f1))"
