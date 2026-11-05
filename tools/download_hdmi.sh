#!/bin/bash
# Download what ndm_gym needs from HDMI (https://github.com/LeCAR-Lab/HDMI) at the pinned commit: the motion clips of its
# 13 human-object interaction tasks, their task configs, and the G1 and object USD assets those tasks use. Files keep
# HDMI's repository layout under data/hdmi/ (git-ignored; HDMI has no license, so nothing of it is committed), each is
# checked against its SHA-256, and any missing or different file stops the script. The USDs reference nothing outside
# themselves but visual MDL materials. Then convert the clips: python tools/convert_hdmi.py
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=32282f6dcf26cae70b814d585ceb12cc38aa1b60
RAW=https://raw.githubusercontent.com/LeCAR-Lab/HDMI/$COMMIT
OUT=data/hdmi
FILES="
28f02a2f4fe3ccd5c8582f301e0b557d5df0ef5ae80bb762e8b1419dba2d2f4b data/motion/data_for_sim/carry_and_place_bread_box-0829/meta.json
ff35558659a4b7fca63df6fc686cfd518b40bbc2cf4e3b5063e69e0ea73d7344 data/motion/data_for_sim/carry_and_place_bread_box-0829/motion.npz
c20c8a5c5094500d8588a9152f77baa65fc397a476c15717d1db78cb1c7093a0 data/motion/data_for_sim/lift_box_over_shoulder/meta.json
01811f0bb0c005601691f130e7b08618b4d5b263d80dd8c4ff31ee14ac14aa76 data/motion/data_for_sim/lift_box_over_shoulder/motion.npz
cc8d514ca200546488472578e6f7a72a295c1f7b723e961296adff0e090e4824 data/motion/data_for_sim/move_foam-climb-0825-2-0_580/meta.json
35306a71a3642c731215c5ea2c62160a37236621c596c315e133d91350c78b9c data/motion/data_for_sim/move_foam-climb-0825-2-0_580/motion.npz
e23a7b6dca796c437b38f29590c1094501a362137e1f3d7b913da2a26be349cc data/motion/data_for_sim/move_stool-climb-0814/meta.json
cbf4a73edbc097776aa85e0041841d802d3cb511652bab99b416d0cf1244887e data/motion/data_for_sim/move_stool-climb-0814/motion.npz
c20c8a5c5094500d8588a9152f77baa65fc397a476c15717d1db78cb1c7093a0 data/motion/g1/omomo/sub1_suitcase_011/meta.json
d6d08c5792fc7396c89629876d91cfec26404fd803ccb9ffc2439c66ce51f3cf data/motion/g1/omomo/sub1_suitcase_011/motion.npz
b91a2d667fca3b76ee3cc12de2e86d498726708673ae56c2e4a9725522e08275 data/motion/data_for_sim/open_door-feet-0828/meta.json
18f2f4bef77c58ab860e58fee9ba706e5c204dce4187fa50790493fbcd822de8 data/motion/data_for_sim/open_door-feet-0828/motion.npz
02e1515ee213eb27f407cfe7a105b0bace48ef38f558854ca249bc807126e46f data/motion/g1/multi_motion_0726/open_chair_and_sit_amass/meta.json
3f56d50deda1728ad4f6be18933982a9e06511eaf569ab5f137c3ea95e7255a2 data/motion/g1/multi_motion_0726/open_chair_and_sit_amass/motion.npz
dffeeed04f3001710bb8ab83061e8e1d1be256d9e2e35a4efa0c6011ae1d4105 data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height/meta.json
8921f589f8470d25ab1dd0fa65f9b9057676ab122fdcc8d87f43d9c527ed461e data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height/motion.npz
b91a2d667fca3b76ee3cc12de2e86d498726708673ae56c2e4a9725522e08275 data/motion/data_for_sim/push_door-hand-0828/meta.json
a4b6027aaea6d4892969fab5464858f914fc6f5bbee165e39af9e39668c4808d data/motion/data_for_sim/push_door-hand-0828/motion.npz
1f3c6af4fb7b44b9065bfc38e008dd6276d4cb5a6b1916babfbb652a27ae367d data/motion/g1/multi_motion_0726/roll_ball_amass/meta.json
9c0d4028550f32d59805cc20cf85923ae75bf0df3e1b04912bdf5dcce2529397 data/motion/g1/multi_motion_0726/roll_ball_amass/motion.npz
49a3e4168cb4f1319ca7d9f0141d5bbcb6fcaa6a86f4c5f703c7af058fdd2222 data/motion/data_for_sim/topple_wood_board_and_cross/meta.json
6fd9d3d696c37a3f938c9cecc58aa78a646d7419e3484d8e7f01cc637739dcc5 data/motion/data_for_sim/topple_wood_board_and_cross/motion.npz
b91a2d667fca3b76ee3cc12de2e86d498726708673ae56c2e4a9725522e08275 data/motion/data_for_sim/traverse_door/meta.json
aa459e598ffd384bc368454925b501dffe9b4ba573d287b02e1dcbe449e9541f data/motion/data_for_sim/traverse_door/motion.npz
70255a6058cfd7c5fc62179d1d9390204eced576b40adb980cb7c2672be3dbba data/motion/data_for_sim/combo-2/meta.json
d7cb56b148096b62762a8641b11170506b148636c8e2ea4a6107b8ef79b9663c data/motion/data_for_sim/combo-2/motion.npz
2911c988b0ca65afa7a6eabc250af3443ef6989160e7bfacc0986194b428f5e7 cfg/task/base/hdmi-base.yaml
a8ff4820b8e51c6a7dce5fbb92257f7d3331ce26d93085c578510bbcc4546561 cfg/task/G1/hdmi/carry_and_place_bread_box.yaml
3d0201d129040621f5fe9ebe453131cce209d3d366045dc4558ccf176f83a2db cfg/task/G1/hdmi/carry_box_over_shoulder.yaml
835bd81ee731c744bbc00d85cf568be7f0aa4012e203f008a649092f8dce0d35 cfg/task/G1/hdmi/move_foam.yaml
124a9d7ab2e0c7f8ba829ff0673dbb02f9f12f6eb4c0f72fa4e8199c2bd9c71c cfg/task/G1/hdmi/move_stool-climb.yaml
28e7e5f02dcaeeafe7732fe9235a1f6bfe5e752343e96d011cc0adc6d93647a5 cfg/task/G1/hdmi/move_suitcase.yaml
a6de41577a2d8e313e916cfe41fb32ef464b9043f4a4dc52b76437a7d89faf98 cfg/task/G1/hdmi/open_door-feet.yaml
6f1d8c698892ca390be91902ddf64f03c6ee0f078111cba8eb3c55c044ef9ed7 cfg/task/G1/hdmi/open_foldchair-sit.yaml
c86a989ec1436b0cfb6cc42c3ac8ff2379e0bcf0f9588891d64654b06bc3933e cfg/task/G1/hdmi/push_box.yaml
4eb676f54e1cd2f9960c44c3ee08791065544c840828e41174e04468bfa1515f cfg/task/G1/hdmi/push_door-hand.yaml
6dd3579019890a379f4d8a2f2d686cbefae90acf486f24d9ea5133a6a43718d6 cfg/task/G1/hdmi/roll_ball-hand.yaml
b6356fb2cc9f97d1197a6c1e76ca51d4be58c75061c94b95f905c635e9ac76ad cfg/task/G1/hdmi/topple_wood_board_and_cross.yaml
ee04110c5e169db3176fa0587f3424b83b7f09d1ed35c99ce03ad1c49e560668 cfg/task/G1/hdmi/traverse_door.yaml
826b9f2012839f69a256242a7364e8e1db581ee2661b7d23219358b7e1707672 cfg/task/G1/hdmi/truman.yaml
faf4d267a7a93fd16186e77e4c2802aa8ea977b5bb971b72fa63dee99c33200d active_adaptation/assets/g1/g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule.usd
68dad86f2aec6a38e8ca378cffdddd2fa713d24e46fce695c5ed643ceb1416a6 active_adaptation/assets/g1/g1_29dof_rubberhand-feet_box-eef_box-body_capsule.usd
dc24fd9fb205be3c558c14015bde5e65827d8b28f5ef536889dd2ca101780912 active_adaptation/assets/g1/g1_29dof_nohand-feet_sphere-eef_L-body_capsule.usd
a1dab021725a8f23d068aaa97018476af81797db5fa18adb2f1718c2fa075943 active_adaptation/assets/objects/bread_box/bread_box.usd
d578ddfae8f168885838bbe46b523cc9fcab65e2818d04f964b8d9534922e523 active_adaptation/assets/objects/support/support0.usd
ec3ad5d53b278f705a7972ae0e5dee0bf4dad715b1a5ba4faaeda22db1e26851 active_adaptation/assets/objects/support/support1.usd
d3c25a338fffa58ddfef084fe2cff06023dead787ab8a3c2aaa1e3a08925cf6b active_adaptation/assets/objects/suitcase/suitcase.usd
37cbf9c6e7a1476d3a3a6390c89111f4cfe106ceac6e20ea266d4b5c9c0a3a52 active_adaptation/assets/objects/foam/foam.usd
4df5a332d32f723c02a5d04f42ee8044d4e53b7896a5205f9565898abd436071 active_adaptation/assets/objects/stool/stool_support.usd
4fa6a9e9395a23f76a12bba23cf329255cdb421b602a1d8335e78522e41d83f1 active_adaptation/assets/objects/stool/stool.usd
02258e14085439c2b565dceaab6f37c3ea0db7748e8a15401757a35015b00979 active_adaptation/assets/objects/door/door.usd
253eaa53476a0da9449485fe99ff8f1c580928d6fcfa69d5f765b843ecdff741 active_adaptation/assets/objects/foldchair/foldchair.usd
5c299ef74daeaebe1e5be13edc0f8e2352cb9da2f09600b919d89a08a9e2147e active_adaptation/assets/objects/box/box.usd
4667330c4e20187ff3e7c07f925da1528913610cb98f3e0eec306fb5d6b26f50 active_adaptation/assets/objects/ball/ball.usd
3d0562f6f181b2ae9a761b389768a0bf26ef8adc0de0e6934823364bb1a9226b active_adaptation/assets/objects/wood_board/wood_board.usd
4467cba198e6d33bfc3676e3afdc61e55db2284524936ee61f73f18b5653b246 active_adaptation/assets/objects/platform/platform0.usd
5fcb1192b9df2ed2d28bc9a9b94a5a66944a34a8fa5b267657f99b0032a68c01 active_adaptation/assets/objects/platform/platform1.usd
db321a753722644c7a192663c5e9c4cbd7a3445ede161ccd144c082ffe63a073 active_adaptation/assets/objects/platform/wall0.usd
692b376b6eb21fd1f2b18ab361a0f578ffd5bb3a4de6d4ff1c4fd721823f403e active_adaptation/assets/objects/stair/stair.usd
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
echo "HDMI @ ${COMMIT:0:7}: $n files in $OUT ($(du -sh "$OUT" | cut -f1))"
