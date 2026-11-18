#!/bin/bash
# The Adroit door's world model as the simulator (benchmarks/adroit_door/results_a100.md), on one GPU with nothing else running:
# the caller sets CUDA_VISIBLE_DEVICES to the card's UUID and pins this process to eight physical cores of that card's NUMA node,
# which the native collector's eight threads use:
#   PY=python benchmarks/adroit_door/scripts/door_speed.sh <in dir> <out dir>
# <in dir>: prep/door23-supported/{door.xml,reference.npz} (tools/prepare_adroit_door.py), prep/native/door_batch.so
# (tools/build_door_native.py) and, per algorithm, <algo>/latest.pt with <algo>/physical-rollouts.npz (supertrack, supermimic:
# a train_door_full run's checkpoint and its eval_door_full evaluation's rollouts).
# ndm_gym.wmbench.speed_door: open-loop errors at 1/8/16/32/64 steps; throughput at 1,024 / 4,096 / 16,384 worlds of MuJoCo on the
# CPU (once: the physics does not depend on the checkpoint), MuJoCo Warp and the world model in fp32 and bf16 (and 65,536 worlds
# for the GPU paths); SuperTrack's checkpoint, then SuperMimic's. A failure ends the script with its exit status. Run from the
# tree's root, with $PY the interpreter. About 15 minutes.
set -u
I=$1 O=$2
mkdir -p $O
P="--model $I/prep/door23-supported/door.xml --reference $I/prep/door23-supported/reference.npz --native-library $I/prep/native/door_batch.so"
first=1
for algo in supertrack supermimic; do
  for n in 1024 4096 16384 65536; do
    what="warp ndm_fp32 ndm_bf16"
    [ $first = 1 ] && [ $n -le 16384 ] && what="cpu $what"
    echo "$(date -u +%FT%TZ) $algo $n: $what" >> $O/speed.log
    $PY -m ndm_gym.wmbench.speed_door --ckpt $I/$algo/latest.pt --rollouts $I/$algo/physical-rollouts.npz $P --nworld $n \
      --what $what --profile --out $O/${algo}_n$n.json > $O/${algo}_n$n.out 2>&1 < /dev/null || exit $?
  done
  first=0
done
echo "$(date -u +%FT%TZ) done" >> $O/speed.log
