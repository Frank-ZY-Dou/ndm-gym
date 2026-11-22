#!/bin/bash
# The Steel comparison of benchmarks/steel/results.md, end to end: bash benchmarks/steel/run.sh <part> <out_dir>
#   record    the recording (wmbench/collect.py --eval_conditions with $POLICY) and its windows (replay_engine export) in <out_dir>
#   accuracy  Steel's settings and MuJoCo Warp in Newton rolled over the windows (steel_g1.py --mode accuracy), scored with the
#             world models against the recording (replay_engine score), then the criteria (gates.py)
#   one       one GPU: every row at N = 4,096, 16,384, 1,024 and 65,536, three fresh processes each, the rows interleaved within
#             each repeat; then the fastest paths doubled from 131,072 worlds until a run fails (that N ends the row)
#   four      Steel's fastest passing setting, MuJoCo Warp in Newton and the MLPs' fastest paths at 16,384 worlds per GPU on 1, 2
#             and 4 GPUs ($GPUS), one process per GPU, started 20 s apart and timing their 1,000 steps together (--barrier)
#   traces    Nsight Systems traces of the fastest paths at 4,096 worlds and at a given N ($TRACE_N), summarized per phase
# Then: python benchmarks/steel/table.py <out_dir>.
# Environment: STEEL_PY (the Steel environment's Python: results.md, Reproducing), MAIN_PY (this repository's, with Isaac Lab),
# PAIR (speed_pair's --pair clip; with --replay only its first frame is read), MJCF (default data/g1_description/g1_29dof.xml),
# POLICY (record: a policy checkpoint), CK_ST and CK_SM (the world models' training checkpoints, lafan1-supertrack-accurate's and
# lafan1-supermimic-accurate's), GPU (one, accuracy, traces: default 0), GPUS (four: default "0 1 2 3"), NSYS (default nsys).
# Every process writes logs/<tag>.log with a start and an end line; a failed process is recorded and the next one runs.
set -uo pipefail
PART=${1:?usage: run.sh <record|accuracy|one|four|traces> <out_dir>}
OUT=${2:?usage: run.sh <part> <out_dir>}
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
MJCF=${MJCF:-$REPO/data/g1_description/g1_29dof.xml}
W=$OUT/windows.npz
GPU=${GPU:-0}
mkdir -p "$OUT/logs"

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }
need() { for v in "$@"; do [ -n "${!v:-}" ] || { say "set $v (the header of run.sh)"; exit 1; }; done; }
proc() {  # <gpu> <tag> <command...>: one process, its log and its status
  local g=$1 tag=$2 rc
  shift 2
  echo "$(date -u +%FT%TZ) start $tag on GPU $g: $*" > "$OUT/logs/$tag.log"
  (cd "$REPO" && CUDA_VISIBLE_DEVICES=$g "$@") >> "$OUT/logs/$tag.log" 2>&1 < /dev/null
  rc=$?
  echo "$(date -u +%FT%TZ) end $tag: exit $rc" >> "$OUT/logs/$tag.log"
  [ "$rc" = 0 ] || say "$tag: exit $rc"
  return "$rc"
}

# Steel's settings: substeps, iterations and options (results.md, Setup)
FAST_OPTS="--opt friction_anchor_beta=0 contact_gap_gate=0 parallel_tree=true reuse_kinematics=true update_mass_matrix_interval=4"
FAST_OPTS+=" --collide control --collapse_fixed"
declare -A SET=([reference]="32 32" [task-physics]="4 12" [g1-preset]="1 2" [fastest-passing]="4 1 $FAST_OPTS")
FAST_MLP="--inference_copy bf16 --fused"
row() {  # <gpu> <name> <n> <tag> <extra...>: one timed process of a row (CSV rows in $OUT/<part>/)
  local g=$1 name=$2 n=$3 tag=$4 sub it opts
  shift 4
  case $name in
    fastest-passing | task-physics | g1-preset)
      read -r sub it opts <<< "${SET[$name]}"
      # shellcheck disable=SC2086
      proc "$g" "$tag" "$STEEL_PY" -I "$HERE/steel_g1.py" --mode speed --windows "$W" --mjcf "$MJCF" --art_max 256 --worlds "$n" \
        --setting "$name" --substeps "$sub" --iterations "$it" $opts --out "$CSV/steel_rows.csv" "$@" ;;
    mujoco-newton)
      proc "$g" "$tag" "$STEEL_PY" -I "$HERE/steel_g1.py" --mode speed --engine mujoco --windows "$W" --mjcf "$MJCF" --worlds "$n" \
        --setting "$name" --substeps 4 --out "$CSV/steel_rows.csv" "$@" ;;
    *)
      local mode=ndm ck="" path=""
      case $name in
        mujoco-isaaclab) mode=physics ;;
        st-fastest) ck=$CK_ST path=$FAST_MLP ;;
        sm-fastest) ck=$CK_SM path=$FAST_MLP ;;
        st-as-trained) ck=$CK_ST ;;
        sm-as-trained) ck=$CK_SM ;;
      esac
      # shellcheck disable=SC2086
      proc "$g" "$tag" "$MAIN_PY" -m ndm_gym.wmbench.speed_pair --pair "$PAIR" --single --num_envs "$n" --warm_seconds 1 --repeats 1 \
        --replay "$W" --mode "$mode" ${ck:+--wm_ckpt "$ck"} $path --out "$CSV/pair_rows.csv" "$@" ;;
  esac
}
ROWS="fastest-passing mujoco-newton mujoco-isaaclab st-fastest sm-fastest st-as-trained sm-as-trained task-physics g1-preset"

case $PART in
  record)
    need MAIN_PY POLICY
    proc "$GPU" record "$MAIN_PY" -m ndm_gym.wmbench.collect --ckpt "$POLICY" --eval_conditions --num_envs 4096 --steps 400 --seed 1 \
      --out "$OUT/recording.pt"
    proc "$GPU" export "$MAIN_PY" -m ndm_gym.wmbench.replay_engine export --recording "$OUT/recording.pt" --windows 8192 --len 32 \
      --seed 0 --out "$W" ;;
  accuracy)
    need STEEL_PY MAIN_PY CK_ST CK_SM
    A=$OUT/accuracy
    mkdir -p "$A"
    rollouts=()
    for name in reference task-physics g1-preset fastest-passing; do
      read -r sub it opts <<< "${SET[$name]}"
      # shellcheck disable=SC2086
      proc "$GPU" "acc_$name" "$STEEL_PY" -I "$HERE/steel_g1.py" --mode accuracy --windows "$W" --mjcf "$MJCF" --substeps "$sub" \
        --iterations "$it" --art_max 256 --watermarks --chunk 4096 $opts --out "$A/steel_$name.npz" && rollouts+=("steel_$name=$A/steel_$name.npz")
    done
    proc "$GPU" acc_mujoco-newton "$STEEL_PY" -I "$HERE/steel_g1.py" --mode accuracy --engine mujoco --windows "$W" --mjcf "$MJCF" --substeps 4 \
      --chunk 4096 --out "$A/mujoco_newton.npz" && rollouts+=("mujoco_newton=$A/mujoco_newton.npz")
    proc "$GPU" score "$MAIN_PY" -m ndm_gym.wmbench.replay_engine score --windows "$W" --rollout "${rollouts[@]}" \
      --ckpt lafan1-supertrack-accurate="$CK_ST" lafan1-supermimic-accurate="$CK_SM" --out "$A/scores.jsonl"
    python3 "$HERE/gates.py" "$A/scores.jsonl" "$A" | tee "$A/criteria.txt" ;;
  one)
    need STEEL_PY MAIN_PY PAIR CK_ST CK_SM
    CSV=$OUT/one
    mkdir -p "$CSV"
    for n in 4096 16384 1024 65536; do
      for rep in 1 2 3; do
        for name in $ROWS; do row "$GPU" "$name" "$n" "${name}_n${n}_r${rep}_g$GPU"; done
      done
      say "N $n done"
    done
    for name in fastest-passing st-fastest sm-fastest; do  # the largest N: doubling until a process fails
      n=131072
      while :; do
        ok=1
        for rep in 1 2 3; do row "$GPU" "$name" "$n" "${name}_n${n}_r${rep}_g$GPU" || { ok=0; break; }; done
        [ "$ok" = 1 ] || break
        n=$((n * 2))
      done
      say "$name: the doubling ended at N $n"
    done ;;
  four)
    need STEEL_PY MAIN_PY PAIR CK_ST CK_SM
    CSV=$OUT/four
    read -r -a CARDS <<< "${GPUS:-0 1 2 3}"
    [ "${#CARDS[@]}" -ge 4 ] || { say "four: GPUS needs four GPUs"; exit 1; }
    mkdir -p "$CSV"
    for rep in 1 2 3; do
      for k in 1 2 4; do
        for name in fastest-passing mujoco-newton st-fastest sm-fastest; do
          pids=()
          for i in $(seq 0 $((k - 1))); do
            row "${CARDS[$i]}" "$name" 16384 "${name}_n16384_r${rep}k${k}_g${CARDS[$i]}" --steps 1000 --barrier "$CSV/barrier/${name}_k${k}_r$rep" \
              --barrier_n "$k" &
            pids+=("$!")
            sleep 20  # one CUDA start at a time
          done
          for p in "${pids[@]}"; do wait "$p"; done
        done
      done
    done ;;
  traces)
    need STEEL_PY MAIN_PY PAIR CK_ST CK_SM
    CSV=$OUT/traces
    NSYS=${NSYS:-nsys}
    mkdir -p "$CSV"
    for n in 4096 ${TRACE_N:-}; do
      for name in fastest-passing st-fastest sm-fastest; do
        tag="trace_${name}_n$n"
        if [ "$name" = fastest-passing ]; then
          read -r sub it opts <<< "${SET[$name]}"
          # shellcheck disable=SC2086
          proc "$GPU" "$tag" "$NSYS" profile --force-overwrite true --trace cuda --cuda-graph-trace node --output "$CSV/$tag" "$STEEL_PY" -I \
            "$HERE/steel_g1.py" --mode speed --windows "$W" --mjcf "$MJCF" --art_max 256 --worlds "$n" --setting "$name" --substeps "$sub" \
            --iterations "$it" $opts --out "$CSV/steel_rows.csv"
        else
          ck=$CK_ST
          [ "$name" = sm-fastest ] && ck=$CK_SM
          # shellcheck disable=SC2086
          proc "$GPU" "$tag" "$NSYS" profile --force-overwrite true --trace cuda --cuda-graph-trace node --output "$CSV/$tag" "$MAIN_PY" -m \
            ndm_gym.wmbench.speed_pair --pair "$PAIR" --single --num_envs "$n" --warm_seconds 1 --repeats 1 --replay "$W" --mode ndm \
            --wm_ckpt "$ck" $FAST_MLP --out "$CSV/pair_rows.csv"
        fi
        "$NSYS" stats --report cuda_gpu_kern_sum --format csv --output "$CSV/$tag" "$CSV/$tag.nsys-rep" > /dev/null 2>&1 \
          || say "$tag: nsys stats failed"
        script=phases_ndm.py
        [ "$name" = fastest-passing ] && script=phases.py
        python3 "$HERE/$script" "$CSV/${tag}_cuda_gpu_kern_sum.csv" > "$CSV/$tag.phases.txt" 2>&1 || say "$tag: $script failed"
      done
    done ;;
  *) say "part record, accuracy, one, four or traces, not $PART"; exit 1 ;;
esac
say "$PART done"
