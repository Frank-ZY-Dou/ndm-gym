#!/usr/bin/env bash
# The benchmark's runs: one algorithm's queue over the three animals and seeds 0-2, fixed iteration budgets (stick 4000,
# fly 6000, rat 10000), the learning rate decayed over the budget, a checkpoint kept every 250 iterations; then each run's
# last checkpoint evaluated on its training clips (0-9, with 4 perturbed repeats) and on held-out clips (10-19).
#   benchmarks/mimic_mjx_animal_tracking/run_formal.sh supertrack|supermimic   (OUT=..., DATA=... to override)
set -euo pipefail  # a run that fails stops the queue
ALGO=$1
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$REPO/logs/mimic_mjx_formal}
DATA=${DATA:-$REPO/data/mimic_mjx}
RUN="python -u"
SYS="--fused_adam --reset_masked --no_sync --compile_eager --collect_graph --update_graphs"
HOI="--pi_opt muon --wm_opt muon --pi_ema 0.995 --pi_lr_mult 2 --lr_decay_ret 0.95 --lr_decay_by 1400 --lr_decay_len 300"
SPEED="--pi_batch 2048 --wm_lr 2e-3 --fused_fk"
if [ "$ALGO" = supertrack ]; then
  OPTS="--algo supertrack --lr 4e-4 --st_fail_w 2 --st_imag_noise 0.2 --st_task_weights $HOI $SYS $SPEED"  # hand-supertrack-fast's
else
  OPTS="--algo supermimic --lr 4e-4 --action_noise 0.3 --gamma 0.99 --bptt_decay 0.95 $HOI $SYS $SPEED"  # hand-supermimic-fast's
fi
declare -A ITERS=([stick]=4000 [fly]=6000 [rat]=10000)
mkdir -p "$OUT"
for animal in stick fly rat; do
  for seed in 0 1 2; do
    name=${animal}_${ALGO}_s${seed}
    [ -f "$OUT/$name/last.pt" ] && [ -f "$OUT/$name/eval_train/metrics.json" ] && continue
    rm -rf "$OUT/$name"
    it=${ITERS[$animal]}
    cd "$REPO"; $RUN -m ndm_gym.train_animal --animal $animal --clip_ids 0-9 --data_dir "$DATA" $OPTS \
      --iterations $it --lr_decay_iters $it --log_every 250 --ckpt_every 250 --ckpt_keep --seed $seed --log_dir "$OUT/$name" > "$OUT/$name.log" 2>&1
    $RUN -m ndm_gym.evaluate_animal --ckpt "$OUT/$name/last.pt" --repeats 4 --out "$OUT/$name/eval_train" > "$OUT/${name}_eval_train.log" 2>&1
    $RUN -m ndm_gym.evaluate_animal --ckpt "$OUT/$name/last.pt" --clip_ids 10-19 --repeats 4 --out "$OUT/$name/eval_heldout" \
      > "$OUT/${name}_eval_heldout.log" 2>&1
    echo "done $name $(date +%H:%M)" >> "$OUT/queue_$ALGO.txt"
  done
done
