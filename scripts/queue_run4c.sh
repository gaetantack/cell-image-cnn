#!/bin/bash
# Run 4, use of the idle GPU after queues 4 and 4b (decision logged in notes/overnight_log_4.md):
#   1. resnet18_pretrained background_only      seeds 1, 2 (tag n4_bgonly; completes the seed-0 row to 3 seeds)
#   2. resnet18_pretrained background_only_disc seed 0     (tag n4_bgdisc; single seed)
# Run-2 settings, 4 classes, VALIDATION. Each run starts only if its worst case (40 min: ResNet18 measured up to
# 36.5 min tonight with CPU jobs in parallel) ends before the deadline. Waits for queue 4b and any training run.
# Run:  nohup bash scripts/queue_run4c.sh > logs/queue_run4c.out 2>&1 &
cd "$(dirname "$0")/.."
source scripts/queue_common.sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
AUG=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['augment'])")
BAL=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['balance'])")
EPOCHS=$(python -c "from src import config; print(config.RUN2_MAX_EPOCHS)")
MODEL=resnet18_pretrained

while pgrep -f "queue_run4b.sh" > /dev/null || pgrep -f "src.train" > /dev/null; do sleep 30; done
echo "$(date '+%H:%M') queue run 4c starts"
train() {  # $1 seed, $2 tag, $3 mask mode
  local RUN=${MODEL}_c4_$2_s$1
  skip_if_done $RUN && return   # before the log redirect
  fits 40 || { echo "$(date '+%H:%M') skip $RUN: would not end before the deadline"; return; }
  BEST=$(python -m scripts.best_cnn_config $MODEL n2_tune)
  for TRY in 1 2 3; do
    echo "$(date '+%H:%M') start $RUN (try $TRY)"
    python -m src.train --model $MODEL --classes 4 --seed $1 $BEST --epochs $EPOCHS --augment $AUG --balance $BAL \
      --run-tag $2 --resume --mask-mode $3 >> logs/${RUN}.log 2>&1 && { echo "$(date '+%H:%M') done $RUN"; return; }
    echo "$(date '+%H:%M') FAILED $RUN (try $TRY)"
  done
}
train 1 n4_bgonly background_only
train 2 n4_bgonly background_only
train 0 n4_bgdisc background_only_disc
echo "$(date '+%H:%M') queue run 4c done"
