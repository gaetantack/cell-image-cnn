#!/bin/bash
# Run 4, follow-up queue (added after the suspicion check on background-only, see notes/overnight_log_4.md):
# SmallCNN trained on "background_only_disc" images (fixed central disc of radius MASK_DISC_RADIUS_PX + the cell
# mask filled, so the hole no longer shows the cell size), seeds 0-2, tag n4_bgdisc, run-2 settings. VALIDATION.
# Waits until scripts/queue_run4.sh has finished (one GPU job at a time). Same guards as queue_run4.sh.
# Run:  nohup bash scripts/queue_run4b.sh > logs/queue_run4b.out 2>&1 &
cd "$(dirname "$0")/.."
source scripts/queue_common.sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
AUG=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['augment'])")
BAL=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['balance'])")
EPOCHS=$(python -c "from src import config; print(config.RUN2_MAX_EPOCHS)")

while pgrep -f "queue_run4.sh" > /dev/null || pgrep -f "src.train" > /dev/null; do sleep 30; done
echo "$(date '+%H:%M') queue run 4b starts"
for SEED in 0 1 2; do
  RUN=smallcnn_c4_n4_bgdisc_s${SEED}
  skip_if_done $RUN && continue   # before the log redirect
  fits 8 || { echo "$(date '+%H:%M') skip $RUN: would not end before the deadline"; continue; }
  BEST=$(python -m scripts.best_cnn_config smallcnn n2_tune)
  for TRY in 1 2 3; do
    echo "$(date '+%H:%M') start $RUN (try $TRY)"
    python -m src.train --model smallcnn --classes 4 --seed $SEED $BEST --epochs $EPOCHS --augment $AUG \
      --balance $BAL --run-tag n4_bgdisc --resume --mask-mode background_only_disc >> logs/${RUN}.log 2>&1 \
      && { echo "$(date '+%H:%M') done $RUN"; break; }
    echo "$(date '+%H:%M') FAILED $RUN (try $TRY)"
  done
done
echo "$(date '+%H:%M') queue run 4b done"
