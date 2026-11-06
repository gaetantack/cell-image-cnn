#!/bin/bash
# Run 4 (overnight 2026-11-06/07): masking experiment (RQ5) + label-shuffle control. VALIDATION only.
# Run-2 settings: AUG*/BAL* from results/run2_aug_bal_choice.json (vflip + weighted), tuned lr/wd/batch
# per family from the n2_tune runs (scripts/best_cnn_config.py), RUN2_MAX_EPOCHS, patience from config.
# Masked images in train AND val (src.train --mask-mode); 4 classes. Order (priority):
#   1. smallcnn            cell_only        seeds 0-2  (tag n4_cellonly)
#   2. smallcnn            background_only  seeds 0-2  (tag n4_bgonly)
#   3. resnet18_pretrained cell_only        seeds 0-2  (tag n4_cellonly)
#   4. resnet18_pretrained background_only  seed 0     (tag n4_bgonly; single seed, report as such)
#   5. smallcnn            original images, permuted TRAINING labels, seeds 0-2 (tag n4_shuffled)
# A run starts only if its worst case (SmallCNN 8 min, ResNet18 33 min; run-2 measurements) ends before
# the deadline (06:30, scripts/queue_common.sh). Finished runs are skipped BEFORE the log redirect;
# interrupted runs continue (--resume). Safe to stop and restart at any time.
# Run:  nohup bash scripts/queue_run4.sh > logs/queue_run4.out 2>&1 &
cd "$(dirname "$0")/.."
source scripts/queue_common.sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
AUG=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['augment'])")
BAL=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['balance'])")
EPOCHS=$(python -c "from src import config; print(config.RUN2_MAX_EPOCHS)")
run_minutes() { if [ "$1" = smallcnn ]; then echo 8; else echo 33; fi; }  # worst case per run

while pgrep -f "src.train" > /dev/null; do sleep 30; done   # one GPU job at a time
echo "$(date '+%H:%M') queue run 4 starts, AUG*=$AUG BAL*=$BAL max epochs $EPOCHS"

train() {  # $1 model, $2 seed, $3 run tag, rest: extra args for src.train
  local MODEL=$1 SEED=$2 TAG=$3; shift 3
  local RUN=${MODEL}_c4_${TAG}_s${SEED}
  skip_if_done $RUN && return   # before the log redirect below
  fits $(run_minutes $MODEL) || { echo "$(date '+%H:%M') skip $RUN: would not end before the deadline"; return; }
  BEST=$(python -m scripts.best_cnn_config $MODEL n2_tune) || { echo "no tuning result for $MODEL"; return; }
  for TRY in 1 2 3; do  # same run failing 3x -> skip it
    echo "$(date '+%H:%M') start $RUN (try $TRY) $BEST $*"
    python -m src.train --model $MODEL --classes 4 --seed $SEED $BEST --epochs $EPOCHS --augment $AUG \
      --balance $BAL --run-tag $TAG --resume "$@" >> logs/${RUN}.log 2>&1 && { echo "$(date '+%H:%M') done $RUN"; return; }
    echo "$(date '+%H:%M') FAILED $RUN (try $TRY), see logs/${RUN}.log"
  done
}

for SEED in 0 1 2; do train smallcnn $SEED n4_cellonly --mask-mode cell_only; done
for SEED in 0 1 2; do train smallcnn $SEED n4_bgonly --mask-mode background_only; done
for SEED in 0 1 2; do train resnet18_pretrained $SEED n4_cellonly --mask-mode cell_only; done
train resnet18_pretrained 0 n4_bgonly --mask-mode background_only
for SEED in 0 1 2; do train smallcnn $SEED n4_shuffled --shuffle-labels; done
echo "$(date '+%H:%M') queue run 4 done"
