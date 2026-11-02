#!/bin/bash
# Run 2: finish the RQ2 learning curve (user decision 2026-11-01): 1/5/20 % of the labelled training cells,
# all 3 families, seeds 0-2, run-2 settings (AUG*/BAL*, tuned config, max epochs, patience from src/config.py).
# Order: ResNet18-scratch seed 0 first (so every family has a seed-0 curve), then seeds 1 and 2.
# Finished runs are skipped (src.train checks experiments.csv), interrupted runs continue (--resume),
# so the script can be stopped and restarted at any time. 21 runs, roughly 6-8 h on MPS. VALIDATION only.
# Run:  nohup bash scripts/queue_run2_lc_finish.sh > logs/queue_run2_lc_finish.out 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
AUG=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['augment'])")
BAL=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['balance'])")
EPOCHS=$(python -c "from src import config; print(config.RUN2_MAX_EPOCHS)")

while pgrep -f "src.train" > /dev/null; do sleep 30; done   # one GPU job at a time
echo "$(date '+%H:%M') learning-curve queue starts, AUG*=$AUG BAL*=$BAL"

train_lc() {  # $1 model, $2 seed, $3 label fraction
  local MODEL=$1 SEED=$2 FRAC=$3 TAG=n2_lf$3
  local RUN_ID=${MODEL}_c4_${TAG}_s${SEED}
  # Skip finished runs here, before the log redirect below would overwrite their training log.
  if python -c "import sys; from src.experiments import has_run; sys.exit(0 if has_run('$RUN_ID') else 1)"; then
    echo "$(date '+%H:%M') skip $RUN_ID: already in experiments.csv"; return
  fi
  BEST=$(python -m scripts.best_cnn_config $MODEL n2_tune) || { echo "no tuning result for $MODEL"; return; }
  echo "$(date '+%H:%M') start $MODEL $TAG seed $SEED $BEST"
  python -m src.train --model $MODEL --classes 4 --seed $SEED $BEST --epochs $EPOCHS --augment $AUG \
    --balance $BAL --run-tag $TAG --label-fraction $FRAC --resume \
    > logs/${MODEL}_c4_${TAG}_s${SEED}.log 2>&1 || echo "FAILED $MODEL $TAG s$SEED"
}

for FRAC in 0.01 0.05 0.2; do train_lc resnet18_scratch 0 $FRAC; done
for SEED in 1 2; do
  for MODEL in smallcnn resnet18_pretrained resnet18_scratch; do
    for FRAC in 0.01 0.05 0.2; do train_lc $MODEL $SEED $FRAC; done
  done
done
python -m scripts.fig_learning_curve --prefix n2_ --suffix _run2
echo "$(date '+%H:%M') learning-curve queue done"
