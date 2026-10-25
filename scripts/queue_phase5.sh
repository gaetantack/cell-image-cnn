#!/bin/bash
# Phase 5 (after phase 3): figures from phase 3, UMAP, Grad-CAM, pseudotime, RQ3 ablation (seed 0,
# preliminary), learning curve seed 0, then learning-curve seeds 1 and 2.
# A run only starts if it should end before 06:30 (ResNet run ≈ 16 min, SmallCNN ≈ 3 min).
# Finished runs are skipped; interrupted runs resume.
# Run:  nohup bash scripts/queue_phase5.sh > logs/queue_phase5.out 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p logs

too_late() {  # $1 = latest start time HHMM; true between that time and 12:00
  local now=$(date +%H%M)
  [ "$((10#$now))" -ge "$((10#$1))" ] && [ "$((10#$now))" -lt 1200 ]
}
latest_start() {  # $1 = model
  if [ "$1" = smallcnn ]; then echo 0625; else echo 0612; fi
}

while pgrep -f queue_phase3.sh > /dev/null; do sleep 60; done
echo "$(date '+%H:%M') phase 3 finished; making figures"
for S in make_results_table make_table1 fig_model_comparison fig_confusion fig_umap fig_gradcam pseudotime; do
  python -m scripts.$S > logs/$S.log 2>&1 || echo "FAILED $S"
done

# RQ3 ablation, SmallCNN, seed 0. augment=basic + balance=weighted is the final run (smallcnn_c4_final_s0).
BEST=$(python -m scripts.best_cnn_config smallcnn)
for AUG in none basic; do
  for BAL in none weighted oversample; do
    [ "$AUG" = basic ] && [ "$BAL" = weighted ] && continue
    too_late $(latest_start smallcnn) && { echo "$(date '+%H:%M') skip ablation $AUG $BAL: too late"; continue; }
    echo "$(date '+%H:%M') start ablation $AUG $BAL"
    python -m src.train --model smallcnn --classes 4 --seed 0 $BEST --augment $AUG --balance $BAL \
      --run-tag abl_${AUG}_${BAL} --resume > logs/smallcnn_c4_abl_${AUG}_${BAL}_s0.log 2>&1 \
      || echo "FAILED ablation $AUG $BAL"
  done
done
python -m scripts.fig_ablation > logs/fig_ablation.log 2>&1 || echo "FAILED fig_ablation"

learning_curve() {  # $1 = seed
  for MODEL in smallcnn resnet18_pretrained resnet18_scratch; do
    BEST=$(python -m scripts.best_cnn_config $MODEL)
    for FRAC in 0.01 0.05 0.2; do
      too_late $(latest_start $MODEL) && { echo "$(date '+%H:%M') skip $MODEL lf$FRAC s$1: too late"; continue; }
      echo "$(date '+%H:%M') start $MODEL lf$FRAC seed $1"
      python -m src.train --model $MODEL --classes 4 --seed $1 $BEST --label-fraction $FRAC \
        --run-tag lf$FRAC --resume > logs/${MODEL}_c4_lf${FRAC}_s$1.log 2>&1 || echo "FAILED $MODEL lf$FRAC s$1"
    done
  done
  python -m scripts.fig_learning_curve > logs/fig_learning_curve.log 2>&1 || echo "FAILED fig_learning_curve"
}

learning_curve 0
learning_curve 1
learning_curve 2
echo "$(date '+%H:%M') phase 5 queue done"
