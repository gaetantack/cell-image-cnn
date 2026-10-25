#!/bin/bash
# Phase 5 (after phase 3): figures from phase 3, UMAP, Grad-CAM, learning curve (seed 0), RQ3 ablation
# (seed 0, preliminary), then learning-curve seeds 1 and 2. No new run starts after 06:30.
# Finished runs are skipped; interrupted runs resume.
# Run:  nohup bash scripts/queue_phase5.sh > logs/queue_phase5.out 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p logs

too_late() {  # true between 06:30 and 12:00
  local now=$(date +%H%M)
  [ "$now" -ge 0630 ] && [ "$now" -lt 1200 ]
}

while pgrep -f queue_phase3.sh > /dev/null; do sleep 60; done
echo "$(date '+%H:%M') phase 3 finished; making figures"
for S in make_results_table fig_model_comparison fig_confusion fig_umap fig_gradcam; do
  python -m scripts.$S > logs/$S.log 2>&1 || echo "FAILED $S"
done

learning_curve() {  # $1 = seed
  for MODEL in smallcnn resnet18_pretrained resnet18_scratch; do
    BEST=$(python -m scripts.best_cnn_config $MODEL)
    for FRAC in 0.01 0.05 0.2; do
      too_late && { echo "$(date '+%H:%M') stop: after 06:30"; exit 0; }
      echo "$(date '+%H:%M') start $MODEL lf$FRAC seed $1"
      python -m src.train --model $MODEL --classes 4 --seed $1 $BEST --label-fraction $FRAC \
        --run-tag lf$FRAC --resume > logs/${MODEL}_c4_lf${FRAC}_s$1.log 2>&1 || echo "FAILED $MODEL lf$FRAC s$1"
    done
  done
  python -m scripts.fig_learning_curve > logs/fig_learning_curve.log 2>&1 || echo "FAILED fig_learning_curve"
}

learning_curve 0

# RQ3 ablation, SmallCNN, seed 0. augment=basic + balance=weighted is the final run (smallcnn_c4_final_s0).
BEST=$(python -m scripts.best_cnn_config smallcnn)
for AUG in none basic; do
  for BAL in none weighted oversample; do
    [ "$AUG" = basic ] && [ "$BAL" = weighted ] && continue
    too_late && { echo "$(date '+%H:%M') stop: after 06:30"; exit 0; }
    echo "$(date '+%H:%M') start ablation $AUG $BAL"
    python -m src.train --model smallcnn --classes 4 --seed 0 $BEST --augment $AUG --balance $BAL \
      --run-tag abl_${AUG}_${BAL} --resume > logs/smallcnn_c4_abl_${AUG}_${BAL}_s0.log 2>&1 \
      || echo "FAILED ablation $AUG $BAL"
  done
done
python -m scripts.fig_ablation > logs/fig_ablation.log 2>&1 || echo "FAILED fig_ablation"

learning_curve 1
learning_curve 2
echo "$(date '+%H:%M') phase 5 queue done"
