#!/bin/bash
# Run 2, Phase A: augmentation x balancing check for SmallCNN (4 classes, VALIDATION only).
# Run-1 best SmallCNN config (lr 2e-3, wd 1e-4, bs 64, read from the run-1 tuning rows),
# augment {none, vflip, basic} x balance {weighted, oversample} x seeds {0, 1, 2} = 18 runs,
# max RUN2_MAX_EPOCHS epochs with early stopping. Run tag n2_A_<augment>_<balance>.
# Finished runs are skipped (run_id already in results/experiments.csv); interrupted runs resume.
# Run:  nohup bash scripts/queue_run2_A.sh > logs/queue_run2_A.out 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p logs
EPOCHS=$(python -c "from src import config; print(config.RUN2_MAX_EPOCHS)")
BEST=$(python -m scripts.best_cnn_config smallcnn)   # run-1 tuning winner
echo "$(date '+%H:%M') Phase A, SmallCNN config: $BEST, max $EPOCHS epochs"
# Seed loop outside, so that after each seed all 6 settings have the same number of seeds
for SEED in 0 1 2; do
  for AUG in none vflip basic; do
    for BAL in weighted oversample; do
      TAG=n2_A_${AUG}_${BAL}
      echo "$(date '+%H:%M') start smallcnn $TAG seed $SEED"
      python -m src.train --model smallcnn --classes 4 --seed $SEED $BEST --epochs $EPOCHS \
        --augment $AUG --balance $BAL --run-tag $TAG --resume \
        > logs/smallcnn_c4_${TAG}_s${SEED}.log 2>&1 || echo "FAILED smallcnn $TAG s$SEED"
    done
  done
done
echo "$(date '+%H:%M') Phase A queue done"
