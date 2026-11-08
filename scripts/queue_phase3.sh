#!/bin/bash
# Phase 3: tune each CNN family on 12 configurations (4 classes, seed 0, TUNE_EPOCHS epochs),
# then train the best configuration with seeds 0, 1, 2 (FINAL_EPOCHS), for 4 and then 7 classes.
# Finished runs are skipped (run_id already in results/experiments.csv); interrupted runs resume.
# Run:  nohup bash scripts/queue_phase3.sh > logs/queue_phase3.out 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p logs
TUNE_EPOCHS=$(python -c "from src import config; print(config.TUNE_EPOCHS)")
N_GRID=$(python -c "from src import config; print(len(config.CNN_GRID))")

for MODEL in smallcnn resnet18_pretrained resnet18_scratch; do
  for K in $(seq 0 $((N_GRID - 1))); do
    ARGS=$(python -c "from src import config; g = config.CNN_GRID[$K]; print(f\"--lr {g['lr']} --weight-decay {g['weight_decay']} --batch-size {g['batch_size']}\")")
    TAG=$(printf "tune%02d" $K)
    echo "$(date '+%H:%M') start $MODEL $TAG $ARGS"
    python -m src.train --model $MODEL --classes 4 --seed 0 $ARGS --epochs $TUNE_EPOCHS \
      --run-tag $TAG --resume > logs/${MODEL}_c4_${TAG}_s0.log 2>&1 || echo "FAILED $MODEL $TAG"
  done
  BEST=$(python -m scripts.best_cnn_config $MODEL)
  echo "$(date '+%H:%M') best $MODEL: $BEST"
  for SEED in 0 1 2; do
    echo "$(date '+%H:%M') start $MODEL final c4 seed $SEED"
    python -m src.train --model $MODEL --classes 4 --seed $SEED $BEST --run-tag final --resume \
      > logs/${MODEL}_c4_final_s${SEED}.log 2>&1 || echo "FAILED $MODEL final c4 s$SEED"
  done
done

# 7 classes: best 4-class configuration of each family, no extra tuning
for MODEL in smallcnn resnet18_pretrained resnet18_scratch; do
  BEST=$(python -m scripts.best_cnn_config $MODEL)
  for SEED in 0 1 2; do
    echo "$(date '+%H:%M') start $MODEL final c7 seed $SEED"
    python -m src.train --model $MODEL --classes 7 --seed $SEED $BEST --run-tag final --resume \
      > logs/${MODEL}_c7_final_s${SEED}.log 2>&1 || echo "FAILED $MODEL final c7 s$SEED"
  done
done
echo "$(date '+%H:%M') phase 3 queue done"
