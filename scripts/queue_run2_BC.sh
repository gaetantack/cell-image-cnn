#!/bin/bash
# Run 2, after Phase A (scripts/queue_run2_A.sh). All numbers are VALIDATION numbers.
#   1. choose AUG*/BAL* from Phase A (scripts/choose_aug_bal.py); orientation check on the CPU
#   2. Phase B: tune every family on CNN_GRID_RUN2 (12 configs, 4 classes, seed 0, TUNE_EPOCHS epochs)
#   3. Phase C: best config per family, seeds 0-2, 4 classes, max RUN2_MAX_EPOCHS with early stopping
#   4. Phase E (if time): 7-class finals, then learning curve 1/5/20 % (seed 0)
# Families in the order SmallCNN, ResNet18 pretrained, ResNet18 scratch. Run tags start with n2_.
# A run only starts if it should end before 06:30 (estimated worst-case minutes below).
# Finished runs are skipped (run_id already in results/experiments.csv); interrupted runs resume.
# Run:  nohup bash scripts/queue_run2_BC.sh > logs/queue_run2_BC.out 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p logs
FAMILIES="smallcnn resnet18_pretrained resnet18_scratch"

# Minutes since 12:00 (so that times after midnight count up: 06:30 -> 1110)
minutes_since_noon() {
  local h=$((10#$(date +%H))) m=$((10#$(date +%M)))
  [ $h -lt 12 ] && h=$((h + 24))
  echo $(( (h - 12) * 60 + m ))
}
# fits <minutes>: true if a job of that length (worst case) ends before 06:30
fits() { [ $(( $(minutes_since_noon) + $1 )) -le 1110 ]; }
# worst-case minutes per run: <kind> <model>
est() {
  case "$1-$2" in
    tune-smallcnn) echo 2 ;;  tune-*) echo 7 ;;
    final-smallcnn) echo 8 ;; final-*) echo 35 ;;
  esac
}

while pgrep -f queue_run2_A.sh > /dev/null; do sleep 60; done
echo "$(date '+%H:%M') Phase A finished; choosing AUG*/BAL*"
python -m scripts.choose_aug_bal > logs/choose_aug_bal.log 2>&1 || { echo "FAILED choose_aug_bal"; exit 1; }
AUG=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['augment'])")
BAL=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['balance'])")
echo "$(date '+%H:%M') AUG*=$AUG BAL*=$BAL"
# Orientation check (CPU) on seed 0 of each augmentation with BAL*, in the background
python -m scripts.orientation_check --runs smallcnn_c4_n2_A_none_${BAL}_s0 smallcnn_c4_n2_A_vflip_${BAL}_s0 \
  smallcnn_c4_n2_A_basic_${BAL}_s0 --out orientation_check_val_run2.csv > logs/orientation_check_run2.log 2>&1 &

# Phase B: tuning
TUNE_EPOCHS=$(python -c "from src import config; print(config.TUNE_EPOCHS)")
N_GRID=$(python -c "from src import config; print(len(config.CNN_GRID_RUN2))")
for MODEL in $FAMILIES; do
  for K in $(seq 0 $((N_GRID - 1))); do
    ARGS=$(python -c "from src import config; g = config.CNN_GRID_RUN2[$K]; print(f\"--lr {g['lr']} --weight-decay {g['weight_decay']} --batch-size {g['batch_size']}\")")
    TAG=$(printf "n2_tune%02d" $K)
    fits $(est tune $MODEL) || { echo "$(date '+%H:%M') skip $MODEL $TAG: too late"; continue; }
    echo "$(date '+%H:%M') start $MODEL $TAG $ARGS"
    python -m src.train --model $MODEL --classes 4 --seed 0 $ARGS --epochs $TUNE_EPOCHS \
      --augment $AUG --balance $BAL --run-tag $TAG --resume > logs/${MODEL}_c4_${TAG}_s0.log 2>&1 \
      || echo "FAILED $MODEL $TAG"
  done
  echo "$(date '+%H:%M') Phase B $MODEL done; best: $(python -m scripts.best_cnn_config $MODEL n2_tune)"
done

# Phase C: finals, 4 classes
EPOCHS=$(python -c "from src import config; print(config.RUN2_MAX_EPOCHS)")
final_runs() {  # $1 = number of classes
  for MODEL in $FAMILIES; do
    BEST=$(python -m scripts.best_cnn_config $MODEL n2_tune) || { echo "no tuning result for $MODEL"; continue; }
    for SEED in 0 1 2; do
      fits $(est final $MODEL) || { echo "$(date '+%H:%M') skip $MODEL final c$1 s$SEED: too late"; continue; }
      echo "$(date '+%H:%M') start $MODEL n2_final c$1 seed $SEED $BEST"
      python -m src.train --model $MODEL --classes $1 --seed $SEED $BEST --epochs $EPOCHS \
        --augment $AUG --balance $BAL --run-tag n2_final --resume \
        > logs/${MODEL}_c$1_n2_final_s${SEED}.log 2>&1 || echo "FAILED $MODEL n2_final c$1 s$SEED"
    done
  done
}
final_runs 4
echo "$(date '+%H:%M') Phase C done"

# Phase E (only if time is left): 7-class finals, then learning curve seed 0
final_runs 7
for MODEL in $FAMILIES; do
  BEST=$(python -m scripts.best_cnn_config $MODEL n2_tune) || continue
  for FRAC in 0.01 0.05 0.2; do
    fits $(est final $MODEL) || { echo "$(date '+%H:%M') skip $MODEL n2_lf$FRAC: too late"; continue; }
    echo "$(date '+%H:%M') start $MODEL n2_lf$FRAC seed 0"
    python -m src.train --model $MODEL --classes 4 --seed 0 $BEST --epochs $EPOCHS --label-fraction $FRAC \
      --augment $AUG --balance $BAL --run-tag n2_lf$FRAC --resume \
      > logs/${MODEL}_c4_n2_lf${FRAC}_s0.log 2>&1 || echo "FAILED $MODEL n2_lf$FRAC"
  done
done
echo "$(date '+%H:%M') run-2 queue done"
