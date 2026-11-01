#!/bin/bash
# Run 2, rest of Phase C + Phase E. Replaces the Phase C/E part of scripts/queue_run2_BC.sh (started 01:44),
# because Phase E must start an ITEM only if the whole item (all its runs) ends before 06:30:
# a lone 7-class ResNet seed is not useful (a family with < 3 seeds is never reported as a result).
#   Phase C: 4-class finals, seeds 0-2 (finished runs are skipped, so this just continues)
#   Phase E1: 7-class finals per family (item = 3 seeds of one family)
#   Phase E2: learning curve 1/5/20 %, seed 0, per family (item = 3 fractions of one family)
# All VALIDATION. Waits for any training run that is still going (e.g. one started by queue_run2_BC.sh).
# Run:  nohup bash scripts/queue_run2_CE.sh > logs/queue_run2_CE.out 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
FAMILIES="smallcnn resnet18_pretrained resnet18_scratch"
AUG=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['augment'])")
BAL=$(python -c "import json; from src import config; print(json.loads(config.RUN2_CHOICE_FILE.read_text())['balance'])")
EPOCHS=$(python -c "from src import config; print(config.RUN2_MAX_EPOCHS)")

minutes_since_noon() {  # 06:30 -> 1110
  local h=$((10#$(date +%H))) m=$((10#$(date +%M)))
  [ $h -lt 12 ] && h=$((h + 24))
  echo $(( (h - 12) * 60 + m ))
}
fits() { [ $(( $(minutes_since_noon) + $1 )) -le 1110 ]; }  # true if $1 minutes from now is before 06:30
run_minutes() { if [ "$1" = smallcnn ]; then echo 8; else echo 33; fi; }  # worst case per run (measured)

while pgrep -f "src.train" > /dev/null; do sleep 30; done   # let a running run finish first
echo "$(date '+%H:%M') queue CE starts, AUG*=$AUG BAL*=$BAL"

train() {  # $1 model, $2 classes, $3 seed, $4 run tag, rest: extra args
  local MODEL=$1 K=$2 SEED=$3 TAG=$4; shift 4
  BEST=$(python -m scripts.best_cnn_config $MODEL n2_tune) || { echo "no tuning result for $MODEL"; return; }
  echo "$(date '+%H:%M') start $MODEL $TAG c$K seed $SEED $BEST $*"
  python -m src.train --model $MODEL --classes $K --seed $SEED $BEST --epochs $EPOCHS --augment $AUG \
    --balance $BAL --run-tag $TAG --resume "$@" > logs/${MODEL}_c${K}_${TAG}_s${SEED}.log 2>&1 \
    || echo "FAILED $MODEL $TAG c$K s$SEED"
}

# Phase C (must): per run, as before
for MODEL in $FAMILIES; do
  for SEED in 0 1 2; do
    fits $(run_minutes $MODEL) || { echo "$(date '+%H:%M') skip $MODEL n2_final c4 s$SEED: too late"; continue; }
    train $MODEL 4 $SEED n2_final
  done
done
echo "$(date '+%H:%M') Phase C done"

# Phase E1: 7-class finals, a family only if all 3 seeds fit
for MODEL in $FAMILIES; do
  fits $(( 3 * $(run_minutes $MODEL) )) || { echo "$(date '+%H:%M') skip $MODEL 7-class finals: 3 seeds do not fit"; continue; }
  for SEED in 0 1 2; do train $MODEL 7 $SEED n2_final; done
done
# Phase E2: learning curve seed 0, a family only if all 3 fractions fit
for MODEL in $FAMILIES; do
  fits $(( 3 * $(run_minutes $MODEL) )) || { echo "$(date '+%H:%M') skip $MODEL learning curve: does not fit"; continue; }
  for FRAC in 0.01 0.05 0.2; do train $MODEL 4 0 n2_lf$FRAC --label-fraction $FRAC; done
done
echo "$(date '+%H:%M') run-2 queue CE done"
