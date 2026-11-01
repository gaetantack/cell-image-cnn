#!/bin/bash
# Run 2, Phase E2 for ResNet18-pretrained (started by hand at 05:12 on 2026-11-01, see notes/overnight_log_2.md):
# learning curve 1/5/20 %, seed 0, run-2 settings, VALIDATION. Finished runs are skipped.
# Run:  nohup bash scripts/queue_run2_lc_pretrained.sh > logs/lc_pretrained_run2.out 2>&1 &
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
for FRAC in 0.01 0.05 0.2; do
  echo "$(date '+%H:%M') start resnet18_pretrained n2_lf$FRAC"
  python -m src.train --model resnet18_pretrained --classes 4 --seed 0 --lr 0.001 --weight-decay 0.0001 --batch-size 64 \
    --epochs 25 --augment vflip --balance weighted --run-tag n2_lf$FRAC --label-fraction $FRAC --resume \
    > logs/resnet18_pretrained_c4_n2_lf${FRAC}_s0.log 2>&1 || echo "FAILED $FRAC"
done
echo "$(date '+%H:%M') lc pretrained done"
