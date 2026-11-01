# Validation summary (mean ± SD over seeds 0, 1, 2)

**Validation results — not the final test numbers.** Made by `scripts/make_results_table.py --tag n2_final` (CNN final runs with run tag `n2_final`; random forest and chance rows from run 1, unchanged).

## 4 classes

| Model | Seeds | Macro-F1 (val) | Balanced acc. (val) | Recall G1 | Recall S | Recall G2 | Recall Mitosis | Train time (s) | Parameters |
|---|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 3 | 0.154 ± 0.000 | 0.250 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | — |
| Chance (stratified) | 3 | 0.257 ± 0.006 | 0.257 ± 0.006 | 0.44 ± 0.01 | 0.28 ± 0.01 | 0.28 ± 0.01 | 0.03 ± 0.01 | 0 ± 0 | — |
| Random forest | 3 | 0.632 ± 0.001 | 0.627 ± 0.001 | 0.84 ± 0.00 | 0.64 ± 0.00 | 0.77 ± 0.00 | 0.26 ± 0.01 | 7 ± 0 | — |
| SmallCNN | 3 | 0.744 ± 0.010 | 0.744 ± 0.012 | 0.86 ± 0.01 | 0.69 ± 0.02 | 0.80 ± 0.02 | 0.63 ± 0.04 | 352 ± 32 | 389,636 |
| ResNet18 (ImageNet) | 3 | 0.781 ± 0.007 | 0.777 ± 0.009 | 0.84 ± 0.03 | 0.71 ± 0.04 | 0.80 ± 0.01 | 0.75 ± 0.03 | 1386 ± 456 | 11,175,428 |
| ResNet18 (scratch) | 3 | 0.745 ± 0.003 | 0.740 ± 0.004 | 0.84 ± 0.00 | 0.70 ± 0.01 | 0.82 ± 0.01 | 0.60 ± 0.02 | 1901 ± 121 | 11,175,428 |
| Random forest + raw DF levels (RQ5 side experiment) | 3 | 0.647 ± 0.003 | 0.644 ± 0.003 | 0.84 ± 0.00 | 0.64 ± 0.00 | 0.78 ± 0.00 | 0.31 ± 0.01 | 10 ± 1 | — |

## 7 classes

| Model | Seeds | Macro-F1 (val) | Balanced acc. (val) | Recall G1 | Recall S | Recall G2 | Recall Prophase | Recall Metaphase | Recall Anaphase | Recall Telophase | Train time (s) | Parameters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 3 | 0.088 ± 0.000 | 0.143 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | — |
| Chance (stratified) | 3 | 0.146 ± 0.003 | 0.146 ± 0.003 | 0.44 ± 0.01 | 0.27 ± 0.01 | 0.28 ± 0.02 | 0.03 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | — |
| Random forest | 3 | 0.559 ± 0.004 | 0.586 ± 0.001 | 0.84 ± 0.00 | 0.63 ± 0.00 | 0.76 ± 0.00 | 0.27 ± 0.01 | 0.10 ± 0.00 | 0.50 ± 0.00 | 1.00 ± 0.00 | 5 ± 0 | — |
| SmallCNN | 3 | 0.593 ± 0.034 | 0.603 ± 0.039 | 0.85 ± 0.06 | 0.61 ± 0.07 | 0.71 ± 0.09 | 0.57 ± 0.13 | 0.07 ± 0.06 | 0.50 ± 0.00 | 0.92 ± 0.14 | 244 ± 91 | 390,407 |
