# Validation summary (mean ± SD over seeds 0, 1, 2)

**Validation results — not the final test numbers.** Made by `scripts/make_results_table.py`.

## 4 classes

| Model | Seeds | Macro-F1 (val) | Balanced acc. (val) | Recall G1 | Recall S | Recall G2 | Recall Mitosis | Train time (s) | Parameters |
|---|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 3 | 0.154 ± 0.000 | 0.250 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | — |
| Chance (stratified) | 3 | 0.257 ± 0.006 | 0.257 ± 0.006 | 0.44 ± 0.01 | 0.28 ± 0.01 | 0.28 ± 0.01 | 0.03 ± 0.01 | 0 ± 0 | — |
| Random forest | 3 | 0.632 ± 0.001 | 0.627 ± 0.001 | 0.84 ± 0.00 | 0.64 ± 0.00 | 0.77 ± 0.00 | 0.26 ± 0.01 | 7 ± 0 | — |
| SmallCNN | 3 | 0.687 ± 0.006 | 0.725 ± 0.020 | 0.82 ± 0.03 | 0.69 ± 0.03 | 0.74 ± 0.03 | 0.65 ± 0.10 | 183 ± 3 | 389,636 |
| ResNet18 (ImageNet) | 3 | 0.761 ± 0.006 | 0.795 ± 0.004 | 0.85 ± 0.01 | 0.72 ± 0.03 | 0.79 ± 0.02 | 0.83 ± 0.01 | 928 ± 18 | 11,175,428 |
| Random forest + raw DF levels (RQ5 side experiment) | 3 | 0.647 ± 0.003 | 0.644 ± 0.003 | 0.84 ± 0.00 | 0.64 ± 0.00 | 0.78 ± 0.00 | 0.31 ± 0.01 | 10 ± 1 | — |

## 7 classes

| Model | Seeds | Macro-F1 (val) | Balanced acc. (val) | Recall G1 | Recall S | Recall G2 | Recall Prophase | Recall Metaphase | Recall Anaphase | Recall Telophase | Train time (s) | Parameters |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 3 | 0.088 ± 0.000 | 0.143 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | — |
| Chance (stratified) | 3 | 0.146 ± 0.003 | 0.146 ± 0.003 | 0.44 ± 0.01 | 0.27 ± 0.01 | 0.28 ± 0.02 | 0.03 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0 ± 0 | — |
| Random forest | 3 | 0.559 ± 0.004 | 0.586 ± 0.001 | 0.84 ± 0.00 | 0.63 ± 0.00 | 0.76 ± 0.00 | 0.27 ± 0.01 | 0.10 ± 0.00 | 0.50 ± 0.00 | 1.00 ± 0.00 | 5 ± 0 | — |
