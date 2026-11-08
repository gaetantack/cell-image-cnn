# Table 2 (test, 4 classes; mean ± SD over seeds 0–2)

**Test numbers (one run).**

| Model | Macro-F1 | Balanced acc. | Recall G1 | Recall S | Recall G2 | Recall Mitosis | PR-AUC Mitosis | Inference (s/1,000 cells, CPU) |
|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 0.154 ± 0.000 | 0.250 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.022 ± 0.000 | 0.000 ± 0.000 |
| Chance (stratified) | 0.257 ± 0.006 | 0.257 ± 0.006 | 0.44 ± 0.01 | 0.28 ± 0.01 | 0.28 ± 0.01 | 0.03 ± 0.01 | 0.022 ± 0.000 | 0.000 ± 0.000 |
| Random forest | 0.659 ± 0.004 | 0.644 ± 0.003 | 0.85 ± 0.00 | 0.65 ± 0.00 | 0.77 ± 0.00 | 0.31 ± 0.01 | 0.382 ± 0.003 | 0.016 ± 0.000 |
| SmallCNN | 0.746 ± 0.003 | 0.742 ± 0.009 | 0.86 ± 0.02 | 0.71 ± 0.03 | 0.77 ± 0.02 | 0.64 ± 0.04 | 0.695 ± 0.008 | 0.891 ± 0.028 |
| ResNet18 (ImageNet) | 0.776 ± 0.004 | 0.778 ± 0.014 | 0.85 ± 0.03 | 0.72 ± 0.05 | 0.78 ± 0.01 | 0.77 ± 0.05 | 0.845 ± 0.014 | 6.825 ± 0.180 |
| ResNet18 (scratch) | 0.735 ± 0.004 | 0.731 ± 0.003 | 0.85 ± 0.01 | 0.68 ± 0.01 | 0.80 ± 0.01 | 0.60 ± 0.01 | 0.674 ± 0.014 | 7.958 ± 1.219 |
| Eulenberg et al. 2017 (reference, different split and metric) | [CHECK] | [CHECK] | [CHECK] | [CHECK] | [CHECK] | [CHECK] | [CHECK] | [CHECK] |

Chance PR-AUC for mitosis = its prevalence in the test split: 0.022. Eulenberg et al. report [CHECK: accuracy on 7 classes? which split? brightfield + darkfield?]; their numbers are not directly comparable (different split, no macro-F1).
