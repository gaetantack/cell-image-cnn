# Table 2 (val, 4 classes; mean ± SD over seeds 0–2)

**Validation dress rehearsal — not the test numbers.**

| Model | Macro-F1 | Balanced acc. | Recall G1 | Recall S | Recall G2 | Recall Mitosis | PR-AUC Mitosis | Inference (s/1,000 cells, CPU) |
|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 0.154 ± 0.000 | 0.250 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.022 ± 0.000 | 0.000 ± 0.000 |
| Chance (stratified) | 0.257 ± 0.006 | 0.257 ± 0.006 | 0.44 ± 0.01 | 0.28 ± 0.01 | 0.28 ± 0.01 | 0.03 ± 0.01 | 0.022 ± 0.000 | 0.000 ± 0.000 |
| Random forest | 0.632 ± 0.001 | 0.627 ± 0.001 | 0.84 ± 0.00 | 0.64 ± 0.00 | 0.77 ± 0.00 | 0.26 ± 0.01 | 0.261 ± 0.006 | 0.019 ± 0.001 |
| SmallCNN | 0.744 ± 0.010 | 0.744 ± 0.012 | 0.86 ± 0.01 | 0.69 ± 0.02 | 0.80 ± 0.02 | 0.63 ± 0.04 | 0.692 ± 0.013 | 1.741 ± 0.086 |
| ResNet18 (ImageNet) | 0.781 ± 0.007 | 0.777 ± 0.009 | 0.84 ± 0.03 | 0.71 ± 0.04 | 0.80 ± 0.01 | 0.75 ± 0.03 | 0.837 ± 0.001 | 13.369 ± 1.311 |
| ResNet18 (scratch) | 0.745 ± 0.003 | 0.740 ± 0.004 | 0.84 ± 0.00 | 0.70 ± 0.01 | 0.82 ± 0.01 | 0.60 ± 0.02 | 0.660 ± 0.014 | 14.389 ± 0.707 |
| Eulenberg et al. 2017 (reference, different split and metric) | [CHECK] | [CHECK] | [CHECK] | [CHECK] | [CHECK] | [CHECK] | [CHECK] | [CHECK] |

Chance PR-AUC for mitosis = its prevalence in the val split: 0.022. Eulenberg et al. report [CHECK: accuracy on 7 classes? which split? brightfield + darkfield?]; their numbers are not directly comparable (different split, no macro-F1).
