# Table 2 (TEST, 4 classes; mean ± SD over seeds 0–2)

**Test numbers, one run on 2026-11-07 (tag `test-frozen`).** Source: `results/table_test_summary.md` (`scripts/final_test.py --split test`); train time from `results/table_val_summary_run2.md` (training on the training set, not repeated). Inference on CPU (4 threads). Made by `scripts/make_table2_test.py`.

| Model | Seeds | Macro-F1 (test) | Balanced acc. (test) | Recall G1 | Recall S | Recall G2 | Recall Mitosis | ROC-AUC macro | PR-AUC macro | Inference (s/1,000 cells) | Train time (s, run 2) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 3 | 0.154 ± 0.000 | 0.250 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.500 ± 0.000 | 0.250 ± 0.000 | 0.000 ± 0.000 | 0 ± 0 |
| Chance (stratified) | 3 | 0.257 ± 0.006 | 0.257 ± 0.006 | 0.44 ± 0.01 | 0.28 ± 0.01 | 0.28 ± 0.01 | 0.03 ± 0.01 | 0.500 ± 0.000 | 0.250 ± 0.000 | 0.000 ± 0.000 | 0 ± 0 |
| Random forest | 3 | 0.659 ± 0.004 | 0.644 ± 0.003 | 0.85 ± 0.00 | 0.65 ± 0.00 | 0.77 ± 0.00 | 0.31 ± 0.01 | 0.921 ± 0.000 | 0.692 ± 0.000 | 0.016 ± 0.000 | 7 ± 0 |
| SmallCNN | 3 | 0.746 ± 0.003 | 0.742 ± 0.009 | 0.86 ± 0.02 | 0.71 ± 0.03 | 0.77 ± 0.02 | 0.64 ± 0.04 | 0.941 ± 0.001 | 0.799 ± 0.003 | 0.891 ± 0.028 | 352 ± 32 |
| ResNet18 (ImageNet) | 3 | 0.776 ± 0.004 | 0.778 ± 0.014 | 0.85 ± 0.03 | 0.72 ± 0.05 | 0.78 ± 0.01 | 0.77 ± 0.05 | 0.946 ± 0.005 | 0.840 ± 0.009 | 6.825 ± 0.180 | 1386 ± 456 |
| ResNet18 (scratch) | 3 | 0.735 ± 0.004 | 0.731 ± 0.003 | 0.85 ± 0.01 | 0.68 ± 0.01 | 0.80 ± 0.01 | 0.60 ± 0.01 | 0.940 ± 0.001 | 0.793 ± 0.003 | 7.958 ± 1.219 | 1901 ± 121 |
| Random forest + raw DF levels (RQ5 side experiment) | 3 | 0.664 ± 0.008 | 0.652 ± 0.007 | 0.85 ± 0.00 | 0.66 ± 0.00 | 0.77 ± 0.00 | 0.33 ± 0.02 | 0.923 ± 0.000 | 0.702 ± 0.001 | 0.018 ± 0.001 | 10 ± 1 |
| Eulenberg et al. 2017, DeepFlow (reference only; see note) | 5-fold CV | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. | n.r. |

Chance PR-AUC (macro) = 0.250; chance PR-AUC for mitosis = its test prevalence 0.022. n.r. = not reported. Eulenberg et al. 2017 (DeepFlow, a 42-layer Inception-style network; brightfield + darkfield input, as here) report only accuracy, from fivefold cross-validation on all 32,266 cells, and no 4-class result: 79.40% ± 0.77% for the seven phases, and 98.73% ± 0.16% for five classes (G1, S and G2 merged into one interphase class, the four mitotic phases separate), against 92.35% for boosting on image features (Blasi et al. 2016) on the same five classes. Not directly comparable with our numbers (other metric, class set and evaluation). For orientation only: the 7-class accuracy of our models on the test set is 0.753 ± 0.001 (forest) and 0.748 ± 0.015 (SmallCNN), computed on 2026-11-08 from the frozen test predictions (`results/preds/finaltest_test/*_c7_*`; no retraining). Accuracy is dominated by interphase cells (97.8% of all cells): on five classes, always predicting "interphase" already gives 97.8% accuracy, which is why we report macro-F1.
