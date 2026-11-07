# Validation summary (mean ± SD over seeds 0, 1, 2), made by scripts/final_test.py --split val

**Validation results — a dress rehearsal of the test-day script, not the final test numbers.**

Frozen models: CNN run-2 finals (`n2_final`); random forest re-fitted on train with the run-1 best configuration; chance = DummyClassifier. ROC-AUC/PR-AUC: one-vs-rest from class probabilities (chance: training class prior → ROC-AUC 0.5, PR-AUC = prevalence). Inference: CPU, 4 threads, s per 1,000 cells. OOB = out-of-bag accuracy of the random forest (eval accuracy next to it).

## 4 classes

| Model | Seeds | Macro-F1 (val) | Balanced acc. (val) | Recall G1 | Recall S | Recall G2 | Recall Mitosis | ROC-AUC macro | PR-AUC macro | Inference (s/1,000 cells) | Accuracy | OOB accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 3 | 0.154 ± 0.000 | 0.250 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.500 ± 0.000 | 0.250 ± 0.000 | 0.000 ± 0.000 | 0.444 ± 0.000 | — |
| Chance (stratified) | 3 | 0.257 ± 0.006 | 0.257 ± 0.006 | 0.44 ± 0.01 | 0.28 ± 0.01 | 0.28 ± 0.01 | 0.03 ± 0.01 | 0.500 ± 0.000 | 0.250 ± 0.000 | 0.000 ± 0.000 | 0.344 ± 0.007 | — |
| Random forest | 3 | 0.632 ± 0.001 | 0.627 ± 0.001 | 0.84 ± 0.00 | 0.64 ± 0.00 | 0.77 ± 0.00 | 0.26 ± 0.01 | 0.915 ± 0.000 | 0.653 ± 0.002 | 0.019 ± 0.001 | 0.753 ± 0.001 | 0.752 ± 0.000 |
| SmallCNN | 3 | 0.744 ± 0.010 | 0.744 ± 0.012 | 0.86 ± 0.01 | 0.69 ± 0.02 | 0.80 ± 0.02 | 0.63 ± 0.04 | 0.941 ± 0.001 | 0.801 ± 0.003 | 1.741 ± 0.086 | 0.791 ± 0.005 | — |
| ResNet18 (ImageNet) | 3 | 0.781 ± 0.007 | 0.777 ± 0.009 | 0.84 ± 0.03 | 0.71 ± 0.04 | 0.80 ± 0.01 | 0.75 ± 0.03 | 0.946 ± 0.003 | 0.842 ± 0.007 | 13.369 ± 1.311 | 0.794 ± 0.006 | — |
| ResNet18 (scratch) | 3 | 0.745 ± 0.003 | 0.740 ± 0.004 | 0.84 ± 0.00 | 0.70 ± 0.01 | 0.82 ± 0.01 | 0.60 ± 0.02 | 0.936 ± 0.001 | 0.789 ± 0.005 | 14.389 ± 0.707 | 0.792 ± 0.000 | — |
| Random forest + raw DF levels (RQ5 side experiment) | 3 | 0.647 ± 0.003 | 0.644 ± 0.003 | 0.84 ± 0.00 | 0.64 ± 0.00 | 0.78 ± 0.00 | 0.31 ± 0.01 | 0.918 ± 0.000 | 0.658 ± 0.000 | 0.019 ± 0.003 | 0.759 ± 0.001 | 0.756 ± 0.001 |

PR-AUC per class (4 classes; the prevalence row is the PR-AUC of a chance model):

| Model | G1 | S | G2 | Mitosis |
|---|---|---|---|---|
| Prevalence (chance PR-AUC) | 0.444 | 0.267 | 0.267 | 0.022 |
| Random forest | 0.932 ± 0.001 | 0.603 ± 0.001 | 0.816 ± 0.001 | 0.261 ± 0.006 |
| SmallCNN | 0.942 ± 0.002 | 0.677 ± 0.003 | 0.895 ± 0.002 | 0.692 ± 0.013 |
| ResNet18 (ImageNet) | 0.943 ± 0.006 | 0.677 ± 0.018 | 0.912 ± 0.005 | 0.837 ± 0.001 |
| ResNet18 (scratch) | 0.946 ± 0.002 | 0.670 ± 0.005 | 0.879 ± 0.005 | 0.660 ± 0.014 |
| Random forest + raw DF levels (RQ5 side experiment) | 0.934 ± 0.001 | 0.613 ± 0.002 | 0.817 ± 0.001 | 0.269 ± 0.003 |

## 7 classes

| Model | Seeds | Macro-F1 (val) | Balanced acc. (val) | Recall G1 | Recall S | Recall G2 | Recall Prophase | Recall Metaphase | Recall Anaphase | Recall Telophase | ROC-AUC macro | PR-AUC macro | Inference (s/1,000 cells) | Accuracy | OOB accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 3 | 0.088 ± 0.000 | 0.143 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.500 ± 0.000 | 0.143 ± 0.000 | 0.000 ± 0.001 | 0.444 ± 0.000 | — |
| Chance (stratified) | 3 | 0.146 ± 0.003 | 0.146 ± 0.003 | 0.44 ± 0.01 | 0.27 ± 0.01 | 0.28 ± 0.02 | 0.03 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.500 ± 0.000 | 0.143 ± 0.000 | 0.000 ± 0.000 | 0.342 ± 0.010 | — |
| Random forest | 3 | 0.559 ± 0.004 | 0.586 ± 0.001 | 0.84 ± 0.00 | 0.63 ± 0.00 | 0.76 ± 0.00 | 0.27 ± 0.01 | 0.10 ± 0.00 | 0.50 ± 0.00 | 1.00 ± 0.00 | 0.890 ± 0.004 | 0.579 ± 0.001 | 0.023 ± 0.008 | 0.749 ± 0.001 | 0.747 ± 0.001 |
| SmallCNN | 3 | 0.593 ± 0.034 | 0.603 ± 0.039 | 0.85 ± 0.06 | 0.61 ± 0.07 | 0.71 ± 0.09 | 0.57 ± 0.13 | 0.07 ± 0.06 | 0.50 ± 0.00 | 0.92 ± 0.14 | 0.936 ± 0.004 | 0.616 ± 0.022 | 1.796 ± 0.207 | 0.741 ± 0.009 | — |

PR-AUC per class (7 classes; the prevalence row is the PR-AUC of a chance model):

| Model | G1 | S | G2 | Prophase | Metaphase | Anaphase | Telophase |
|---|---|---|---|---|---|---|---|
| Prevalence (chance PR-AUC) | 0.444 | 0.267 | 0.267 | 0.019 | 0.002 | 0.000 | 0.001 |
| Random forest | 0.933 ± 0.000 | 0.596 ± 0.001 | 0.792 ± 0.001 | 0.193 ± 0.002 | 0.039 ± 0.005 | 0.500 ± 0.000 | 1.000 ± 0.000 |
| SmallCNN | 0.926 ± 0.005 | 0.615 ± 0.026 | 0.812 ± 0.016 | 0.359 ± 0.062 | 0.100 ± 0.062 | 0.502 ± 0.000 | 1.000 ± 0.000 |

Random forest mtry: max_features 9 of 28 features per split (sklearn default sqrt would be 5.3).
