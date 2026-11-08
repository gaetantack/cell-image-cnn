# TEST summary (mean ± SD over seeds 0, 1, 2), made by scripts/final_test.py --split test

Frozen models: CNN run-2 finals (`n2_final`); random forest re-fitted on train with the run-1 best configuration; chance = DummyClassifier. ROC-AUC/PR-AUC: one-vs-rest from class probabilities (chance: training class prior → ROC-AUC 0.5, PR-AUC = prevalence). Inference: CPU, 4 threads, s per 1,000 cells. OOB = out-of-bag accuracy of the random forest (eval accuracy next to it).

## 4 classes

| Model | Seeds | Macro-F1 (test) | Balanced acc. (test) | Recall G1 | Recall S | Recall G2 | Recall Mitosis | ROC-AUC macro | PR-AUC macro | Inference (s/1,000 cells) | Accuracy | OOB accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 3 | 0.154 ± 0.000 | 0.250 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.500 ± 0.000 | 0.250 ± 0.000 | 0.000 ± 0.000 | 0.444 ± 0.000 | — |
| Chance (stratified) | 3 | 0.257 ± 0.006 | 0.257 ± 0.006 | 0.44 ± 0.01 | 0.28 ± 0.01 | 0.28 ± 0.01 | 0.03 ± 0.01 | 0.500 ± 0.000 | 0.250 ± 0.000 | 0.000 ± 0.000 | 0.344 ± 0.007 | — |
| Random forest | 3 | 0.659 ± 0.004 | 0.644 ± 0.003 | 0.85 ± 0.00 | 0.65 ± 0.00 | 0.77 ± 0.00 | 0.31 ± 0.01 | 0.921 ± 0.000 | 0.692 ± 0.000 | 0.016 ± 0.000 | 0.762 ± 0.001 | 0.752 ± 0.000 |
| SmallCNN | 3 | 0.746 ± 0.003 | 0.742 ± 0.009 | 0.86 ± 0.02 | 0.71 ± 0.03 | 0.77 ± 0.02 | 0.64 ± 0.04 | 0.941 ± 0.001 | 0.799 ± 0.003 | 0.891 ± 0.028 | 0.788 ± 0.002 | — |
| ResNet18 (ImageNet) | 3 | 0.776 ± 0.004 | 0.778 ± 0.014 | 0.85 ± 0.03 | 0.72 ± 0.05 | 0.78 ± 0.01 | 0.77 ± 0.05 | 0.946 ± 0.005 | 0.840 ± 0.009 | 6.825 ± 0.180 | 0.792 ± 0.007 | — |
| ResNet18 (scratch) | 3 | 0.735 ± 0.004 | 0.731 ± 0.003 | 0.85 ± 0.01 | 0.68 ± 0.01 | 0.80 ± 0.01 | 0.60 ± 0.01 | 0.940 ± 0.001 | 0.793 ± 0.003 | 7.958 ± 1.219 | 0.784 ± 0.003 | — |
| Random forest + raw DF levels (RQ5 side experiment) | 3 | 0.664 ± 0.008 | 0.652 ± 0.007 | 0.85 ± 0.00 | 0.66 ± 0.00 | 0.77 ± 0.00 | 0.33 ± 0.02 | 0.923 ± 0.000 | 0.702 ± 0.001 | 0.018 ± 0.001 | 0.765 ± 0.003 | 0.756 ± 0.001 |

PR-AUC per class (4 classes; the prevalence row is the PR-AUC of a chance model):

| Model | G1 | S | G2 | Mitosis |
|---|---|---|---|---|
| Prevalence (chance PR-AUC) | 0.444 | 0.267 | 0.267 | 0.022 |
| Random forest | 0.941 ± 0.000 | 0.616 ± 0.000 | 0.828 ± 0.002 | 0.382 ± 0.003 |
| SmallCNN | 0.951 ± 0.001 | 0.658 ± 0.003 | 0.892 ± 0.007 | 0.695 ± 0.008 |
| ResNet18 (ImageNet) | 0.949 ± 0.006 | 0.661 ± 0.011 | 0.905 ± 0.012 | 0.845 ± 0.014 |
| ResNet18 (scratch) | 0.950 ± 0.001 | 0.661 ± 0.007 | 0.887 ± 0.002 | 0.674 ± 0.014 |
| Random forest + raw DF levels (RQ5 side experiment) | 0.943 ± 0.000 | 0.626 ± 0.001 | 0.834 ± 0.001 | 0.405 ± 0.005 |

## 7 classes

| Model | Seeds | Macro-F1 (test) | Balanced acc. (test) | Recall G1 | Recall S | Recall G2 | Recall Prophase | Recall Metaphase | Recall Anaphase | Recall Telophase | ROC-AUC macro | PR-AUC macro | Inference (s/1,000 cells) | Accuracy | OOB accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Chance (majority) | 3 | 0.088 ± 0.000 | 0.143 ± 0.000 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.500 ± 0.000 | 0.143 ± 0.000 | 0.000 ± 0.000 | 0.444 ± 0.000 | — |
| Chance (stratified) | 3 | 0.146 ± 0.003 | 0.146 ± 0.003 | 0.44 ± 0.01 | 0.27 ± 0.01 | 0.28 ± 0.02 | 0.03 ± 0.02 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.500 ± 0.000 | 0.143 ± 0.000 | 0.000 ± 0.000 | 0.342 ± 0.010 | — |
| Random forest | 3 | 0.501 ± 0.002 | 0.515 ± 0.001 | 0.84 ± 0.00 | 0.64 ± 0.00 | 0.76 ± 0.00 | 0.37 ± 0.01 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 0.901 ± 0.001 | 0.531 ± 0.002 | 0.018 ± 0.001 | 0.753 ± 0.001 | 0.747 ± 0.001 |
| SmallCNN | 3 | 0.514 ± 0.006 | 0.528 ± 0.007 | 0.85 ± 0.05 | 0.63 ± 0.07 | 0.71 ± 0.10 | 0.59 ± 0.19 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.92 ± 0.14 | 0.930 ± 0.006 | 0.550 ± 0.027 | 1.143 ± 0.078 | 0.748 ± 0.015 | — |

PR-AUC per class (7 classes; the prevalence row is the PR-AUC of a chance model):

| Model | G1 | S | G2 | Prophase | Metaphase | Anaphase | Telophase |
|---|---|---|---|---|---|---|---|
| Prevalence (chance PR-AUC) | 0.444 | 0.267 | 0.267 | 0.019 | 0.002 | 0.000 | 0.001 |
| Random forest | 0.940 ± 0.000 | 0.612 ± 0.001 | 0.803 ± 0.001 | 0.312 ± 0.010 | 0.021 ± 0.001 | 0.028 ± 0.002 | 1.000 ± 0.000 |
| SmallCNN | 0.936 ± 0.005 | 0.602 ± 0.027 | 0.818 ± 0.018 | 0.431 ± 0.089 | 0.054 ± 0.058 | 0.005 ± 0.002 | 1.000 ± 0.000 |

Random forest mtry: max_features 9 of 28 features per split (sklearn default sqrt would be 5.3).
