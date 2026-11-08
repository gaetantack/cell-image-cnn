# Masking check, inference only (val, 4 classes)

Inference only: run-2 final models (trained on original images) scored on masked val images. Mask = brightfield cell mask dilated by 3 px; fill per channel = median training background (BF 94, DF 39). 1 val cell has an empty mask. Chance rows: run-1 dummy baselines. 'Mask area only' = logistic regression on log mask area (1 fit, deterministic). VALIDATION numbers. All 9 checkpoints present.

| Model | Input | n seeds | Macro-F1 | Bal. acc. | Recall G1 | Recall S | Recall G2 | Recall Mitosis |
|---|---|---|---|---|---|---|---|---|
| SmallCNN | original | 3 | 0.744 ± 0.010 | 0.744 ± 0.012 | 0.857 ± 0.015 | 0.689 ± 0.016 | 0.797 ± 0.023 | 0.632 ± 0.039 |
| SmallCNN | cell_only | 3 | 0.420 ± 0.027 | 0.521 ± 0.018 | 0.363 ± 0.036 | 0.287 ± 0.046 | 0.869 ± 0.007 | 0.567 ± 0.051 |
| SmallCNN | background_only | 3 | 0.160 ± 0.003 | 0.249 ± 0.001 | 0.979 ± 0.003 | 0.008 ± 0.002 | 0.005 ± 0.002 | 0.003 ± 0.005 |
| ResNet18 (ImageNet) | original | 3 | 0.781 ± 0.007 | 0.777 ± 0.009 | 0.843 ± 0.030 | 0.708 ± 0.040 | 0.804 ± 0.012 | 0.754 ± 0.030 |
| ResNet18 (ImageNet) | cell_only | 3 | 0.156 ± 0.031 | 0.323 ± 0.015 | 0.041 ± 0.043 | 0.026 ± 0.021 | 0.673 ± 0.033 | 0.551 ± 0.025 |
| ResNet18 (ImageNet) | background_only | 3 | 0.258 ± 0.026 | 0.295 ± 0.009 | 0.638 ± 0.056 | 0.321 ± 0.118 | 0.080 ± 0.101 | 0.143 ± 0.075 |
| ResNet18 (scratch) | original | 3 | 0.745 ± 0.003 | 0.740 ± 0.004 | 0.843 ± 0.003 | 0.699 ± 0.011 | 0.819 ± 0.010 | 0.598 ± 0.016 |
| ResNet18 (scratch) | cell_only | 3 | 0.456 ± 0.045 | 0.514 ± 0.024 | 0.292 ± 0.081 | 0.357 ± 0.015 | 0.946 ± 0.012 | 0.461 ± 0.014 |
| ResNet18 (scratch) | background_only | 3 | 0.221 ± 0.031 | 0.256 ± 0.013 | 0.787 ± 0.149 | 0.149 ± 0.082 | 0.085 ± 0.076 | 0.003 ± 0.005 |
| logreg_mask_area | mask area only | 1 | 0.483 | 0.563 | 0.758 | 0.517 | 0.389 | 0.589 |
| Chance (majority) | - | 3 | 0.154 ± 0.000 | 0.250 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Chance (stratified) | - | 3 | 0.257 ± 0.006 | 0.257 ± 0.006 | 0.440 ± 0.008 | 0.277 ± 0.006 | 0.280 ± 0.014 | 0.031 ± 0.014 |
