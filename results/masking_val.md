# Masking experiment (RQ5), validation, 4 classes

Mask = brightfield cell mask (`src.features.cell_mask`) dilated by 3 px; fill = median training background per channel (BF 94, DF 39). *Trained on masked*: train and val images masked the same way, run-2 settings (vflip + weighted, lr 1e-3, wd 1e-4, batch 64, ≤ 25 epochs, patience 6), no re-tuning. *Evaluated on masked*: run-2 finals trained on original images, only the val images masked. *Fixed disc*: central disc of radius 22 px plus the cell mask filled, so the hole no longer shows the cell size (control added in run 4). Mean ± SD over seeds; n seeds per row. Made by `scripts/masking_summary.py`.

| Kind | Model | Input | n seeds | Macro-F1 (val) | Bal. acc. (val) | Recall G1 | Recall S | Recall G2 | Recall Mitosis |
|---|---|---|---|---|---|---|---|---|---|
| original | SmallCNN | original | 3 | 0.744 ± 0.010 | 0.744 ± 0.012 | 0.857 ± 0.015 | 0.689 ± 0.016 | 0.797 ± 0.023 | 0.632 ± 0.039 |
| trained on masked | SmallCNN | cell only | 3 | 0.729 ± 0.019 | 0.713 ± 0.025 | 0.852 ± 0.008 | 0.688 ± 0.013 | 0.771 ± 0.047 | 0.539 ± 0.076 |
| trained on masked | SmallCNN | background only | 3 | 0.558 ± 0.028 | 0.601 ± 0.037 | 0.826 ± 0.019 | 0.545 ± 0.042 | 0.584 ± 0.174 | 0.449 ± 0.219 |
| trained on masked | SmallCNN | background only, fixed disc | 3 | 0.450 ± 0.007 | 0.492 ± 0.019 | 0.727 ± 0.060 | 0.408 ± 0.060 | 0.441 ± 0.017 | 0.393 ± 0.065 |
| original | ResNet18 (ImageNet) | original | 3 | 0.781 ± 0.007 | 0.777 ± 0.009 | 0.843 ± 0.030 | 0.708 ± 0.040 | 0.804 ± 0.012 | 0.754 ± 0.030 |
| trained on masked | ResNet18 (ImageNet) | cell only | 3 | 0.769 ± 0.004 | 0.754 ± 0.005 | 0.864 ± 0.016 | 0.673 ± 0.022 | 0.807 ± 0.032 | 0.673 ± 0.019 |
| trained on masked | ResNet18 (ImageNet) | background only | 3 | 0.601 ± 0.007 | 0.621 ± 0.028 | 0.777 ± 0.036 | 0.615 ± 0.056 | 0.714 ± 0.029 | 0.380 ± 0.144 |
| trained on masked | ResNet18 (ImageNet) | background only, fixed disc | 3 | 0.468 ± 0.003 | 0.478 ± 0.006 | 0.723 ± 0.023 | 0.422 ± 0.019 | 0.577 ± 0.056 | 0.190 ± 0.011 |
| evaluated on masked | SmallCNN | cell only | 3 | 0.420 ± 0.027 | 0.521 ± 0.018 | 0.363 ± 0.036 | 0.287 ± 0.046 | 0.869 ± 0.007 | 0.567 ± 0.051 |
| evaluated on masked | SmallCNN | background only | 3 | 0.160 ± 0.003 | 0.249 ± 0.001 | 0.979 ± 0.003 | 0.008 ± 0.002 | 0.005 ± 0.002 | 0.003 ± 0.005 |
| evaluated on masked | ResNet18 (ImageNet) | cell only | 3 | 0.156 ± 0.031 | 0.323 ± 0.015 | 0.041 ± 0.043 | 0.026 ± 0.021 | 0.673 ± 0.033 | 0.551 ± 0.025 |
| evaluated on masked | ResNet18 (ImageNet) | background only | 3 | 0.258 ± 0.026 | 0.295 ± 0.009 | 0.638 ± 0.056 | 0.321 ± 0.118 | 0.080 ± 0.101 | 0.143 ± 0.075 |
| evaluated on masked | ResNet18 (scratch) | cell only | 3 | 0.456 ± 0.045 | 0.514 ± 0.024 | 0.292 ± 0.081 | 0.357 ± 0.015 | 0.946 ± 0.012 | 0.461 ± 0.014 |
| evaluated on masked | ResNet18 (scratch) | background only | 3 | 0.221 ± 0.031 | 0.256 ± 0.013 | 0.787 ± 0.149 | 0.149 ± 0.082 | 0.085 ± 0.076 | 0.003 ± 0.005 |
| reference | Logistic regression | log mask area only | 1 | 0.483 | 0.563 | 0.758 | 0.517 | 0.389 | 0.589 |
| control | SmallCNN | original images, shuffled training labels | 3 | 0.282 ± 0.020 | 0.324 ± 0.041 | 0.541 ± 0.276 | 0.297 ± 0.263 | 0.359 ± 0.264 | 0.097 ± 0.167 |
| chance | Chance (majority) | — | 3 | 0.154 ± 0.000 | 0.250 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| chance | Chance (stratified) | — | 3 | 0.257 ± 0.006 | 0.257 ± 0.006 | 0.440 ± 0.008 | 0.277 ± 0.006 | 0.280 ± 0.014 | 0.031 ± 0.014 |
