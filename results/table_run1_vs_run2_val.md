# Run 1 vs run 2, CNN finals, 4 classes (VALIDATION, mean ± SD over seeds)

Made by `scripts/compare_runs.py`. Best epoch is 0-based; "epochs run" = max epochs means the run never stopped early.

| Family | Run | Seeds | Configuration | Macro-F1 (val) | Mitosis recall (val) | Best epoch per seed | Epochs run per seed | Train time (s) |
|---|---|---|---|---|---|---|---|---|
| SmallCNN | run 1 | 3 | lr 0.002, wd 0.0001, bs 64, aug basic, bal weighted, max 12 ep | 0.687 ± 0.006 | 0.65 ± 0.10 | 7 / 11 / 7 | 12 / 12 / 12 | 183 |
| SmallCNN | run 2 | 3 | lr 0.001, wd 0.0001, bs 64, aug vflip, bal weighted, max 25 ep | 0.744 ± 0.010 | 0.63 ± 0.04 | 19 / 21 / 14 | 25 / 25 / 21 | 352 |
| | change | | | +0.057 | | | | |
| ResNet18 (ImageNet) | run 1 | 3 | lr 0.002, wd 0.0001, bs 64, aug basic, bal weighted, max 12 ep | 0.761 ± 0.006 | 0.83 ± 0.01 | 7 / 11 / 8 | 12 / 12 / 12 | 928 |
| ResNet18 (ImageNet) | run 2 | 0 | not run | | | | | |
| ResNet18 (scratch) | run 1 | 3 | lr 0.002, wd 0.0001, bs 64, aug basic, bal weighted, max 12 ep | 0.696 ± 0.012 | 0.64 ± 0.11 | 10 / 11 / 6 | 12 / 12 / 12 | 900 |
| ResNet18 (scratch) | run 2 | 0 | not run | | | | | |

Random forest (unchanged, run 1): 0.632 ± 0.001 val macro-F1.
