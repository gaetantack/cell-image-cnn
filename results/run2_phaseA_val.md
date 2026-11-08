# Run 2, Phase A: augmentation × balancing, SmallCNN, 4 classes (VALIDATION)

Run-1 best SmallCNN config (lr 2e-3, wd 1e-4, batch 64), max 25 epochs, early stopping (patience 6) on val macro-F1. Mean ± SD over the seeds listed. Best epoch is 0-based. Made by `scripts/choose_aug_bal.py`.

| Augment | Balance | Seeds | Macro-F1 (val) | Mitosis recall (val) | Macro-F1 per seed | Best epoch per seed | Epochs run per seed | Train time (s) |
|---|---|---|---|---|---|---|---|---|
| vflip | weighted | 3 | 0.743 ± 0.013 | 0.66 ± 0.01 | 0.757 / 0.741 / 0.730 | 22 / 24 / 23 | 25 / 25 / 25 | 448 |
| vflip | oversample | 3 | 0.741 ± 0.010 | 0.57 ± 0.10 | 0.735 / 0.735 / 0.752 | 12 / 8 / 19 | 19 / 15 / 25 | 333 |
| none | weighted | 3 | 0.737 ± 0.015 | 0.59 ± 0.05 | 0.748 / 0.744 / 0.720 | 23 / 22 / 5 | 25 / 25 / 12 | 336 |
| basic | weighted | 3 | 0.723 ± 0.030 | 0.64 ± 0.12 | 0.689 / 0.733 / 0.746 | 9 / 18 / 18 | 16 / 25 / 25 | 450 |
| none | oversample | 3 | 0.716 ± 0.008 | 0.60 ± 0.08 | 0.717 / 0.708 / 0.724 | 7 / 8 / 10 | 14 / 15 / 17 | 248 |
| basic | oversample | 3 | 0.715 ± 0.053 | 0.54 ± 0.14 | 0.721 / 0.765 / 0.659 | 4 / 17 / 2 | 11 / 24 / 9 | 275 |

**Chosen: augment = vflip, balance = weighted.** best two vflip/weighted and vflip/oversample differ by 0.002 <= one seed SD 0.013; mitosis recall also within one SD (0.090 <= 0.101); tie-break prefers vflip/weighted (vflip > none > basic, then weighted > oversample).
