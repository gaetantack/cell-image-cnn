# RQ3 ablation — PRELIMINARY (seed 0 only), SmallCNN, 4 classes, validation

| Augment | Balance | Macro-F1 (val) | Balanced acc. (val) | Recall mitosis | Recall G1 | Recall S | Recall G2 |
|---|---|---|---|---|---|---|---|
| none | none | 0.742 | 0.725 | 0.57 | 0.89 | 0.61 | 0.83 |
| none | weighted | 0.734 | 0.742 | 0.64 | 0.85 | 0.66 | 0.81 |
| none | oversample | 0.738 | 0.718 | 0.53 | 0.85 | 0.67 | 0.82 |
| basic | none | 0.682 | 0.656 | 0.30 | 0.90 | 0.61 | 0.82 |
| basic | weighted | 0.690 | 0.707 | 0.56 | 0.78 | 0.72 | 0.77 |
| basic | oversample | 0.746 | 0.758 | 0.69 | 0.85 | 0.68 | 0.81 |
