# Biological check (val): size, shape, recall and pseudotime per phase

Description only. Area/eccentricity: brightfield mask (hand-crafted feature table; the empty-mask cell has no value). Recall: run-2 finals, mean over seeds 0–2. Pseudotime: PC1 of the standardised space (fitted on train), median percentile rank of the phase's val cells (0 = earliest, 1 = latest). Indented rows = mitotic sub-phases (merged into Mitosis for the 4-class models). Made by `scripts/bio_check.py`.

| Phase | n val | Median area (px) | Median eccentricity | Recall SmallCNN | Recall ResNet18 (ImageNet) | PC1 pseudotime, hand-crafted (median pct.) | PC1 pseudotime, ResNet18 emb. (median pct.) |
|---|---|---|---|---|---|---|---|
| G1 | 2150 | 480 | 0.28 | 0.86 | 0.84 | 0.34 | 0.23 |
| S | 1293 | 570 | 0.28 | 0.69 | 0.71 | 0.55 | 0.57 |
| G2 | 1290 | 670 | 0.28 | 0.80 | 0.80 | 0.75 | 0.86 |
| Mitosis | 107 | 727 | 0.29 | 0.63 | 0.75 | 0.81 | 0.74 |
|   Prophase | 91 | 731 | 0.28 |  |  | 0.81 | 0.74 |
|   Metaphase | 10 | 676 | 0.30 |  |  | 0.80 | 0.78 |
|   Anaphase | 2 | 698 | 0.49 |  |  | 0.65 | 0.72 |
|   Telophase | 4 | 812 | 0.82 |  |  | 0.81 | 0.59 |
