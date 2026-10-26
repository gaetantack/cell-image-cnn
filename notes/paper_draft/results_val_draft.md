# Results (DRAFT)

> **Validation results — not the final test numbers.** Every number below is a validation-set number from `results/experiments.csv` (tables: `results/table_val_summary.md`). The test set has not been touched. Claude wrote this overnight on 2026-10-25/26; check every sentence. Places still waiting for a run are marked [PENDING: ...].

All models were selected and early-stopped on the validation set, so these numbers are optimistic estimates. Mean ± SD over seeds 0, 1 and 2 unless stated otherwise. The validation set has 4,840 cells: 2,150 G1, 1,293 S, 1,290 G2 and 107 mitotic (91 prophase, 10 metaphase, 2 anaphase, 4 telophase).

## RQ1 — CNNs vs hand-crafted features + random forest

On the 4-class task, the random forest on 28 hand-crafted features reached a validation macro-F1 of 0.632 ± 0.001. SmallCNN reached 0.687 ± 0.006 and the ImageNet-pretrained ResNet18 reached 0.761 ± 0.006 (Fig. model_comparison_val). For reference, always predicting G1 gives 0.154 and random guessing in class proportions gives 0.257 ± 0.006.

Most of the gain is in mitosis. Mitosis recall was 0.26 ± 0.01 for the forest, 0.65 ± 0.10 for SmallCNN and 0.83 ± 0.01 for ResNet18. G1, S and G2 recall rose only a little: from 0.84 / 0.64 / 0.77 for the forest to 0.85 / 0.72 / 0.79 for ResNet18 (Fig. confusion_val_4class). In both models most errors are G1↔S and S↔G2 confusions. Missed mitotic cells are almost always called G2: 76 of 107 for the forest and 19 of 107 for ResNet18 (seed 0).

The forest relied mostly on cell size. Shuffling `area` lowered the validation macro-F1 by 0.113 ± 0.014, and no other feature lowered it by more than 0.04 (Fig. rf_importance). Next came brightfield texture (`Ch3_glcm_correlation`, 0.037) and the cell's axis lengths (0.026 and 0.021).

On the 7-class task the forest reached 0.559 ± 0.004. Recall on single mitotic phases rests on very few validation cells (2 anaphase, 4 telophase), so it can only change in steps of 0.5 or 0.25 and should not be interpreted per phase. [PENDING: 7-class CNN finals.]

*Take-home (Fig. model_comparison_val):* on validation, both CNNs beat the hand-crafted features, mainly because they find far more mitotic cells.

## RQ2 — ImageNet transfer vs training from scratch; learning curve

With the same tuning grid and epoch caps, ResNet18 trained from scratch reached at most 0.649 in tuning (4 epochs, seed 0). Its pretrained version reached 0.722 in the same setting. [PENDING: scratch finals, 3 seeds.]

[PENDING: learning curve at 1 / 5 / 20 / 100% of the training labels, SmallCNN (Fig. learning_curve_val).]

*Caveat:* the epoch cap (12 epochs for final runs) favours models that start closer to a good solution. With longer training the scratch model may close part of the gap.

## RQ3 — Augmentation and class balancing (preliminary, seed 0)

[PENDING: six SmallCNN runs (augmentation none/basic × balancing none/weighted/oversample), seed 0 only (Fig. ablation_val). Report macro-F1 and mitosis recall; with one seed, differences below about 0.01–0.02 macro-F1 are within seed noise (SmallCNN SD 0.006, mitosis recall SD 0.10).]

## RQ4 — Is the continuous cell cycle visible in the learned features?

In a UMAP of the SmallCNN embedding of the validation cells, the cells lie along an arc in the order G1 → S → G2 → mitosis (Fig. umap_val). The network was only trained on the 4 classes, not on the 7 phases. In the hand-crafted feature space, the phases overlap more.

The first principal component (PC1, fitted on training cells) served as a simple pseudotime. Its Spearman correlation with phase order (G1 = 0 … telophase = 6) on validation cells was 0.45 for the hand-crafted features and 0.83 for the SmallCNN embedding (Fig. pseudotime_val; one run, seed 0). Part of this is expected by construction: the network was trained to separate G1, S, G2 and mitosis. [PENDING: same for the best ResNet18.]

*Take-home (Fig. umap_val):* the CNN embedding orders cells along the cell cycle, though it was only trained on discrete classes.

## RQ5 — Does the network look at the cell?

The SmallCNN numbers below come from the seed-0 final run. Grad-CAM maps for the true 4-class label were computed for all validation cells (Fig. gradcam_val; results/gradcam_val_summary.csv). For each phase, the median share of the heat inside the brightfield cell mask was:

| Phase | Cells | Heat in mask | Mask area | Heat in border ring | Ring area |
|---|---|---|---|---|---|
| G1 | 2,150 | 0.38 | 0.11 | 0.25 | 0.43 |
| S | 1,293 | 0.18 | 0.13 | 0.44 | 0.43 |
| G2 | 1,290 | 0.49 | 0.15 | 0.21 | 0.43 |
| Prophase | 91 | 0.64 | 0.17 | 0.04 | 0.43 |

For G1, G2 and prophase the heat is concentrated on the cell, at 3–4 times the mask's share of the image. For S phase the heat is not on the cell. It lies in the 8-pixel border ring at about the ring's share of the area (0.44 vs 0.43), often as a frame along the image edge. This could be a shortcut (the background differs by phase) or an artefact of zero padding in the convolutions; we cannot yet tell them apart. In training cell 30439 (corner artefact), the heat does not fall on the artefact (Fig. gradcam_cell30439_train).

For the forest, adding the raw darkfield intensity levels, which track the background, raised the 4-class macro-F1 from 0.632 ± 0.001 to 0.647 ± 0.003. That gain is small but consistent over seeds.

[PENDING: Grad-CAM for the best model (probably ResNet18-pretrained); its maps are 4 × 4 before upsampling, so they are coarser.]

*Take-home (Fig. gradcam_val):* for most phases the network looks at the cell; for S phase it does not, and this needs a follow-up check.
