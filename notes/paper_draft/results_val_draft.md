# Results (DRAFT, run 2)

> **Validation results — not the final test numbers.** Every number below is a validation-set number from `results/experiments.csv` (tables: `results/table_val_summary_run2.md`, `results/run2_phaseA_val.md`, `results/table_run1_vs_run2_val.md`). The test set has not been touched. Claude rewrote this overnight on 2026-10-31/11-01 with the run-2 CNN settings (vflip augmentation, weighted loss, widened learning-rate grid, up to 25 epochs with early stopping); check every sentence. Run-1 numbers are only in the note at the end.

All models were selected and early-stopped on the validation set, so these numbers are optimistic estimates. Mean ± SD over seeds 0, 1 and 2 unless stated otherwise. The validation set has 4,840 cells: 2,150 G1, 1,293 S, 1,290 G2 and 107 mitotic (91 prophase, 10 metaphase, 2 anaphase, 4 telophase).

## RQ1 — CNNs vs hand-crafted features + random forest

[PENDING: ResNet18 finals; fill in from `results/table_val_summary_run2.md`.]

On the 4-class task, the random forest on 28 hand-crafted features reached a validation macro-F1 of 0.632 ± 0.001. SmallCNN reached 0.744 ± 0.010. For reference, always predicting G1 gives 0.154 and random guessing in class proportions gives 0.257 ± 0.006.

Most of the gain is in mitosis. Mitosis recall was 0.26 ± 0.01 for the forest and 0.63 ± 0.04 for SmallCNN. G1, S and G2 recall rose less: from 0.84 / 0.64 / 0.77 for the forest to 0.86 / 0.69 / 0.80 for SmallCNN.

The forest relied mostly on cell size. Shuffling `area` lowered the validation macro-F1 by 0.113 ± 0.014, and no other feature lowered it by more than 0.04 (Fig. rf_importance). Next came brightfield texture (`Ch3_glcm_correlation`, 0.037) and the cell's axis lengths (0.026 and 0.021).

## RQ2 — ImageNet transfer vs training from scratch

[PENDING: ResNet18 pretrained vs scratch finals.]

## RQ3 — Augmentation and class balancing

SmallCNN was trained with three augmentation options (none; vflip = up-down flips + intensity jitter; basic = 90° rotations + both flips + jitter) and two balancing options (weighted loss; oversampling), three seeds each, with fixed hyperparameters (lr 2e-3, weight decay 1e-4, batch 64; up to 25 epochs with early stopping; Fig. phaseA_val_run2; `results/run2_phaseA_val.md`). Validation macro-F1 and mitosis recall:

| Augmentation | Balancing | Macro-F1 (val) | Mitosis recall (val) |
|---|---|---|---|
| vflip | weighted | 0.743 ± 0.013 | 0.66 ± 0.01 |
| vflip | oversampling | 0.741 ± 0.010 | 0.57 ± 0.10 |
| none | weighted | 0.737 ± 0.015 | 0.59 ± 0.05 |
| basic | weighted | 0.723 ± 0.030 | 0.64 ± 0.12 |
| none | oversampling | 0.716 ± 0.008 | 0.60 ± 0.08 |
| basic | oversampling | 0.715 ± 0.053 | 0.54 ± 0.14 |

All six settings lie within 0.03 of each other, and single seeds of the same setting differed by up to 0.11 (basic with oversampling: 0.659–0.765). Up-down flips with weighted loss had the highest mean and by far the most stable mitosis recall (SD 0.01). It was chosen, by a rule fixed before the runs, for all further CNN runs. Full rotation/flip augmentation ("basic") was on average slightly worse and much less stable across seeds. Oversampling runs reached their best epoch early (epochs 2–19) and were then stopped by early stopping, whereas weighted-loss runs mostly improved until the last epochs.

*Why direction matters (`scripts/orientation_check.py`).* In the darkfield channel, the left half of the image is brighter than the right half in 80.9% of training images, against 44.7% for brightfield. We attribute this to the side-scatter illumination coming from one side of the instrument [CHECK: confirm with the ImageStream description]. To test whether models use this direction, the seed-0 models were evaluated on flipped and rotated validation images (`results/orientation_check_val_run2.csv`):

| Model (seed 0) | Original | Up-down flip | Left-right flip | Rotated 180° |
|---|---|---|---|---|
| no augmentation, weighted | 0.748 | 0.713 | 0.571 | 0.544 |
| vflip, weighted (chosen) | 0.757 | 0.743 | 0.621 | 0.629 |
| basic, weighted | 0.689 | 0.692 | 0.698 | 0.696 |

The vflip model is nearly unaffected by up-down flips but loses 0.14 on left-right flips. Its mitosis recall falls from 0.67 to 0.19, so it uses the left/right direction, above all for mitosis. The basic model scores the same on every transformed image. Because the direction is the same for all images of the instrument, it is information the model may legitimately use, not leakage. It ties the model to this instrument's geometry, though.

*Take-home (Fig. phaseA_val_run2):* the choice of augmentation and balancing changes validation macro-F1 by at most about 0.03 on average, less than the seed-to-seed spread of the less stable settings. Keeping the darkfield direction (no left-right flips or rotations) with weighted loss gave the best and most stable results.

## RQ4 — Is the continuous cell cycle visible in the learned features?

[PENDING: UMAP and pseudotime with the run-2 models (`figures/umap_val_run2`, `results/pseudotime_val_run2.csv`).]

## RQ5 — Does the network look at the cell?

*SmallCNN (seed 0; `python -m scripts.fig_gradcam --model smallcnn --tag n2_final --suffix _run2`; `results/gradcam_val_summary_smallcnn_run2.csv`).* Grad-CAM maps for the true 4-class label were computed for all validation cells. Median share of the heat per phase:

| Phase | Cells | Heat in mask | Heat in 8-px border ring |
|---|---|---|---|
| G1 | 2,150 | 0.51 | 0.09 |
| S | 1,293 | 0.43 | 0.16 |
| G2 | 1,290 | 0.54 | 0.12 |
| Prophase | 91 | 0.57 | 0.05 |

The mask covers 0.11–0.17 of the image, the ring 0.43. For every phase the heat is concentrated on the cell, at three to five times the mask's share of the image, and the border ring gets far less than its area share. In S-phase cells the heat often lies on the right half of the cell.

[PENDING: best CNN Grad-CAM; border swap test on run-2 models.]

For the forest, adding the raw darkfield intensity levels, which track the background, raised the 4-class macro-F1 from 0.632 ± 0.001 to 0.647 ± 0.003. That gain is small but consistent over seeds.

---

## Note: run 1 → run 2

[PENDING: complete after Phase C from `results/table_run1_vs_run2_val.md`.]
