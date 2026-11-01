# Results (DRAFT, run 2)

> **Validation results — not the final test numbers.** Every number below is a validation-set number from `results/experiments.csv` (tables: `results/table_val_summary_run2.md`, `results/run2_phaseA_val.md`, `results/table_run1_vs_run2_val.md`). The test set has not been touched. Claude rewrote this overnight on 2026-10-31/11-01 with the run-2 CNN settings (vflip augmentation, weighted loss, widened learning-rate grid, up to 25 epochs with early stopping); check every sentence. Run-1 (pilot) numbers are only in the note at the end. Decisions confirmed by the user on 2026-11-01: run 2 is the paper result, run 1 the pilot; see PROJECT.md decisions log.

All models were selected and early-stopped on the validation set, so these numbers are optimistic estimates. Mean ± SD over seeds 0, 1 and 2 unless stated otherwise. The validation set has 4,840 cells: 2,150 G1, 1,293 S, 1,290 G2 and 107 mitotic (91 prophase, 10 metaphase, 2 anaphase, 4 telophase).

## RQ1 — CNNs vs hand-crafted features + random forest

On the 4-class task, the random forest on 28 hand-crafted features reached a validation macro-F1 of 0.632 ± 0.001. SmallCNN reached 0.744 ± 0.010, the ResNet18 trained from scratch 0.745 ± 0.003 and the ImageNet-pretrained ResNet18 0.781 ± 0.007 (Fig. model_comparison_val_run2; `results/table_val_summary_run2.md`). For reference, always predicting G1 gives 0.154 and random guessing in class proportions gives 0.257 ± 0.006.

Most of the gain is in mitosis. Mitosis recall was 0.26 ± 0.01 for the forest, 0.63 ± 0.04 for SmallCNN, 0.60 ± 0.02 for the scratch ResNet18 and 0.75 ± 0.03 for the pretrained ResNet18. The pretrained network was also the most precise when it predicted mitosis: 77–83 of its 91–111 mitosis predictions were correct (precision 0.74–0.85 over seeds), against 0.57–0.69 for the other two CNNs. G1, S and G2 recall rose less: from 0.84 / 0.64 / 0.77 for the forest to 0.84 / 0.71 / 0.80 for the pretrained ResNet18. Missed mitotic cells were almost always called G2, in both the forest and the CNNs: 71% of mitotic cells for the forest and 28% for the pretrained ResNet18 (seed 0; Fig. confusion_val_4class_run2). The remaining errors are mostly G1↔S and S↔G2.

The forest relied mostly on cell size. Shuffling `area` lowered the validation macro-F1 by 0.113 ± 0.014, and no other feature lowered it by more than 0.04 (Fig. rf_importance). Next came brightfield texture (`Ch3_glcm_correlation`, 0.037) and the cell's axis lengths (0.026 and 0.021).

*Take-home (Fig. model_comparison_val_run2):* on validation, all three CNNs beat the hand-crafted features by 0.11–0.15 macro-F1, mainly because they find two to three times more mitotic cells.

On the 7-class task (secondary result; 4-class hyperparameters reused), the forest reached 0.559 ± 0.004 and SmallCNN 0.593 ± 0.034. The 7-class task is reported for the forest and SmallCNN only; the ResNet18 7-class runs were dropped, because the 7-class score depends on 6 validation cells (2 anaphase, 4 telophase) and the compute went to the learning curve. Recall on single mitotic phases rests on very few validation cells (10 metaphase, 2 anaphase, 4 telophase), so it changes in steps of 0.1, 0.5 or 0.25 and should not be interpreted per phase. Prophase (91 cells) was recognised better by SmallCNN (recall 0.57 ± 0.13) than by the forest (0.27). Two of the three SmallCNN 7-class runs stopped early (best epochs 4 and 3 of 25), which explains part of the seed spread.

## RQ2 — ImageNet transfer vs training from scratch

With the same tuning grid, augmentation, balancing and training length (up to 25 epochs with early stopping), the pretrained ResNet18 reached 0.781 ± 0.007 and the same architecture trained from scratch 0.745 ± 0.003, a difference of 0.036, about five times the larger SD. The difference is larger for mitosis: recall 0.75 ± 0.03 vs 0.60 ± 0.02. The scratch network matched SmallCNN (0.744 ± 0.010) despite 29 times more parameters. All three scratch runs used the full 25 epochs, but their best epochs were 18–20, so they were not still improving at the end. Two of the three pretrained runs stopped early (best epochs 6 and 11). In tuning (4 epochs), all three families selected the same learning rate (1e-3) and weight decay (1e-4), inside the grid.

*Learning curve (currently seed 0 only; seeds 1–2 and the scratch ResNet18 subsets are running, `scripts/queue_run2_lc_finish.sh` — update numbers when done; Fig. learning_curve_val_run2; `results/learning_curve_val_run2.csv`).* Each subset run had as many gradient steps per epoch as a full run, up to 25 epochs with early stopping. Validation macro-F1 at 1 / 5 / 20 / 100% of the labelled training cells (100% = mean of the 3 final runs):
- SmallCNN: 0.503 / 0.598 / 0.664 / 0.744
- pretrained ResNet18: 0.540 / 0.639 / 0.721 / 0.781
- scratch ResNet18: only 100% (0.745); subsets running.

At every fraction the pretrained network was ahead of SmallCNN: by 0.04 at 1–5% and by 0.06 at 20% (0.721 vs 0.664). With 5% of the labels (1,129 cells) the pretrained ResNet18 reached the level of the random forest trained on all labels (0.632). These are single-seed points, and several small-subset runs stopped very early (SmallCNN 5%: best epoch 2), so differences of a few hundredths are within noise.

*Take-home:* on validation, ImageNet pretraining gives a modest but clear gain in macro-F1 and a larger gain in mitosis recall, with the same training budget.

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

In UMAPs of the CNN embeddings of the validation cells (seed-0 runs; Fig. umap_val_run2), the pretrained ResNet18 embedding forms an arc from G1 through S to G2, with the mitotic cells on a separate branch at the G2 end; the four telophase cells lie at its tip. The SmallCNN embedding shows a G1 → S → G2 gradient with the mitotic cells clustered at the G2 side. The networks were trained only on the 4 classes, not on the 7 phases. In the hand-crafted feature space, G1 and G2 lie at opposite sides, but S and mitotic cells are mixed in with them.

The first principal component (PC1, fitted on training cells) served as a simple pseudotime. Its Spearman correlation with phase order (G1 = 0 … telophase = 6) on validation cells was 0.45 for the hand-crafted features, 0.85 for the SmallCNN embedding and 0.85 for the pretrained ResNet18 embedding (Fig. pseudotime_val_run2; `results/pseudotime_val_run2.csv`). Part of this is expected by construction, because the networks were trained to separate G1, S, G2 and mitosis. Correlations with the 4-class order are identical to three decimals, since only 16 validation cells are in metaphase, anaphase or telophase.

*Take-home (Fig. umap_val_run2):* the CNN embeddings order cells along the cell cycle, though they were only trained on discrete classes.

## RQ5 — Does the network look at the cell?

*SmallCNN (seed 0; `python -m scripts.fig_gradcam --model smallcnn --tag n2_final --suffix _run2`; `results/gradcam_val_summary_smallcnn_run2.csv`).* Grad-CAM maps for the true 4-class label were computed for all validation cells. Median share of the heat per phase:

| Phase | Cells | Heat in mask | Heat in 8-px border ring |
|---|---|---|---|
| G1 | 2,150 | 0.51 | 0.09 |
| S | 1,293 | 0.43 | 0.16 |
| G2 | 1,290 | 0.54 | 0.12 |
| Prophase | 91 | 0.57 | 0.05 |

The mask covers 0.11–0.17 of the image, the ring 0.43. For every phase the heat is concentrated on the cell, at three to five times the mask's share of the image, and the border ring gets far less than its area share. In S-phase cells the heat often lies on the right half of the cell.

*Best model (pretrained ResNet18, seed 0; Fig. gradcam_val_run2; `results/gradcam_val_summary_run2.csv`).* The heat lies on the cell for every phase: the median share inside the mask is 0.34–0.44 against a mask area of 0.11–0.19, and the share in the border ring is 0.09–0.12 for G1/S/G2 and 0.14–0.16 for the mitotic phases, far below the ring's 0.43 of the area. For mitotic cells the heat covers the whole cell, and for telophase both daughter cells. For G1, S and G2, the heat often peaks at the same spot at the lower or lower-right edge of the cell, whatever the cell looks like. Because this model was trained with random up-down flips, a fixed lower spot cannot come from a learned up/down image direction. It may be a property of the very coarse 4 × 4 Grad-CAM grid of ResNet18 at 128 px input; we have not tested it.

*Border test (`scripts/border_check.py --tag n2_final`, `results/border_check_val_run2.csv`, seed-0 runs).* To test whether the networks use the background, we changed only the outer 8-pixel ring of the validation images and left the cell in the centre unchanged. On the unedited images the script reproduced the logged scores (0.755, 0.782 and 0.748 for SmallCNN, pretrained and scratch ResNet18). Every edit lowered the macro-F1:
- setting the ring to its own median: −0.008, −0.038 and −0.011;
- setting it to one common brightness: −0.055, −0.107 and −0.066;
- copying in the ring of another cell of the same class: −0.150, −0.086 and −0.071.

The last edit is the control. Copying a ring from a cell of a different class gave almost the same scores: +0.006, −0.010 and +0.002 relative to the same-class copy. The drops therefore come from the edit itself (an unnatural seam at the ring edge, or parts of the donor cell inside the ring), not from losing background information that belongs to the right class. Within the precision of this test, we find no evidence that the CNNs use a phase-specific background in the border ring. They are, however, sensitive to changes there, and predict mitosis more often on edited images. One seed, one random draw of donor cells.

*Take-home (Fig. gradcam_val_run2):* all three networks put most Grad-CAM heat on the cell in every phase, and swapping the border between classes does not change their scores, so we found no sign of a phase-specific background shortcut.

For the forest, adding the raw darkfield intensity levels, which track the background, raised the 4-class macro-F1 from 0.632 ± 0.001 to 0.647 ± 0.003. That gain is small but consistent over seeds.

---

## Limitations (to move to the Discussion)

- *Instrument geometry.* The models use the fixed left/right direction of the darkfield images (left-right flips cost the chosen SmallCNN 0.14 macro-F1 and most of its mitosis recall). This direction comes from the instrument and is the same for every cell, so it is not leakage, but the models may not transfer to another cytometer or another mounting of the optics. A direction-free model (rotations and both flips) scored about 0.02 lower and varied more between seeds.
- *Augmentation hardly matters.* All six augmentation × balancing settings were within 0.03 of each other; the chosen setting won a tie, not a clear contest.
- *Early stopping on a noisy curve.* Models were early-stopped on validation macro-F1 with patience 6. The curve is noisy, especially for mitosis (107 validation cells), and some runs kept a very early epoch (best epochs 2–5), most often in the small learning-curve subsets. A smoothed or fixed-epoch selection was not tried, because it would require re-running all models.
- *Validation used for selection.* All numbers in this draft are on the validation set that was also used for model selection and early stopping, so they are optimistic; the test set gives the final numbers.
- *Seven classes.* The 7-class score depends on 6 anaphase/telophase validation cells, and was only run for the forest and SmallCNN.

## Note: pilot run (run 1) → run 2

Run 1 was a pilot. Its main finding was the orientation problem: a SmallCNN without augmentation beat the one with rotations and flips (0.742 vs 0.690, seed 0) and lost much of its score on left-right-flipped images, which led to the darkfield-direction analysis and the vflip augmentation used in run 2.


Run 1 (2026-10-25/26) used rotation/flip augmentation ("basic"), weighted loss, a learning-rate grid {1e-4, 5e-4, 2e-3} × weight decay × batch size {64, 128}, and at most 12 epochs (patience 5). Run 2 changed augmentation to vflip, widened the learning-rate grid, fixed batch size 64, and allowed up to 25 epochs (patience 6). Validation macro-F1, 4 classes, mean ± SD over 3 seeds (`results/table_run1_vs_run2_val.md`):

| Model | Run 1 | Run 2 | Change | Mitosis recall run 1 → run 2 |
|---|---|---|---|---|
| SmallCNN | 0.687 ± 0.006 | 0.744 ± 0.010 | +0.057 | 0.65 → 0.63 |
| ResNet18 (ImageNet) | 0.761 ± 0.006 | 0.781 ± 0.007 | +0.021 | 0.83 → 0.75 |
| ResNet18 (scratch) | 0.696 ± 0.012 | 0.745 ± 0.003 | +0.050 | 0.64 → 0.60 |
| Random forest | 0.632 ± 0.001 | (unchanged) | | 0.26 |

Several things changed at once (augmentation, learning rate, number of epochs, patience), so the gains cannot be assigned to one cause. The Phase A runs (fixed lr 2e-3) suggest that the longer schedule contributes: basic + weighted gave 0.723 ± 0.030 with 25 epochs against 0.687 ± 0.006 with 12 epochs in run 1. The pretrained ResNet18 gained least in macro-F1 and lost mitosis recall, while becoming more precise for mitosis (seed 0: precision 0.63 → 0.85). The ranking of the models did not change: pretrained ResNet18 > scratch ResNet18 ≈ SmallCNN > random forest. The pretrained − scratch gap shrank from 0.065 to 0.036, as run 1 suspected (scratch was limited by the 12-epoch cap). Run-1 results not repeated in run 2: learning curve (1 / 5 / 20 / 100 %: SmallCNN 0.547 / 0.631 / 0.689 / 0.687, pretrained 0.552 / 0.660 / 0.739 / 0.761, scratch 0.539 / 0.609 / 0.678 / 0.696; seed 0 except SmallCNN), border swap test on the run-1 models (no evidence of a phase-specific background shortcut), and the run-1 SmallCNN S-phase border pattern in Grad-CAM, which is gone in run 2.
