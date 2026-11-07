# Results (DRAFT, run 2; test headline numbers added 2026-11-07)

> **Test and validation numbers, labelled.** The main RQ1/RQ2 numbers (4- and 7-class macro-F1, mitosis recall, mitosis PR-AUC, pretrained vs scratch) are **test** numbers from the single test run on 2026-11-07 (`scripts/final_test.py --split test`, tag `test-frozen`; `results/table_test_summary.md`, `notes/paper_draft/table2_test.md`, Fig. 2 = `figures/fig2_test`); they are labelled "test". Every other number (all analyses, learning curve, RQ3–RQ5) is a **validation** number from `results/experiments.csv` (tables: `results/table_val_summary_run2.md`, `results/run2_phaseA_val.md`, `results/table_run1_vs_run2_val.md`). Before 2026-11-07 the test set had not been touched; nothing was changed after the test run. Claude rewrote this overnight on 2026-10-31/11-01 with the run-2 CNN settings (vflip augmentation, weighted loss, widened learning-rate grid, up to 25 epochs with early stopping); check every sentence. Run-1 (pilot) numbers are only in the note at the end. Decisions confirmed by the user on 2026-11-01: run 2 is the paper result, run 1 the pilot; see PROJECT.md decisions log.

All models were selected and early-stopped on the validation set, so validation numbers are optimistic estimates; the test set was evaluated once, with the configuration frozen beforehand (PROJECT.md §7a). Mean ± SD over seeds 0, 1 and 2 unless stated otherwise; the SD reflects training seeds only, not the sampling of cells into a split. The validation and test sets each have 4,840 cells with the same class counts: 2,150 G1, 1,293 S, 1,290 G2 and 107 mitotic (91 prophase, 10 metaphase, 2 anaphase, 4 telophase).

## RQ1 — CNNs vs hand-crafted features + random forest

On the 4-class **test** set, the random forest on 28 hand-crafted features reached a macro-F1 of 0.659 ± 0.004. SmallCNN reached 0.746 ± 0.003, the ResNet18 trained from scratch 0.735 ± 0.004 and the ImageNet-pretrained ResNet18 0.776 ± 0.004 (Table 2, Fig. 2; `results/table_test_summary.md`). Always predicting G1 gives 0.154 and random guessing in class proportions 0.257 ± 0.006 (test). Mitosis recall (test) was 0.31 ± 0.01 for the forest, 0.64 ± 0.04 for SmallCNN, 0.60 ± 0.01 for the scratch ResNet18 and 0.77 ± 0.05 for the pretrained ResNet18; mitosis PR-AUC (test; chance = prevalence 0.022) was 0.382 ± 0.003, 0.695 ± 0.008, 0.674 ± 0.014 and 0.845 ± 0.014.

*Test vs validation.* The validation macro-F1 values were 0.632 ± 0.001 (forest), 0.744 ± 0.010 (SmallCNN), 0.745 ± 0.003 (scratch) and 0.781 ± 0.007 (pretrained) (`results/table_val_summary_run2.md`). SmallCNN (+0.003) and the pretrained ResNet18 (−0.005) differ by less than one seed SD. Two models differ by more than 2 seed SDs: the forest scored 0.027 higher on test than on validation (mitosis recall 0.31 vs 0.26), and the scratch ResNet18 0.010 lower (2.8 times the larger SD). The ranking of the four models is the same on test and validation except that SmallCNN (0.746) is now ahead of the scratch ResNet18 (0.735).

Most of the gain is in mitosis. On validation, mitosis recall was 0.26 ± 0.01 for the forest, 0.63 ± 0.04 for SmallCNN, 0.60 ± 0.02 for the scratch ResNet18 and 0.75 ± 0.03 for the pretrained ResNet18. The pretrained network was also the most precise when it predicted mitosis: 77–83 of its 91–111 mitosis predictions were correct (precision 0.74–0.85 over seeds), against 0.57–0.69 for the other two CNNs. G1, S and G2 recall rose less: from 0.84 / 0.64 / 0.77 for the forest to 0.84 / 0.71 / 0.80 for the pretrained ResNet18. Missed mitotic cells were almost always called G2, in both the forest and the CNNs: 71% of mitotic cells for the forest and 28% for the pretrained ResNet18 (seed 0; Fig. confusion_val_4class_run2). The remaining errors are mostly G1↔S and S↔G2. On test (seed 0; Fig. 2C–D), missed mitotic cells were again mostly called G2: 66% for the forest and 25% for the pretrained ResNet18.

The forest relied mostly on cell size. Shuffling `area` lowered the validation macro-F1 by 0.113 ± 0.014, and no other feature lowered it by more than 0.04 (Fig. rf_importance). Next came brightfield texture (`Ch3_glcm_correlation`, 0.037) and the cell's axis lengths (0.026 and 0.021).

*Take-home (Table 2, Fig. 2):* on test, all three CNNs beat the hand-crafted features by 0.08–0.12 macro-F1 (validation: 0.11–0.15), mainly because they find about two to two and a half times more mitotic cells (test).

On the 7-class task (secondary result; 4-class hyperparameters reused), the forest reached 0.501 ± 0.002 and SmallCNN 0.514 ± 0.006 on **test** (validation 0.559 ± 0.004 and 0.593 ± 0.034). Both are lower on test than on validation by more than 2 seed SDs (−0.058 and −0.079). On test, neither model recognised any metaphase (10 cells) or anaphase (2 cells) cell (recall 0.00 for every seed), whereas on validation the forest found 1 of 10 metaphase and 1 of 2 anaphase cells (recall 0.10 and 0.50) and SmallCNN 0.07 ± 0.06 and 0.50. Telophase recall (4 cells) was 1.00 (forest) and 0.92 ± 0.14 (SmallCNN) on test. Prophase recall on test: forest 0.37 ± 0.01, SmallCNN 0.59 ± 0.19. The validation details follow. The 7-class task is reported for the forest and SmallCNN only; the ResNet18 7-class runs were dropped, because the 7-class score depends on 6 validation cells (2 anaphase, 4 telophase) and the compute went to the learning curve. Recall on single mitotic phases rests on very few validation cells (10 metaphase, 2 anaphase, 4 telophase), so it changes in steps of 0.1, 0.5 or 0.25 and should not be interpreted per phase. On validation, prophase (91 cells) was recognised better by SmallCNN (recall 0.57 ± 0.13) than by the forest (0.27). Two of the three SmallCNN 7-class runs stopped early (best epochs 4 and 3 of 25), which explains part of the seed spread.

## RQ2 — ImageNet transfer vs training from scratch

With the same tuning grid, augmentation, balancing and training length (up to 25 epochs with early stopping), the pretrained ResNet18 reached 0.776 ± 0.004 on **test** and the same architecture trained from scratch 0.735 ± 0.004, a difference of 0.041, about ten times the SD (validation: 0.781 ± 0.007 vs 0.745 ± 0.003, difference 0.036). The difference is larger for mitosis: test recall 0.77 ± 0.05 vs 0.60 ± 0.01 (validation 0.75 ± 0.03 vs 0.60 ± 0.02), test mitosis PR-AUC 0.845 ± 0.014 vs 0.674 ± 0.014. On validation the scratch network matched SmallCNN (0.745 vs 0.744 ± 0.010) despite 29 times more parameters; on test it scored 0.011 below SmallCNN (0.735 vs 0.746). On the training/validation runs: all three scratch runs used the full 25 epochs, but their best epochs were 18–20, so they were not still improving at the end. Two of the three pretrained runs stopped early (best epochs 6 and 11). In tuning (4 epochs), all three families selected the same learning rate (1e-3) and weight decay (1e-4), inside the grid.

*Learning curve (Fig. learning_curve_val_run2; `results/learning_curve_val_run2.csv`).* Each subset run had as many gradient steps per epoch as a full run, up to 25 epochs with early stopping. Validation macro-F1, mean ± SD over seeds 0, 1 and 2, at 1 / 5 / 20 / 100% of the labelled training cells (225 / 1,129 / 4,517 / 22,586 cells; 100% = the 3 final runs):

| Model | 1% | 5% | 20% | 100% |
|---|---|---|---|---|
| SmallCNN | 0.515 ± 0.012 | 0.588 ± 0.010 | 0.647 ± 0.020 | 0.744 ± 0.010 |
| ResNet18, pretrained | 0.547 ± 0.007 | 0.648 ± 0.008 | 0.717 ± 0.015 | 0.781 ± 0.007 |
| ResNet18, scratch | 0.517 ± 0.011 | 0.597 ± 0.006 | 0.658 ± 0.009 | 0.745 ± 0.003 |

At every fraction, the pretrained network was ahead of the same architecture trained from scratch: by 0.030 at 1%, 0.052 at 5%, 0.059 at 20% and 0.036 at 100%. The gap was largest at 5–20%, where it is about 4–7 times the larger SD. With 5% of the labels (1,129 cells), the pretrained ResNet18 (0.648) scored about as well as SmallCNN and the scratch ResNet18 with 20% (0.647 and 0.658), and above the random forest trained on all labels (0.632, val). SmallCNN and the scratch ResNet18 were within one SD of each other at every fraction. At 1% (225 cells), the scratch ResNet18 did not learn beyond its first epoch (0.517 ± 0.011), whereas the pretrained ResNet18 reached 0.547 ± 0.007. The scratch network fitted the 225 training cells within the first epoch (one epoch shows each cell about 100 times; train loss ≤ 0.0002 from epoch 1–3 on), and its validation score did not improve afterwards: best epoch 0 for seeds 0 and 1, and flat at 0.515–0.523 for seed 2. SmallCNN at 1% scored about the same as the scratch ResNet18 (0.515 ± 0.012). Many small-subset runs reached their best epoch early (best epoch 6 or earlier in 14 of the 18 runs at 1–5%), so early stopping on the noisy validation curve adds to the seed spread there.

*Take-home:* on test (0.041) and on validation (0.036), ImageNet pretraining gives a modest but clear gain in macro-F1 and a larger gain in mitosis recall, with the same training budget; the learning curve (validation) shows the largest gain with 5–20% of the labels.

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

*Class separation (run 4; `scripts/silhouette.py`, `results/silhouette_val.md`).* The silhouette score of the true 4-class labels, computed on the high-dimensional vectors (not on the UMAP coordinates, which distort distances), was 0.006 for the 28 standardised hand-crafted features, 0.150 ± 0.003 for the SmallCNN embedding and 0.164 ± 0.017 for the pretrained ResNet18 embedding (validation cells, seeds 0–2). All values are low because neighbouring phases form a continuum rather than separate clusters; the classes overlap almost completely in the hand-crafted space, although a random forest still separates them reasonably (macro-F1 0.632), since it can weight a few informative features while Euclidean distance weights all 28 equally.

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

*Masking (run 4; Fig. masking_val, `results/masking_val.md`, `results/masking_check_val.md`, `results/bgonly_investigation_val.md`; validation, mean ± SD over seeds 0–2 unless stated).* Networks trained and evaluated on **cell-only** images (everything outside the 3-px-dilated brightfield mask set to the training background median) scored close to the originals: SmallCNN 0.729 ± 0.019 vs 0.744 ± 0.010, pretrained ResNet18 0.769 ± 0.004 vs 0.781 ± 0.007. Mitosis recall dropped more for SmallCNN (0.54 ± 0.08 vs 0.63 ± 0.04) than for ResNet18 (0.67 ± 0.02 vs 0.75 ± 0.03). Networks trained on **background-only** images (inside the mask filled) scored well above chance (stratified 0.257): SmallCNN 0.558 ± 0.028, pretrained ResNet18 0.601 ± 0.007. The filled hole has the cell's own size and shape, and the log mask area alone gave 0.483 with a logistic regression (mitosis recall 0.59, as for the background-only SmallCNN); mask area plus the per-cell background level and spread gave 0.554 with a random forest, while the background level and spread alone gave 0.27–0.31. When a fixed central disc (radius 22 px, containing 99.5 % of training masks) was filled together with the mask, so that the hole no longer showed the cell size, SmallCNN still reached 0.450 ± 0.007 (recall G1 0.73, S 0.41, G2 0.44, mitosis 0.39) and pretrained ResNet18 0.468 ± 0.003 (mitosis recall 0.19 ± 0.01). The darkfield background in these images is scaled by a per-image factor that differs between phases (training medians G1 45, S 37, G2 32, mitosis 32.5 grey levels; background SD/median constant at 0.066 in every class); radial intensity profiles outside the disc gave 0.31–0.33 with a random forest, so part of the 0.45 is not explained by these simple statistics. Run-2 models trained on original images and evaluated on masked images only collapsed (cell-only 0.16–0.46, background-only 0.16–0.26), mostly predicting G2/mitosis for cell-only images, i.e. they did not transfer to the flat grey surround. A SmallCNN trained on shuffled training labels scored 0.282 ± 0.020. The three-seed border-ring swap (`results/border_check_val_run2.csv`) gave other-class minus same-class differences of 0.000 ± 0.005 (SmallCNN), −0.002 ± 0.010 (pretrained) and +0.003 ± 0.002 (scratch).


*Take-home (Fig. gradcam_val_run2, Fig. 6b; `results/masking_val.md`, user wording 2026-11-07):* The networks do not need the background (cell only ≈ full image: SmallCNN 0.729 vs 0.744, ResNet18 0.769 vs 0.781, val), but in this dataset the background is not free of phase information: a network trained on the background alone, with the cell's size hidden by a fixed disc, still reaches 0.45–0.47 macro-F1 (chance 0.26), apparently through a phase-dependent scaling of the darkfield intensity.

For the forest, adding the raw darkfield intensity levels, which track the background, raised the 4-class macro-F1 from 0.632 ± 0.001 to 0.647 ± 0.003. That gain is small but consistent over seeds.

---

## Limitations (to move to the Discussion)
- *Selection on validation (run 4):* early stopping keeps the epoch with the best val macro-F1, so every reported val number is optimistically selected. A SmallCNN trained on shuffled labels reached 0.282 ± 0.020 (chance 0.257) although its median epoch scored 0.16–0.20. The test set, evaluated once, removes this bias.

- *Darkfield scaling (run 4).* The darkfield image of each cell is scaled by a per-image factor that differs between phases (background SD/median 0.066 in every class; training background medians G1 45, S 37, G2 32 grey levels; `results/bgonly_investigation_val.md`). Its origin (instrument, acquisition or export) is unknown [CHECK BBBC048 documentation].
- *Instrument geometry.* The models use the fixed left/right direction of the darkfield images (left-right flips cost the chosen SmallCNN 0.14 macro-F1 and most of its mitosis recall). This direction comes from the instrument and is the same for every cell, so it is not leakage, but the models may not transfer to another cytometer or another mounting of the optics. A direction-free model (rotations and both flips) scored about 0.02 lower and varied more between seeds.
- *Augmentation hardly matters.* All six augmentation × balancing settings were within 0.03 of each other; the chosen setting won a tie, not a clear contest.
- *Early stopping on a noisy curve.* Models were early-stopped on validation macro-F1 with patience 6. The curve is noisy, especially for mitosis (107 validation cells), and some runs kept a very early epoch (best epochs 0–5), most often in the small learning-curve subsets (the 1% scratch ResNet18 runs fitted their 225 training cells within the first epoch). A smoothed or fixed-epoch selection was not tried, because it would require re-running all models.
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
