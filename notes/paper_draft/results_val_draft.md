# Results (DRAFT)

> **Validation results — not the final test numbers.** Every number below is a validation-set number from `results/experiments.csv` (tables: `results/table_val_summary.md`). The test set has not been touched. Claude wrote this overnight on 2026-10-25/26; check every sentence. 

All models were selected and early-stopped on the validation set, so these numbers are optimistic estimates. Mean ± SD over seeds 0, 1 and 2 unless stated otherwise. The validation set has 4,840 cells: 2,150 G1, 1,293 S, 1,290 G2 and 107 mitotic (91 prophase, 10 metaphase, 2 anaphase, 4 telophase).

## RQ1 — CNNs vs hand-crafted features + random forest

On the 4-class task, the random forest on 28 hand-crafted features reached a validation macro-F1 of 0.632 ± 0.001. SmallCNN reached 0.687 ± 0.006 and the ImageNet-pretrained ResNet18 reached 0.761 ± 0.006 (Fig. model_comparison_val). For reference, always predicting G1 gives 0.154 and random guessing in class proportions gives 0.257 ± 0.006.

Most of the gain is in mitosis. Mitosis recall was 0.26 ± 0.01 for the forest, 0.65 ± 0.10 for SmallCNN and 0.83 ± 0.01 for ResNet18. G1, S and G2 recall rose only a little: from 0.84 / 0.64 / 0.77 for the forest to 0.85 / 0.72 / 0.79 for ResNet18 (Fig. confusion_val_4class). In both models most errors are G1↔S and S↔G2 confusions. Missed mitotic cells are almost always called G2: 76 of 107 for the forest and 19 of 107 for ResNet18 (seed 0).

The forest relied mostly on cell size. Shuffling `area` lowered the validation macro-F1 by 0.113 ± 0.014, and no other feature lowered it by more than 0.04 (Fig. rf_importance). Next came brightfield texture (`Ch3_glcm_correlation`, 0.037) and the cell's axis lengths (0.026 and 0.021).

On the 7-class task the forest reached 0.559 ± 0.004. Recall on single mitotic phases rests on very few validation cells (2 anaphase, 4 telophase), so it can only change in steps of 0.5 or 0.25 and should not be interpreted per phase. With the 4-class hyperparameters reused, the CNNs reached 0.560 ± 0.007 (SmallCNN), 0.581 ± 0.042 (pretrained ResNet18) and 0.468 ± 0.016 (scratch ResNet18). In this task the forest and SmallCNN were equal and the pretrained ResNet18 was only slightly ahead, with a large seed spread. The scratch ResNet18 was clearly below the forest. We checked this gap in the confusion matrices (seed 0). It comes almost entirely from the anaphase and telophase cells: the forest predicted telophase 4 times, all correct, while the scratch ResNet18 predicted telophase 9 times and anaphase 9 times, with no anaphase correct. With 2 and 4 validation cells, the F1 of these two classes, and with it two-sevenths of the macro-F1, hinges on single cells. The 7-class ranking on validation is therefore not reliable. Prophase, the largest mitotic phase (91 cells), was recognised far better by all CNNs (recall 0.53–0.64) than by the forest (0.27), which called 70% of prophase cells G2 (Fig. confusion_val_7class).

*Take-home (Fig. model_comparison_val):* on validation, both CNNs beat the hand-crafted features, mainly because they find far more mitotic cells.

## RQ2 — ImageNet transfer vs training from scratch; learning curve

With the same tuning grid and epoch caps, ResNet18 trained from scratch reached at most 0.649 in tuning (4 epochs, seed 0). Its pretrained version reached 0.722 in the same setting. In the final runs, the scratch ResNet18 reached 0.696 ± 0.012, compared with 0.761 ± 0.006 for the pretrained network. That is a difference of 0.065, about five times the larger SD. The scratch network matched SmallCNN (0.687 ± 0.006) despite 29 times more parameters, and its mitosis recall (0.64 ± 0.11) was also at the SmallCNN level rather than the pretrained level (0.83 ± 0.01). Two of the three scratch runs reached their best validation score in the last two of their 12 epochs. The scratch model is therefore the one most limited by the epoch cap.

*Learning curve (Fig. learning_curve_val; results/learning_curve_val.csv).* Each subset run had as many gradient steps as a full run. Seeds: SmallCNN 3 at every fraction; both ResNet18 variants seed 0 only for the subsets. Validation macro-F1 at 1 / 5 / 20 / 100% of the labelled training cells:
- SmallCNN: 0.547 ± 0.015 / 0.631 ± 0.016 / 0.689 ± 0.008 / 0.687 ± 0.006
- pretrained ResNet18: 0.552 / 0.660 / 0.739 / 0.761 ± 0.006
- scratch ResNet18: 0.539 / 0.609 / 0.678 / 0.696 ± 0.012

At 1% (225 cells, about 5 of them mitotic) all three networks were within 0.015 of each other. With 5% of the labels the pretrained ResNet18 already exceeded the random forest trained on all labels (0.632). The advantage of pretraining appeared from 20% onwards, and SmallCNN did not improve beyond 20%. Pretraining therefore did not help most when labels were scarcest. The single-seed points carry a seed spread of about 0.015 (SmallCNN SD at 1–5%), so the differences at 1% are within noise.

*Take-home (Fig. learning_curve_val):* on validation, ImageNet pretraining helps once a few thousand labelled cells are available, while a small network saturates at about 4,500 cells.

*Caveat:* the epoch cap (12 epochs for final runs) favours models that start closer to a good solution. With longer training the scratch model may close part of the gap.

## RQ3 — Augmentation and class balancing (preliminary, seed 0)

Six SmallCNN runs crossed augmentation (none / basic) with class balancing (none / weighted loss / oversampling). They used the selected hyperparameters and seed 0 only (Fig. ablation_val; results/ablation_val.md). Validation macro-F1:

| Augmentation | No balancing | Weighted loss | Oversampling |
|---|---|---|---|
| none | 0.742 | 0.734 | 0.738 |
| basic | 0.682 | 0.690 | 0.746 |

Mitosis recall: without augmentation 0.57 / 0.64 / 0.53; with augmentation 0.30 / 0.56 / 0.69.

Training *without* augmentation gave a higher macro-F1 (0.734–0.742) than the setting used for all final runs (basic augmentation with weighted loss, 0.690). That difference is far larger than the seed SD of SmallCNN (0.006). Augmentation combined with oversampling, however, reached the same level (0.746) and had the highest mitosis recall (0.69). With augmentation, balancing mattered: without it, mitosis recall fell to 0.30. Without augmentation, balancing changed little. The setting fixed before tuning (basic + weighted) was thus one of the weakest combinations.

*Why augmentation can hurt here (extra check, `scripts/orientation_check.py`).* Our augmentation assumes that a cell image has no preferred orientation. For the darkfield channel this is false: the left half of the image is brighter than the right half in 81% of training images, against 45% for brightfield. The side-scatter light presumably reaches the camera from one direction (instrument geometry, [CHECK: confirm with the BBBC048 / ImageStream description]). The model trained without augmentation uses this direction. On left-right-flipped validation images its macro-F1 fell from 0.742 to 0.608, and its mitosis recall from 0.57 to 0.16. On images flipped top to bottom it fell only to 0.710. The augmented model scored the same on all flipped and rotated images (0.681–0.698). Because the direction is a property of the instrument and identical in all images, it is information the model may legitimately use, not leakage. But 90° rotations and left-right flips remove it.

These results are preliminary: one seed, and the hyperparameters were tuned with augmentation switched on.

*Take-home (Fig. ablation_val):* the rotation/flip augmentation removes a real directional signal in the darkfield images. Without augmentation, or with augmentation plus oversampling, SmallCNN reached about 0.74 validation macro-F1, against 0.69 for the setting used in all final runs. With one seed, which choice is best remains open.

## RQ4 — Is the continuous cell cycle visible in the learned features?

In UMAPs of the CNN embeddings of the validation cells, the cells lie along an arc in the order G1 → S → G2 → mitosis (Fig. umap_val). The arc is clearest for the pretrained ResNet18, where G1 and G2 occupy the two ends and the mitotic cells sit at the G2 end. The networks were trained only on the 4 classes, not on the 7 phases. In the hand-crafted feature space, G1 and G2 lie at opposite sides, but S and mitotic cells are mixed in with them.

The first principal component (PC1, fitted on training cells) served as a simple pseudotime. Its Spearman correlation with phase order (G1 = 0 … telophase = 6) on validation cells was 0.45 for the hand-crafted features, 0.83 for the SmallCNN embedding and 0.86 for the pretrained ResNet18 embedding (Fig. pseudotime_val; seed-0 runs). Part of this is expected by construction, because the networks were trained to separate G1, S, G2 and mitosis. Correlations with the 4-class order are identical to three decimals, since only 16 validation cells are in metaphase, anaphase or telophase.

*Take-home (Fig. umap_val):* the CNN embedding orders cells along the cell cycle, though it was only trained on discrete classes.

## RQ5 — Does the network look at the cell?

*SmallCNN (seed 0; `python -m scripts.fig_gradcam --model smallcnn`, results/gradcam_val_summary_smallcnn.csv).* Grad-CAM maps for the true 4-class label were computed for all validation cells (Fig. gradcam_val; results/gradcam_val_summary.csv). For each phase, the median share of the heat inside the brightfield cell mask was:

| Phase | Cells | Heat in mask | Mask area | Heat in border ring | Ring area |
|---|---|---|---|---|---|
| G1 | 2,150 | 0.38 | 0.11 | 0.25 | 0.43 |
| S | 1,293 | 0.18 | 0.13 | 0.44 | 0.43 |
| G2 | 1,290 | 0.49 | 0.15 | 0.21 | 0.43 |
| Prophase | 91 | 0.64 | 0.17 | 0.04 | 0.43 |

For G1, G2 and prophase the heat is concentrated on the cell, at 3–4 times the mask's share of the image. For S phase the heat is not on the cell. It lies in the 8-pixel border ring at about the ring's share of the area (0.44 vs 0.43), often as a frame along the image edge. This could be a shortcut (the background differs by phase) or an artefact of zero padding in the convolutions; we cannot yet tell them apart. In training cell 30439 (corner artefact), the heat does not fall on the artefact (Fig. gradcam_cell30439_train).

For the forest, adding the raw darkfield intensity levels, which track the background, raised the 4-class macro-F1 from 0.632 ± 0.001 to 0.647 ± 0.003. That gain is small but consistent over seeds.

*Border test (extra analysis, preliminary).* To test whether the networks use the background, we changed only the outer 8-pixel ring of the validation images (`scripts/border_check.py`, results/border_check_val.csv, seed-0 runs) and left the cell in the centre unchanged. Every edit lowered the macro-F1:
- setting the ring to its own median: −0.007 for SmallCNN, −0.036 for the pretrained ResNet18, +0.001 for the scratch ResNet18;
- setting it to one common brightness: −0.065, −0.112 and −0.040;
- copying in the ring of another cell of the same class: −0.137, −0.098 and −0.070.

The last edit is the control. Copying a ring from a cell of a different class gave almost the same scores: −0.014, +0.003 and +0.012 relative to the same-class copy. The drops therefore come from the edit itself (an unnatural seam at the ring edge, or parts of the donor cell inside the ring), not from losing background information that belongs to the right class. Within the precision of this test, we find no evidence that the CNNs use a phase-specific background in the border ring. The networks are, however, sensitive to changes there, and they then predict mitosis more often (mitosis recall rises with the common-brightness and both swap edits). The Grad-CAM border heat for S phase thus does not seem to reflect a background shortcut. Zero padding remains a possible cause, untested. One seed, one random draw of donor cells.

*Best model (pretrained ResNet18, seed 0).* The figure `gradcam_val` was remade with the best CNN by mean 4-class validation macro-F1. Its heat lies on the cell for every phase. The median share of heat inside the mask is 0.36–0.49, against a mask area of 0.11–0.19. The share in the border ring is 0.06–0.12 (0.22 for the 4 telophase cells), far below the ring's 0.43 of the area. The S-phase border pattern of SmallCNN does not occur here. For G1, S and G2, the heat often peaks at the same spot at the lower right of the cell edge, whatever the cell's orientation. This may be a property of the very coarse 4 × 4 Grad-CAM grid rather than of the cells; we have not tested it. For mitotic cells the heat covers the whole cell, and for telophase both daughter cells.

*Take-home (Fig. gradcam_val):* the best network (pretrained ResNet18) looks at the cell in every phase. For SmallCNN, the S-phase heat lies at the image border, but swapping the border between classes does not change the predictions, so we found no sign that the CNNs use a phase-specific background.
