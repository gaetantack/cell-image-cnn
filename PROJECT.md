# PROJECT.md — Seeing cells: CNNs for cell cycle phase from label-free images

## 1. Goal
Predict the cell cycle phase of single Jurkat cells from label-free brightfield and darkfield imaging flow cytometry images. Compare hand-crafted morphological features with a random forest against CNNs trained from scratch and fine-tuned from ImageNet, using class-balanced metrics under a fixed, equal-budget protocol. Then check what the networks have learned, both whether the continuous cell cycle can be recovered from their features and whether their predictions rely on the cell rather than on artefacts.

## 2. Data
- **Dataset:** Jurkat cell cycle imaging flow cytometry, BBBC048 (Eulenberg et al., 2017), 7 phases.
- **Input channels:** brightfield + darkfield only.
- **Main result:** 4 classes: G1, S, G2, and merged mitosis.
- **Secondary result:** all 7 classes.
- **Leakage rule:** fluorescence channels are never used as model input, because they were used to make the labels.
- AML-Cytomorphology_LMU is out of scope unless time is left after the paper is finished (Tier 3).

## 3. Research questions
1. How much do CNNs improve over hand-crafted morphological features + random forest?
2. Does ImageNet transfer learning help compared with training from scratch, especially with few labelled cells?
3. How do data augmentation and class balancing affect performance on rare (mitotic) classes?
4. Can the continuous cell cycle be recovered from features learned for a discrete classification task?
5. For which classes (if any) does the network base its predictions on shortcuts such as background or image borders rather than on the cell?

## 4. Models
| Model | Description |
|---|---|
| Chance baselines | Majority class + stratified random |
| Random forest | Hand-crafted morphological features (e.g. area, eccentricity, texture) |
| SmallCNN | Small CNN trained from scratch |
| ResNet18 (scratch) | Random initialisation |
| ResNet18 (pretrained) | ImageNet-pretrained, fine-tuned |

## 5. Rigour rules
- **Split:** stratified 70/15/15 train/val/test, seed 42, made once and never changed. Saved as `splits/splits.json` (image indices only) and committed to git so it is reproducible.
- **Test set:** used once, in week 5.
- **Seeds:** 3 seeds (0, 1, 2), reported as mean ± SD.
- **Tuning budget:** 12 configurations per model, evaluated on validation.
- **Metrics:** macro-F1, balanced accuracy, per-class recall, confusion matrices. Chance baselines are reported next to every model.
- **Runtime:** logged for every run.
- **No fluorescence input** (see §2).

## 6. Figure and table plan (Bioinformatics template, 6–8 pages)
| Item | Content | RQ |
|---|---|---|
| Table 1 | Methods overview: model, input, pretrained, parameters, training time | Setup |
| Fig 1 | Example cells per phase, class counts, pipeline | Setup |
| Fig 2 | Macro-F1 / balanced accuracy per model with chance line; confusion matrices RF vs best CNN | RQ1 |
| Table 2 | Test metrics, per-class recall, runtime, plus Eulenberg 2017 numbers for reference | RQ1, RQ2 |
| Fig 3 | Learning curve: % labelled cells vs macro-F1 | RQ2 |
| Table 3 + Fig 4 | Augmentation × balancing ablation, rare-class recall | RQ3 |
| Fig 5 | UMAP of CNN embedding and of RF features; pseudotime vs phase | RQ4 |
| Fig 6 | 6a Grad-CAM grid per class; 6b masking, main text, small: one panel, four bars per model (full image, cell only, background only, background only + fixed disc; all trained on that input, mean ± SD over 3 seeds) with a dashed chance line (`scripts/fig_masking.py` → `figures/fig6b_masking_val`). Supplementary: evaluate-only masking rows (`figures/masking_val`), masking examples (`figures/masking_examples_val`), darkfield background investigation (`results/bgonly_investigation_val.md`) | RQ5 |

## 7. Priority tiers
- **Tier 1 (must):** RQ1; RQ2 including the learning curve; RQ4 UMAP.
- **Tier 2 (should):** RQ3 ablation; Grad-CAM for RQ5.
- **Tier 3 (if time):** pseudotime for RQ4; masking for RQ5; AML dataset.

## 7a. Final configuration per model (frozen before the test)
Drafted overnight 2026-11-07 by Claude (run 4). The plan requires this to be fixed **before** the test set is touched. Nothing below may change after it is confirmed; `scripts/final_test.py` implements exactly this.

| Model | What is evaluated on test | Settings | Why this one |
|---|---|---|---|
| Chance (majority, stratified) | `DummyClassifier(most_frequent / stratified)`, fitted on train labels, seeds 0–2 | as `scripts/run_baselines.py` | Lower bound next to every model (rigour rule) |
| Random forest | Re-fitted on train, seeds 0–2, 4 and 7 classes | feature set `main` (28 features), 500 trees, `max_features` 0.33 (= 9 features per split), `min_samples_leaf` 5, `class_weight` balanced, median imputation fitted on train; config of `rf_c4_tune08_s0`, best of 12 (run 1) | Selected on val with the same 12-config budget as the CNNs; unchanged in run 2 |
| Random forest + raw DF levels (RQ5 side) | as RF, feature set `raw_dark`, 4 classes | same configuration as the RF | Side experiment only |
| SmallCNN | Checkpoints `smallcnn_c4_n2_final_s{0,1,2}` (and `smallcnn_c7_n2_final_s{0,1,2}`) | lr 1e-3, wd 1e-4, batch 64, vflip + weighted loss, max 25 epochs, early stopping patience 6 on val macro-F1 (best-epoch weights) | Run 2 = paper result (user 2026-11-01); config = best of 12 (`n2_tune`) |
| ResNet18 (ImageNet) | Checkpoints `resnet18_pretrained_c4_n2_final_s{0,1,2}` | as SmallCNN; 66 → 128 px, conv1 = mean of RGB filters | same |
| ResNet18 (scratch) | Checkpoints `resnet18_scratch_c4_n2_final_s{0,1,2}` | as SmallCNN; 66 → 128 px | same |

Not evaluated on test: tuning runs, Phase A runs, learning-curve runs, masked (`n4_`) runs, 7-class ResNets (dropped). Metrics: macro-F1 (primary), balanced accuracy, per-class recall, confusion matrices, one-vs-rest ROC-AUC and PR-AUC (mitosis PR-AUC next to its prevalence), inference time per 1,000 cells. Reported as mean ± SD over seeds 0–2. The val dress rehearsal (`results/table_val_summary_finaltest.md`, `results/final_test_val_check.csv`) reproduces every logged val macro-F1. **Status: Confirmed by user 2026-11-07.**

## 8. Decisions log
| Date | Decision | Reason |
|---|---|---|
| 2026-10-22 | All settings in `src/config.py` (plain Python) | One place for config; no hard-coded values |
| 2026-10-22 | Split saved as `splits/splits.json` and committed | Small (indices only); keeps the split reproducible |
| 2026-10-24 | Main result on 4 classes (G1, S, G2, mitosis merged); 7 classes as secondary result | Anaphase (15 cells) and telophase (27) are too few for a reliable test score (about 2 and 4 test cells) |
| 2026-10-24 | Keep every cell, including the empty-mask cell and tiny/failed masks; no cell is removed from the split | The split stays fixed; mask failures are handled at the feature level |
| 2026-10-24 | Random forest: missing features filled with the training-set median per feature, fitted on train rows only | One cell (12686) has an empty mask; fitting on train only avoids leakage from val/test |
| 2026-10-24 | Darkfield (Ch6) background-corrected intensity features added (mean and percentiles minus the median outside the mask); raw features kept | Lets us compare raw vs corrected, because Ch6 background differs by phase |
| 2026-10-24 | Mask: if every object reaches the border ring, keep the object nearest the image centre (not the largest); Otsu threshold = min(whole-image Otsu, Otsu within 25 px of the centre) | Fixes the 5 masks that sat on artefacts: nearest-centre alone fixed none (the cell was never detected), central-only Otsu shrank 46 blurry cells to their bright core; the min fixes all 5 and changes only 13 train/val cells by >20% |
| 2026-10-24 | Two feature sets in `config.FEATURE_SETS`: `main` = background-robust (shape, brightfield intensity, GLCM both channels, Ch6 SD, Ch6 background-corrected percentiles; no raw Ch6 mean/percentiles, no corrected Ch6 mean); `raw_dark` = `main` + raw Ch6 mean/percentiles | Main random forest uses `main`; `raw_dark` is a side experiment for RQ5 (does the darkfield background shortcut help?). Ch6 SD is kept because subtracting a constant background does not change it |
| 2026-10-25 | Epoch caps: CNN tuning runs 4 epochs, final runs 12 epochs, identical for all families (made overnight by Claude, to confirm) | 12 configs × 3 families at full length did not fit in one night; deviation from the plan |
| 2026-10-25 | RF uses feature set `main` (28 features); RF grid max_features {sqrt, 0.33} × min_samples_leaf {1, 5, 20} × class_weight {balanced, balanced_subsample}, 500 trees (made overnight by Claude, to confirm) | Background-robust features (decision 2026-10-24); 12 configs = same budget as CNNs |
| 2026-10-25 | CNN grid lr {1e-4, 5e-4, 2e-3} × weight decay {1e-4, 1e-2} × batch size {64, 128}, augment = basic and balance = weighted fixed; best config re-trained with seeds 0–2 (made overnight by Claude, to confirm) | Same grid for every family (fair comparison) |
| 2026-10-25 | 7-class runs reuse the 4-class configuration without re-tuning (made overnight by Claude, to confirm) | Time; 7 classes is the secondary result |
| 2026-10-25 | ResNet18: input upscaled 66 → 128 px; pretrained conv1 = mean of the RGB filters for both channels (made overnight by Claude, to confirm) | 66 px leaves a 3 × 3 map; the mean keeps the grey-level filters |
| 2026-10-25 | Learning curve: stratified subsets (≥ 1 cell per class), every epoch draws len(full train) samples (made overnight by Claude, to confirm) | Equal gradient steps, so the curve measures labels, not training length |
| 2026-10-25 | umap-learn 0.5.12 installed (environment.yml) (made overnight by Claude, to confirm) | UMAP for RQ4 |
| 2026-10-25 | Grad-CAM self-written, true 4-class label, last conv block; summaries heat-in-mask and heat-in-8-px-border (made overnight by Claude, to confirm) | Explainable in the oral |
| 2026-10-25 | experiments.csv gets columns run_id, device, git_commit (made overnight by Claude, to confirm) | Skip finished runs; traceability |
| 2026-10-25 | Extra analyses: raw_dark RF, PC1 pseudotime, border swap test, orientation check (made overnight by Claude, to confirm) | Cheap checks for RQ3–RQ5 |
| 2026-11-01 | Run 2 (all CNN runs tagged `n2_`; run-1 rows kept): new augmentation option `vflip` (up-down flips + jitter, no rotations/left-right flips); `EARLY_STOP_PATIENCE` 5 → 6; final and Phase A runs max 25 epochs (made overnight by Claude; confirmed by user 2026-11-01) | Darkfield images have a fixed left/right direction (run-1 orientation check); 12 epochs was a cap for scratch ResNet |
| 2026-11-01 | AUG*/BAL* = vflip + weighted loss for all CNN families, chosen from 3 augmentations × 2 balancings × 3 seeds (SmallCNN) by a rule fixed beforehand (highest mean val macro-F1; tie within 1 SD → mitosis recall; tie → vflip > none > basic, weighted > oversample; "1 SD" = the larger of the two settings' SDs) (made overnight by Claude; confirmed by user 2026-11-01) | Best two settings tied (0.743 vs 0.741); vflip + weighted most stable mitosis recall |
| 2026-11-01 | Run-2 CNN grid lr {5e-4, 1e-3, 2e-3, 3.5e-3, 5e-3, 1e-2} × wd {1e-4, 1e-2}, batch size fixed 64, 4-epoch tuning, same for all families (made overnight by Claude; confirmed by user 2026-11-01) | Run 1 picked the grid edge (2e-3) everywhere; 64 beat 128 in every run-1 pair. All families now pick lr 1e-3, wd 1e-4 (inside the grid) |
| 2026-11-01 | Phase E items started only if all runs of the item (3 seeds or 3 label fractions of one family) end before 06:30 (`scripts/queue_run2_CE.sh`) (made overnight by Claude, to confirm) | A family with < 3 seeds is never reported as a result |
| 2026-11-01 | ResNet18-pretrained learning curve (run 2, seed 0) started by hand outside the worst-case time guard; each run started only when it could end by 06:30 (made overnight by Claude, to confirm) | GPU otherwise idle; RQ2 is Tier 1 |
| 2026-11-01 | Random forest and chance baselines not re-run in run 2; run-2 tables use their run-1 rows (made overnight by Claude, to confirm) | Their settings did not change |
| 2026-11-01 | Keep vflip + weighted loss for the paper; report honestly that all six augmentation × balancing settings were within 0.03, so augmentation hardly matters (user) | Tied for best, most stable mitosis recall (SD 0.01) |
| 2026-11-01 | Keep the darkfield left/right direction (no rotations or left-right flips); state it as a limitation: it comes from the instrument and is the same for every cell (not leakage), but may not carry over to another cytometer (user) | Direction-free `basic` scored ≈ 0.02 lower and was less stable |
| 2026-11-01 | Run 2 is the result in the paper; run 1 is presented as the pilot that found the orientation problem (user) | Clean main result; good story for the oral |
| 2026-11-01 | Finish the RQ2 learning curve with 3 seeds for all three families (`scripts/queue_run2_lc_finish.sh`); drop the 7-class ResNet18 runs (7-class reported for RF and SmallCNN only) (user) | RQ2 is must-have and needs 3 seeds; 7-class is secondary and its score depends on 6 cells; both do not fit in one night |
| 2026-11-01 | Keep early-stopping patience 6; state as a limitation that early stopping on the noisy val curve sometimes picks a very early epoch (user) | Changing it means re-running everything for a small effect |
| 2026-11-07 | Run 4 (tags `n4_`): masking experiment for RQ5. Mask = `cell_mask` (brightfield) dilated by 3 px; fill = one constant per channel, the median training background (BF 94, DF 39); masked train AND val; run-2 settings, no re-tuning (made overnight by Claude; confirmed by user 2026-11-07) | Prompt design; a constant fill removes the phase-dependent darkfield level |
| 2026-11-07 | Added a `background_only_disc` control (fixed central disc r = 22 px + cell mask filled), SmallCNN and ResNet18-pre, seeds 0–2 (made overnight by Claude; confirmed by user 2026-11-07) | Background-only scored far above chance; the hole showed the cell size (mask area alone 0.483) |
| 2026-11-07 | Extra GPU queues `queue_run4c.sh`/`queue_run4d.sh`: ResNet18-pre background-only seeds 1–2 and disc control seeds 0–2 (made overnight by Claude; confirmed by user 2026-11-07) | GPU idle after the planned runs; turns a single-seed row into 3 seeds |
| 2026-11-07 | Queue time guard now uses real clock times with a deadline fixed at queue start (`scripts/queue_common.sh`, `QUEUE_DEADLINE` override) (made overnight by Claude; confirmed by user 2026-11-07) | The old guard counted minutes from noon and failed after midnight / in the daytime |
| 2026-11-07 | `scripts/final_test.py` = the test-day script (frozen models of §7a; refuses test without `--i-understand-this-is-the-one-test-run`; refuses a second test run); val dress rehearsal reproduces 33/33 logged val macro-F1. Extra metrics (ROC-AUC, PR-AUC, OOB, inference time) are stored in the settings JSON of the appended rows, so the experiments.csv columns stay unchanged; chance ROC/PR-AUC use the training class prior as probability (made overnight by Claude; confirmed by user 2026-11-07) | Rigour rules; Table 2 needs inference time; keeps the CSV schema |
| 2026-11-07 | Fig 1 shows all 4 mitotic sub-phases (pro, meta, ana, telo); test class counts read from `splits.json` indices (labels only); `make_table1.py` default tag is run 2 (`n2_final`, `--tag final` = run 1); commits with the existing author identity via env variables, `try_mask.py` left untracked (made overnight by Claude; confirmed by user 2026-11-07) | Run-4 report decisions 6–8 |
| 2026-11-07 | RQ5 message rephrased: The networks do not need the background (cell only ≈ full image: SmallCNN 0.729 vs 0.744, ResNet18 0.769 vs 0.781, val), but in this dataset the background is not free of phase information: a network trained on the background alone, with the cell's size hidden by a fixed disc, still reaches 0.45–0.47 macro-F1 (chance 0.26), apparently through a phase-dependent scaling of the darkfield intensity. Origin of the darkfield scaling listed as a limitation [CHECK BBBC048 documentation] (user) | Masking results (`results/masking_val.md`) |
| 2026-11-07 | §7a frozen configuration and `notes/test_day_checklist.md` confirmed (user) | Required before the test |
| 2026-11-07 | Masking in the main text, small (Fig 6b, trained-on-input rows only); evaluate-only rows and the darkfield investigation in the Supplementary (user) | 3 seeds in every row; keeps the main text short |

## 9. Observations
| Date | Observation | Relevance |
|---|---|---|
| 2026-10-22 | Prophase cell 30439 has a bright rectangle artefact in the top-right corner, in both Ch3 and Ch6 (`figures/channel_check.png`). | RQ5 shortcut candidate |
| 2026-10-22 | Ch6 (darkfield) background level varies a lot between cells, from near-black to mid-grey. | RQ5 shortcut candidate |
| 2026-10-24 | Corner/border artefacts: the brightfield mask found an extra object in the 3 px border ring in 860 of 32,266 cells (2.7%); similar rate in G1/S/G2/prophase (2.1–2.7%). | RQ5 shortcut candidate; dropped from the mask |
| 2026-10-24 | Ch6 background depends on phase: median border level G1 45, S 37, G2 32, prophase 33, metaphase 24 (train+val; `figures/eda_intensity_hist.png`). Subtracting it flattens Ch6 p90 across phases, but the corrected Ch6 mean is still strongly tied to the background (r = −0.92, raw r = +0.82), because the cell interior is dark regardless. | RQ5 shortcut candidate; corrected mean is not shortcut-free |
| 2026-10-24 | 52 train/val cells have a mask < 150 px or empty (`figures/tiny_masks.png`), 39 of them G1. Most are small, blurry cells where the mask catches only the bright core. In about 5 (e.g. 18117, 25221, 39984, 44192, 48404) the mask sits on a corner/line artefact instead of a pale cell on a bright background. | Features of those cells describe the artefact; kept per decision above |
| 2026-10-24 | After the mask change (see decisions log), the 5 artefact masks are on the cell (areas 306–621 px; 48404 still partial). 47 train/val cells remain < 150 px or empty, 37 of them G1, all small/blurry cells (`figures/tiny_masks.png`); border objects dropped in 864 cells (2.7%). | Mask failures now limited to blurry cells; 12686 still empty |
| 2026-10-24 | Tiny masks are mostly real small cells: 46 of the 47 train/val cells with a mask < 150 px (about 45 of 52 before the mask fix) are small cells, mostly G1 (37), often out of focus, where the mask captures only the bright centre, so their area is underestimated (0.2% of train/val cells; the 47th is the empty-mask cell 12686). Accepted, not fixed. | Limitation of hand-crafted features: mention in the paper |
| 2026-10-26 | Darkfield images have a fixed left/right direction: left half brighter in 81% of training images (brightfield 45%). SmallCNN without augmentation reaches val macro-F1 0.742 vs 0.690 with rotations/flips + weighted loss, and drops to 0.608 on left-right-flipped val images (`scripts/orientation_check.py`; seed 0) (overnight, Claude) | Augmentation choice (RQ3); rotations/left-right flips remove a real instrument signal |
| 2026-10-26 | Border swap test: pasting the 8-px border of a same-class vs other-class cell gives the same val macro-F1 (difference −0.014 to +0.012) for all CNNs (`scripts/border_check.py`) (overnight, Claude) | RQ5: the 8-px border ring carries no phase-specific information the networks use; the background as a whole does (masking, 2026-11-07) |
| 2026-10-26 | 7-class val macro-F1 hinges on 6 cells (2 anaphase, 4 telophase); scratch ResNet18 0.09 below RF because of them (overnight, Claude) | 7-class ranking unreliable |
| 2026-11-01 | Augmentation × balancing (SmallCNN, 3 seeds, val): all six settings within 0.03 in mean (0.715–0.743), seed spread up to 0.11 within one setting; the run-1 one-seed gap "none 0.05 better than basic" shrinks to 0.014 with 3 seeds (`results/run2_phaseA_val.md`) (overnight, Claude) | RQ3: augmentation choice matters less than seed noise; vflip + weighted most stable |
| 2026-11-01 | vflip-trained SmallCNN is invariant to up-down flips (0.757 → 0.743) but still uses the left/right darkfield direction (left-right flip → 0.621, mitosis recall 0.67 → 0.19) (`results/orientation_check_val_run2.csv`) (overnight, Claude) | RQ3/RQ5: model tied to instrument geometry |
| 2026-11-01 | Run-2 SmallCNN Grad-CAM: S-phase border pattern of run 1 gone (S heat in border 0.44 → 0.16, in mask 0.18 → 0.43). ResNet18-pretrained: "lower-right spot" on G1/S/G2 maps remains despite up-down flip training → probably a 4 × 4 grid/stride property (overnight, Claude) | RQ5 |
| 2026-11-01 | Run-2 ResNet18-pretrained: macro-F1 up (0.761 → 0.781) but mitosis recall down (0.83 → 0.75) with higher mitosis precision (s0 0.63 → 0.85) (overnight, Claude) | Report recall and precision, not only macro-F1 |
| 2026-11-07 | Masking (val, 3 seeds): cell-only networks nearly keep the score (SmallCNN 0.729 ± 0.019 vs 0.744; ResNet18-pre 0.769 ± 0.004 vs 0.781); background-only SmallCNN 0.558 ± 0.028, ResNet18-pre 0.601 ± 0.007; fixed-disc background-only SmallCNN 0.450 ± 0.007, ResNet18-pre 0.468 ± 0.003; mask area alone 0.483; label-shuffle 0.282 ± 0.020 (`results/masking_val.md`) (overnight, Claude, to confirm) | RQ5: the networks do not need the background, but the background is not free of phase information (user wording 2026-11-07) |
| 2026-11-07 | Darkfield images are scaled by a per-image factor that differs by phase: background SD/median = 0.066 in every class, level vs SD ρ = 0.93, background G1 45 > S 37 > G2 32 (`results/bgonly_investigation_val.md`) (overnight, Claude) | RQ5 shortcut source; origin unknown [CHECK BBBC048 documentation] |
| 2026-11-07 | Early stopping on val inflates val scores: label-shuffled SmallCNN 0.282 ± 0.020 vs chance 0.257, median epoch 0.16–0.20 (overnight, Claude) | Limitation; test set removes it |
| 2026-11-07 | Inference-only masking collapses the run-2 models (cell-only 0.16–0.46): a flat surround is out of distribution, so inference-only masking alone cannot show a shortcut (overnight, Claude) | Method lesson for RQ5 |
