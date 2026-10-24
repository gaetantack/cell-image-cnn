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
| Fig 6 | Grad-CAM grid per class; masking result | RQ5 |

## 7. Priority tiers
- **Tier 1 (must):** RQ1; RQ2 including the learning curve; RQ4 UMAP.
- **Tier 2 (should):** RQ3 ablation; Grad-CAM for RQ5.
- **Tier 3 (if time):** pseudotime for RQ4; masking for RQ5; AML dataset.

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
