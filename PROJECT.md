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
