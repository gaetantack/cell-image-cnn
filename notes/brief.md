# Project 19 — Seeing cells: CNNs for imaging flow cytometry and cell morphology

## Official brief

**Background.** Imaging flow cytometry (e.g. Amnis ImageStream) combines the throughput of flow cytometry with microscopy: it records brightfield, darkfield and fluorescence images of every single cell passing the laser. In haematology, blood and bone marrow smears are still examined by eye to find abnormal cells, such as leukaemic blasts. Such image data are ideal for convolutional neural networks (CNNs), which learn features directly from pixels instead of relying on hand-crafted measurements like area or intensity. Eulenberg et al. (2017) showed that a deep network can predict cell cycle phase from label-free brightfield and darkfield images alone, and even reconstruct the continuous cell cycle trajectory from the learned features. Matek et al. (2019) reached expert-level classification of white blood cells in AML smears. This project gives hands-on experience with deep learning for images: data augmentation, transfer learning, class imbalance, and interpreting what a network has learned.

**Machine learning project.** Students train image classifiers for single-cell images: (i) a baseline using hand-crafted morphological features (area, eccentricity, texture from scikit-image or CellProfiler) with a random forest; (ii) a small CNN trained from scratch; (iii) a pretrained network (ResNet or EfficientNet) fine-tuned by transfer learning. They evaluate with class-balanced metrics and confusion matrices, visualize learned features with UMAP, and use saliency methods (Grad-CAM) to check that the network looks at biologically meaningful structures rather than artefacts.

**Data.** Jurkat cell cycle imaging flow cytometry dataset of Eulenberg et al. (2017) (about 32,000 cells, brightfield and darkfield, 7 phases; public); AML-Cytomorphology_LMU dataset (Matek et al., 2019; about 18,000 single white blood cell images in 15 classes); optionally the BBBC datasets.

**Suggested tools.** PyTorch and torchvision (or timm), scikit-image, scikit-learn, pytorch-grad-cam; GPU via Google Colab or Kaggle.

## My scope (agreed with the lecturer, 20 Oct)
- Core dataset: Jurkat cell cycle, BBBC048. Brightfield + darkfield only, 7 phases.
- Main result on 4 classes (G1, S, G2, merged mitosis); 7 classes as secondary result.
- AML dataset: only if time is left after the paper is finished. The lecturer said to start with one dataset in depth.

## Research questions
1. How much do CNNs improve over hand-crafted morphological features + random forest?
2. Does ImageNet transfer learning help compared with training from scratch, especially with few labelled cells?
3. How do data augmentation and class balancing affect performance on rare (mitotic) classes?
4. Can the continuous cell cycle be recovered from features learned for a discrete classification task?
5. For which classes (if any) does the network base its predictions on shortcuts such as background or image borders rather than on the cell?

## Models
Chance baselines (majority + stratified random), random forest on hand-crafted features, SmallCNN from scratch, ResNet18 from scratch, ResNet18 ImageNet-pretrained.

## Rigour rules
- Stratified 70/15/15 train/val/test split, seed 42, made once and never changed
- Test set used once, in week 5
- 3 seeds (0, 1, 2), reported as mean ± SD
- Same tuning budget: 12 configurations on validation per model
- Metrics: macro-F1, balanced accuracy, per-class recall, confusion matrices
- Chance baselines next to every model
- Runtime logged for every run
- Fluorescence channels never used as input (they were used to make the labels → leakage)

## Priority tiers
- Tier 1 (must): RQ1, RQ2 incl. learning curve, RQ4 UMAP
- Tier 2 (should): RQ3 ablation, Grad-CAM for RQ5
- Tier 3 (if time): pseudotime for RQ4, masking for RQ5, AML

## Paper plan (Bioinformatics template, 6–8 pages)
- Table 1: methods overview (model, input, pretrained, parameters, training time)
- Fig 1: example cells per phase, class counts, pipeline
- Fig 2 (RQ1): macro-F1/balanced accuracy per model with chance line; confusion matrices RF vs best CNN
- Table 2 (RQ1, RQ2): test metrics, per-class recall, runtime + Eulenberg 2017 numbers
- Fig 3 (RQ2): learning curve, % labelled cells vs macro-F1
- Table 3 + Fig 4 (RQ3): augmentation × balancing, rare-class recall
- Fig 5 (RQ4): UMAP of CNN embedding and of RF features, pseudotime vs phase
- Fig 6 (RQ5): Grad-CAM grid per class, masking result
