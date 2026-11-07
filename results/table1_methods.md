# Table 1. Methods overview

CNN rows: final runs with run tag `n2_final`. Training time: wall-clock seconds per final 4-class run (mean ± SD over seeds 0–2; CNNs on Apple M2 GPU (MPS), random forest on 8 CPU cores), including the per-epoch validation pass. Inference: CPU (4 threads), seconds per 1,000 cells, measured by `scripts/final_test.py` on val while a training job used the GPU (re-measured on test day). BF = brightfield (Ch3), DF = darkfield (Ch6). Chance baselines (majority, stratified) have no input features.

| Model | Input | Pretrained | Parameters (4 classes) | Training settings | Best epoch / epochs run | Training time (s) | Inference (s/1,000 cells) |
|---|---|---|---|---|---|---|---|
| Random forest | 28 hand-crafted features (BF mask: shape, BF intensity, BF + DF texture, DF spread) | — | — | 500 trees, max_features 0.33, min_samples_leaf 5, class_weight balanced | — | 7 ± 0 | 0.019 ± 0.001 |
| SmallCNN | BF + DF image, 2 × 66 × 66 | No | 389,636 | AdamW lr 0.001, wd 0.0001, batch 64, cosine schedule, max 25 epochs, early stopping (patience 6), aug. vflip, weighted loss | 19/25 · 21/25 · 14/21 | 352 ± 32 | 1.741 ± 0.086 |
| ResNet18 (scratch) | BF + DF image, upscaled to 2 × 128 × 128 | No | 11,175,428 | AdamW lr 0.001, wd 0.0001, batch 64, cosine schedule, max 25 epochs, early stopping (patience 6), aug. vflip, weighted loss | 20/25 · 20/25 · 18/25 | 1901 ± 121 | 14.389 ± 0.707 |
| ResNet18 (ImageNet) | BF + DF image, upscaled to 2 × 128 × 128 | ImageNet (timm resnet18.a1_in1k) | 11,175,428 | AdamW lr 0.001, wd 0.0001, batch 64, cosine schedule, max 25 epochs, early stopping (patience 6), aug. vflip, weighted loss | 19/25 · 6/13 · 11/18 | 1386 ± 456 | 13.369 ± 1.311 |

Random forest: 500 trees; its size (number of nodes) is not comparable to network parameters. Best epoch is 0-based; seeds 0 · 1 · 2. The tuning budget was 12 configurations per model (val).
