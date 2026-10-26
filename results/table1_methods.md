# Table 1. Methods overview

Training time: wall-clock seconds per final 4-class run (mean ± SD over seeds 0–2; CNNs on Apple M2 GPU (MPS), random forest on 8 CPU cores), including the per-epoch validation pass. BF = brightfield (Ch3), DF = darkfield (Ch6). Chance baselines (majority, stratified) have no input features.

| Model | Input | Pretrained | Parameters (4 classes) | Training time (s) |
|---|---|---|---|---|
| Random forest | 28 hand-crafted features (BF mask: shape, BF intensity, BF + DF texture, DF spread) | — | — | 7 ± 0 |
| SmallCNN | BF + DF image, 2 × 66 × 66 | No | 389,636 | 183 ± 3 |
| ResNet18 (scratch) | BF + DF image, upscaled to 2 × 128 × 128 | No | 11,175,428 | 900 ± 7 |
| ResNet18 (ImageNet) | BF + DF image, upscaled to 2 × 128 × 128 | ImageNet (timm resnet18.a1_in1k) | 11,175,428 | 928 ± 18 |

Random forest: 500 trees; its size (number of nodes) is not comparable to network parameters.
