# Silhouette score of the true 4-class labels, validation cells (Euclidean, high-dimensional space)

s = (b − a) / max(a, b) per cell, a = mean distance to its own class, b = to the nearest other class; mean over 4,840 val cells. Range −1 to 1; 0 = classes overlap. Made by `scripts/silhouette.py`.

| Space | Dimension | n seeds | Silhouette (val) |
|---|---|---|---|
| hand-crafted features (28, standardised) | 28 | 1 | 0.006 |
| SmallCNN embedding | 256 | 3 | 0.150 ± 0.003 |
| ResNet18 (ImageNet) embedding | 512 | 3 | 0.164 ± 0.017 |
