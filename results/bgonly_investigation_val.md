# Why is background-only above chance? Simple per-cell numbers (val, 4 classes)

Fitted on train, scored on val. Features from the dilated brightfield mask: log mask area (the size of the filled hole), median and SD of the pixels outside the mask per channel. Made by `scripts/bgonly_investigation.py`.

Median per class (training cells):

| cls | log area | BF bg median | DF bg median | BF bg SD | DF bg SD |
|---|---|---|---|---|---|
| G1 | 6.59 | 92.0 | 45.0 | 2.78 | 3.05 |
| S | 6.73 | 93.0 | 37.0 | 2.66 | 2.46 |
| G2 | 6.86 | 97.0 | 32.0 | 2.66 | 2.18 |
| Mitosis | 6.94 | 98.0 | 32.5 | 2.61 | 2.23 |

| classifier | features | macro_f1 | balanced_accuracy | recall_G1 | recall_S | recall_G2 | recall_Mitosis |
|---|---|---|---|---|---|---|---|
| logreg | area | 0.485 | 0.565 | 0.76 | 0.52 | 0.4 | 0.59 |
| logreg | bg level (BF, DF) | 0.265 | 0.307 | 0.48 | 0.21 | 0.19 | 0.35 |
| logreg | bg level + spread | 0.27 | 0.301 | 0.49 | 0.2 | 0.22 | 0.29 |
| logreg | area + bg level | 0.488 | 0.569 | 0.76 | 0.53 | 0.39 | 0.59 |
| logreg | area + bg level + spread | 0.487 | 0.568 | 0.76 | 0.53 | 0.39 | 0.59 |
| rf | area | 0.484 | 0.555 | 0.74 | 0.51 | 0.43 | 0.54 |
| rf | bg level (BF, DF) | 0.28 | 0.308 | 0.38 | 0.28 | 0.34 | 0.23 |
| rf | bg level + spread | 0.309 | 0.318 | 0.46 | 0.25 | 0.48 | 0.08 |
| rf | area + bg level | 0.54 | 0.561 | 0.77 | 0.53 | 0.64 | 0.31 |
| rf | area + bg level + spread | 0.554 | 0.557 | 0.81 | 0.56 | 0.7 | 0.16 |
