"""RQ4 (run 4): how well do the true 4-class phases separate in each feature space? Silhouette score, val cells.

For one cell: a = mean Euclidean distance to the other cells of its own class, b = mean distance to the cells
of the nearest other class; s = (b - a) / max(a, b), from -1 (closer to another class) to +1 (well inside
its own class). The score is the mean s over all VALIDATION cells (sklearn silhouette_score, Euclidean).
Spaces (same as scripts/fig_umap.py, but here the HIGH-DIMENSIONAL vectors, never UMAP's 2-D coordinates,
because UMAP distorts distances):
  - the 28 hand-crafted `main` features, median-imputed and standardised with TRAINING medians / mean / SD
    (one value: no training seed involved)
  - SmallCNN embedding (penultimate layer, 256-D), run-2 finals n2_final, seeds 0-2
  - ResNet18 (ImageNet) embedding (512-D), run-2 finals n2_final, seeds 0-2
Embeddings are used as they come out of the network (not standardised). Runs on the CPU.
Writes results/silhouette_val.csv and results/silhouette_val.md. VALIDATION only.
Run: python -m scripts.silhouette
"""
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scripts.run_rf import load_features
from src import config
from src.data import load_dataset
from src.interpret import embeddings, load_trained
from src.plotting import MODEL_LABELS
from src.splits import load_split

torch.set_num_threads(4)
train, val = load_split("train"), load_split("val")
X_img, y4, _ = load_dataset(4)
y_val = y4[val]

rows = []
F, _ = load_features(4)
imputer = SimpleImputer(strategy="median").fit(F[train])  # training cells only
scaler = StandardScaler().fit(imputer.transform(F[train]))
Z = scaler.transform(imputer.transform(F[val]))
rows.append({"space": "hand-crafted features (28, standardised)", "seed": "-", "dim": Z.shape[1],
             "silhouette": silhouette_score(Z, y_val, metric="euclidean")})
print(rows[-1], flush=True)
for model in ("smallcnn", "resnet18_pretrained"):
    for seed in config.SEEDS:
        run_id = f"{model}_c4_n2_final_s{seed}"
        net, mean, sd = load_trained(run_id, device=torch.device("cpu"))
        Z = embeddings(net, X_img[val], mean, sd)
        rows.append({"space": f"{MODEL_LABELS[model]} embedding", "seed": seed, "dim": Z.shape[1],
                     "silhouette": silhouette_score(Z, y_val, metric="euclidean")})
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv(config.RESULTS_DIR / "silhouette_val.csv", index=False)
lines = ["# Silhouette score of the true 4-class labels, validation cells (Euclidean, high-dimensional space)", "",
         "s = (b − a) / max(a, b) per cell, a = mean distance to its own class, b = to the nearest other class; "
         "mean over 4,840 val cells. Range −1 to 1; 0 = classes overlap. Made by `scripts/silhouette.py`.", "",
         "| Space | Dimension | n seeds | Silhouette (val) |", "|---|---|---|---|"]
for space, g in df.groupby("space", sort=False):
    v = g["silhouette"].to_numpy()
    val_str = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}"
    lines.append(f"| {space} | {g['dim'].iloc[0]} | {len(v)} | {val_str} |")
md = "\n".join(lines) + "\n"
(config.RESULTS_DIR / "silhouette_val.md").write_text(md)
print(md)
