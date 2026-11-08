"""Biological sanity check (plan week 6; run 4): do size/shape and the models follow known cell-cycle biology?

Per phase, VALIDATION cells:
- median mask area (px) and eccentricity (0 = circle, 1 = line) from the hand-crafted feature table
  (src.features on the brightfield mask; data/features.parquet). Expected: cells grow G1 -> S -> G2,
  mitotic cells round up (low eccentricity) except anaphase/telophase (two daughters -> elongated mask).
- per-class recall (4 classes) of the run-2 SmallCNN and ResNet18 (ImageNet), mean over seeds 0-2 (experiments.csv)
- PC1 pseudotime (same method as scripts/pseudotime.py: standardise and PCA fitted on TRAIN cells, sign so that
  train G1 is lowest), as the median percentile rank (0-1) of the cells of each phase, for the hand-crafted
  features and the ResNet18 (ImageNet) embedding (n2_final seed 0). Expected order G1 < S < G2 < mitosis.
Description only. Writes results/bio_check_val.md. Run: python -m scripts.bio_check
"""
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from scripts.make_results_table import final_runs
from scripts.run_rf import load_features
from src import config
from src.data import load_dataset
from src.experiments import read_log
from src.interpret import embeddings, load_trained
from src.splits import load_split

torch.set_num_threads(4)
train, val = load_split("train"), load_split("val")
X_img, y7, _ = load_dataset(7)
y4 = np.asarray(config.MAP_7_TO_4)[y7]
feat = pd.read_parquet(config.FEATURES_FILE).sort_values("row").set_index("row")


def pc1_percentile(Z_train, Z_val, y_train):
    """PC1 coordinate of val cells (fitted on train), sign so that train G1 is lowest, as percentile rank 0-1."""
    scaler = StandardScaler().fit(Z_train)
    pca = PCA(n_components=1, random_state=0).fit(scaler.transform(Z_train))
    t_train = pca.transform(scaler.transform(Z_train))[:, 0]
    t_val = pca.transform(scaler.transform(Z_val))[:, 0]
    if t_train[y_train == 0].mean() > t_train.mean():
        t_val = -t_val
    return (rankdata(t_val) - 1) / (len(t_val) - 1)


F, _ = load_features(7)
imp = SimpleImputer(strategy="median").fit(F[train])
pt_hand = pc1_percentile(imp.transform(F[train]), imp.transform(F[val]), y7[train])
net, mean, sd = load_trained("resnet18_pretrained_c4_n2_final_s0", device=torch.device("cpu"))
pt_resnet = pc1_percentile(embeddings(net, X_img[train], mean, sd), embeddings(net, X_img[val], mean, sd), y7[train])

log = read_log()
recall = {}
for model in ("smallcnn", "resnet18_pretrained"):
    r = pd.DataFrame([json.loads(x) for x in final_runs(log, model, 4, "n2_final")["recall"]])
    recall[model] = r.mean()

rows = []
groups = [(c, y4[val] == k) for k, c in enumerate(config.CLASSES_4)]
groups += [(f"  {c}", y7[val] == k) for k, c in enumerate(config.CLASSES_7) if k >= 3]  # mitotic sub-phases
for name, sel in groups:
    cls4 = name.strip() if name.strip() in config.CLASSES_4 else None
    f = feat.loc[val[sel]]
    rows.append({"Phase": name, "n val": int(sel.sum()), "Median area (px)": f"{f['area'].median():.0f}",
                 "Median eccentricity": f"{f['eccentricity'].median():.2f}",
                 "Recall SmallCNN": f"{recall['smallcnn'][cls4]:.2f}" if cls4 else "",
                 "Recall ResNet18 (ImageNet)": f"{recall['resnet18_pretrained'][cls4]:.2f}" if cls4 else "",
                 "PC1 pseudotime, hand-crafted (median pct.)": f"{np.median(pt_hand[sel]):.2f}",
                 "PC1 pseudotime, ResNet18 emb. (median pct.)": f"{np.median(pt_resnet[sel]):.2f}"})
table = pd.DataFrame(rows)
cols = list(table.columns)
md = ["# Biological check (val): size, shape, recall and pseudotime per phase", "",
      "Description only. Area/eccentricity: brightfield mask (hand-crafted feature table; the empty-mask cell has no value). "
      "Recall: run-2 finals, mean over seeds 0–2. Pseudotime: PC1 of the standardised space (fitted on train), "
      "median percentile rank of the phase's val cells (0 = earliest, 1 = latest). Indented rows = mitotic sub-phases "
      "(merged into Mitosis for the 4-class models). Made by `scripts/bio_check.py`.", "",
      "| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
md += ["| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False)]
text = "\n".join(md) + "\n"
(config.RESULTS_DIR / "bio_check_val.md").write_text(text)
print(text)
