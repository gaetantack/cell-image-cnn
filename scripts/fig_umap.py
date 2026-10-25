"""RQ4: UMAP of the VALIDATION cells in three feature spaces, coloured by the 7 phases.

1. Random forest features: the 28 hand-crafted features, median-imputed and standardised with
   statistics from the TRAINING cells.
2. Best SmallCNN (4-class final run, seed 0): penultimate-layer embedding (model.embed).
3. Best ResNet18 (pretrained or scratch, whichever has the higher mean 4-class val macro-F1), seed 0.
The networks never saw the 7-phase labels, only 4 classes. Phases are coloured in cell cycle order
(G1 -> S -> G2 -> pro -> meta -> ana -> telo) with an ordered colour map, mitotic cells drawn on top.
Writes figures/umap_val.png/.pdf and results/umap_val.npz (coordinates).
"""
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from scripts.make_results_table import final_runs
from scripts.run_rf import load_features
from src import config
from src.data import load_dataset
from src.experiments import read_log
from src.interpret import embeddings, load_trained
from src.plotting import CM, MODEL_LABELS, TWO_COLUMNS, save, set_style
from src.splits import load_split


def best_resnet(log):
    scores = {m: final_runs(log, m, 4)["macro_f1"].mean() for m in ("resnet18_pretrained", "resnet18_scratch")
              if len(final_runs(log, m, 4)) == len(config.SEEDS)}
    return max(scores, key=scores.get) if scores else None


log = read_log()
train, val = load_split("train"), load_split("val")
X_img, y7, _ = load_dataset(7)
y_val = y7[val]

spaces = {}
F_all, _ = load_features(7)
imputer = SimpleImputer(strategy="median").fit(F_all[train])  # medians from training cells only
scaler = StandardScaler().fit(imputer.transform(F_all[train]))  # mean and SD from training cells only
spaces["Random forest features (28)"] = scaler.transform(imputer.transform(F_all[val]))
for model in ("smallcnn", best_resnet(log)):
    if model is None or not (config.CHECKPOINT_DIR / f"{model}_c4_final_s0_best.pt").exists():
        print(f"skip {model}: no final 4-class seed-0 checkpoint")
        continue
    net, mean, sd = load_trained(f"{model}_c4_final_s0")
    spaces[f"{MODEL_LABELS[model]} embedding"] = embeddings(net, X_img[val], mean, sd)

set_style()
colors = plt.get_cmap("viridis")(np.linspace(0, 0.95, len(config.CLASSES_7)))
fig, axes = plt.subplots(1, len(spaces), figsize=(TWO_COLUMNS, 6.5 * CM), squeeze=False)
coords = {}
for ax, (name, Z) in zip(axes[0], spaces.items()):
    xy = umap.UMAP(n_neighbors=config.UMAP_N_NEIGHBORS, min_dist=config.UMAP_MIN_DIST,
                   random_state=config.UMAP_SEED).fit_transform(Z)
    coords[name] = xy
    for c, phase in enumerate(config.CLASSES_7):  # G1 first, so rare mitotic phases end on top
        sel = y_val == c
        ax.scatter(xy[sel, 0], xy[sel, 1], s=1.5 if c < 3 else 6, color=colors[c], linewidths=0,
                   label=f"{phase} (n={sel.sum()})", rasterized=True)
    ax.set_title(f"{name}\n(dimension {Z.shape[1]})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, markerscale=3)
fig.suptitle("UMAP of validation cells, coloured by cell cycle phase (networks trained on 4 classes)",
             fontsize=7, y=1.03)
save(fig, "umap_val")
np.savez(config.RESULTS_DIR / "umap_val.npz", rows=val, y7=y_val,
         **{name.split(" ")[0].lower() + f"_{i}": xy for i, (name, xy) in enumerate(coords.items())})
