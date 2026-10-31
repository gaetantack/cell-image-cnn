"""RQ4 (Tier 3, simple version): does one direction in feature space order the cells along the cell cycle?

For each feature space (hand-crafted features, SmallCNN embedding, best ResNet18 embedding; validation cells):
1. Standardise the features (mean and SD from the TRAINING cells) and fit PCA on the TRAINING cells.
2. Pseudotime of a validation cell = its coordinate on the first principal component (PC1). The sign is
   chosen so that the training G1 cells have the lowest mean (the sign of a PC is arbitrary).
3. Spearman correlation between pseudotime and phase order (G1=0, S=1, G2=2, pro=3, meta=4, ana=5, telo=6)
   on the validation cells. 1 = perfect ordering, 0 = no ordering.
PC1 is the simplest possible "trajectory"; a real pseudotime method (e.g. diffusion pseudotime) could do
better. Writes results/pseudotime_val.csv and figures/pseudotime_val.png/.pdf. VALIDATION numbers only.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from scripts.make_results_table import best_resnet
from scripts.run_rf import load_features
from src import config
from src.data import load_dataset
from src.experiments import read_log
from src.interpret import embeddings, load_trained
from src.plotting import CM, MODEL_LABELS, TWO_COLUMNS, save, set_style
from src.splits import load_split

log = read_log()
train, val = load_split("train"), load_split("val")
X_img, y7, _ = load_dataset(7)

spaces = {}
F, _ = load_features(7)
F = SimpleImputer(strategy="median").fit(F[train]).transform(F)  # medians from training cells only
spaces["Hand-crafted features"] = F
for model in ("smallcnn", best_resnet(log)):
    if model is None or not (config.CHECKPOINT_DIR / f"{model}_c4_final_s0_best.pt").exists():
        continue
    net, mean, sd = load_trained(f"{model}_c4_final_s0")
    rows = np.concatenate([train, val])
    Z = np.zeros((len(X_img), net.embed(mean.new_zeros(1, 2, 66, 66)).shape[1]), np.float32)
    Z[rows] = embeddings(net, X_img[rows], mean, sd)
    spaces[f"{MODEL_LABELS[model]} embedding"] = Z

results, pseudo = [], {}
for name, Z in spaces.items():
    scaler = StandardScaler().fit(Z[train])
    pca = PCA(n_components=1, random_state=0).fit(scaler.transform(Z[train]))
    t_train = pca.transform(scaler.transform(Z[train]))[:, 0]
    sign = -1 if t_train[y7[train] == 0].mean() > t_train.mean() else 1
    t_val = sign * pca.transform(scaler.transform(Z[val]))[:, 0]
    rho = spearmanr(t_val, y7[val]).statistic
    rho_4 = spearmanr(t_val, np.asarray(config.MAP_7_TO_4)[y7[val]]).statistic
    pseudo[name] = t_val
    results.append({"space": name, "spearman_7_phases": round(rho, 3), "spearman_4_classes": round(rho_4, 3),
                    "pc1_variance_explained": round(pca.explained_variance_ratio_[0], 3)})
table = pd.DataFrame(results)
table.to_csv(config.RESULTS_DIR / "pseudotime_val.csv", index=False)
print(table.to_string(index=False))

set_style()
colors = plt.get_cmap("viridis")([0.0, 0.22, 0.45, 0.62, 0.75, 0.87, 0.97])
fig, axes = plt.subplots(1, len(pseudo), figsize=(TWO_COLUMNS, 5 * CM), squeeze=False)
for ax, (name, t) in zip(axes[0], pseudo.items()):
    data = [t[y7[val] == c] for c in range(len(config.CLASSES_7))]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.8)
    ax.scatter(range(1, 8), [np.median(d) for d in data], s=8, color="white", edgecolor="black", lw=0.5, zorder=3)
    ax.set_xticks(range(1, 8), ["G1", "S", "G2", "Pro", "Meta", "Ana", "Telo"])
    rho = table.set_index("space").loc[name, "spearman_7_phases"]
    ax.set_title(f"{name}\nSpearman ρ = {rho:.2f}")
axes[0][0].set_ylabel("PC1 pseudotime (validation cells)")
fig.suptitle("Pseudotime = first principal component; violins per phase, dots = median (validation)",
             fontsize=7, y=1.12)
save(fig, "pseudotime_val")
