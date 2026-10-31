"""Run 2, Phase A figure (RQ3): SmallCNN, augmentation {none, vflip, basic} x balancing {weighted, oversample},
seeds 0-2, VALIDATION. Left: macro-F1, right: mitosis recall. Small markers = single seeds, large marker
with bar = mean ± SD. Weighted loss = filled circles, oversampling = open squares (shape, not colour, tells
the balancing apart). Reads results/run2_phaseA_val.csv (made by scripts/choose_aug_bal.py) for the chosen
setting and results/experiments.csv for the single seeds. Writes figures/phaseA_val_run2.png/.pdf.
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.experiments import read_log
from src.plotting import CM, INK, MUTED, ONE_COLUMN, save, set_style

MARKERS = {"weighted": ("o", INK), "oversample": ("s", "white")}  # marker, face colour
OFFSET = {"weighted": -0.15, "oversample": 0.15}

log = read_log()
choice = json.loads(config.RUN2_CHOICE_FILE.read_text())
set_style()
fig, axes = plt.subplots(1, 2, figsize=(ONE_COLUMN * 1.3, 5 * CM))
for ax, (metric, title) in zip(axes, (("macro_f1", "Macro-F1 (val)"), ("mitosis", "Mitosis recall (val)"))):
    for i, aug in enumerate(config.RUN2_AUGMENTS):
        for bal in config.RUN2_BALANCES:
            ids = [f"smallcnn_c4_{config.RUN2_TAG}A_{aug}_{bal}_s{s}" for s in config.SEEDS]
            runs = log[log["run_id"].isin(ids)]
            if len(runs) == 0:
                continue
            if metric == "macro_f1":
                v = runs["macro_f1"].to_numpy(dtype=float)
            else:
                v = np.array([json.loads(r)["Mitosis"] for r in runs["recall"]])
            x = i + OFFSET[bal]
            marker, face = MARKERS[bal]
            ax.scatter(np.full(len(v), x + 0.07), v, s=6, marker=marker, facecolor=face, edgecolor=MUTED, lw=0.5)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, fmt=marker, ms=4.5,
                        mfc=face, mec=INK, ecolor=INK, elinewidth=0.8, capsize=2,
                        label=f"{bal}" if i == 0 else None)
            if (aug, bal) == (choice["augment"], choice["balance"]):
                ax.annotate("chosen", (x, v.mean()), xytext=(0, -14), textcoords="offset points",
                            ha="center", fontsize=6, color=INK)
    ax.set_xticks(range(len(config.RUN2_AUGMENTS)), config.RUN2_AUGMENTS)
    ax.set_xlim(-0.5, len(config.RUN2_AUGMENTS) - 0.5)
    ax.set_xlabel("Augmentation")
    ax.set_title(title)
axes[0].legend(title="Balancing", frameon=False, loc="lower left")
fig.suptitle("SmallCNN, augmentation × balancing, validation, mean ± SD over 3 seeds (small marks: seeds)",
             fontsize=7)
fig.tight_layout()
save(fig, "phaseA_val_run2")
