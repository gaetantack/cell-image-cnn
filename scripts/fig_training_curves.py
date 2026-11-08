"""Supplementary figure: training curves of the run-2 final runs (4 classes, n2_final, seeds 0-2).

Top row: validation macro-F1 per epoch; bottom row: training loss per epoch (mean weighted cross-entropy over
the epoch's batches); one column per CNN family, one line per seed, the best epoch (the weights that were kept)
marked with a dot. Validation loss was not logged during training (logs/history/<run_id>.json holds only
train_loss and val_macro_f1), so it cannot be shown without re-running.
Writes figures/training_curves_val_run2.png/.pdf. VALIDATION numbers.
Run: python -m scripts.fig_training_curves
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.plotting import CM, MODEL_COLORS, MODEL_LABELS, TWO_COLUMNS, save, set_style

FAMILIES = ("smallcnn", "resnet18_pretrained", "resnet18_scratch")
STYLES = ("-", "--", ":")

set_style()
fig, axes = plt.subplots(2, 3, figsize=(TWO_COLUMNS, 10 * CM), sharex=True)
for c, model in enumerate(FAMILIES):
    for seed, ls in zip(config.SEEDS, STYLES):
        h = json.loads((config.LOG_DIR / "history" / f"{model}_c4_n2_final_s{seed}.json").read_text())
        ep = np.array([e["epoch"] for e in h])
        f1 = np.array([e["val_macro_f1"] for e in h])
        loss = np.array([e["train_loss"] for e in h])
        best = int(np.argmax(f1))  # same rule as src.train: first epoch with the highest val macro-F1
        color = MODEL_COLORS[model]
        axes[0, c].plot(ep, f1, ls=ls, color=color, lw=0.8, label=f"seed {seed} (best epoch {ep[best]})")
        axes[0, c].plot(ep[best], f1[best], "o", color=color, ms=3)
        axes[1, c].plot(ep, loss, ls=ls, color=color, lw=0.8)
        axes[1, c].plot(ep[best], loss[best], "o", color=color, ms=3)
    axes[0, c].set_title(MODEL_LABELS[model])
    axes[0, c].legend(frameon=False, loc="lower right", fontsize=6)
    axes[0, c].set_ylim(0.3, 0.85)
    axes[1, c].set_xlabel("Epoch (0-based)")
axes[0, 0].set_ylabel("Val macro-F1")
axes[1, 0].set_ylabel("Training loss")
fig.suptitle("Run-2 final runs (4 classes): val macro-F1 and training loss per epoch; dot = kept (best) epoch", fontsize=7)
fig.tight_layout()
save(fig, "training_curves_val_run2")
