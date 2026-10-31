"""RQ3 ablation (PRELIMINARY, seed 0 only): SmallCNN with augment {none, basic} x balance {none, weighted,
oversample}, best tuned learning rate / weight decay / batch size. The basic + weighted cell is the final run.
Writes results/ablation_val.csv/.md and figures/ablation_val.png/.pdf. VALIDATION numbers only.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.experiments import read_log
from src.plotting import CM, ONE_COLUMN, save, set_style

AUGMENTS, BALANCES = ("none", "basic"), ("none", "weighted", "oversample")

log = read_log().set_index("run_id")
rows = []
for aug in AUGMENTS:
    for bal in BALANCES:
        run_id = "smallcnn_c4_final_s0" if (aug, bal) == ("basic", "weighted") else f"smallcnn_c4_abl_{aug}_{bal}_s0"
        if run_id not in log.index:
            continue
        r = log.loc[run_id]
        recall = json.loads(r["recall"])
        rows.append({"augment": aug, "balance": bal, "run_id": run_id, "macro_f1": r["macro_f1"],
                     "balanced_accuracy": r["balanced_accuracy"], "recall_mitosis": recall["Mitosis"],
                     "recall_G1": recall["G1"], "recall_S": recall["S"], "recall_G2": recall["G2"]})
table = pd.DataFrame(rows)
table.to_csv(config.RESULTS_DIR / "ablation_val.csv", index=False)
md = ["# RQ3 ablation — PRELIMINARY (seed 0 only), SmallCNN, 4 classes, validation", "",
      "| Augment | Balance | Macro-F1 (val) | Balanced acc. (val) | Recall mitosis | Recall G1 | Recall S | Recall G2 |",
      "|---|---|---|---|---|---|---|---|"]
md += [f"| {r.augment} | {r.balance} | {r.macro_f1:.3f} | {r.balanced_accuracy:.3f} | {r.recall_mitosis:.2f} | "
       f"{r.recall_G1:.2f} | {r.recall_S:.2f} | {r.recall_G2:.2f} |" for r in table.itertuples()]
(config.RESULTS_DIR / "ablation_val.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

set_style()
fig, axes = plt.subplots(1, 2, figsize=(ONE_COLUMN, 4.2 * CM), sharey=True)
for ax, (col, title) in zip(axes, (("macro_f1", "Macro-F1"), ("recall_mitosis", "Mitosis recall"))):
    grid = np.full((len(AUGMENTS), len(BALANCES)), np.nan)
    for r in table.itertuples():
        grid[AUGMENTS.index(r.augment), BALANCES.index(r.balance)] = getattr(r, col)
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(AUGMENTS)):
        for j in range(len(BALANCES)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}" if col == "macro_f1" else f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="white" if grid[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(BALANCES)), ["none", "weight", "over-\nsample"])
    ax.tick_params(length=0)
    ax.set_xlabel("Class balancing")
    ax.set_title(f"{title} (val)")
    ax.spines[:].set_visible(False)
axes[0].set_yticks(range(len(AUGMENTS)), AUGMENTS)
axes[0].set_ylabel("Augmentation")
fig.suptitle("SmallCNN ablation, validation, seed 0 only (preliminary)", fontsize=7)
fig.tight_layout()
save(fig, "ablation_val")
