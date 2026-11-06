"""Fig 1: example cells and class counts (setup figure). Writes figures/fig1_cells_val.png/.pdf.

A: two VALIDATION cells per phase (seeded random choice), brightfield (Ch3) above darkfield (Ch6), for the
   three interphase classes and the mitotic sub-phases that are merged into "Mitosis" in the 4-class task.
B: number of cells per 7-class phase in train / val / test (log scale). The test counts are LABEL COUNTS
   ONLY: the test row indices are read from splits/splits.json and only their labels are counted;
   no test image is loaded or shown.
Run: python -m scripts.fig1_cells
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from src import config
from src.data import load_dataset
from src.plotting import CM, INK, MUTED, TWO_COLUMNS, save, set_style
from src.splits import load_split

N_PER_PHASE = 2
set_style()
X, y7, cell_ids = load_dataset(7)
val = load_split("val")
split_rows = json.loads(config.SPLITS_FILE.read_text())  # indices only; used for label counts
counts = {s: np.bincount(y7[np.array(split_rows[s])], minlength=7) for s in ("train", "val", "test")}

rng = np.random.default_rng(0)
fig = plt.figure(figsize=(TWO_COLUMNS, 13 * CM))
gs = GridSpec(2 * N_PER_PHASE + 2, 7, figure=fig, height_ratios=[1] * (2 * N_PER_PHASE) + [0.25, 1.8])
for k, phase in enumerate(config.CLASSES_7):
    chosen = rng.choice(val[y7[val] == k], N_PER_PHASE, replace=False)
    for n, row in enumerate(chosen):
        for c, ch in enumerate(("BF", "DF")):
            ax = fig.add_subplot(gs[2 * n + c, k])
            ax.imshow(X[row, c], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([])
            ax.spines[:].set_visible(False)
            if n == 0 and c == 0:
                ax.set_title(f"\n{phase}" if k < 3 else f"Mitosis:\n{phase}", fontsize=7)
            if k == 0:
                ax.set_ylabel(ch, rotation=0, labelpad=8, va="center")
fig.text(0.01, 0.97, "A", fontweight="bold")

ax = fig.add_subplot(gs[-1, :])
x = np.arange(7)
for i, (s, color) in enumerate(zip(("train", "val", "test"), (INK, MUTED, "#b9b8b3"))):
    bars = ax.bar(x + (i - 1) * 0.27, counts[s], width=0.27, color=color, label=f"{s} (n={counts[s].sum():,})")
    for b, v in zip(bars, counts[s]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.1, str(v), ha="center", va="bottom", fontsize=5)
ax.set_yscale("log")
ax.set_ylim(1, 3e4)
ax.set_xticks(x, config.CLASSES_7)
ax.set_ylabel("Cells (log scale)")
ax.legend(frameon=False, ncol=3, loc="upper right")
ax.text(-0.06, 1.02, "B", transform=ax.transAxes, fontweight="bold")
fig.tight_layout()
save(fig, "fig1_cells_val")
print({s: c.tolist() for s, c in counts.items()})
