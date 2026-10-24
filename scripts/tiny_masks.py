"""Fig: every train/val cell with a tiny or empty brightfield mask, mask outline on Ch3 and Ch6, smallest first."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import load_dataset
from src.features import cell_mask
from src.splits import load_split

N_COLS = 8  # cells per figure row; each cell takes two panels

X, y7, cell_ids = load_dataset(7)
df = pd.read_parquet(config.FEATURES_FILE)
df = df.iloc[np.concatenate([load_split("train"), load_split("val")])]
tiny = df[(df["area"] < config.TINY_MASK_AREA_PX) | df["area"].isna()].sort_values("area", na_position="first")

n_rows = -(-len(tiny) // N_COLS)
fig, axes = plt.subplots(n_rows, 2 * N_COLS, figsize=(2 * N_COLS * 1.1, n_rows * 1.3))
for ax in axes.ravel():
    ax.set_axis_off()
for k, (i, area) in enumerate(zip(tiny["row"], tiny["area"])):
    mask, _ = cell_mask(X[i, 0])
    r, col = divmod(k, N_COLS)
    for c, ch in enumerate(config.INPUT_CHANNELS):
        ax = axes[r, 2 * col + c]
        ax.imshow(X[i, c], cmap="gray", vmin=0, vmax=255)
        if mask.any():
            ax.contour(mask, levels=[0.5], colors="#eb6834", linewidths=0.8)
        area_txt = "empty" if np.isnan(area) else f"{area:.0f} px"
        ax.set_title(f"{ch} · {config.CLASSES_7[y7[i]]} · {cell_ids[i]} · {area_txt}", fontsize=4.5)

fig.suptitle(f"Train/val cells with mask area < {config.TINY_MASK_AREA_PX} px or empty (n={len(tiny)}), "
             f"mask outline in orange, smallest first", fontsize=8)
fig.tight_layout()
config.FIGURES_DIR.mkdir(exist_ok=True)
out = config.FIGURES_DIR / "tiny_masks.png"
fig.savefig(out, dpi=200)
print(f"{len(tiny)} cells; saved {out.relative_to(config.ROOT)}")
