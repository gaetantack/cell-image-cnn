"""Fig: 20 random training cells with their brightfield mask outline, to check the segmentation."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.data import load_dataset
from src.features import cell_mask
from src.splits import load_split

N_CELLS = 20
N_COLS = 5
SEED = 0

X, y7, cell_ids = load_dataset(7)
rng = np.random.default_rng(SEED)
picks = rng.choice(load_split("train"), N_CELLS, replace=False)

n_rows = N_CELLS // N_COLS
fig, axes = plt.subplots(n_rows, 2 * N_COLS, figsize=(2 * N_COLS * 1.2, n_rows * 1.4))
for k, i in enumerate(picks):
    mask, _ = cell_mask(X[i, 0])
    r, col = divmod(k, N_COLS)
    for c, ch in enumerate(config.INPUT_CHANNELS):
        ax = axes[r, 2 * col + c]
        ax.imshow(X[i, c], cmap="gray", vmin=0, vmax=255)
        ax.contour(mask, levels=[0.5], colors="#eb6834", linewidths=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{ch} · {config.CLASSES_7[y7[i]]} · {cell_ids[i]}", fontsize=5)

fig.suptitle(f"Brightfield Otsu mask (orange) on Ch3 and Ch6, {N_CELLS} random training cells (seed {SEED})", fontsize=8)
fig.tight_layout()
config.FIGURES_DIR.mkdir(exist_ok=True)
out = config.FIGURES_DIR / "mask_check.png"
fig.savefig(out, dpi=200)
print(f"saved {out.relative_to(config.ROOT)}")
