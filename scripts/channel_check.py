"""Fig: 3 random cells per class, Ch3 and Ch6 side by side, to check which is brightfield / darkfield."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.data import load_dataset

N_PER_CLASS = 3
SEED = 0

X, y, cell_ids = load_dataset(7)
rng = np.random.default_rng(SEED)

fig, axes = plt.subplots(len(config.CLASSES_7), 2 * N_PER_CLASS, figsize=(2 * N_PER_CLASS * 1.3, len(config.CLASSES_7) * 1.4))
for label, cls in enumerate(config.CLASSES_7):
    picks = rng.choice(np.flatnonzero(y == label), N_PER_CLASS, replace=False)
    for k, i in enumerate(picks):
        for c, ch in enumerate(config.INPUT_CHANNELS):
            ax = axes[label, 2 * k + c]
            ax.imshow(X[i, c], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{ch} · id {cell_ids[i]}", fontsize=6)
            if k == 0 and c == 0:
                ax.set_ylabel(cls, fontsize=8)

fig.suptitle("Raw Ch3 vs Ch6 (shared 0-255 grey scale), 3 random cells per class", fontsize=9)
fig.tight_layout()
config.FIGURES_DIR.mkdir(exist_ok=True)
out = config.FIGURES_DIR / "channel_check.png"
fig.savefig(out, dpi=200)
print(f"saved {out.relative_to(config.ROOT)}")
