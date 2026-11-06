"""Figure figures/masking_examples_val.png/.pdf: what the masked inputs look like (run 4, RQ5).

One VALIDATION cell per 4-class phase (chosen at random with a fixed seed among cells whose dilated
mask is at least config.TINY_MASK_AREA_PX px), shown as original / cell only / background only,
for brightfield (Ch3) and darkfield (Ch6). The dilated mask outline is drawn on the original images.
The fill value comes from the TRAINING cells (src.masking.background_fill).
Run: python -m scripts.fig_masking_examples
"""
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.data import load_dataset
from src.masking import apply_mask, background_fill, load_masks
from src.plotting import CM, TWO_COLUMNS, save, set_style
from src.splits import load_split

set_style()
X, y4, cell_ids = load_dataset(4)
train, val = load_split("train"), load_split("val")
fill = background_fill(X[train], load_masks(train), np.arange(len(train)))
val_masks = load_masks(val)

rng = np.random.default_rng(0)
modes = ("original", "cell_only", "background_only")
titles = {"original": "original", "cell_only": "cell only", "background_only": "background only"}
fig, axes = plt.subplots(4, 6, figsize=(TWO_COLUMNS, 12 * CM))
for k, cls in enumerate(config.CLASSES_4):
    candidates = np.flatnonzero((y4[val] == k) & (val_masks.sum(axis=(1, 2)) >= config.TINY_MASK_AREA_PX))
    i = rng.choice(candidates)
    img, mask = X[val[i]][None], val_masks[i][None]
    for m, mode in enumerate(modes):
        out = apply_mask(img, mask, mode, fill)[0]
        for c, ch in enumerate(("brightfield", "darkfield")):
            ax = axes[k, c * 3 + m]
            ax.imshow(out[c], cmap="gray", vmin=0, vmax=255)
            if mode == "original":
                ax.contour(mask[0], levels=[0.5], colors="#eb6834", linewidths=0.5)
            ax.set_xticks([]), ax.set_yticks([])
            if k == 0:
                ax.set_title(f"{ch}\n{titles[mode]}")
    axes[k, 0].set_ylabel(f"{cls}\ncell {cell_ids[val[i]]}")
fig.suptitle(f"Masked inputs (val cells). Mask = brightfield cell mask dilated by {config.MASK_DILATION_PX} px; "
             f"fill = training background median (BF {fill[0]}, DF {fill[1]})", y=0.995)
fig.tight_layout()
save(fig, "masking_examples_val")
