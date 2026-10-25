"""RQ5: Grad-CAM for the best CNN (4-class final run, seed 0) on VALIDATION cells.

Figure figures/gradcam_val.png/.pdf: for each of the 7 phases, GRADCAM_CELLS_PER_CLASS randomly chosen
(seeded) validation cells. Per cell: brightfield with the Grad-CAM map on top, then brightfield and
darkfield alone. The map is computed for the cell's TRUE 4-class label.

Numbers (results/gradcam_val_summary.csv), for ALL validation cells:
- heat_in_mask: share of the heat inside the brightfield cell mask (src.features.cell_mask),
  next to mask_area_share, the share of the image the mask covers (what uniform heat would give);
- heat_in_border: share of the heat in the outer GRADCAM_BORDER_PX pixels, next to the ring's area share.
Also figures/gradcam_cell30439_train.png: training cell 30439 (corner artefact, PROJECT.md Observations).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scripts.fig_confusion import best_cnn
from src import config
from src.data import load_dataset
from src.experiments import read_log
from src.features import cell_mask
from src.interpret import border_fraction, grad_cam, load_trained, mask_fraction
from src.plotting import CM, MODEL_LABELS, TWO_COLUMNS, save, set_style
from src.splits import load_split
from src.train import to_input

ARTEFACT_CELL = 30439

model_name = best_cnn(read_log())
run_id = f"{model_name}_c4_final_s0"
model, mean, sd = load_trained(run_id)
X, y7, cell_ids = load_dataset(7)
y4 = np.asarray(config.MAP_7_TO_4)[y7]
val = load_split("val")


def cam_for_row(row):
    x = to_input(torch.from_numpy(X[row:row + 1]).to(mean.device), mean, sd)
    return grad_cam(model, x, int(y4[row]))


# 1. Numbers for every validation cell
ring_share = 1 - ((config.IMAGE_SIZE - 2 * config.GRADCAM_BORDER_PX) / config.IMAGE_SIZE) ** 2
records = []
for row in val:
    cam = cam_for_row(row)
    mask, _ = cell_mask(X[row, 0])
    records.append({"row": row, "cell_id": cell_ids[row], "phase": config.CLASSES_7[y7[row]],
                    "heat_in_mask": mask_fraction(cam, mask), "mask_area_share": mask.mean(),
                    "heat_in_border": border_fraction(cam), "border_area_share": ring_share})
per_cell = pd.DataFrame(records)
summary = per_cell.groupby("phase", sort=False)[["heat_in_mask", "mask_area_share", "heat_in_border",
                                                 "border_area_share"]].median().round(3)
summary.insert(0, "n_cells", per_cell.groupby("phase", sort=False).size())
summary = summary.reindex(config.CLASSES_7)
summary.to_csv(config.RESULTS_DIR / "gradcam_val_summary.csv")
per_cell.to_csv(config.RESULTS_DIR / "gradcam_val_per_cell.csv", index=False)
print(f"Grad-CAM, {run_id}, validation cells (medians per phase):\n{summary}")

# 2. Grid figure
set_style()
rng = np.random.default_rng(0)
k = config.GRADCAM_CELLS_PER_CLASS
fig, axes = plt.subplots(len(config.CLASSES_7), 3 * k, figsize=(TWO_COLUMNS, 11 * CM))
for r, phase in enumerate(config.CLASSES_7):
    rows = val[y7[val] == r]
    chosen = np.sort(rng.choice(rows, size=min(k, len(rows)), replace=False))
    for j in range(k):
        cell_axes = axes[r, 3 * j:3 * j + 3]
        for ax in cell_axes:
            ax.set_axis_off()
        if j >= len(chosen):
            continue
        row = chosen[j]
        cam = cam_for_row(row)
        cell_axes[0].imshow(X[row, 0], cmap="gray")
        cell_axes[0].imshow(cam, cmap="inferno", alpha=0.5, vmin=0, vmax=1)
        cell_axes[1].imshow(X[row, 0], cmap="gray")
        cell_axes[2].imshow(X[row, 1], cmap="gray")
        cell_axes[1].set_title(f"{cell_ids[row]}", fontsize=5, pad=1)
    axes[r, 0].text(-0.15, 0.5, phase, transform=axes[r, 0].transAxes, ha="right", va="center")
for j in range(k):
    for c, label in enumerate(("Grad-CAM", "BF", "DF")):
        axes[0, 3 * j + c].text(0.5, 1.45, label, transform=axes[0, 3 * j + c].transAxes, ha="center",
                                fontsize=6)
fig.suptitle(f"Grad-CAM for the true 4-class label, {MODEL_LABELS[model_name]} (seed 0), validation cells",
             fontsize=7, y=1.0)
fig.subplots_adjust(wspace=0.05, hspace=0.35)
save(fig, "gradcam_val")

# 3. The training cell with a corner artefact (not a validation cell; shown for the shortcut check only)
row = int(np.flatnonzero(cell_ids == ARTEFACT_CELL)[0])
if row in set(load_split("train").tolist()):
    cam = cam_for_row(row)
    fig, axes = plt.subplots(1, 3, figsize=(8.5 * CM, 3.2 * CM))
    axes[0].imshow(X[row, 0], cmap="gray")
    axes[0].imshow(cam, cmap="inferno", alpha=0.5, vmin=0, vmax=1)
    axes[1].imshow(X[row, 0], cmap="gray")
    axes[2].imshow(X[row, 1], cmap="gray")
    for ax, t in zip(axes, ("Grad-CAM", "BF", "DF")):
        ax.set_title(t)
        ax.set_axis_off()
    fig.suptitle(f"Cell {ARTEFACT_CELL} ({config.CLASSES_7[y7[row]]}, TRAINING cell): heat in border "
                 f"{border_fraction(cam):.2f} (ring area {ring_share:.2f})", fontsize=7)
    save(fig, "gradcam_cell30439_train")
