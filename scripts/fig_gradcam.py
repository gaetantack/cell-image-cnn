"""RQ5: Grad-CAM for the best CNN (4-class final run, seed 0) on VALIDATION cells.

Figure figures/gradcam_val.png/.pdf: for each of the 7 phases, GRADCAM_CELLS_PER_CLASS randomly chosen
(seeded) validation cells. Per cell: brightfield with the Grad-CAM map on top, then brightfield and
darkfield alone. The map is computed for the cell's TRUE 4-class label.

Numbers (results/gradcam_val_summary.csv), for ALL validation cells:
- heat_in_mask: share of the heat inside the brightfield cell mask (src.features.cell_mask),
  next to mask_area_share, the share of the image the mask covers (what uniform heat would give);
- heat_in_border: share of the heat in the outer GRADCAM_BORDER_PX pixels, next to the ring's area share.
Also figures/gradcam_cell30439_train.png: training cell 30439 (corner artefact, PROJECT.md Observations).

Default: the best CNN. `--model smallcnn` makes the same outputs for another CNN, with "_smallcnn" added to
every file name; `--device cpu` runs on the CPU (e.g. while the GPU is busy with training).
`--tag n2_final --suffix _run2`: run-2 final runs (best CNN among them by default); files get the suffix too.
`--correct-wrong` (run 4, Fig 6a): instead of the outputs above, figures/gradcam_val_correct_wrong<suffix>.png/.pdf
with CORRECT_PER_CLASS correctly and WRONG_PER_CLASS wrongly predicted val cells per 4-class phase (seeded
random choice; predictions from results/preds/<run_id>.npz), and results/gradcam_val_correct_wrong<suffix>.csv
with the median heat in mask / border for correct vs wrong cells per class (from the per-cell CSV of a normal run).
"""
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="CNN to explain (default: best CNN)")
parser.add_argument("--device", default=None, help="cpu or mps (default: best available)")
parser.add_argument("--tag", default="final", help="run tag of the CNN final runs (run 2: n2_final)")
parser.add_argument("--suffix", default="", help="added to the output file names, e.g. _run2")
parser.add_argument("--correct-wrong", action="store_true", help="Fig 6a: correct and wrong cells per class")
args = parser.parse_args()
CORRECT_PER_CLASS, WRONG_PER_CLASS = 4, 2
model_name = args.model or best_cnn(read_log(), args.tag)
suffix = (f"_{args.model}" if args.model else "") + args.suffix
run_id = f"{model_name}_c4_{args.tag}_s0"
model, mean, sd = load_trained(run_id, device=torch.device(args.device) if args.device else None)
X, y7, cell_ids = load_dataset(7)
y4 = np.asarray(config.MAP_7_TO_4)[y7]
val = load_split("val")


def cam_for_row(row):
    x = to_input(torch.from_numpy(X[row:row + 1]).to(mean.device), mean, sd)
    return grad_cam(model, x, int(y4[row]))


def correct_wrong_figure():
    """Fig 6a: per 4-class phase, CORRECT_PER_CLASS correct and WRONG_PER_CLASS wrong val cells (true-label Grad-CAM)."""
    p = np.load(config.PREDS_DIR / f"{run_id}.npz")
    pred = pd.Series(p["y_pred"], index=p["rows"])
    rng = np.random.default_rng(0)
    n = CORRECT_PER_CLASS + WRONG_PER_CLASS
    set_style()
    fig, axes = plt.subplots(4, 2 * n, figsize=(TWO_COLUMNS, 8.5 * CM))
    for r, phase in enumerate(config.CLASSES_4):
        rows = val[y4[val] == r]
        ok = rows[pred.loc[rows].to_numpy() == r]
        bad = rows[pred.loc[rows].to_numpy() != r]
        chosen = list(rng.choice(ok, CORRECT_PER_CLASS, replace=False)) + list(rng.choice(bad, WRONG_PER_CLASS, replace=False))
        for j, row in enumerate(chosen):
            cam = cam_for_row(row)
            a_cam, a_df = axes[r, 2 * j], axes[r, 2 * j + 1]
            a_cam.imshow(X[row, 0], cmap="gray")
            a_cam.imshow(cam, cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            a_df.imshow(X[row, 1], cmap="gray")
            verdict = "correct" if j < CORRECT_PER_CLASS else f"wrong → {config.CLASSES_4[pred.loc[row]][:3]}"
            a_cam.set_title(f"{cell_ids[row]}\n{verdict}", fontsize=6, pad=1,
                            color="black" if j < CORRECT_PER_CLASS else "#c0392b")
            for ax in (a_cam, a_df):
                ax.set_axis_off()
        axes[r, 0].text(-0.15, 0.5, phase, transform=axes[r, 0].transAxes, ha="right", va="center")
    fig.suptitle(f"Grad-CAM (true label) on BF, with DF next to it: {MODEL_LABELS[model_name]}, seed 0, val cells; "
                 f"{CORRECT_PER_CLASS} correct + {WRONG_PER_CLASS} wrong per phase", fontsize=7, y=1.0)
    fig.subplots_adjust(wspace=0.05, hspace=0.45)
    save(fig, f"gradcam_val_correct_wrong{suffix}")
    # Numbers for ALL val cells: correct vs wrong per class, from the per-cell CSV of a normal run
    per_cell_file = config.RESULTS_DIR / f"gradcam_val_per_cell{suffix}.csv"
    if per_cell_file.exists():
        pc = pd.read_csv(per_cell_file)
        pc["class4"] = [config.CLASSES_4[y4[row]] for row in pc["row"]]
        pc["prediction"] = np.where(pred.loc[pc["row"]].to_numpy() == y4[pc["row"]], "correct", "wrong")
        table = pc.groupby(["class4", "prediction"])[["heat_in_mask", "mask_area_share", "heat_in_border"]].median().round(3)
        table.insert(0, "n_cells", pc.groupby(["class4", "prediction"]).size())
        table = table.reindex(config.CLASSES_4, level=0)
        table.to_csv(config.RESULTS_DIR / f"gradcam_val_correct_wrong{suffix}.csv")
        print(table)


if args.correct_wrong:
    correct_wrong_figure()
    raise SystemExit

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
summary.to_csv(config.RESULTS_DIR / f"gradcam_val_summary{suffix}.csv")
per_cell.to_csv(config.RESULTS_DIR / f"gradcam_val_per_cell{suffix}.csv", index=False)
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
        cell_axes[1].set_title(f"{cell_ids[row]}", fontsize=7, pad=1)
    axes[r, 0].text(-0.15, 0.5, phase, transform=axes[r, 0].transAxes, ha="right", va="center")
for j in range(k):
    for c, label in enumerate(("Grad-CAM", "BF", "DF")):
        axes[0, 3 * j + c].text(0.5, 1.45, label, transform=axes[0, 3 * j + c].transAxes, ha="center",
                                fontsize=7)
fig.suptitle(f"Grad-CAM for the true 4-class label, {MODEL_LABELS[model_name]} (seed 0, run tag {args.tag}), validation cells",
             fontsize=7, y=1.0)
fig.subplots_adjust(wspace=0.05, hspace=0.35)
save(fig, f"gradcam_val{suffix}")

# 3. The training cell with a corner artefact (not a validation cell; shown for the shortcut check only)
row = int(np.flatnonzero(cell_ids == ARTEFACT_CELL)[0])
if row in set(load_split("train").tolist()):
    cam = cam_for_row(row)
    fig, axes = plt.subplots(1, 3, figsize=(8.5 * CM, 3.6 * CM))
    axes[0].imshow(X[row, 0], cmap="gray")
    axes[0].imshow(cam, cmap="inferno", alpha=0.5, vmin=0, vmax=1)
    axes[1].imshow(X[row, 0], cmap="gray")
    axes[2].imshow(X[row, 1], cmap="gray")
    for ax, t in zip(axes, ("Grad-CAM", "BF", "DF")):
        ax.set_title(t)
        ax.set_axis_off()
    fig.suptitle(f"Cell {ARTEFACT_CELL} ({config.CLASSES_7[y7[row]]}, TRAINING cell): heat in border "
                 f"{border_fraction(cam):.2f} (ring area {ring_share:.2f})", fontsize=7, y=1.12)
    save(fig, f"gradcam_cell30439_train{suffix}")
