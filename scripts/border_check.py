"""RQ5 follow-up (extra, preliminary): does the CNN use the image border, i.e. the background?

Grad-CAM put S-phase heat in the outer ring of the image. Here we change only that ring
(outer GRADCAM_BORDER_PX pixels, the same ring as in fig_gradcam.py) in the VALIDATION images and see
whether predictions change. The cell itself sits in the centre and is left as it is.
- "original":       unchanged images
- "ring_flattened": ring pixels set to the median of that image's own ring, per channel
                    (keeps the background LEVEL, removes structure such as edges and artefacts)
- "ring_common":    ring pixels set to one common value per channel, the median ring level over all
                    TRAINING cells (removes the background level too)
- "ring_same_class":  ring copied from a random other validation cell of the SAME 4-class label
- "ring_other_class": ring copied from a random validation cell of a DIFFERENT 4-class label
The two swaps both put a real background with a seam at the ring edge into the image; the only
difference is whether that background comes from the right class. So "same class" is the control for
the seam, and the drop from "same class" to "other class" is what the background itself contributes.
If recall of a class drops a lot when only the ring changes, the network uses the background for it.
Four-class final runs of every CNN, seed 0 by default (`--seeds 0 1 2` for all seeds, run 4), on the CPU
(the GPU is used by the training queue). With several seeds the CSV also gets rows seed = "mean" and "sd".
The donor cells of the swaps are drawn once (rng seed 0), so every model and seed sees the same edited images.
`--tag n2_final --suffix _run2`: the run-2 final runs, written to results/border_check_val_run2.csv.
Writes results/border_check_val.csv. VALIDATION numbers only.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from src import config
from src.data import load_dataset
from src.evaluate import evaluate
from src.interpret import load_trained
from src.models import MODEL_NAMES
from src.splits import load_split
from src.train import predict

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="final", help="run tag of the CNN final runs (run 2: n2_final)")
parser.add_argument("--suffix", default="", help="added to the output file name, e.g. _run2")
parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="training seeds of the final runs")
args = parser.parse_args()

w = config.GRADCAM_BORDER_PX
X, y7, _ = load_dataset(7)
y4 = np.asarray(config.MAP_7_TO_4)[y7]
train, val = load_split("train"), load_split("val")

ring = np.ones((config.IMAGE_SIZE, config.IMAGE_SIZE), bool)
ring[w:-w, w:-w] = False

# Median ring level per image and channel, shape (N, 2)
ring_median = np.median(X[:, :, ring], axis=2)
common = np.median(ring_median[train], axis=0)  # from training cells only

X_val = X[val]
flattened = X_val.copy()
flattened[:, :, ring] = np.round(ring_median[val])[:, :, None].astype(np.uint8)
common_img = X_val.copy()
common_img[:, :, ring] = np.round(common)[None, :, None].astype(np.uint8)

rng = np.random.default_rng(0)
labels = y4[val]
donor_same = np.empty(len(val), int)
donor_other = np.empty(len(val), int)
for i, label in enumerate(labels):
    same = np.flatnonzero(labels == label)
    donor_same[i] = rng.choice(same[same != i])
    donor_other[i] = rng.choice(np.flatnonzero(labels != label))
same_img = X_val.copy()
same_img[:, :, ring] = X_val[donor_same][:, :, ring]
other_img = X_val.copy()
other_img[:, :, ring] = X_val[donor_other][:, :, ring]
versions = {"original": X_val, "ring_flattened": flattened, "ring_common": common_img,
            "ring_same_class": same_img, "ring_other_class": other_img}

rows = []
for model_name in MODEL_NAMES:
    for seed in args.seeds:
        run_id = f"{model_name}_c4_{args.tag}_s{seed}"
        if not (config.CHECKPOINT_DIR / f"{run_id}_best.pt").exists():
            print(f"skip {run_id}: no checkpoint")
            continue
        model, mean, sd = load_trained(run_id, device=torch.device("cpu"))
        pred_original = None
        for version, images in versions.items():
            with torch.no_grad():
                pred = predict(model, torch.from_numpy(images), mean, sd)
            if pred_original is None:
                pred_original = pred
            m = evaluate(y4[val], pred, config.CLASSES_4)
            row = {"model": model_name, "seed": seed, "images": version, "macro_f1": round(m["macro_f1"], 3)}
            for c, r in m["recall"].items():
                row[f"recall_{c}"] = round(r, 3)
            row["share_predictions_changed"] = round(float((pred != pred_original).mean()), 3)
            rows.append(row)
            print(row, flush=True)

table = pd.DataFrame(rows)
if len(args.seeds) > 1:  # add mean and SD over seeds per model and image version
    numeric = [c for c in table.columns if c not in ("model", "seed", "images")]
    grouped = table.groupby(["model", "images"], sort=False)[numeric]
    for stat, frame in (("mean", grouped.mean()), ("sd", grouped.std(ddof=1))):
        frame = frame.round(3).reset_index()
        frame.insert(1, "seed", stat)
        table = pd.concat([table, frame], ignore_index=True)
table.to_csv(config.RESULTS_DIR / f"border_check_val{args.suffix}.csv", index=False)
print(f"ring = outer {w} px; common ring level per channel (train median): {common}")
print(table.to_string(index=False))
