"""RQ3 follow-up (extra, preliminary): do the images have a preferred orientation?

Our augmentation rotates and flips cells, assuming a cell has no "up". In imaging flow cytometry, however,
cells flow in one direction and the darkfield (side scatter) light comes from one side, so the IMAGE may
have a fixed orientation even if the cell does not. Two checks:
1. Data: mean brightness of the left, right, top and bottom halves of each channel over all TRAINING
   images (cell-centred images; a big left/right difference means a preferred direction).
2. Model: the SmallCNN trained WITHOUT augmentation (ablation run smallcnn_c4_abl_none_none_s0) and the
   final SmallCNN (with augmentation) are evaluated on VALIDATION images that are rotated by 90/180/270
   degrees or flipped. A model that relies on orientation loses accuracy on rotated images.
Runs on the CPU. Writes results/orientation_check_val.csv. VALIDATION numbers only.
"""
import numpy as np
import pandas as pd
import torch

from src import config
from src.data import load_dataset
from src.evaluate import evaluate
from src.interpret import load_trained
from src.splits import load_split
from src.train import predict

X, y7, _ = load_dataset(7)
y4 = np.asarray(config.MAP_7_TO_4)[y7]
train, val = load_split("train"), load_split("val")

# 1. Brightness of image halves, training images
half = config.IMAGE_SIZE // 2
for c, channel in enumerate(("Ch3 brightfield", "Ch6 darkfield")):
    img = X[train, c].astype(np.float64)
    left, right = img[:, :, :half].mean(), img[:, :, half:].mean()
    top, bottom = img[:, :half, :].mean(), img[:, half:, :].mean()
    # share of training images whose left half is brighter than their right half (0.5 = no preference)
    left_brighter = (img[:, :, :half].mean(axis=(1, 2)) > img[:, :, half:].mean(axis=(1, 2))).mean()
    print(f"{channel}: mean left {left:.1f}, right {right:.1f}, top {top:.1f}, bottom {bottom:.1f}; "
          f"left half brighter in {left_brighter:.1%} of training images")

# 2. Models on rotated / flipped validation images
transforms = {
    "original": lambda x: x,
    "rotate_90": lambda x: np.rot90(x, 1, axes=(2, 3)),
    "rotate_180": lambda x: np.rot90(x, 2, axes=(2, 3)),
    "rotate_270": lambda x: np.rot90(x, 3, axes=(2, 3)),
    "flip_left_right": lambda x: x[:, :, :, ::-1],
    "flip_up_down": lambda x: x[:, :, ::-1, :],
}
rows = []
for run_id in ("smallcnn_c4_abl_none_none_s0", "smallcnn_c4_final_s0"):
    model, mean, sd = load_trained(run_id, device=torch.device("cpu"))
    for name, transform in transforms.items():
        images = torch.from_numpy(np.ascontiguousarray(transform(X[val])))
        with torch.no_grad():
            pred = predict(model, images, mean, sd)
        m = evaluate(y4[val], pred, config.CLASSES_4)
        row = {"run_id": run_id, "images": name, "macro_f1": round(m["macro_f1"], 3)}
        row.update({f"recall_{c}": round(r, 3) for c, r in m["recall"].items()})
        rows.append(row)
        print(row, flush=True)

table = pd.DataFrame(rows)
table.to_csv(config.RESULTS_DIR / "orientation_check_val.csv", index=False)
print(table.to_string(index=False))
