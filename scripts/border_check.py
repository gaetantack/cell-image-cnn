"""RQ5 follow-up (extra, preliminary): does the CNN use the image border, i.e. the background?

Grad-CAM put S-phase heat in the outer ring of the image. Here we change only that ring
(outer GRADCAM_BORDER_PX pixels, the same ring as in fig_gradcam.py) in the VALIDATION images and see
whether predictions change. The cell itself sits in the centre and is left as it is.
- "original":       unchanged images
- "ring_flattened": ring pixels set to the median of that image's own ring, per channel
                    (keeps the background LEVEL, removes structure such as edges and artefacts)
- "ring_common":    ring pixels set to one common value per channel, the median ring level over all
                    TRAINING cells (removes the background level too)
If recall of a class drops a lot when only the ring changes, the network uses the background for it.
Seed-0 four-class final run of every CNN, run on the CPU (the GPU is used by the training queue).
Writes results/border_check_val.csv. VALIDATION numbers only.
"""
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
versions = {"original": X_val, "ring_flattened": flattened, "ring_common": common_img}

rows = []
for model_name in MODEL_NAMES:
    run_id = f"{model_name}_c4_final_s0"
    if not (config.CHECKPOINT_DIR / f"{run_id}_best.pt").exists():
        continue
    model, mean, sd = load_trained(run_id, device=torch.device("cpu"))
    pred_original = None
    for version, images in versions.items():
        with torch.no_grad():
            pred = predict(model, torch.from_numpy(images), mean, sd)
        if pred_original is None:
            pred_original = pred
        m = evaluate(y4[val], pred, config.CLASSES_4)
        row = {"model": model_name, "images": version, "macro_f1": round(m["macro_f1"], 3)}
        for c, r in m["recall"].items():
            row[f"recall_{c}"] = round(r, 3)
        row["share_predictions_changed"] = round(float((pred != pred_original).mean()), 3)
        rows.append(row)
        print(row, flush=True)

table = pd.DataFrame(rows)
table.to_csv(config.RESULTS_DIR / "border_check_val.csv", index=False)
print(f"ring = outer {w} px; common ring level per channel (train median): {common}")
print(table.to_string(index=False))
