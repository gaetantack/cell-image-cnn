"""RQ5 masking check, inference only (run 4): run-2 final CNNs on masked VALIDATION images.

The 9 run-2 final models (smallcnn, resnet18_pretrained, resnet18_scratch; tag n2_final; seeds 0-2),
trained on ORIGINAL images, are scored on val images in three modes (src.masking.apply_mask):
original / cell_only / background_only. Mask = cached dilated brightfield mask (scripts/make_masks.py),
fill = median training background per channel. Nothing is retrained; runs on the CPU.

Reference row "mask area only": a logistic regression (balanced class weights) on log(dilated mask area),
fitted on train, scored on val. It shows how much the mask SIZE alone predicts. This matters because the
filled hole in a background_only image still shows the cell's outline, i.e. its size and shape.
It is a fitted model, so it appends one row to results/experiments.csv (tag n4_maskarea).

Writes results/masking_check_val.csv (one row per model x seed x mode) and results/masking_check_val.md
(mean +- SD over seeds, chance rows from experiments.csv). VALIDATION numbers only.
Run: python -m scripts.masking_check
"""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import load_dataset
from src.evaluate import evaluate
from src.experiments import has_run, log_experiment
from src.interpret import load_trained
from src.masking import apply_mask, background_fill, load_masks
from src.models import MODEL_NAMES
from src.plotting import MODEL_LABELS
from src.splits import load_split
from src.train import predict

TAG = "n2_final"
torch.set_num_threads(4)  # leave CPU for the training queue


def metric_row(**info_and_metrics):
    m = info_and_metrics.pop("m")
    row = dict(info_and_metrics, macro_f1=round(m["macro_f1"], 4), balanced_accuracy=round(m["balanced_accuracy"], 4))
    row.update({f"recall_{c}": round(r, 4) for c, r in m["recall"].items()})
    return row


def chance_rows():
    """Run-1 chance baselines (4 classes, val) from experiments.csv, one row per seed."""
    from src.experiments import read_log
    import json
    log = read_log()
    rows = []
    for name in ("dummy_majority", "dummy_stratified"):
        for _, r in log[(log["model"] == name) & (log["n_classes"] == 4) & (log["eval_split"] == "val")].iterrows():
            rec = json.loads(r["recall"])
            rows.append({"model": name, "seed": int(r["seed"]), "mode": "-", "macro_f1": r["macro_f1"],
                         "balanced_accuracy": r["balanced_accuracy"],
                         **{f"recall_{c}": rec[c] for c in config.CLASSES_4}})
    return pd.DataFrame(rows)


def summary_markdown(df, title, note=""):
    """mean +- SD over seeds per (model, mode); n = number of seeds."""
    cols = ["macro_f1", "balanced_accuracy"] + [f"recall_{c}" for c in config.CLASSES_4]
    lines = [f"# {title}", "", note, "",
             "| Model | Input | n seeds | Macro-F1 | Bal. acc. | " + " | ".join(f"Recall {c}" for c in config.CLASSES_4) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for (model, mode), g in df.groupby(["model", "mode"], sort=False):
        cells = [f"{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}" if len(g) > 1 else f"{g[c].mean():.3f}" for c in cols]
        lines.append(f"| {MODEL_LABELS.get(model, model)} | {mode} | {len(g)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    X, y4, _ = load_dataset(4)
    train, val = load_split("train"), load_split("val")
    train_masks, val_masks = load_masks(train), load_masks(val)
    fill = background_fill(X[train], train_masks, np.arange(len(train)))
    empty = int((val_masks.sum(axis=(1, 2)) == 0).sum())
    print(f"fill per channel {fill.tolist()}; {empty} val cell(s) with an empty mask "
          f"(cell_only -> fully grey, background_only -> unchanged)")
    images = {mode: torch.from_numpy(apply_mask(X[val], val_masks, mode, fill)) for mode in config.MASK_MODES}

    rows, missing = [], []
    for model_name in MODEL_NAMES:
        for seed in config.SEEDS:
            run_id = f"{model_name}_c4_{TAG}_s{seed}"
            if not (config.CHECKPOINT_DIR / f"{run_id}_best.pt").exists():
                missing.append(run_id)
                print(f"MISSING checkpoint {run_id}")
                continue
            model, mean, sd = load_trained(run_id, device=torch.device("cpu"))
            for mode, imgs in images.items():
                t0 = time.perf_counter()
                m = evaluate(y4[val], predict(model, imgs, mean, sd), config.CLASSES_4)
                rows.append(metric_row(model=model_name, seed=seed, mode=mode, run_id=run_id, m=m))
                print(f"{run_id:38s} {mode:16s} macro-F1 {m['macro_f1']:.4f} ({time.perf_counter() - t0:.0f} s)",
                      flush=True)

    # Reference: how much does mask size alone predict? (fitted on train, scored on val)
    area = lambda masks: np.log1p(masks.sum(axis=(1, 2)))[:, None]
    run_id = "logreg_mask_area_c4_n4_maskarea_s0"
    t0 = time.perf_counter()
    lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(area(train_masks), y4[train])
    m = evaluate(y4[val], lr.predict(area(val_masks)), config.CLASSES_4)
    if not has_run(run_id):
        log_experiment("logreg_mask_area", 4, "val", 0, {"feature": "log1p(dilated mask area)",
                       "class_weight": "balanced", "run_tag": "n4_maskarea"}, m, time.perf_counter() - t0,
                       run_id=run_id, device="cpu")
    rows.append(metric_row(model="logreg_mask_area", seed=0, mode="mask area only", run_id=run_id, m=m))
    print(f"mask area only: macro-F1 {m['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "masking_check_val.csv", index=False)
    note = ("Inference only: run-2 final models (trained on original images) scored on masked val images. "
            f"Mask = brightfield cell mask dilated by {config.MASK_DILATION_PX} px; fill per channel = median "
            f"training background (BF {fill[0]}, DF {fill[1]}). {empty} val cell has an empty mask. "
            "Chance rows: run-1 dummy baselines. 'Mask area only' = logistic regression on log mask area "
            "(1 fit, deterministic). VALIDATION numbers."
            + (f" Missing checkpoints: {', '.join(missing)}." if missing else " All 9 checkpoints present."))
    md = summary_markdown(pd.concat([df, chance_rows()], ignore_index=True),
                          "Masking check, inference only (val, 4 classes)", note)
    (config.RESULTS_DIR / "masking_check_val.md").write_text(md)
    print(md)
