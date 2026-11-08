"""Investigation (run 4): why does a CNN trained on background-only images score far above chance?

Suspicion-rule check. A background-only image still contains (1) the OUTLINE of the filled hole, i.e. the size and
shape of the cell, and (2) the background itself, whose darkfield level is known to depend on the phase
(PROJECT.md observations). This script asks how much of the score simple per-cell numbers already explain.
Per cell, from the cached dilated mask (train + val only):
  area        = log(1 + dilated mask area)
  bg level    = median of the pixels outside the mask, per channel (BF, DF)
  bg spread   = SD of the pixels outside the mask, per channel
Classifiers fitted on TRAIN, scored on VAL through src.evaluate (4 classes):
  - logistic regression (balanced class weights, standardised inputs)
  - random forest with the main RF settings (500 trees, max_features 0.33, min_samples_leaf 5, balanced)
on the feature groups: area | bg level | bg level + spread | area + bg level | area + bg level + spread.
Part 2 (after the fixed-disc control "background_only_disc" was still above chance): what is left OUTSIDE the
disc (radius MASK_DISC_RADIUS_PX) and the cell mask? Radial intensity profile per channel: mean of the visible
pixels in rings 22-25, 25-28, 28-31, 31-34 and > 34 px from the centre (corners), plus their SD; random forest,
for BF only, DF only and both. Also the same with only the outermost region (> 28 px), far from the cell.
Each fit appends one row to results/experiments.csv (tag n4_bginv). Writes results/bgonly_investigation_val.md.
Run: python -m scripts.bgonly_investigation
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import load_dataset
from src.evaluate import evaluate
from src.experiments import has_run, log_experiment
from src.masking import load_masks
from src.splits import load_split


def per_cell_stats(X, masks):
    """log area, and median / SD of the background pixels per channel, for every cell."""
    out = np.zeros((len(X), 5))
    out[:, 0] = np.log1p(masks.sum(axis=(1, 2)))
    for i in range(len(X)):
        bg = ~masks[i]
        for c in range(2):
            v = X[i, c][bg]
            out[i, 1 + c] = np.median(v)
            out[i, 3 + c] = v.std()
    return out


GROUPS = {"area": [0], "bg level (BF, DF)": [1, 2], "bg level + spread": [1, 2, 3, 4],
          "area + bg level": [0, 1, 2], "area + bg level + spread": [0, 1, 2, 3, 4]}
MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
    "rf": lambda: RandomForestClassifier(n_estimators=500, max_features=0.33, min_samples_leaf=5,
                                         class_weight="balanced", random_state=0, n_jobs=4),
}

def md_table(df):
    """Plain markdown table (pandas.to_markdown needs the optional tabulate package)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|---" * len(cols) + "|"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


if __name__ == "__main__":
    X, y, _ = load_dataset(4)
    train, val = load_split("train"), load_split("val")
    S_train, S_val = per_cell_stats(X[train], load_masks(train)), per_cell_stats(X[val], load_masks(val))
    names = ["log area", "BF bg median", "DF bg median", "BF bg SD", "DF bg SD"]
    by_class = pd.DataFrame(S_train, columns=names).assign(cls=np.array(config.CLASSES_4)[y[train]])
    med = by_class.groupby("cls").median().reindex(config.CLASSES_4).round(2)
    print(med)
    rows = []
    for mname, make in MODELS.items():
        for gname, cols in GROUPS.items():
            run_id = f"bginv_{mname}_{gname.replace(' ', '').replace('(', '').replace(')', '').replace(',', '').replace('+', '_')}_c4_n4_bginv_s0"
            t0 = time.perf_counter()
            clf = make().fit(S_train[:, cols], y[train])
            m = evaluate(y[val], clf.predict(S_val[:, cols]), config.CLASSES_4)
            if not has_run(run_id):
                log_experiment(f"bginv_{mname}", 4, "val", 0, {"features": [names[c] for c in cols], "run_tag": "n4_bginv"},
                               m, time.perf_counter() - t0, run_id=run_id, device="cpu")
            rows.append({"classifier": mname, "features": gname, "macro_f1": round(m["macro_f1"], 3),
                         "balanced_accuracy": round(m["balanced_accuracy"], 3),
                         **{f"recall_{c}": round(v, 2) for c, v in m["recall"].items()}})
            print(rows[-1], flush=True)
    # Part 2: radial profiles outside the disc + mask (what background_only_disc images still show)
    from src.masking import central_disc
    yy, xx = np.mgrid[0:config.IMAGE_SIZE, 0:config.IMAGE_SIZE]
    radius = np.hypot(yy - (config.IMAGE_SIZE - 1) / 2, xx - (config.IMAGE_SIZE - 1) / 2)
    edges = [config.MASK_DISC_RADIUS_PX, 25, 28, 31, 34, 99]
    disc = central_disc(config.IMAGE_SIZE, config.MASK_DISC_RADIUS_PX)

    def radial(Xs, masks):
        """(N, 2 channels x 5 rings x {mean, SD}) from the pixels outside disc and mask."""
        out = np.zeros((len(Xs), 2, len(edges) - 1, 2))
        for i in range(len(Xs)):
            visible = ~(disc | masks[i])
            for r in range(len(edges) - 1):
                ring = visible & (radius > edges[r]) & (radius <= edges[r + 1])
                for c in range(2):
                    v = Xs[i, c][ring].astype(float)
                    out[i, c, r] = (v.mean(), v.std()) if len(v) else (np.nan, np.nan)
        return out

    R_train, R_val = radial(X[train], load_masks(train)), radial(X[val], load_masks(val))
    fill_nan = np.nanmedian(R_train, axis=0)
    R_train, R_val = np.where(np.isnan(R_train), fill_nan, R_train), np.where(np.isnan(R_val), fill_nan, R_val)
    by_ring = pd.DataFrame(R_train[:, 1, :, 0], columns=[f"DF mean r{edges[r]}-{edges[r + 1]}" for r in range(5)])
    print(by_ring.assign(cls=np.array(config.CLASSES_4)[y[train]]).groupby("cls").median().reindex(config.CLASSES_4).round(1))
    variants = {"radial BF": (slice(0, 1), slice(None)), "radial DF": (slice(1, 2), slice(None)),
                "radial BF + DF": (slice(None), slice(None)), "radial BF + DF, rings > 28 px only": (slice(None), slice(2, None))}
    for vname, (ch, rings) in variants.items():
        A, B = R_train[:, ch, rings].reshape(len(train), -1), R_val[:, ch, rings].reshape(len(val), -1)
        run_id = f"bginv_rf_{vname.replace(' ', '').replace('+', '_').replace(',', '').replace('>', 'gt')}_c4_n4_bginv_s0"
        t0 = time.perf_counter()
        clf = MODELS["rf"]().fit(A, y[train])
        m = evaluate(y[val], clf.predict(B), config.CLASSES_4)
        if not has_run(run_id):
            log_experiment("bginv_rf", 4, "val", 0, {"features": vname, "run_tag": "n4_bginv"}, m,
                           time.perf_counter() - t0, run_id=run_id, device="cpu")
        rows.append({"classifier": "rf", "features": vname + " (outside disc + mask)", "macro_f1": round(m["macro_f1"], 3),
                     "balanced_accuracy": round(m["balanced_accuracy"], 3),
                     **{f"recall_{c}": round(v, 2) for c, v in m["recall"].items()}})
        print(rows[-1], flush=True)
    ring_medians = md_table(by_ring.assign(cls=np.array(config.CLASSES_4)[y[train]]).groupby("cls").median()
                            .reindex(config.CLASSES_4).round(1).reset_index())

    df = pd.DataFrame(rows)
    md = ["# Why is background-only above chance? Simple per-cell numbers (val, 4 classes)", "",
          "Fitted on train, scored on val. Features from the dilated brightfield mask: log mask area (the size of the "
          "filled hole), median and SD of the pixels outside the mask per channel. Made by `scripts/bgonly_investigation.py`.", "",
          "Median per class (training cells):", "", md_table(med.reset_index()), "",
          md_table(df), "",
          "Median DF ring mean outside disc + mask, per class (training cells):", "", ring_medians, ""]
    (config.RESULTS_DIR / "bgonly_investigation_val.md").write_text("\n".join(md))
    print("\n".join(md))
