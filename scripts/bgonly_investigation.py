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
    df = pd.DataFrame(rows)
    md = ["# Why is background-only above chance? Simple per-cell numbers (val, 4 classes)", "",
          "Fitted on train, scored on val. Features from the dilated brightfield mask: log mask area (the size of the "
          "filled hole), median and SD of the pixels outside the mask per channel. Made by `scripts/bgonly_investigation.py`.", "",
          "Median per class (training cells):", "", md_table(med.reset_index()), "",
          md_table(df), ""]
    (config.RESULTS_DIR / "bgonly_investigation_val.md").write_text("\n".join(md))
    print("\n".join(md))
