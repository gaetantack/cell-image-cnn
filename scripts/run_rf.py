"""Random forest on hand-crafted features (RQ1 baseline), validation only.

1. Tuning: the 12 configurations in config.RF_GRID, 4 classes, seed 0, scored on val macro-F1.
2. Final: the best configuration with seeds 0, 1, 2, for 4 classes and (no extra tuning) 7 classes.
Every run appends one row to results/experiments.csv; final runs also save val predictions to results/preds/.
Runs already in the log (same run_id) are skipped, so the script can be re-run safely.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src import config
from src.evaluate import evaluate
from src.experiments import has_run, log_experiment, read_log
from src.splits import load_split

CLASS_NAMES = {4: config.CLASSES_4, 7: config.CLASSES_7}


def load_features(n_classes, feature_set=config.RF_FEATURE_SET):
    """Feature matrix (rows in load_dataset() order) and labels for the chosen label set."""
    df = pd.read_parquet(config.FEATURES_FILE).sort_values("row")
    X = df[list(config.FEATURE_SETS[feature_set])].to_numpy(np.float64)
    y = df["label_7"].to_numpy()
    if n_classes == 4:
        y = np.asarray(config.MAP_7_TO_4)[y]
    return X, y


def make_rf(params, seed):
    # The imputer is inside the pipeline, so its medians are fitted on the training rows only
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=config.RF_N_ESTIMATORS, random_state=seed, n_jobs=-1, **params),
    )


def run(params, n_classes, seed, run_id, save_preds=False, feature_set=config.RF_FEATURE_SET,
        model_name="random_forest"):
    if has_run(run_id):
        print(f"skip {run_id} (already in log)")
        return
    X, y = load_features(n_classes, feature_set)
    train, val = load_split("train"), load_split("val")
    t0 = time.perf_counter()
    rf = make_rf(params, seed).fit(X[train], y[train])
    train_time = time.perf_counter() - t0
    pred = rf.predict(X[val])
    m = evaluate(y[val], pred, CLASS_NAMES[n_classes])
    settings = {"feature_set": feature_set, "n_estimators": config.RF_N_ESTIMATORS, **params}
    log_experiment(model_name, n_classes, "val", seed, settings, m, train_time, run_id=run_id, device="cpu")
    if save_preds:
        config.PREDS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(config.PREDS_DIR / f"{run_id}.npz", y_true=y[val], y_pred=pred, rows=val)
    print(f"{run_id}: val macro-F1 {m['macro_f1']:.4f}  bal. acc {m['balanced_accuracy']:.4f}  {train_time:.0f} s")


def best_params():
    """Configuration with the highest val macro-F1 among the 12 tuning runs (4 classes, seed 0)."""
    log = read_log()
    tune = log[log["run_id"].astype(str).str.startswith("rf_c4_tune")]
    best = tune.loc[tune["macro_f1"].idxmax()]
    settings = json.loads(best["settings"])
    return {k: settings[k] for k in ("max_features", "min_samples_leaf", "class_weight")}, best["run_id"]


if __name__ == "__main__":
    for k, params in enumerate(config.RF_GRID):
        run(params, 4, seed=0, run_id=f"rf_c4_tune{k:02d}_s0")
    params, best_id = best_params()
    print(f"best configuration: {best_id} {params}")
    for n_classes in (4, 7):
        for seed in config.SEEDS:
            run(params, n_classes, seed, run_id=f"rf_c{n_classes}_final_s{seed}", save_preds=True)
