"""Experiment log: every run appends one row to results/experiments.csv."""
import datetime
import json

import pandas as pd

from src import config

COLUMNS = ("date", "model", "n_classes", "eval_split", "seed", "settings",
           "macro_f1", "balanced_accuracy", "recall", "train_time_s")


def log_experiment(model, n_classes, eval_split, seed, settings, metrics, train_time_s):
    """Append one run. `metrics` is the dict returned by src.evaluate.evaluate()."""
    if eval_split == "test":
        raise PermissionError("test results are logged only at the final test (week 5)")
    row = {
        "date": datetime.date.today().isoformat(),
        "model": model,
        "n_classes": n_classes,
        "eval_split": eval_split,
        "seed": seed,
        "settings": json.dumps(settings, sort_keys=True),
        "macro_f1": round(metrics["macro_f1"], 4),
        "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
        "recall": json.dumps({k: round(v, 4) for k, v in metrics["recall"].items()}),
        "train_time_s": round(train_time_s, 3),
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    new = not config.EXPERIMENTS_FILE.exists()
    pd.DataFrame([row], columns=COLUMNS).to_csv(config.EXPERIMENTS_FILE, mode="a", header=new, index=False)
