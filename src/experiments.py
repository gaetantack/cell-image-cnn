"""Experiment log: every run appends one row to results/experiments.csv."""
import datetime
import json
import subprocess

import pandas as pd

from src import config

# run_id, device and git_commit were added on 2026-10-25; older rows have them empty.
COLUMNS = ("date", "model", "n_classes", "eval_split", "seed", "settings",
           "macro_f1", "balanced_accuracy", "recall", "train_time_s",
           "run_id", "device", "git_commit")


def git_commit():
    """Short hash of the current commit, with '-dirty' if there are uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=config.ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return ""


def read_log():
    """The experiment log as a DataFrame (empty if there is no log yet)."""
    if not config.EXPERIMENTS_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(config.EXPERIMENTS_FILE)


def has_run(run_id):
    """True if a row with this run_id is already in the log (used to skip finished runs)."""
    log = read_log()
    return "run_id" in log.columns and (log["run_id"] == run_id).any()


def log_experiment(model, n_classes, eval_split, seed, settings, metrics, train_time_s,
                   run_id="", device="cpu"):
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
        "run_id": run_id,
        "device": device,
        "git_commit": git_commit(),
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    log = read_log()
    if list(log.columns) != list(COLUMNS):
        # Old log without the newer columns: rewrite it once with those columns left empty
        log.reindex(columns=COLUMNS).to_csv(config.EXPERIMENTS_FILE, index=False)
    new = not config.EXPERIMENTS_FILE.exists()
    pd.DataFrame([row], columns=COLUMNS).to_csv(config.EXPERIMENTS_FILE, mode="a", header=new, index=False)
