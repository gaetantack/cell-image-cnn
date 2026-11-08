"""Validation summary table: per model, mean ± SD over seeds 0, 1, 2 of the final runs.

Columns: macro-F1, balanced accuracy, recall per class, training time, number of parameters.
Chance baselines (majority, stratified) are included. Written to results/table_val_summary.csv and .md.
All numbers are VALIDATION numbers.

`--tag` selects the CNN final runs by run tag (run 1: "final", the default; run 2: "n2_final") and `--suffix`
is added to the output file names, e.g.
    python -m scripts.make_results_table --tag n2_final --suffix _run2   -> results/table_val_summary_run2.*
The random forest and chance baselines were not re-run in run 2, so they always use their run-1 rows.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.experiments import read_log
from src.plotting import MODEL_LABELS

MODELS = ("dummy_majority", "dummy_stratified", "random_forest", "smallcnn", "resnet18_pretrained",
          "resnet18_scratch", "random_forest_raw_dark")
RUN_PREFIX = {"random_forest": "rf", "random_forest_raw_dark": "rf_raw_dark"}  # run_id prefix if not the model name
CLASS_NAMES = {4: config.CLASSES_4, 7: config.CLASSES_7}
CNNS = ("smallcnn", "resnet18_pretrained", "resnet18_scratch")


def final_runs(log, model, n_classes, tag="final"):
    """Rows of the final (selected-configuration) runs of one model, one per seed.
    `tag` is the run tag of the CNN final runs; RF rows always use "final"."""
    rows = log[(log["model"] == model) & (log["n_classes"] == n_classes) & (log["eval_split"] == "val")]
    if model.startswith("dummy"):
        return rows[rows["seed"].isin(config.SEEDS)]
    prefix = RUN_PREFIX.get(model, model)
    tag = tag if model in CNNS else "final"
    ids = [f"{prefix}_c{n_classes}_{tag}_s{s}" for s in config.SEEDS]
    return rows[rows["run_id"].isin(ids)]


def best_resnet(log, tag="final"):
    """The ResNet18 variant with the higher mean 4-class val macro-F1 (only if all 3 final seeds exist)."""
    scores = {m: final_runs(log, m, 4, tag)["macro_f1"].mean() for m in ("resnet18_pretrained", "resnet18_scratch")
              if len(final_runs(log, m, 4, tag)) == len(config.SEEDS)}
    return max(scores, key=scores.get) if scores else None


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), (values.std(ddof=1) if len(values) > 1 else np.nan)


def fmt(mean, sd, digits=3):
    return f"{mean:.{digits}f}" if np.isnan(sd) else f"{mean:.{digits}f} ± {sd:.{digits}f}"


def summarise(log, tag="final"):
    rows = []
    for n_classes in (4, 7):
        for model in MODELS:
            runs = final_runs(log, model, n_classes, tag)
            if len(runs) == 0:
                continue
            settings = json.loads(runs.iloc[0]["settings"])
            row = {"n_classes": n_classes, "model": model, "n_seeds": len(runs),
                   "n_params": settings.get("n_params", np.nan)}
            for metric in ("macro_f1", "balanced_accuracy", "train_time_s"):
                row[f"{metric}_mean"], row[f"{metric}_sd"] = mean_sd(runs[metric])
            recalls = pd.DataFrame([json.loads(r) for r in runs["recall"]])
            for cls in CLASS_NAMES[n_classes]:
                row[f"recall_{cls}_mean"], row[f"recall_{cls}_sd"] = mean_sd(recalls[cls])
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(table, tag="final"):
    lines = ["# Validation summary (mean ± SD over seeds 0, 1, 2)", "",
             "**Validation results — not the final test numbers.** Made by `scripts/make_results_table.py "
             f"--tag {tag}` (CNN final runs with run tag `{tag}`; random forest and chance rows from run 1, unchanged).", ""]
    for n_classes in (4, 7):
        sub = table[table["n_classes"] == n_classes]
        classes = CLASS_NAMES[n_classes]
        lines += [f"## {n_classes} classes", "",
                  "| Model | Seeds | Macro-F1 (val) | Balanced acc. (val) | "
                  + " | ".join(f"Recall {c}" for c in classes) + " | Train time (s) | Parameters |",
                  "|---" * (6 + len(classes)) + "|"]
        for _, r in sub.iterrows():
            params = "—" if np.isnan(r["n_params"]) else f"{int(r['n_params']):,}"
            lines.append(
                f"| {MODEL_LABELS[r['model']]} | {r['n_seeds']} | {fmt(r['macro_f1_mean'], r['macro_f1_sd'])} | "
                f"{fmt(r['balanced_accuracy_mean'], r['balanced_accuracy_sd'])} | "
                + " | ".join(fmt(r[f'recall_{c}_mean'], r[f'recall_{c}_sd'], 2) for c in classes)
                + f" | {fmt(r['train_time_s_mean'], r['train_time_s_sd'], 0)} | {params} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="final", help="run tag of the CNN final runs (run 2: n2_final)")
    parser.add_argument("--suffix", default="", help="added to the output file names, e.g. _run2")
    args = parser.parse_args()
    table = summarise(read_log(), args.tag)
    table.to_csv(config.RESULTS_DIR / f"table_val_summary{args.suffix}.csv", index=False)
    md = to_markdown(table, args.tag)
    (config.RESULTS_DIR / f"table_val_summary{args.suffix}.md").write_text(md)
    print(md)
