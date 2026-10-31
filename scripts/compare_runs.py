"""Run 1 vs run 2 per CNN family, 4 classes, VALIDATION (mean ± SD over seeds 0-2 of the final runs).

Run 1 = run tag "final" (augment basic, balance weighted, 12 epochs, early-stop patience 5, run-1 grid).
Run 2 = run tag "n2_final" (AUG*/BAL* from Phase A, max 25 epochs, patience 6, run-2 grid).
Per run: the training settings, macro-F1, mitosis recall, best epoch per seed and epochs run per seed.
A family with fewer than 3 run-2 seeds is shown but marked "incomplete" (not a final result).
Writes results/table_run1_vs_run2_val.md.
"""
import json

import numpy as np

from scripts.make_results_table import CNNS, final_runs, fmt, mean_sd
from src import config
from src.experiments import read_log
from src.plotting import MODEL_LABELS

RUNS = (("run 1", "final"), ("run 2", "n2_final"))


def describe(runs):
    s = [json.loads(x) for x in runs.sort_values("seed")["settings"]]
    f1 = mean_sd(runs["macro_f1"])
    mit = mean_sd([json.loads(r)["Mitosis"] for r in runs["recall"]])
    return {
        "config": f"lr {s[0]['lr']:g}, wd {s[0]['weight_decay']:g}, bs {s[0]['batch_size']}, "
                  f"aug {s[0]['augment']}, bal {s[0]['balance']}, max {s[0]['max_epochs']} ep",
        "f1": fmt(*f1), "mit": fmt(*mit, 2), "f1_mean": f1[0],
        "best": " / ".join(str(x["best_epoch"]) for x in s),
        "ran": " / ".join(str(x["epochs_run"]) for x in s),
        "time": f"{runs['train_time_s'].mean():.0f}",
    }


log = read_log()
lines = ["# Run 1 vs run 2, CNN finals, 4 classes (VALIDATION, mean ± SD over seeds)", "",
         "Made by `scripts/compare_runs.py`. Best epoch is 0-based; \"epochs run\" = max epochs means the run "
         "never stopped early.", "",
         "| Family | Run | Seeds | Configuration | Macro-F1 (val) | Mitosis recall (val) | Best epoch per seed | "
         "Epochs run per seed | Train time (s) |",
         "|---|---|---|---|---|---|---|---|---|"]
for model in CNNS:
    means = {}
    for name, tag in RUNS:
        runs = final_runs(log, model, 4, tag)
        if len(runs) == 0:
            lines.append(f"| {MODEL_LABELS[model]} | {name} | 0 | not run | | | | | |")
            continue
        d = describe(runs)
        means[name] = d["f1_mean"]
        flag = "" if len(runs) == len(config.SEEDS) else " (incomplete)"
        lines.append(f"| {MODEL_LABELS[model]} | {name}{flag} | {len(runs)} | {d['config']} | {d['f1']} | "
                     f"{d['mit']} | {d['best']} | {d['ran']} | {d['time']} |")
    if len(means) == 2:
        lines.append(f"| | change | | | {means['run 2'] - means['run 1']:+.3f} | | | | |")
rf = final_runs(log, "random_forest", 4)
lines += ["", f"Random forest (unchanged, run 1): {fmt(*mean_sd(rf['macro_f1']))} val macro-F1.", ""]
md = "\n".join(lines)
(config.RESULTS_DIR / "table_run1_vs_run2_val.md").write_text(md)
print(md)
