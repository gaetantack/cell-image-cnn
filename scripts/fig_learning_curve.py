"""RQ2 learning curve: validation macro-F1 vs fraction of labelled training cells (4 classes).

Fractions 1%, 5%, 20% are the runs tagged lf<fraction>; 100% is the final run. Each point is the mean over
the seeds available (n shown in results/learning_curve_val.csv), error bars = SD when n > 1.
Writes figures/learning_curve_val.png/.pdf. VALIDATION numbers only.
`--prefix n2_ --suffix _run2`: run-2 runs (n2_lf<fraction>, n2_final), written with the suffix.
"""
import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.experiments import read_log
from src.plotting import INK, MODEL_COLORS, MODEL_LABELS, ONE_COLUMN, CM, save, set_style

MODELS = ("smallcnn", "resnet18_pretrained", "resnet18_scratch")

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="", help="run-tag prefix (run 2: n2_)")
parser.add_argument("--suffix", default="", help="added to the output file names, e.g. _run2")
args = parser.parse_args()
log = read_log()
rows = []
for model in MODELS:
    for frac in config.LABEL_FRACTIONS:
        tag = args.prefix + ("final" if frac == 1.0 else f"lf{frac}")
        ids = [f"{model}_c4_{tag}_s{s}" for s in config.SEEDS]
        runs = log[log["run_id"].isin(ids)]
        if len(runs):
            n_train = json.loads(runs.iloc[0]["settings"])["n_train"]
            rows.append({"model": model, "label_fraction": frac, "n_train": n_train, "n_seeds": len(runs),
                         "macro_f1_mean": runs["macro_f1"].mean(),
                         "macro_f1_sd": runs["macro_f1"].std(ddof=1) if len(runs) > 1 else float("nan")})
table = pd.DataFrame(rows)
table.to_csv(config.RESULTS_DIR / f"learning_curve_val{args.suffix}.csv", index=False)
print(table.to_string(index=False))



def seed_text(t):
    """Legend note on how many seeds each point has."""
    if t["n_seeds"].nunique() == 1:
        return f"{t['n_seeds'].iloc[0]} seeds"
    return "seed 0; 100%: 3 seeds"


set_style()
fig, ax = plt.subplots(figsize=(ONE_COLUMN, 6 * CM))
for model in MODELS:
    t = table[table["model"] == model]
    if len(t) == 0:
        continue
    ax.errorbar(t["label_fraction"] * 100, t["macro_f1_mean"], yerr=t["macro_f1_sd"].fillna(0),
                color=MODEL_COLORS[model], marker="o", ms=4, lw=1.5, capsize=2, elinewidth=0.8,
                label=f"{MODEL_LABELS[model]} ({seed_text(t)})")
rf = log[log["run_id"].isin([f"rf_c4_final_s{s}" for s in config.SEEDS])]["macro_f1"].mean()
ax.axhline(rf, color=MODEL_COLORS["random_forest"], lw=0.8, ls="--", label="Random forest, 100%")
ax.set_xscale("log")
ax.set_xticks([1, 5, 20, 100], ["1%", "5%", "20%", "100%"])
ax.set_xlabel("Labelled training cells used")
ax.set_ylabel("Macro-F1 (validation)")
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="lower right")
ax.set_title(f"Learning curve, 4 classes (validation){' — run 2' if args.prefix else ''}", color=INK)
save(fig, f"learning_curve_val{args.suffix}")
