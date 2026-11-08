"""RQ5 masking experiment, summary table and figure (run 4). VALIDATION numbers, 4 classes.

Two kinds of rows, kept apart:
- TRAINED on masked images (train AND val masked; src.train --mask-mode; tags n4_cellonly, n4_bgonly, n4_bgdisc),
  next to the original run-2 finals (n2_final), from results/experiments.csv;
- EVALUATED on masked images only: the run-2 finals (trained on original images) scored on masked val images,
  from results/masking_check_val.csv (scripts/masking_check.py).
Reference rows: chance (run-1 dummy baselines), "mask area only" (logistic regression on log mask area), and the
label-shuffle control (n4_shuffled: SmallCNN trained on permuted training labels).
Mean ± SD over the seeds that exist; n seeds is given in every row (single-seed rows have no SD).
Writes results/masking_val.md and figures/masking_val.png/.pdf.
Run: python -m scripts.masking_summary
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.experiments import read_log
from src.plotting import CM, MODEL_COLORS, MODEL_LABELS, MUTED, TWO_COLUMNS, save, set_style

CLASSES = config.CLASSES_4
TRAINED = [("original", "n2_final"), ("cell only", "n4_cellonly"), ("background only", "n4_bgonly"),
           ("background only, fixed disc", "n4_bgdisc")]
MODELS = ("smallcnn", "resnet18_pretrained")


def runs_with_tag(log, model, tag):
    ids = [f"{model}_c4_{tag}_s{s}" for s in config.SEEDS]
    return log[log["run_id"].isin(ids)]


def as_records(rows):
    """experiments.csv rows -> list of dicts with macro_f1, balanced_accuracy, recall_<class>."""
    out = []
    for _, r in rows.iterrows():
        rec = json.loads(r["recall"])
        out.append({"macro_f1": r["macro_f1"], "balanced_accuracy": r["balanced_accuracy"],
                    **{f"recall_{c}": rec[c] for c in CLASSES}})
    return pd.DataFrame(out)


def fmt(v):
    v = np.asarray(v, float)
    return "—" if len(v) == 0 else (f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v[0]:.3f}")


def line(kind, label, inp, df):
    cols = ["macro_f1", "balanced_accuracy"] + [f"recall_{c}" for c in CLASSES]
    return f"| {kind} | {label} | {inp} | {len(df)} | " + " | ".join(fmt(df[c]) if len(df) else "—" for c in cols) + " |"


if __name__ == "__main__":
    log = read_log()
    check = pd.read_csv(config.RESULTS_DIR / "masking_check_val.csv")
    header = ("| Kind | Model | Input | n seeds | Macro-F1 (val) | Bal. acc. (val) | "
              + " | ".join(f"Recall {c}" for c in CLASSES) + " |")
    md = ["# Masking experiment (RQ5), validation, 4 classes", "",
          "Mask = brightfield cell mask (`src.features.cell_mask`) dilated by "
          f"{config.MASK_DILATION_PX} px; fill = median training background per channel (BF 94, DF 39). "
          "*Trained on masked*: train and val images masked the same way, run-2 settings (vflip + weighted, lr 1e-3, "
          "wd 1e-4, batch 64, ≤ 25 epochs, patience 6), no re-tuning. *Evaluated on masked*: run-2 finals trained on "
          f"original images, only the val images masked. *Fixed disc*: central disc of radius {config.MASK_DISC_RADIUS_PX} px "
          "plus the cell mask filled, so the hole no longer shows the cell size (control added in run 4). "
          "Mean ± SD over seeds; n seeds per row. Made by `scripts/masking_summary.py`.", "",
          header, "|---" * (6 + len(CLASSES)) + "|"]
    bars = []  # (model, label, kind, values)
    for model in MODELS:
        for label, tag in TRAINED:
            df = as_records(runs_with_tag(log, model, tag))
            if len(df):
                kind = "original" if tag == "n2_final" else "trained on masked"
                md.append(line(kind, MODEL_LABELS[model], label, df))
                bars.append((model, label, kind, df["macro_f1"].to_numpy()))
    for model in ("smallcnn", "resnet18_pretrained", "resnet18_scratch"):
        for mode, label in (("cell_only", "cell only"), ("background_only", "background only")):
            df = check[(check["model"] == model) & (check["mode"] == mode)]
            md.append(line("evaluated on masked", MODEL_LABELS[model], label, df))
            if model in MODELS:
                bars.append((model, label, "evaluated on masked", df["macro_f1"].to_numpy()))
    ref = check[check["model"] == "logreg_mask_area"]
    md.append(line("reference", "Logistic regression", "log mask area only", ref))
    shuffled = as_records(runs_with_tag(log, "smallcnn", "n4_shuffled"))
    md.append(line("control", "SmallCNN", "original images, shuffled training labels", shuffled))
    chance = {}
    for name in ("dummy_majority", "dummy_stratified"):
        df = as_records(log[(log["model"] == name) & (log["n_classes"] == 4) & (log["eval_split"] == "val")
                            & log["run_id"].isna()])
        chance[name] = df["macro_f1"].mean()
        md.append(line("chance", MODEL_LABELS[name], "—", df))
    (config.RESULTS_DIR / "masking_val.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # Figure: macro-F1 per model and input; solid = trained on that input, hatched = evaluated only
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMNS, 7 * CM), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = [b for b in bars if b[0] == model]
        for i, (_, label, kind, v) in enumerate(sub):
            hatch = "////" if kind == "evaluated on masked" else None
            ax.bar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, color="white" if hatch else MODEL_COLORS[model],
                   edgecolor=MODEL_COLORS[model], hatch=hatch, width=0.7, capsize=2, error_kw={"elinewidth": 0.6})
            ax.scatter(np.full(len(v), i), v, s=4, color="black", zorder=3)
            ax.text(i, v.mean() + 0.03, f"n={len(v)}", ha="center", fontsize=5, color=MUTED)
        ax.set_xticks(range(len(sub)), [f"{l}\n({'eval only' if k == 'evaluated on masked' else 'trained'})"
                                        if k != "original" else "original" for _, l, k, _ in sub],
                      rotation=30, ha="right", fontsize=6)
        ax.axhline(chance["dummy_stratified"], color=MUTED, ls="--", lw=0.7, label="chance (stratified)")
        ax.axhline(ref["macro_f1"].mean(), color=MUTED, ls=":", lw=0.9, label="mask area only (logistic regr.)")
        ax.set_title(MODEL_LABELS[model])
        ax.set_ylim(0, 0.9)
    axes[0].set_ylabel("Macro-F1 (val)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=6)
    fig.suptitle("Masking: filled bars = trained and evaluated on that input, hatched = run-2 model evaluated on masked "
                 "val images only", fontsize=7)
    fig.tight_layout()
    save(fig, "masking_val")
