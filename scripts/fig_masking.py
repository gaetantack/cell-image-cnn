"""Fig 6b (main text): RQ5 masking, VALIDATION macro-F1, 4 classes.

One panel, four bars per model: full image, cell only, background only, background only + fixed disc.
Every bar is a model TRAINED and evaluated on that input (tags n2_final, n4_cellonly, n4_bgonly, n4_bgdisc),
mean ± SD over seeds 0–2, seed values as dots; dashed line = stratified chance.
The evaluate-only rows and the darkfield investigation are Supplementary (scripts/masking_summary.py,
scripts/bgonly_investigation.py). Numbers are the same as in results/masking_val.md.
Writes figures/fig6b_masking_val.png/.pdf.
Run: python -m scripts.fig_masking
"""
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.experiments import read_log
from src.plotting import CM, MODEL_COLORS, MODEL_LABELS, MUTED, ONE_COLUMN, save, set_style
from scripts.masking_summary import as_records, runs_with_tag

INPUTS = [("Full image", "n2_final"), ("Cell only", "n4_cellonly"), ("Background only", "n4_bgonly"),
          ("Background only\n+ fixed disc", "n4_bgdisc")]
MODELS = ("smallcnn", "resnet18_pretrained")
SHADES = (1.0, 0.7, 0.45, 0.25)  # opacity per input: full image darkest


def chance_stratified(log):
    """Mean val macro-F1 of the run-1 stratified dummy baseline (4 classes), as in masking_summary."""
    rows = log[(log["model"] == "dummy_stratified") & (log["n_classes"] == 4) & (log["eval_split"] == "val")
               & log["run_id"].isna()]
    return rows["macro_f1"].mean()


if __name__ == "__main__":
    log = read_log()
    set_style()
    fig, ax = plt.subplots(figsize=(ONE_COLUMN, 6 * CM))
    ticks, labels = [], []
    for g, model in enumerate(MODELS):
        for i, ((label, tag), alpha) in enumerate(zip(INPUTS, SHADES)):
            v = as_records(runs_with_tag(log, model, tag))["macro_f1"].to_numpy()
            assert len(v) == len(config.SEEDS), f"{model} {tag}: {len(v)} seeds"
            x = g * (len(INPUTS) + 1) + i
            ax.bar(x, v.mean(), yerr=v.std(ddof=1), width=0.8, color=MODEL_COLORS[model], alpha=alpha,
                   edgecolor=MODEL_COLORS[model], linewidth=0.6, capsize=1.5, error_kw={"elinewidth": 0.6})
            ax.scatter(np.full(len(v), x), v, s=3, color="black", zorder=3)
            ax.text(x, v.mean() + v.std(ddof=1) + 0.02, f"{v.mean():.2f}", ha="center", fontsize=6, color=MUTED)
            ticks.append(x)
            labels.append(label)
            print(f"{model:20s} {label.replace(chr(10), ' '):30s} {v.mean():.3f} ± {v.std(ddof=1):.3f} (n={len(v)})")
        ax.text(g * (len(INPUTS) + 1) + (len(INPUTS) - 1) / 2, 0.95, MODEL_LABELS[model], ha="center", fontsize=7)
    chance = chance_stratified(log)
    ax.axhline(chance, color=MUTED, ls="--", lw=0.7)
    ax.text(ticks[-1] + 0.5, chance + 0.01, "chance", ha="right", va="bottom", fontsize=6, color=MUTED)
    ax.set_xticks(ticks, labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro-F1 (val)")
    fig.tight_layout()
    save(fig, "fig6b_masking_val")
