"""Validation macro-F1 per model: mean ± SD over seeds 0, 1, 2 (bar + error bar), single seeds as dots,
chance baselines as horizontal lines. One panel per task (4 and 7 classes).
Writes figures/model_comparison_val.png/.pdf. VALIDATION numbers only.
"""
import matplotlib.pyplot as plt
import numpy as np

from scripts.make_results_table import final_runs
from src.experiments import read_log
from src.plotting import INK, MODEL_COLORS, MODEL_LABELS, MUTED, TWO_COLUMNS, CM, save, set_style

MODELS = ("random_forest", "smallcnn", "resnet18_pretrained", "resnet18_scratch")

log = read_log()
set_style()
fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMNS, 6 * CM), sharey=True)
for ax, n_classes in zip(axes, (4, 7)):
    models = [m for m in MODELS if len(final_runs(log, m, n_classes))]
    for i, model in enumerate(models):
        f1 = final_runs(log, model, n_classes)["macro_f1"].to_numpy()
        sd = f1.std(ddof=1) if len(f1) > 1 else 0
        ax.bar(i, f1.mean(), width=0.6, color=MODEL_COLORS[model], yerr=sd,
               error_kw={"elinewidth": 0.8, "ecolor": INK, "capsize": 2})
        ax.scatter(np.full(len(f1), i) + np.linspace(-0.12, 0.12, len(f1)), f1, s=8, color="white",
                   edgecolor=INK, linewidth=0.5, zorder=3)
        ax.text(i, f1.mean() + sd + 0.02, f"{f1.mean():.3f}", ha="center", va="bottom", color=INK)
    for chance, style in (("dummy_majority", ":"), ("dummy_stratified", "--")):
        level = final_runs(log, chance, n_classes)["macro_f1"].mean()
        ax.axhline(level, color=MUTED, lw=0.8, ls=style, label=MODEL_LABELS[chance], zorder=0)
    ax.set_xticks(range(len(models)), [MODEL_LABELS[m].replace(" (", "\n(").replace("Random forest", "Random\nforest") for m in models])
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_title(f"{n_classes} classes")
axes[0].set_ylabel("Macro-F1 (validation)")
axes[0].legend(loc="upper left", frameon=False)
fig.suptitle("Validation macro-F1, mean ± SD over 3 seeds (dots: single seeds)", fontsize=7, y=1.02)
save(fig, "model_comparison_val")
