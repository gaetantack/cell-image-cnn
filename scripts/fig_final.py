"""Fig 2 and Table 2 from the outputs of scripts/final_test.py (val tonight as a dress rehearsal; test on test day).

    python -m scripts.fig_final --split val     # reads results/table_val_summary_per_run.csv
    python -m scripts.fig_final --split test    # test day, after final_test.py --split test

Fig 2 (figures/fig2_<split>.png/.pdf), 4 classes:
  A: macro-F1 per model (bar = mean over seeds 0-2, error bar = SD, dots = single seeds), chance lines
  B: balanced accuracy, same layout
  C, D: row-normalised confusion matrices, random forest vs the best CNN, seed 0
The best CNN is the one with the highest mean VALIDATION macro-F1 of the run-2 finals (chosen on val, never on test).
Table 2 (results/table2_<split>.md): 4-class metrics per model + an Eulenberg et al. 2017 reference row
with [CHECK] placeholders (their numbers must be copied from the paper by hand and checked).
Reads only final_test.py outputs; loads no images.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.fig_confusion import SHORT, best_cnn
from scripts.make_results_table import fmt, mean_sd
from src import config
from src.evaluate import evaluate
from src.experiments import read_log
from src.plotting import CM, INK, MODEL_COLORS, MODEL_LABELS, MUTED, TWO_COLUMNS, save, set_style

MODELS = ("random_forest", "smallcnn", "resnet18_pretrained", "resnet18_scratch")


def bars(ax, df, metric, label):
    for i, model in enumerate(MODELS):
        v = df.loc[df["model"] == model, metric].to_numpy()
        ax.bar(i, v.mean(), yerr=v.std(ddof=1), color=MODEL_COLORS[model], width=0.65, capsize=2,
               error_kw={"elinewidth": 0.6, "capthick": 0.6})
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=5, color=INK, zorder=3)
    for name, style in (("dummy_majority", ":"), ("dummy_stratified", "--")):
        ax.axhline(df.loc[df["model"] == name, metric].mean(), color=MUTED, ls=style, lw=0.7,
                   label=MODEL_LABELS[name])
    ax.set_xticks(range(len(MODELS)), [MODEL_LABELS[m].replace(" (", "\n(") for m in MODELS])
    ax.set_ylabel(label)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")


def confusion(ax, run_file, model, split):
    p = np.load(run_file)
    m = evaluate(p["y_true"], p["y_pred"], config.CLASSES_4)
    cm, counts = m["confusion_matrix_norm"], m["confusion_matrix"].sum(axis=1)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.6 else "black")
    labels = [SHORT.get(n, n) for n in config.CLASSES_4]
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(4), [f"{l} (n={c})" for l, c in zip(labels, counts)])
    ax.set_xlabel("Predicted")
    ax.set_ylabel(f"True ({split})")
    ax.set_title(f"{MODEL_LABELS[model]}, seed 0\nmacro-F1 {m['macro_f1']:.3f}")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)


def table2(df, split):
    classes = config.CLASSES_4
    lines = [f"# Table 2 ({split}, 4 classes; mean ± SD over seeds 0–2)", "",
             ("**Validation dress rehearsal — not the test numbers.**" if split == "val" else "**Test numbers (one run).**"),
             "", "| Model | Macro-F1 | Balanced acc. | " + " | ".join(f"Recall {c}" for c in classes)
             + " | PR-AUC Mitosis | Inference (s/1,000 cells, CPU) |", "|---" * (5 + len(classes)) + "|"]
    for model in ("dummy_majority", "dummy_stratified") + MODELS:
        g = df[df["model"] == model]
        lines.append(f"| {MODEL_LABELS[model]} | {fmt(*mean_sd(g['macro_f1']))} | {fmt(*mean_sd(g['balanced_accuracy']))} | "
                     + " | ".join(fmt(*mean_sd(g[f'recall_{c}']), 2) for c in classes)
                     + f" | {fmt(*mean_sd(g['pr_auc_Mitosis']))} | {fmt(*mean_sd(g['infer_s_per_1000']))} |")
    lines.append("| Eulenberg et al. 2017 (reference, different split and metric) | [CHECK] | [CHECK] | "
                 + " | ".join("[CHECK]" for _ in classes) + " | [CHECK] | [CHECK] |")
    prev = df["prevalence_Mitosis"].iloc[0]
    lines += ["", f"Chance PR-AUC for mitosis = its prevalence in the {split} split: {prev:.3f}. "
              "Eulenberg et al. report [CHECK: accuracy on 7 classes? which split? brightfield + darkfield?]; "
              "their numbers are not directly comparable (different split, no macro-F1)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    set_style()
    df = pd.read_csv(config.RESULTS_DIR / f"table_{args.split}_summary_per_run.csv")
    df = df[df["n_classes"] == 4]
    cnn = best_cnn(read_log(), "n2_final")  # chosen on VALIDATION
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COLUMNS, 14 * CM))
    bars(axes[0, 0], df, "macro_f1", f"Macro-F1 ({args.split})")
    bars(axes[0, 1], df, "balanced_accuracy", f"Balanced accuracy ({args.split})")
    for ax, model in zip(axes[1], ("random_forest", cnn)):
        name = "rf" if model == "random_forest" else model
        confusion(ax, config.PREDS_DIR / f"finaltest_{args.split}" / f"{name}_c4_n4_finaltest_{args.split}_s0.npz",
                  model, args.split)
    for ax, letter in zip(axes.flat, "ABCD"):
        ax.text(-0.15, 1.05, letter, transform=ax.transAxes, fontweight="bold")
    fig.tight_layout()
    save(fig, f"fig2_{args.split}")
    (config.RESULTS_DIR / f"table2_{args.split}.md").write_text(table2(df, args.split))
    print(table2(df, args.split))
