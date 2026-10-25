"""Row-normalised validation confusion matrices: random forest vs the best CNN (highest mean val macro-F1
over seeds on 4 classes), seed 0. Writes figures/confusion_val_4class and figures/confusion_val_7class.
Matrices are recomputed from the saved predictions (results/preds/) through src.evaluate.
"""
import matplotlib.pyplot as plt
import numpy as np

from scripts.make_results_table import final_runs
from src import config
from src.evaluate import evaluate
from src.experiments import read_log
from src.plotting import CM, MODEL_LABELS, TWO_COLUMNS, save, set_style

CNNS = ("smallcnn", "resnet18_pretrained", "resnet18_scratch")
CLASS_NAMES = {4: config.CLASSES_4, 7: config.CLASSES_7}
SHORT = {"Prophase": "Pro", "Metaphase": "Meta", "Anaphase": "Ana", "Telophase": "Telo", "Mitosis": "M"}


def best_cnn(log):
    scores = {m: final_runs(log, m, 4)["macro_f1"].mean() for m in CNNS if len(final_runs(log, m, 4))}
    return max(scores, key=scores.get) if scores else None


def plot(n_classes, models, seed=0):
    names = CLASS_NAMES[n_classes]
    fig, axes = plt.subplots(1, len(models), figsize=(TWO_COLUMNS * (0.5 if n_classes == 4 else 0.75),
                                                      (6.5 if n_classes == 4 else 8.5) * CM),
                             squeeze=False)
    for ax, model in zip(axes[0], models):
        run_id = f"{'rf' if model == 'random_forest' else model}_c{n_classes}_final_s{seed}"
        p = np.load(config.PREDS_DIR / f"{run_id}.npz")
        m = evaluate(p["y_true"], p["y_pred"], names)
        cm, counts = m["confusion_matrix_norm"], m["confusion_matrix"].sum(axis=1)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6 if n_classes == 7 else 7,
                        color="white" if cm[i, j] > 0.6 else "black")
        labels = [SHORT.get(n, n) for n in names]
        ax.set_xticks(range(len(names)), labels)
        ax.set_yticks(range(len(names)), [f"{l} (n={c})" for l, c in zip(labels, counts)])
        ax.set_xlabel("Predicted")
        ax.set_title(f"{MODEL_LABELS[model]}\nmacro-F1 {m['macro_f1']:.3f}")
        ax.spines[:].set_visible(False)
    axes[0][0].set_ylabel("True (validation)")
    fig.suptitle(f"Validation confusion matrices, {n_classes} classes, seed {seed} (rows sum to 1)", fontsize=7)
    fig.tight_layout()
    save(fig, f"confusion_val_{n_classes}class")


if __name__ == "__main__":
    set_style()
    log = read_log()
    cnn = best_cnn(log)
    models = ["random_forest"] + ([cnn] if cnn else [])
    print(f"best CNN (4-class mean val macro-F1): {cnn}")
    for n_classes in (4, 7):
        plot(n_classes, [m for m in models if (config.PREDS_DIR / f"{'rf' if m == 'random_forest' else m}_c{n_classes}_final_s0.npz").exists()])
