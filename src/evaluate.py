"""All metrics for the project. Every model is scored only through evaluate()."""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)


def evaluate(y_true, y_pred, class_names):
    """Macro-F1, balanced accuracy, per-class recall and confusion matrices.

    Classes are 0..len(class_names)-1, so a class that is never predicted still gets a
    row/column and an F1 of 0 instead of being silently dropped.
    """
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": dict(zip(class_names, recall.tolist())),
        "confusion_matrix": cm,
        "confusion_matrix_norm": cm_norm,  # rows = true class, each row sums to 1
    }


def ranking_metrics(y_true, proba, class_names):
    """One-vs-rest ROC-AUC and PR-AUC (average precision) per class from class probabilities, plus macro means.

    proba: (N, K) scores, column k = probability of class k (softmax for CNNs, predict_proba for the RF).
    For class k the cells of class k are "positive" and all others "negative"; the score is proba[:, k].
    ROC-AUC: chance = 0.5. PR-AUC: chance = the prevalence of class k (share of cells in class k), so it is
    the more informative number for a rare class such as mitosis. A class absent from y_true gets NaN.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    roc, pr = {}, {}
    for k, name in enumerate(class_names):
        positive = y_true == k
        if positive.all() or not positive.any():
            roc[name] = pr[name] = float("nan")
            continue
        roc[name] = float(roc_auc_score(positive, proba[:, k]))
        pr[name] = float(average_precision_score(positive, proba[:, k]))
    return {"roc_auc": roc, "pr_auc": pr,
            "roc_auc_macro": float(np.nanmean(list(roc.values()))), "pr_auc_macro": float(np.nanmean(list(pr.values()))),
            "prevalence": {name: float((y_true == k).mean()) for k, name in enumerate(class_names)}}
