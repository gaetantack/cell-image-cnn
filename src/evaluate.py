"""All metrics for the project. Every model is scored only through evaluate()."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score


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
