import numpy as np
import pytest

from src.evaluate import evaluate

NAMES = ("a", "b", "c")


def test_perfect_prediction():
    y = np.array([0, 0, 1, 2])
    m = evaluate(y, y, NAMES)
    assert m["macro_f1"] == m["balanced_accuracy"] == 1.0
    assert m["recall"] == {"a": 1.0, "b": 1.0, "c": 1.0}
    assert np.array_equal(m["confusion_matrix"], np.diag([2, 1, 1]))


def test_hand_computed_example():
    # true a a a a b b c c ; pred a a a b b a c a
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 2, 0])
    m = evaluate(y_true, y_pred, NAMES)
    assert m["recall"] == pytest.approx({"a": 0.75, "b": 0.5, "c": 0.5})
    assert m["balanced_accuracy"] == pytest.approx((0.75 + 0.5 + 0.5) / 3)
    # F1 a: P=3/5, R=3/4 -> 2/3; b: P=1/2, R=1/2 -> 1/2; c: P=1, R=1/2 -> 2/3
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1 / 2 + 2 / 3) / 3)
    assert np.array_equal(m["confusion_matrix"], [[3, 1, 0], [1, 1, 0], [1, 0, 1]])
    assert np.allclose(m["confusion_matrix_norm"].sum(axis=1), 1)


def test_majority_classifier_scores_chance():
    """Always predicting class a: balanced accuracy = 1/3, rare classes get recall and F1 of 0."""
    y_true = np.array([0] * 8 + [1, 2])
    m = evaluate(y_true, np.zeros_like(y_true), NAMES)
    assert m["balanced_accuracy"] == pytest.approx(1 / 3)
    assert m["recall"]["b"] == m["recall"]["c"] == 0
    assert m["macro_f1"] == pytest.approx((2 * 0.8 / 1.8) / 3)
    assert m["confusion_matrix"].shape == (3, 3)
