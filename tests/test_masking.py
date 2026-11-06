"""Tests for src/masking.py (run-4 masking experiment) and the --shuffle-labels option."""
import numpy as np
import pytest

from src.masking import apply_mask, background_fill

rng = np.random.default_rng(0)


def toy(n=6, size=12):
    X = rng.integers(0, 256, size=(n, 2, size, size), dtype=np.uint8)
    masks = np.zeros((n, size, size), bool)
    masks[:, 3:8, 4:9] = True  # a square "cell"
    masks[-1] = False  # one cell with an empty mask
    return X, masks


def test_background_only_never_changes_pixels_outside_mask():
    X, masks = toy()
    out = apply_mask(X, masks, "background_only", np.array([7, 9], np.uint8))
    outside = ~masks[:, None].repeat(2, axis=1)
    assert np.array_equal(out[outside], X[outside])
    assert (out[:, 0][masks] == 7).all() and (out[:, 1][masks] == 9).all()
    assert np.array_equal(out[-1], X[-1])  # empty mask -> unchanged


def test_cell_only_never_changes_pixels_inside_mask():
    X, masks = toy()
    out = apply_mask(X, masks, "cell_only", np.array([7, 9], np.uint8))
    inside = masks[:, None].repeat(2, axis=1)
    assert np.array_equal(out[inside], X[inside])
    assert (out[:, 0][~masks] == 7).all() and (out[:, 1][~masks] == 9).all()
    assert (out[-1, 0] == 7).all() and (out[-1, 1] == 9).all()  # empty mask -> fully grey


def test_original_and_input_untouched():
    X, masks = toy()
    before = X.copy()
    assert apply_mask(X, masks, "original", np.array([0, 0])) is X
    apply_mask(X, masks, "cell_only", np.array([0, 0], np.uint8))
    assert np.array_equal(X, before)  # apply_mask works on a copy
    with pytest.raises(ValueError):
        apply_mask(X, masks, "bogus", np.array([0, 0]))


def test_fill_comes_from_training_cells_only():
    X, masks = toy(n=10)
    train_pos = np.arange(5)
    X[:5, 0][~masks[:5]] = 40  # training background levels
    X[:5, 1][~masks[:5]] = 90
    fill = background_fill(X, masks, train_pos)
    assert fill.tolist() == [40, 90]
    X2 = X.copy()
    X2[5:] = 255  # change every non-training cell completely
    assert background_fill(X2, masks, train_pos).tolist() == [40, 90]
    X3 = X.copy()
    X3[:5, 0][masks[:5]] = 0  # pixels INSIDE the training masks do not count either
    assert background_fill(X3, masks, train_pos).tolist() == [40, 90]


def test_fill_is_exact_median():
    X = np.zeros((1, 1, 2, 3), np.uint8)
    X[0, 0] = [[1, 2, 3], [4, 5, 6]]
    masks = np.zeros((1, 2, 3), bool)
    masks[0, 1, 2] = True  # value 6 excluded -> background 1..5, median 3
    assert background_fill(X, masks, np.array([0])).tolist() == [3]


def test_shuffle_labels_keeps_class_counts():
    from src.train import parse_args
    y = rng.integers(0, 4, 500)
    perm = np.random.default_rng(0).permutation(y)  # same call as in src.train
    assert np.array_equal(np.bincount(perm), np.bincount(y)) and not np.array_equal(perm, y)
    assert parse_args(["--model", "smallcnn", "--shuffle-labels"]).shuffle_labels
    assert parse_args(["--model", "smallcnn"]).mask_mode == "original"  # default changes nothing
