import json

import numpy as np
import pytest

from src import config
from src.data import load_dataset
from src.splits import SPLIT_NAMES, load_split

# Checks only split membership and labels; no test-split images are used.
pytestmark = pytest.mark.skipif(not config.SPLITS_FILE.exists(), reason="splits.json not present")


@pytest.fixture(scope="module")
def splits():
    return {name: np.array(idx) for name, idx in json.loads(config.SPLITS_FILE.read_text()).items()
            if name in SPLIT_NAMES}


@pytest.fixture(scope="module")
def y7():
    return load_dataset(7)[1]


def test_no_cell_in_two_splits_and_all_covered(splits, y7):
    all_idx = np.concatenate(list(splits.values()))
    assert len(all_idx) == len(np.unique(all_idx)) == len(y7)
    assert np.array_equal(np.sort(all_idx), np.arange(len(y7)))


def test_fractions(splits, y7):
    for name, frac in zip(SPLIT_NAMES, config.SPLIT_FRACTIONS):
        assert abs(len(splits[name]) / len(y7) - frac) < 0.01


@pytest.mark.parametrize("n_classes", [7, 4])
def test_every_class_in_every_split(splits, y7, n_classes):
    y = y7 if n_classes == 7 else np.asarray(config.MAP_7_TO_4)[y7]
    for name in SPLIT_NAMES:
        assert set(np.unique(y[splits[name]])) == set(range(n_classes)), name


@pytest.mark.parametrize("n_classes", [7, 4])
def test_class_proportions_within_1_percent(splits, y7, n_classes):
    """Each class's share in each split is within 1 percentage point of its share overall."""
    y = y7 if n_classes == 7 else np.asarray(config.MAP_7_TO_4)[y7]
    overall = np.bincount(y, minlength=n_classes) / len(y)
    for name in SPLIT_NAMES:
        share = np.bincount(y[splits[name]], minlength=n_classes) / len(splits[name])
        assert np.all(np.abs(share - overall) < 0.01), name


def test_test_split_is_locked():
    with pytest.raises(PermissionError):
        load_split("test")
    assert len(load_split("train")) > 0
