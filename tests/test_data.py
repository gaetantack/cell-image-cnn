import numpy as np
import pytest

from src import config
from src.data import load_dataset

# Counted from the raw class folders (one Ch3/Ch4/Ch6 triple per cell)
EXPECTED_7 = [14333, 8616, 8601, 606, 68, 15, 27]  # G1, S, G2, Pro, Meta, Ana, Telo
EXPECTED_4 = [14333, 8616, 8601, 716]

pytestmark = pytest.mark.skipif(not config.RAW_DIR.exists(), reason="raw data not present")


def test_no_fluorescence_input():
    assert config.INPUT_CHANNELS == ("Ch3", "Ch6")
    assert not set(config.INPUT_CHANNELS) & set(config.FORBIDDEN_CHANNELS)


@pytest.mark.parametrize("n_classes, expected", [(7, EXPECTED_7), (4, EXPECTED_4)])
def test_shapes_and_counts(n_classes, expected):
    X, y, cell_ids = load_dataset(n_classes)
    n = sum(expected)
    assert X.shape == (n, 2, 66, 66) and X.dtype == np.uint8
    assert y.shape == cell_ids.shape == (n,)
    assert np.bincount(y, minlength=n_classes).tolist() == expected


def test_4_class_is_merge_of_7_class():
    X7, y7, ids7 = load_dataset(7)
    X4, y4, ids4 = load_dataset(4)
    assert np.array_equal(ids7, ids4) and np.array_equal(X7, X4)
    assert np.array_equal(y4, np.minimum(y7, 3))
