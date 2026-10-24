"""Fixed train/val/test split: row indices into load_dataset() order, stored in splits/splits.json."""
import json

import numpy as np
from sklearn.model_selection import train_test_split

from src import config

SPLIT_NAMES = ("train", "val", "test")


def make_splits(y7):
    """Stratified (on the 7 phases) train/val/test row indices, seeded by config.SPLIT_SEED.

    Test is split off first, then val from the rest, so the fractions match config.SPLIT_FRACTIONS.
    """
    _, f_val, f_test = config.SPLIT_FRACTIONS
    idx = np.arange(len(y7))
    rest, test = train_test_split(idx, test_size=f_test, stratify=y7, random_state=config.SPLIT_SEED)
    train, val = train_test_split(rest, test_size=f_val / (1 - f_test), stratify=y7[rest],
                                  random_state=config.SPLIT_SEED)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def save_splits(splits, n_cells):
    if config.SPLITS_FILE.exists():
        raise FileExistsError(f"{config.SPLITS_FILE} exists; the split is never regenerated")
    config.SPLITS_FILE.parent.mkdir(exist_ok=True)
    payload = {
        "seed": config.SPLIT_SEED,
        "fractions": list(config.SPLIT_FRACTIONS),
        "stratified_on": "7-class labels",
        "n_cells": n_cells,
        **{name: splits[name].tolist() for name in SPLIT_NAMES},
    }
    config.SPLITS_FILE.write_text(json.dumps(payload))


def load_split(name, final_test=False):
    """Row indices of one split. The test split needs final_test=True (week 5 only)."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    if name == "test" and not final_test:
        raise PermissionError("test split is locked until the final test (week 5)")
    return np.array(json.loads(config.SPLITS_FILE.read_text())[name], dtype=np.int64)
