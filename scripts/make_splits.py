"""Make the one fixed stratified 70/15/15 split and write splits/splits.json. Refuses to overwrite."""
import numpy as np

from src import config
from src.data import load_dataset
from src.splits import SPLIT_NAMES, make_splits, save_splits

_, y7, _ = load_dataset(7)
splits = make_splits(y7)
save_splits(splits, len(y7))

# Sizes only; per-class counts are checked in tests/test_splits.py
for name in SPLIT_NAMES:
    print(f"{name}: {len(splits[name])} cells ({100 * len(splits[name]) / len(y7):.1f}%)")
print(f"saved {config.SPLITS_FILE.relative_to(config.ROOT)}")
