"""Compute hand-crafted features for every cell and save them to data/features.parquet.

Per-cell and unfitted, so computing them for test cells leaks nothing; models still use only the split rows they are allowed.
"""
import time

from src import config
from src.data import load_dataset
from src.features import compute_features

X, y7, cell_ids = load_dataset(7)
t0 = time.time()
df = compute_features(X, cell_ids, y7)
df.to_parquet(config.FEATURES_FILE, index=False)

n_empty = df["area"].isna().sum()
print(f"{len(df)} cells x {df.shape[1]} columns in {time.time() - t0:.0f} s")
print(f"empty masks: {n_empty}; >1 object: {(df['mask_n_objects'] > 1).sum()}; "
      f"border objects dropped in: {(df['mask_border_objects_dropped'] > 0).sum()}; "
      f"touching border: {df['mask_touches_border'].sum()}")
print(f"saved {config.FEATURES_FILE.relative_to(config.ROOT)}")
