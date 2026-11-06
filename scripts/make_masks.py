"""Compute and cache the dilated brightfield cell masks for every TRAIN and VAL cell (masking experiment, RQ5).

Mask = src.features.cell_mask(brightfield), dilated by config.MASK_DILATION_PX px (src.masking.dilated_cell_mask).
Test cells are never loaded here. Writes config.MASKS_FILE (gitignored data folder) with
  rows   : row indices (load_dataset() order) of the train + val cells, sorted
  masks  : (n, 66, 66) bool, dilated masks
  area_raw, area_dilated : mask areas in pixels before / after dilation
and prints, per 7-class phase and split, how many masks are empty or smaller than config.TINY_MASK_AREA_PX.
Run: python -m scripts.make_masks
"""
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import config
from src.data import load_dataset
from src.masking import dilated_cell_mask
from src.splits import load_split


def raw_and_dilated(bf):
    raw = dilated_cell_mask(bf, px=0)
    return raw.sum(), dilated_cell_mask(bf)


if __name__ == "__main__":
    t0 = time.perf_counter()
    X, y7, _ = load_dataset(7)
    train, val = load_split("train"), load_split("val")
    rows = np.sort(np.concatenate([train, val]))
    with Pool(6) as pool:  # cell_mask is per cell, so it parallelises trivially
        out = pool.map(raw_and_dilated, [X[r, 0] for r in rows], chunksize=500)
    area_raw = np.array([a for a, _ in out])
    masks = np.stack([m for _, m in out])
    area_dil = masks.sum(axis=(1, 2))
    config.MASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.MASKS_FILE, rows=rows, masks=masks, area_raw=area_raw, area_dilated=area_dil)

    split = np.where(np.isin(rows, train), "train", "val")
    df = pd.DataFrame({"split": split, "phase": np.array(config.CLASSES_7)[y7[rows]],
                       "empty": area_raw == 0, "tiny_raw": (area_raw > 0) & (area_raw < config.TINY_MASK_AREA_PX),
                       "tiny_dilated": (area_dil > 0) & (area_dil < config.TINY_MASK_AREA_PX)})
    summary = df.groupby(["phase", "split"])[["empty", "tiny_raw", "tiny_dilated"]].sum()
    summary["n_cells"] = df.groupby(["phase", "split"]).size()
    print(summary.to_string())
    print(f"total: {len(rows)} cells, empty {int(df["empty"].sum())}, < {config.TINY_MASK_AREA_PX} px before dilation "
          f"{int(df["tiny_raw"].sum())}, after dilation {int(df["tiny_dilated"].sum())}")
    print(f"median area raw {np.median(area_raw):.0f} px, dilated {np.median(area_dil):.0f} px "
          f"(image {config.IMAGE_SIZE ** 2} px); {time.perf_counter() - t0:.0f} s")
