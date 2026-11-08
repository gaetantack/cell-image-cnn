"""Masking experiment (RQ5): show the network only the cell, or only the background.

The cell mask is src.features.cell_mask on the brightfield image, dilated by config.MASK_DILATION_PX
pixels (scripts/make_masks.py computes and caches it for the TRAIN and VAL cells; never for test).

apply_mask(X, masks, mode, fill):
- "original":        images unchanged
- "cell_only":       every pixel OUTSIDE the dilated mask is set to `fill` (the cell on a flat grey)
- "background_only": every pixel INSIDE the dilated mask is set to `fill` (the cell replaced by grey)
- "background_only_disc": every pixel inside a FIXED central disc (radius config.MASK_DISC_RADIUS_PX) OR inside the
                     dilated mask is set to `fill`. Added after "background_only" scored far above chance: there the
                     filled hole has the cell's own size and shape, so the network can read the cell size from it.
                     With the disc, the hole looks the same for 99.5% of cells (only cells reaching beyond it differ).
`fill` is one constant per channel, the same for every cell: the median background pixel (outside the
masks) over all TRAINING cells. Because it is the same for every cell, the phase-dependent background
level of the darkfield channel (PROJECT.md observations) is removed in cell_only images.
A cell with an empty mask becomes fully grey in cell_only and stays unchanged in background_only.
"""
import numpy as np
from scipy import ndimage as ndi

from src import config
from src.features import cell_mask


def dilated_cell_mask(bf, px=config.MASK_DILATION_PX):
    """src.features.cell_mask of one brightfield image, dilated by `px` pixels (a disc of radius px)."""
    mask, _ = cell_mask(bf)
    if px == 0 or not mask.any():
        return mask
    yy, xx = np.mgrid[-px:px + 1, -px:px + 1]
    return ndi.binary_dilation(mask, structure=(yy ** 2 + xx ** 2) <= px ** 2)


def central_disc(size, radius):
    """Boolean (size, size) disc of the given radius around the image centre."""
    yy, xx = np.mgrid[0:size, 0:size]
    centre = (size - 1) / 2
    return np.hypot(yy - centre, xx - centre) <= radius


def background_fill(X, masks, train_pos):
    """Fill value per channel: median of all pixels outside the masks, over the TRAINING cells only.

    X (N, C, H, W) uint8, masks (N, H, W) bool, train_pos = positions in X / masks of the training cells.
    Returns a uint8 array (C,). The median is exact (from a 256-bin histogram), so it is a whole number.
    """
    bg = ~masks[train_pos]  # (n_train, H, W): True = background pixel
    fill = []
    for c in range(X.shape[1]):
        counts = np.bincount(X[train_pos, c][bg], minlength=256)
        cum = np.cumsum(counts)
        fill.append(int(np.searchsorted(cum, cum[-1] / 2)))  # first level where half the pixels are below
    return np.array(fill, dtype=np.uint8)


def apply_mask(X, masks, mode, fill):
    """Masked copy of X (N, C, H, W) uint8; masks (N, H, W) bool; fill (C,) values. See module docstring."""
    if mode not in config.MASK_MODES:
        raise ValueError(f"mode must be one of {config.MASK_MODES}")
    if mode == "original":
        return X
    if masks.shape != (X.shape[0], X.shape[2], X.shape[3]):
        raise ValueError("one mask per image is needed")
    if mode == "cell_only":
        replace = ~masks  # pixels that are set to the fill value
    elif mode == "background_only":
        replace = masks
    else:  # background_only_disc: fixed central disc united with each cell's own mask
        replace = masks | central_disc(X.shape[2], config.MASK_DISC_RADIUS_PX)[None]
    out = X.copy()
    for c in range(X.shape[1]):
        channel = out[:, c]  # a view, so this writes into `out`
        channel[replace] = fill[c]
    return out


def load_masks(rows):
    """Cached dilated masks (scripts/make_masks.py) for the given rows of load_dataset() order.

    Only TRAIN and VAL rows are in the cache; asking for any other row is an error.
    """
    d = np.load(config.MASKS_FILE)
    cached_rows, masks = d["rows"], d["masks"]
    pos = np.searchsorted(cached_rows, rows)
    if np.any(pos >= len(cached_rows)) or np.any(cached_rows[np.minimum(pos, len(cached_rows) - 1)] != rows):
        raise KeyError("some rows have no cached mask (only train and val rows are cached)")
    return masks[pos]


def masked_split_images(X, mode, train_rows, rows_list):
    """Convenience for training/inference: masked images for each row array in rows_list.

    The fill comes from the cached masks of `train_rows` (the TRAINING cells). Returns (list of arrays, fill).
    """
    if mode == "original":
        return [X[r] for r in rows_list], None
    train_masks = load_masks(train_rows)
    fill = background_fill(X[train_rows], train_masks, np.arange(len(train_rows)))
    out = []
    for r in rows_list:
        out.append(apply_mask(X[r], load_masks(r), mode, fill))
    return out, fill
