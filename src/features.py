"""Hand-crafted features for the random forest: brightfield mask, shape, intensity and GLCM texture.

Every feature is computed from one cell's own pixels; nothing is fitted across cells.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology

from src import config

SHAPE_PROPS = ("area", "perimeter", "eccentricity", "solidity",
               "axis_major_length", "axis_minor_length", "extent")
# Identifier, label and QC columns in features.parquet: never model inputs
NON_FEATURE_COLUMNS = ("row", "cell_id", "label_7", "mask_n_objects", "mask_border_objects_dropped",
                       "mask_touches_border", *(f"{ch}_background" for ch in config.BG_CORRECTED_CHANNELS))


def border_ring(shape, width=config.MASK_BORDER_PX):
    ring = np.ones(shape, bool)
    ring[width:-width, width:-width] = False
    return ring


def border_median(img):
    return np.median(img[border_ring(img.shape)])


def central_disc(shape, radius=config.MASK_OTSU_RADIUS_PX):
    yy, xx = np.indices(shape)
    return np.hypot(yy - (shape[0] - 1) / 2, xx - (shape[1] - 1) / 2) < radius


def cell_mask(bf):
    """Boolean cell mask from a brightfield image.

    The cell is both darker (rim) and brighter (interior) than the grey background, so Otsu is
    applied to the absolute difference from the background level, not to the raw intensity.
    The threshold is the lower of the whole-image and central-disc Otsu thresholds: a high-contrast
    corner artefact raises the whole-image one above a pale cell's contrast (central wins), while for
    a blurry cell filling the disc the central one splits core from halo (whole-image wins).
    Objects reaching into the border ring (corner artefacts) are dropped unless nothing else is left,
    then the object nearest the image centre is kept. All other objects are kept, so telophase
    doublets keep both daughters. Returns (mask, number of border objects dropped).
    """
    diff = np.abs(bf.astype(np.float64) - border_median(bf))
    diff = filters.gaussian(diff, sigma=config.MASK_SMOOTH_SIGMA, preserve_range=True)
    threshold = min(filters.threshold_otsu(diff), filters.threshold_otsu(diff[central_disc(diff.shape)]))
    mask = diff > threshold
    mask = ndi.binary_closing(mask, structure=morphology.disk(config.MASK_CLOSING_RADIUS))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, max_size=config.MASK_MIN_OBJECT_PX)

    labels = measure.label(mask)
    edge = np.unique(labels[border_ring(labels.shape)])
    edge = edge[edge > 0]
    inner = mask & ~np.isin(labels, edge)
    if inner.any():
        return inner, len(edge)
    if labels.max() == 0:
        return mask, 0
    centre = (np.array(labels.shape) - 1) / 2
    props = measure.regionprops(labels)
    nearest = min(props, key=lambda p: np.hypot(*(np.array(p.centroid) - centre))).label
    return labels == nearest, len(edge) - 1


def shape_features(mask):
    """Region properties of the whole mask, treated as one region."""
    props = measure.regionprops(mask.astype(np.uint8))[0]
    return {p: float(getattr(props, p)) for p in SHAPE_PROPS}


def intensity_features(img, mask, ch):
    vals = img[mask].astype(np.float64)
    out = {f"{ch}_mean": vals.mean(), f"{ch}_sd": vals.std()}
    for q, v in zip(config.INTENSITY_PERCENTILES, np.percentile(vals, config.INTENSITY_PERCENTILES)):
        out[f"{ch}_p{q}"] = v
    return out


def bgcorr_intensity_features(img, mask, ch):
    """Mean and percentiles inside the mask minus the median outside it (SD is unchanged, so not repeated).

    Also returns the background level itself as `{ch}_background`, a QC column, not a model feature.
    """
    bg = float(np.median(img[~mask]))
    raw = intensity_features(img, mask, ch)
    out = {f"{ch}_background": bg}
    out.update({f"{ch}_bgcorr_{k.removeprefix(ch + '_')}": v - bg for k, v in raw.items() if not k.endswith("_sd")})
    return out


def glcm_features(img, mask, ch):
    """GLCM texture inside the mask: background pixels get level 0, which is dropped from the matrix."""
    levels = config.GLCM_LEVELS
    q = 1 + (img.astype(np.int64) * levels) // 256  # 1..levels
    q[~mask] = 0
    glcm = feature.graycomatrix(q.astype(np.uint8), distances=config.GLCM_DISTANCES,
                                angles=np.deg2rad(config.GLCM_ANGLES_DEG), levels=levels + 1,
                                symmetric=True, normed=False)[1:, 1:].astype(np.float64)
    glcm /= glcm.sum(axis=(0, 1), keepdims=True)
    return {f"{ch}_glcm_{p}": float(feature.graycoprops(glcm, p).mean()) for p in config.GLCM_PROPS}


def cell_features(cell):
    """All features for one cell, cell = (2, H, W) uint8 [brightfield, darkfield].

    Returns QC columns plus features; features are NaN if the mask is empty.
    """
    mask, n_dropped = cell_mask(cell[0])
    out = {
        "mask_n_objects": int(measure.label(mask).max()),
        "mask_border_objects_dropped": n_dropped,
        "mask_touches_border": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
    }
    if not mask.any():
        return out
    out.update(shape_features(mask))
    for c, ch in enumerate(config.INPUT_CHANNELS):
        out.update(intensity_features(cell[c], mask, ch))
        out.update(glcm_features(cell[c], mask, ch))
        if ch in config.BG_CORRECTED_CHANNELS:
            out.update(bgcorr_intensity_features(cell[c], mask, ch))
    return out


def compute_features(X, cell_ids, y7, progress_every=5000):
    """One row per cell, in load_dataset() row order (column `row`), with cell_id and 7-class label."""
    rows = []
    for i in range(len(X)):
        rows.append(cell_features(X[i]))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"{i + 1}/{len(X)} cells")
    df = pd.DataFrame(rows)
    df.insert(0, "row", np.arange(len(X)))
    df.insert(1, "cell_id", cell_ids)
    df.insert(2, "label_7", y7)
    return df
