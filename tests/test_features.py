import numpy as np
import pytest

from src import config
from src.features import cell_features, cell_mask


def _synthetic_cell(radius=12, corner_artefact=False):
    """Grey background, dark ring and bright interior (like brightfield), optional dark corner block."""
    size = config.IMAGE_SIZE
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    img = np.full((size, size), 100, np.uint8)
    img[r < radius] = 40
    img[r < radius - 3] = 160
    if corner_artefact:
        img[size - 9:size - 2, 2:9] = 30
    return img, (r < radius)


def test_mask_finds_the_disc():
    img, disc = _synthetic_cell()
    mask, n_dropped = cell_mask(img)
    overlap = (mask & disc).sum() / (mask | disc).sum()
    assert overlap > 0.85 and n_dropped == 0


def test_corner_artefact_is_dropped():
    img, disc = _synthetic_cell(corner_artefact=True)
    mask, n_dropped = cell_mask(img)
    assert n_dropped == 1
    assert not mask[-12:, :12].any()


def test_cell_features_columns():
    img, _ = _synthetic_cell()
    out = cell_features(np.stack([img, img]))
    for ch in config.INPUT_CHANNELS:
        for p in config.GLCM_PROPS:
            assert np.isfinite(out[f"{ch}_glcm_{p}"])
    assert 0 <= out["eccentricity"] < 0.3 and out["solidity"] > 0.9


def test_darkfield_background_correction():
    """Raising the whole darkfield image by a constant changes raw intensities but not corrected ones."""
    img, _ = _synthetic_cell()
    a = cell_features(np.stack([img, img]))
    b = cell_features(np.stack([img, img + 20]))
    assert b["Ch6_mean"] == pytest.approx(a["Ch6_mean"] + 20)
    assert b["Ch6_background"] == a["Ch6_background"] + 20
    for k in ("mean", "p10", "p50", "p90"):
        assert b[f"Ch6_bgcorr_{k}"] == pytest.approx(a[f"Ch6_bgcorr_{k}"])
    assert "Ch3_bgcorr_mean" not in a and "Ch6_bgcorr_sd" not in a


def test_empty_image_gives_no_features():
    out = cell_features(np.full((2, config.IMAGE_SIZE, config.IMAGE_SIZE), 100, np.uint8))
    assert "area" not in out
