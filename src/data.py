"""Load the Jurkat BBBC048 cell cycle images: brightfield + darkfield only."""
import numpy as np
from PIL import Image

from src import config


def _read(path):
    img = np.asarray(Image.open(path))
    if img.shape != (config.IMAGE_SIZE, config.IMAGE_SIZE) or img.dtype != np.uint8:
        raise ValueError(f"{path}: unexpected image {img.shape} {img.dtype}")
    return img


def _load_raw():
    """Read all cells from the class folders, sorted by class then cell id."""
    assert not set(config.INPUT_CHANNELS) & set(config.FORBIDDEN_CHANNELS)
    images, labels, cell_ids = [], [], []
    for label, cls in enumerate(config.CLASSES_7):
        folder = config.RAW_DIR / cls
        ids = sorted(int(p.name.split("_")[0]) for p in folder.glob("*_Ch3.ome.jpg"))
        for cid in ids:
            images.append(np.stack([
                _read(folder / f"{cid}_{ch}.ome.jpg") for ch in config.INPUT_CHANNELS
            ]))
            labels.append(label)
            cell_ids.append(cid)
    return np.stack(images), np.array(labels, dtype=np.int64), np.array(cell_ids, dtype=np.int64)


def load_dataset(n_classes=7, use_cache=True):
    """Return X (N, 2, H, W) uint8 [brightfield, darkfield], y (N,) int64, cell_ids (N,).

    n_classes=7: labels follow config.CLASSES_7.
    n_classes=4: mitotic phases merged into one class (config.CLASSES_4).
    Row order is identical for both, so one split serves both label sets.
    """
    if n_classes not in (4, 7):
        raise ValueError("n_classes must be 4 or 7")
    if use_cache and config.CACHE_FILE.exists():
        d = np.load(config.CACHE_FILE)
        X, y, cell_ids = d["X"], d["y"], d["cell_ids"]
    else:
        X, y, cell_ids = _load_raw()
        if use_cache:
            config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(config.CACHE_FILE, X=X, y=y, cell_ids=cell_ids)
    if n_classes == 4:
        y = np.asarray(config.MAP_7_TO_4)[y]
    return X, y, cell_ids
