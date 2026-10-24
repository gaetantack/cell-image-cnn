"""EDA figures: class counts, example cells per class (Ch3, Ch6), intensity histograms, image-size check.

Run as `python -m scripts.eda` from the repo root; notebooks/01_eda.ipynb imports the same functions.
Uses all cells: no split exists yet, so nothing here is fitted or tuned.
"""
from collections import Counter

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src import config
from src.data import load_dataset

N_PER_CLASS = 8
SEED = 0
# Default categorical palette slots 1-4 (dataviz skill, validated light-mode order)
CLASS_COLORS_4 = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def class_count_table(y7):
    """One row per 7-class phase, with its 4-class label and counts/percentages for both."""
    n7 = np.bincount(y7, minlength=len(config.CLASSES_7))
    n4 = np.bincount(np.asarray(config.MAP_7_TO_4)[y7], minlength=len(config.CLASSES_4))
    df = pd.DataFrame({
        "class_7": config.CLASSES_7,
        "class_4": [config.CLASSES_4[j] for j in config.MAP_7_TO_4],
        "n_7": n7,
        "pct_7": 100 * n7 / n7.sum(),
    })
    df["n_4"] = [n4[j] for j in config.MAP_7_TO_4]
    df["pct_4"] = 100 * df["n_4"] / n7.sum()
    df["G1_to_class_ratio_7"] = n7.max() / n7
    return df.round(2)


def pick_examples(y7, rng):
    """N_PER_CLASS random row indices per 7-class phase, drawn in class order from one RNG."""
    return {label: rng.choice(np.flatnonzero(y7 == label), N_PER_CLASS, replace=False)
            for label in range(len(config.CLASSES_7))}


def plot_examples(X, cell_ids, picks, channel_idx):
    """Grid: rows = phases, columns = the picked cells, one channel, shared 0-255 grey scale."""
    ch = config.INPUT_CHANNELS[channel_idx]
    fig, axes = plt.subplots(len(config.CLASSES_7), N_PER_CLASS,
                             figsize=(N_PER_CLASS * 1.1, len(config.CLASSES_7) * 1.2))
    for label, cls in enumerate(config.CLASSES_7):
        for k, i in enumerate(picks[label]):
            ax = axes[label, k]
            ax.imshow(X[i, channel_idx], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"id {cell_ids[i]}", fontsize=6)
            if k == 0:
                ax.set_ylabel(cls, fontsize=8)
    fig.suptitle(f"{ch}: {N_PER_CLASS} random cells per phase (seed {SEED}, shared 0-255 grey scale)", fontsize=9)
    fig.tight_layout()
    return fig


def plot_intensity_histograms(X, y7):
    """Per channel: pixel-intensity density for each 4-class group (log y-axis)."""
    y4 = np.asarray(config.MAP_7_TO_4)[y7]
    bins = np.arange(257)
    fig, axes = plt.subplots(1, len(config.INPUT_CHANNELS), figsize=(10, 3.6), sharey=True)
    for c, (ax, ch) in enumerate(zip(axes, config.INPUT_CHANNELS)):
        for j, (cls, color) in enumerate(zip(config.CLASSES_4, CLASS_COLORS_4)):
            counts = np.bincount(X[y4 == j, c].ravel(), minlength=256)
            ax.stairs(counts / counts.sum(), bins, color=color, linewidth=2, label=f"{cls} (n={np.sum(y4 == j)})")
        ax.set_yscale("log")
        ax.set_xlim(0, 255)
        ax.set_xlabel("pixel intensity (0-255)")
        ax.set_title(ch, fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("fraction of pixels (log)")
    axes[-1].legend(fontsize=8, frameon=False)
    fig.suptitle("Pixel-intensity distribution per channel, all pixels of all cells", fontsize=10)
    fig.tight_layout()
    return fig


def background_table(X, y7, border=5):
    """Per phase and channel: median over cells of each cell's median border-pixel intensity."""
    ring = np.ones(X.shape[2:], bool)
    ring[border:-border, border:-border] = False
    bg = np.median(X[:, :, ring], axis=2)
    df = pd.DataFrame(bg, columns=[f"bg_{ch}" for ch in config.INPUT_CHANNELS])
    df["class_7"] = pd.Categorical(np.asarray(config.CLASSES_7)[y7], config.CLASSES_7)
    return df.groupby("class_7", observed=True).median()


def image_size_check():
    """Read the header of every raw Ch3/Ch6 file (not the cache) and tally (width, height, mode)."""
    sizes = {ch: Counter() for ch in config.INPUT_CHANNELS}
    missing = []
    for cls in config.CLASSES_7:
        folder = config.RAW_DIR / cls
        for p3 in folder.glob("*_Ch3.ome.jpg"):
            cid = p3.name.split("_")[0]
            for ch in config.INPUT_CHANNELS:
                path = folder / f"{cid}_{ch}.ome.jpg"
                if not path.exists():
                    missing.append(str(path.relative_to(config.ROOT)))
                    continue
                with Image.open(path) as im:
                    sizes[ch][(*im.size, im.mode)] += 1
    rows = [{"channel": ch, "width": w, "height": h, "mode": m, "n_files": n}
            for ch, tally in sizes.items() for (w, h, m), n in sorted(tally.items())]
    return pd.DataFrame(rows), missing


def save(fig, name):
    config.FIGURES_DIR.mkdir(exist_ok=True)
    out = config.FIGURES_DIR / name
    fig.savefig(out, dpi=200)
    print(f"saved {out.relative_to(config.ROOT)}")


def main():
    X, y7, cell_ids = load_dataset(7)
    print(class_count_table(y7).to_string(index=False))

    picks = pick_examples(y7, np.random.default_rng(SEED))
    for c, ch in enumerate(config.INPUT_CHANNELS):
        save(plot_examples(X, cell_ids, picks, c), f"eda_examples_{ch}.png")
    save(plot_intensity_histograms(X, y7), "eda_intensity_hist.png")
    print(background_table(X, y7).to_string())

    table, missing = image_size_check()
    print(table.to_string(index=False))
    print(f"missing channel files: {len(missing)}")
    print(f"cached array X: shape {X.shape}, dtype {X.dtype}")


if __name__ == "__main__":
    main()
