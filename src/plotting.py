"""Shared figure style for the paper (Bioinformatics: 1 column = 8.5 cm, 2 columns = 17.5 cm, fonts >= 7 pt)."""
import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt

from src import config

CM = 1 / 2.54  # inches per cm
ONE_COLUMN = 8.5 * CM
TWO_COLUMNS = 17.5 * CM

# One fixed colour per model, the same in every figure
MODEL_COLORS = {
    "random_forest": "#2a78d6",
    "smallcnn": "#eb6834",
    "resnet18_pretrained": "#1baf7a",
    "resnet18_scratch": "#6250d6",
}
MODEL_LABELS = {
    "dummy_majority": "Chance (majority)",
    "dummy_stratified": "Chance (stratified)",
    "random_forest": "Random forest",
    "smallcnn": "SmallCNN",
    "resnet18_pretrained": "ResNet18 (ImageNet)",
    "resnet18_scratch": "ResNet18 (scratch)",
    "random_forest_raw_dark": "Random forest + raw DF levels (RQ5 side experiment)",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def set_style():
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "font.family": "sans-serif",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def save(fig, name):
    """Save as PNG and PDF in figures/."""
    config.FIGURES_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(config.FIGURES_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}.png and .pdf")
