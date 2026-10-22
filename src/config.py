"""All project settings. Import from here; never hard-code these elsewhere."""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "BBBC048v1" / "CellCycle"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_FILE = PROCESSED_DIR / "jurkat_bf_df.npz"
SPLITS_FILE = ROOT / "splits" / "splits.json"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Channels (file suffix per cell: <id>_<channel>.ome.jpg)
# Ch3 = brightfield, Ch6 = darkfield (side scatter), Ch4 = fluorescence DNA stain.
# Ch4 was used to make the labels, so it is NEVER model input.
INPUT_CHANNELS = ("Ch3", "Ch6")  # order of axis 1 in X: brightfield, darkfield
FORBIDDEN_CHANNELS = ("Ch4",)
IMAGE_SIZE = 66  # all images are 66 x 66, 8-bit grayscale

# Classes: label i = CLASSES_7[i]; names match the raw folder names
CLASSES_7 = ("G1", "S", "G2", "Prophase", "Metaphase", "Anaphase", "Telophase")
CLASSES_4 = ("G1", "S", "G2", "Mitosis")
MAP_7_TO_4 = (0, 1, 2, 3, 3, 3, 3)

# Split and seeds
SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train, val, test
SEEDS = (0, 1, 2)
TUNING_BUDGET = 12
