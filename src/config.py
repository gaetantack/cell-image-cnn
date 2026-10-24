"""All project settings. Import from here; never hard-code these elsewhere."""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "BBBC048v1" / "CellCycle"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_FILE = PROCESSED_DIR / "jurkat_bf_df.npz"
FEATURES_FILE = ROOT / "data" / "features.parquet"
SPLITS_FILE = ROOT / "splits" / "splits.json"
RESULTS_DIR = ROOT / "results"
EXPERIMENTS_FILE = RESULTS_DIR / "experiments.csv"
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

# Hand-crafted features (src/features.py)
# Mask: Otsu threshold on the brightfield |pixel - background|, background = median of the border ring
MASK_BORDER_PX = 3  # width of the border ring used to estimate the background
MASK_SMOOTH_SIGMA = 1.0  # Gaussian blur before thresholding
MASK_OTSU_RADIUS_PX = 25  # threshold = min(Otsu on whole image, Otsu on pixels within this radius of the centre)
MASK_CLOSING_RADIUS = 2
MASK_MIN_OBJECT_PX = 30  # objects this size or smaller are removed
INTENSITY_PERCENTILES = (10, 25, 50, 75, 90)
# Also store mean/percentiles minus the median outside the mask (raw versions are kept too)
BG_CORRECTED_CHANNELS = ("Ch6",)
TINY_MASK_AREA_PX = 150  # masks smaller than this are shown in figures/tiny_masks.png

GLCM_LEVELS = 32  # grey levels after quantising 0-255
GLCM_DISTANCES = (1,)
GLCM_ANGLES_DEG = (0, 45, 90, 135)  # GLCM properties are averaged over these
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")

# Feature sets for the random forest (column names in data/features.parquet)
_SHAPE = ("area", "perimeter", "eccentricity", "solidity", "axis_major_length", "axis_minor_length", "extent")
_PCTS = tuple(f"p{q}" for q in INTENSITY_PERCENTILES)
_BF_INTENSITY = tuple(f"Ch3_{k}" for k in ("mean", "sd", *_PCTS))
_GLCM = tuple(f"{ch}_glcm_{p}" for ch in ("Ch3", "Ch6") for p in GLCM_PROPS)
_DF_RAW_LEVEL = tuple(f"Ch6_{k}" for k in ("mean", *_PCTS))  # shift with the darkfield background
_DF_BGCORR_PCTS = tuple(f"Ch6_bgcorr_{k}" for k in _PCTS)
# Ch6_sd is unchanged by subtracting a constant background, so it counts as background-robust.
# Ch6_bgcorr_mean is left out: it still tracks the background (r = -0.92, see PROJECT.md Observations).
FEATURE_SETS = {
    "main": _SHAPE + _BF_INTENSITY + _GLCM + ("Ch6_sd",) + _DF_BGCORR_PCTS,  # main random forest
    "raw_dark": _SHAPE + _BF_INTENSITY + _GLCM + ("Ch6_sd",) + _DF_BGCORR_PCTS + _DF_RAW_LEVEL,  # RQ5 side experiment
}

# Split and seeds
SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train, val, test
SEEDS = (0, 1, 2)
TUNING_BUDGET = 12
