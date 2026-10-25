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

# Random forest (scripts/run_rf.py). Missing features -> train-set median (see PROJECT.md decisions).
RF_FEATURE_SET = "main"
RF_N_ESTIMATORS = 500
# 12 configurations = TUNING_BUDGET: max_features x min_samples_leaf x class_weight (max_depth unlimited;
# min_samples_leaf limits tree size instead). Both class_weight options re-weight the rare classes.
RF_GRID = [
    {"max_features": mf, "min_samples_leaf": leaf, "class_weight": cw}
    for mf in ("sqrt", 0.33) for leaf in (1, 5, 20) for cw in ("balanced", "balanced_subsample")
]
RF_PERMUTATION_REPEATS = 10
RF_TOP_FEATURES = 15
PREDS_DIR = RESULTS_DIR / "preds"  # val predictions per run, for confusion matrices

# CNNs (src/models.py, src/train.py)
SMALLCNN_CHANNELS = (32, 64, 128, 256)  # output channels of the 4 conv blocks
SMALLCNN_DROPOUT = 0.3
RESNET_INPUT_SIZE = 128  # 66 px images are upscaled (bilinear) to this before ResNet18
CHECKPOINT_DIR = ROOT / "checkpoints"  # gitignored
LOG_DIR = ROOT / "logs"  # gitignored
EVAL_BATCH_SIZE = 512
EARLY_STOP_PATIENCE = 5  # epochs without val macro-F1 improvement before stopping
# Epoch caps (set by the overnight time budget, 2026-10-25; same for every CNN family)
TUNE_EPOCHS = 4
FINAL_EPOCHS = 12
# Augmentation "basic" (train only): random 90-degree rotation, horizontal/vertical flip, and per-image,
# per-channel intensity jitter x * (1 + g) + b with g, b uniform in +-AUG_JITTER (in normalised units)
AUG_JITTER = 0.1
# CNN tuning grid: 12 configurations = TUNING_BUDGET, same grid for every family.
# augment = basic and balance = weighted are fixed during tuning (their ablation is RQ3).
CNN_GRID = [
    {"lr": lr, "weight_decay": wd, "batch_size": bs}
    for lr in (1e-4, 5e-4, 2e-3) for wd in (1e-4, 1e-2) for bs in (64, 128)
]
# Learning curve (RQ2): stratified fraction of the TRAIN split, seeded by the run seed
LABEL_FRACTIONS = (0.01, 0.05, 0.2, 1.0)
