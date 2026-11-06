"""Table 1 (methods overview): model, input, pretrained, parameters, training settings, epochs, training time
(mean ± SD over the three final 4-class runs) and inference time per 1,000 cells (from scripts/final_test.py).
Written to results/table1_methods.md and notes/paper_draft/table1_methods.md.

Default: the run-2 final runs (`--tag n2_final`, the paper result; updated in run 4). `--tag final` gives the
run-1 version (pilot). The random forest is the same in both runs (run-1 rows).
"""
import argparse
import json

import numpy as np
import pandas as pd

from scripts.make_results_table import final_runs
from src import config
from src.experiments import read_log
from src.models import build_model, count_parameters
from src.plotting import MODEL_LABELS

ROWS = [  # model, input, pretrained
    ("random_forest", "28 hand-crafted features (BF mask: shape, BF intensity, BF + DF texture, DF spread)", "—"),
    ("smallcnn", "BF + DF image, 2 × 66 × 66", "No"),
    ("resnet18_scratch", "BF + DF image, upscaled to 2 × 128 × 128", "No"),
    ("resnet18_pretrained", "BF + DF image, upscaled to 2 × 128 × 128", "ImageNet (timm resnet18.a1_in1k)"),
]


def mean_sd_str(t, digits=0):
    t = np.asarray(t, float)
    if len(t) == 0:
        return "not run yet"
    return f"{t.mean():.{digits}f} ± {t.std(ddof=1):.{digits}f}" if len(t) > 1 else f"{t[0]:.{digits}f}"


parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="n2_final", help="run tag of the CNN final runs (run 1: final)")
args = parser.parse_args()

log = read_log()
infer_file = config.RESULTS_DIR / "table_val_summary_per_run.csv"  # written by scripts/final_test.py --split val
infer = pd.read_csv(infer_file) if infer_file.exists() else None
lines = ["# Table 1. Methods overview", "",
         f"CNN rows: final runs with run tag `{args.tag}`. Training time: wall-clock seconds per final 4-class run "
         "(mean ± SD over seeds 0–2; CNNs on Apple M2 GPU (MPS), random forest on 8 CPU cores), including the "
         "per-epoch validation pass. Inference: CPU (4 threads), seconds per 1,000 cells, measured by "
         "`scripts/final_test.py` on val while a training job used the GPU (re-measured on test day). "
         "BF = brightfield (Ch3), DF = darkfield (Ch6). Chance baselines (majority, stratified) have no input features.", "",
         "| Model | Input | Pretrained | Parameters (4 classes) | Training settings | Best epoch / epochs run | "
         "Training time (s) | Inference (s/1,000 cells) |", "|---|---|---|---|---|---|---|---|"]
for model, inp, pre in ROWS:
    runs = final_runs(log, model, 4, args.tag)
    params = "—" if model == "random_forest" else f"{count_parameters(build_model(model, 4)):,}"
    s = [json.loads(x) for x in runs["settings"]]
    if model == "random_forest":
        setting = (f"500 trees, max_features {s[0]['max_features']}, min_samples_leaf {s[0]['min_samples_leaf']}, "
                   f"class_weight {s[0]['class_weight']}") if s else "—"
        epochs = "—"
    elif s:
        setting = (f"AdamW lr {s[0]['lr']:g}, wd {s[0]['weight_decay']:g}, batch {s[0]['batch_size']}, cosine schedule, "
                   f"max {s[0]['max_epochs']} epochs, early stopping (patience {config.EARLY_STOP_PATIENCE}), "
                   f"aug. {s[0]['augment']}, {s[0]['balance']} loss")
        epochs = " · ".join(f"{x['best_epoch']}/{x['epochs_run']}" for x in s)
    else:
        setting, epochs = "—", "—"
    inf = "—" if infer is None else mean_sd_str(
        infer.loc[(infer["model"] == model) & (infer["n_classes"] == 4), "infer_s_per_1000"], 3)
    lines.append(f"| {MODEL_LABELS[model]} | {inp} | {pre} | {params} | {setting} | {epochs} | "
                 f"{mean_sd_str(runs['train_time_s'])} | {inf} |")
lines.append("\nRandom forest: 500 trees; its size (number of nodes) is not comparable to network parameters. "
             "Best epoch is 0-based; seeds 0 · 1 · 2. The tuning budget was 12 configurations per model (val).")
text = "\n".join(lines) + "\n"
(config.RESULTS_DIR / "table1_methods.md").write_text(text)
(config.ROOT / "notes" / "paper_draft" / "table1_methods.md").write_text(text)
print(text)
