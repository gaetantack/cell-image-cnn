"""Table 1 (methods overview): model, input, pretrained, parameters, training time (mean ± SD over the
three final 4-class runs). Written to results/table1_methods.md and notes/paper_draft/table1_methods.md."""
import numpy as np

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

log = read_log()
lines = ["# Table 1. Methods overview", "",
         "Training time: wall-clock seconds per final 4-class run (mean ± SD over seeds 0–2; CNNs on Apple M2 GPU "
         "(MPS), random forest on 8 CPU cores), including the per-epoch validation pass. BF = brightfield (Ch3), "
         "DF = darkfield (Ch6). Chance baselines (majority, stratified) have no input features.", "",
         "| Model | Input | Pretrained | Parameters (4 classes) | Training time (s) |", "|---|---|---|---|---|"]
for model, inp, pre in ROWS:
    params = "—" if model == "random_forest" else f"{count_parameters(build_model(model, 4)):,}"
    t = final_runs(log, model, 4)["train_time_s"].to_numpy()
    time_str = "not run yet" if len(t) == 0 else (f"{t.mean():.0f} ± {t.std(ddof=1):.0f}" if len(t) > 1 else f"{t[0]:.0f}")
    lines.append(f"| {MODEL_LABELS[model]} | {inp} | {pre} | {params} | {time_str} |")
lines.append("\nRandom forest: 500 trees; its size (number of nodes) is not comparable to network parameters.")
text = "\n".join(lines) + "\n"
(config.RESULTS_DIR / "table1_methods.md").write_text(text)
(config.ROOT / "notes" / "paper_draft" / "table1_methods.md").write_text(text)
print(text)
