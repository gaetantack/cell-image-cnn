"""Print the training arguments of the best tuning run of one CNN family (highest val macro-F1).

Used by scripts/queue_phase3.sh:  python -m scripts.best_cnn_config smallcnn
-> "--lr 0.0005 --weight-decay 0.0001 --batch-size 128"
Only 4-class tuning runs (run_tag tuneNN, seed 0) are considered.
"""
import json
import sys

from src.experiments import read_log

model = sys.argv[1]
log = read_log()
tune = log[(log["model"] == model) & (log["n_classes"] == 4)
           & log["run_id"].astype(str).str.match(rf"^{model}_c4_tune\d\d_s0$")]
if len(tune) == 0:
    sys.exit(f"no tuning runs for {model}")
best = json.loads(tune.loc[tune["macro_f1"].idxmax(), "settings"])
print(f"--lr {best['lr']} --weight-decay {best['weight_decay']} --batch-size {best['batch_size']}")
