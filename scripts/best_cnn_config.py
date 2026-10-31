"""Print the training arguments of the best tuning run of one CNN family (highest val macro-F1).

Used by the queue scripts:
    python -m scripts.best_cnn_config smallcnn            -> run-1 tuning runs (run tag tune00..tune11)
    python -m scripts.best_cnn_config smallcnn n2_tune    -> run-2 tuning runs (run tag n2_tune00..n2_tune11)
Output, e.g. "--lr 0.0005 --weight-decay 0.0001 --batch-size 128".
Only 4-class tuning runs (seed 0) are considered.
"""
import json
import sys

from src.experiments import read_log

model = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "tune"  # run-tag prefix of the tuning runs
log = read_log()
tune = log[(log["model"] == model) & (log["n_classes"] == 4)
           & log["run_id"].astype(str).str.match(rf"^{model}_c4_{prefix}\d\d_s0$")]
if len(tune) == 0:
    sys.exit(f"no tuning runs for {model} with prefix {prefix}")
best = json.loads(tune.loc[tune["macro_f1"].idxmax(), "settings"])
print(f"--lr {best['lr']} --weight-decay {best['weight_decay']} --batch-size {best['batch_size']}")
