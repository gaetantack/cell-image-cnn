"""Run 2, Phase A: summarise the augmentation x balancing check and choose AUG*/BAL*.

Reads the SmallCNN rows with run tag n2_A_<augment>_<balance> (seeds 0, 1, 2; VALIDATION) from
results/experiments.csv and writes
  results/run2_phaseA_val.csv / .md   one row per setting: mean ± SD of val macro-F1 and mitosis recall,
                                      best epochs and epochs run (to see whether 25 epochs is a cap)
  results/run2_aug_bal_choice.json    the chosen setting and the reason (read by scripts/queue_run2_BC.sh)

Choice rule (fixed in the overnight prompt before the runs):
1. Rank the settings by mean val macro-F1.
2. If the best two differ by more than one seed SD, take the best. "One seed SD" = the larger of the two
   settings' SDs (if either setting's seed noise covers the gap, the gap is not meaningful).
3. Otherwise take the one with the higher mean mitosis recall, if that difference is larger than one SD
   (again the larger of the two mitosis-recall SDs).
4. Otherwise prefer, in this order: augment vflip > none > basic (vflip still regularises; basic removes the
   darkfield direction found in run 1), then balance weighted > oversample (simpler).
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.experiments import read_log

AUG_PREFERENCE = {"vflip": 0, "none": 1, "basic": 2}  # lower = preferred in a tie
BAL_PREFERENCE = {"weighted": 0, "oversample": 1}


def phase_a_table(log):
    rows = []
    for aug in config.RUN2_AUGMENTS:
        for bal in config.RUN2_BALANCES:
            ids = [f"smallcnn_c4_{config.RUN2_TAG}A_{aug}_{bal}_s{s}" for s in config.SEEDS]
            runs = log[log["run_id"].isin(ids)].sort_values("seed")
            if len(runs) == 0:
                continue
            f1 = runs["macro_f1"].to_numpy(dtype=float)
            mit = np.array([json.loads(r)["Mitosis"] for r in runs["recall"]])
            settings = [json.loads(s) for s in runs["settings"]]
            rows.append({
                "augment": aug, "balance": bal, "n_seeds": len(runs),
                "macro_f1_mean": f1.mean(), "macro_f1_sd": f1.std(ddof=1) if len(f1) > 1 else np.nan,
                "mitosis_recall_mean": mit.mean(), "mitosis_recall_sd": mit.std(ddof=1) if len(mit) > 1 else np.nan,
                "macro_f1_per_seed": " / ".join(f"{v:.3f}" for v in f1),
                "best_epoch_per_seed": " / ".join(str(s["best_epoch"]) for s in settings),
                "epochs_run_per_seed": " / ".join(str(s["epochs_run"]) for s in settings),
                "train_time_s_mean": runs["train_time_s"].mean(),
            })
    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def choose(table):
    """Apply the rule in the module docstring to the two best settings. Returns (row, reason)."""
    a, b = table.iloc[0], table.iloc[1]
    name = lambda r: f"{r['augment']}/{r['balance']}"
    gap, sd = a["macro_f1_mean"] - b["macro_f1_mean"], max(a["macro_f1_sd"], b["macro_f1_sd"])
    if gap > sd:
        return a, (f"{name(a)} has the highest mean val macro-F1 ({a['macro_f1_mean']:.3f}); "
                   f"gap to {name(b)} {gap:.3f} > one seed SD {sd:.3f}")
    reason = f"best two {name(a)} and {name(b)} differ by {gap:.3f} <= one seed SD {sd:.3f}; "
    mgap = a["mitosis_recall_mean"] - b["mitosis_recall_mean"]
    msd = max(a["mitosis_recall_sd"], b["mitosis_recall_sd"])
    if abs(mgap) > msd:
        win = a if mgap > 0 else b
        return win, reason + (f"{name(win)} has the higher mean mitosis recall "
                              f"(difference {abs(mgap):.3f} > SD {msd:.3f})")
    key = lambda r: (AUG_PREFERENCE[r["augment"]], BAL_PREFERENCE[r["balance"]])
    win = min((a, b), key=key)
    return win, reason + (f"mitosis recall also within one SD ({abs(mgap):.3f} <= {msd:.3f}); "
                          f"tie-break prefers {name(win)} (vflip > none > basic, then weighted > oversample)")


def to_markdown(table, chosen, reason):
    lines = ["# Run 2, Phase A: augmentation × balancing, SmallCNN, 4 classes (VALIDATION)", "",
             "Run-1 best SmallCNN config (lr 2e-3, wd 1e-4, batch 64), max 25 epochs, early stopping "
             f"(patience {config.EARLY_STOP_PATIENCE}) on val macro-F1. Mean ± SD over the seeds listed. "
             "Best epoch is 0-based. Made by `scripts/choose_aug_bal.py`.", "",
             "| Augment | Balance | Seeds | Macro-F1 (val) | Mitosis recall (val) | Macro-F1 per seed | "
             "Best epoch per seed | Epochs run per seed | Train time (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['augment']} | {r['balance']} | {r['n_seeds']} | {r['macro_f1_mean']:.3f} ± "
                     f"{r['macro_f1_sd']:.3f} | {r['mitosis_recall_mean']:.2f} ± {r['mitosis_recall_sd']:.2f} | "
                     f"{r['macro_f1_per_seed']} | {r['best_epoch_per_seed']} | {r['epochs_run_per_seed']} | "
                     f"{r['train_time_s_mean']:.0f} |")
    lines += ["", f"**Chosen: augment = {chosen['augment']}, balance = {chosen['balance']}.** {reason}.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    table = phase_a_table(read_log())
    if len(table) < 2:
        raise SystemExit("fewer than two Phase A settings in the log")
    if (table["n_seeds"] < len(config.SEEDS)).any():
        print("WARNING: some settings have fewer than 3 seeds:", table[["augment", "balance", "n_seeds"]].to_dict("records"))
    chosen, reason = choose(table)
    table.to_csv(config.RESULTS_DIR / "run2_phaseA_val.csv", index=False)
    md = to_markdown(table, chosen, reason)
    (config.RESULTS_DIR / "run2_phaseA_val.md").write_text(md)
    config.RUN2_CHOICE_FILE.write_text(json.dumps(
        {"augment": chosen["augment"], "balance": chosen["balance"], "reason": reason}, indent=2))
    print(md)
