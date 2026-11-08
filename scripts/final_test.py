"""The one final evaluation of the frozen models (test day, 14 Nov). Tonight (run 4) it is run on VAL only.

    python -m scripts.final_test                     # --split val (default): a dress rehearsal on val
    python -m scripts.final_test --split test --i-understand-this-is-the-one-test-run   # test day ONLY

Without the flag, `--split test` stops with an error before ANY data is loaded.

What it evaluates (frozen beforehand, see PROJECT.md "Final configuration per model"):
- CNNs: the run-2 final checkpoints (run tag n2_final, seeds 0-2): smallcnn, resnet18_pretrained,
  resnet18_scratch on 4 classes; smallcnn also on 7 classes (the 7-class ResNets were dropped, user 2026-11-01).
  Nothing is re-trained; the normalisation is recomputed from the same training cells (src.interpret.load_trained).
- Random forest: re-fitted on TRAIN with the configuration of the best run-1 tuning run (scripts/run_rf.py),
  seeds 0-2, 4 and 7 classes, plus the raw_dark side experiment (4 classes). oob_score=True: the out-of-bag
  accuracy (each tree scored on the training cells left out of its bootstrap sample) is logged next to the
  eval accuracy. OOB does not change the fitted trees.
- Chance: DummyClassifier majority and stratified (as scripts/run_baselines.py), seeds 0-2. For ROC-AUC/PR-AUC
  the chance "probabilities" are the training class prior for every cell, so ROC-AUC = 0.5 and
  PR-AUC = class prevalence in the evaluated split.

Per model and seed: macro-F1, balanced accuracy, per-class recall (src.evaluate.evaluate), one-vs-rest
ROC-AUC and PR-AUC per class + macro (src.evaluate.ranking_metrics), and inference time per 1,000 cells
(CPU, after one warm-up batch, data already in memory).

Outputs (<split> = val or test):
- results/table_test_summary.md (test) or results/table_val_summary_finaltest.md (val; table_val_summary.md is
  the run-1 table of make_results_table.py) + results/table_<split>_summary_per_run.csv: mean +- SD over seeds, same layout as table_val_summary_run2.md
  plus ROC-AUC/PR-AUC/inference-time columns and a per-class PR-AUC table with the prevalence row.
- results/preds/finaltest_<split>/<run_id>.npz: y_true, y_pred, proba, rows (for confusion matrices).
- val: one row per model and seed appended to results/experiments.csv (eval_split val, run tag
  n4_finaltest_val; extra metrics in the settings JSON, so the CSV columns stay unchanged), and
  results/final_test_val_check.csv: macro-F1 here vs the logged run of the same model/seed (must match to 3 decimals).
- test: results/test_results.csv (src.experiments refuses to log test rows into experiments.csv). Refuses to
  run if that file already exists: the test set is evaluated once.
"""
import argparse
import sys

TEST_FLAG = "--i-understand-this-is-the-one-test-run"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument(TEST_FLAG, dest="test_ok", action="store_true", help="required for --split test (test day only)")
    args = p.parse_args(argv)
    if args.split == "test" and not args.test_ok:
        p.error(f"refusing to evaluate the TEST split without {TEST_FLAG} (test day only, see "
                "notes/test_day_checklist.md)")
    return args


if __name__ == "__main__":
    ARGS = parse_args()  # the refusal happens here, before any data or model is loaded

import json  # noqa: E402  (imports after the refusal on purpose)
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402

from scripts.make_results_table import fmt, mean_sd  # noqa: E402
from scripts.run_rf import best_params, load_features, make_rf  # noqa: E402
from src import config  # noqa: E402
from src.data import load_dataset  # noqa: E402
from src.evaluate import evaluate, ranking_metrics  # noqa: E402
from src.experiments import COLUMNS, has_run, log_experiment, read_log  # noqa: E402
from src.interpret import load_trained  # noqa: E402
from src.plotting import MODEL_LABELS  # noqa: E402
from src.splits import load_split  # noqa: E402
from src.train import predict_proba  # noqa: E402

CNN_TAG = "n2_final"
CLASS_NAMES = {4: config.CLASSES_4, 7: config.CLASSES_7}
# (model, n_classes) in table order; RF feature set per model name
EVALUATED = [("dummy_majority", 4), ("dummy_stratified", 4), ("random_forest", 4), ("smallcnn", 4),
             ("resnet18_pretrained", 4), ("resnet18_scratch", 4), ("random_forest_raw_dark", 4),
             ("dummy_majority", 7), ("dummy_stratified", 7), ("random_forest", 7), ("smallcnn", 7)]
RF_FEATURE_SET = {"random_forest": "main", "random_forest_raw_dark": "raw_dark"}
RF_PREFIX = {"random_forest": "rf", "random_forest_raw_dark": "rf_raw_dark"}
CPU = torch.device("cpu")


def source_run_id(model, k, seed):
    """run_id of the logged val run that this evaluation should reproduce (None for chance: no run_id)."""
    if model in RF_PREFIX:
        return f"{RF_PREFIX[model]}_c{k}_final_s{seed}"
    if model.startswith("dummy"):
        return None
    return f"{model}_c{k}_{CNN_TAG}_s{seed}"


def per_1000(seconds, n):
    return 1000 * seconds / n


def eval_cnn(model, k, seed, X, rows):
    net, mean, sd = load_trained(source_run_id(model, k, seed), device=CPU)
    imgs = torch.from_numpy(X[rows])
    predict_proba(net, imgs[:64], mean, sd)  # warm-up
    t0 = time.perf_counter()
    proba = predict_proba(net, imgs, mean, sd)
    return proba, per_1000(time.perf_counter() - t0, len(rows)), {}


def eval_rf(model, k, seed, rf_params, train, rows):
    F, y = load_features(k, RF_FEATURE_SET[model])
    pipe = make_rf(rf_params, seed)
    pipe.steps[-1][1].set_params(oob_score=True)  # out-of-bag accuracy; the fitted trees are unchanged
    t0 = time.perf_counter()
    pipe.fit(F[train], y[train])
    fit_s = time.perf_counter() - t0
    pipe.predict_proba(F[rows][:64])  # warm-up
    t0 = time.perf_counter()
    proba = pipe.predict_proba(F[rows])
    infer = per_1000(time.perf_counter() - t0, len(rows))
    forest = pipe.steps[-1][1]
    n_features = F.shape[1]
    mf = forest.max_features
    mtry = int(max(1, mf * n_features)) if isinstance(mf, float) else int(np.sqrt(n_features))
    extra = {"oob_accuracy": round(float(forest.oob_score_), 4), "fit_time_s": round(fit_s, 1),
             "n_features": n_features, "mtry": mtry}
    return proba, infer, extra


def eval_dummy(model, k, seed, y, train, rows):
    strategy = "most_frequent" if model == "dummy_majority" else "stratified"
    clf = DummyClassifier(strategy=strategy, random_state=seed).fit(np.zeros((len(train), 1)), y[train])
    t0 = time.perf_counter()
    pred = clf.predict(np.zeros((len(rows), 1)))
    infer = per_1000(time.perf_counter() - t0, len(rows))
    prior = np.bincount(y[train], minlength=k) / len(train)  # class prior: ROC-AUC 0.5, PR-AUC = prevalence
    return pred, np.tile(prior, (len(rows), 1)), infer


def main(args):
    split = args.split
    torch.set_num_threads(4)
    rows = load_split(split, final_test=(split == "test"))
    train = load_split("train")
    out_dir = config.PREDS_DIR / f"finaltest_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_file = config.RESULTS_DIR / "test_results.csv"
    if split == "test" and test_file.exists():
        sys.exit(f"{test_file} exists: the test set has already been evaluated once. Nothing was run.")
    rf_params, rf_source = best_params()
    print(f"split {split}: {len(rows)} cells; RF configuration from {rf_source}: {rf_params}")
    X = None
    results = []
    for model, k in EVALUATED:
        _, y, _ = load_dataset(k)
        for seed in config.SEEDS:
            src_id = source_run_id(model, k, seed)
            name = RF_PREFIX.get(model, model)
            run_id = f"{name}_c{k}_n4_finaltest_{split}_s{seed}"
            extra = {}
            if model.startswith("dummy"):
                pred, proba, infer = eval_dummy(model, k, seed, y, train, rows)
            elif model in RF_PREFIX:
                proba, infer, extra = eval_rf(model, k, seed, rf_params, train, rows)
                pred = proba.argmax(1)
            else:
                if X is None:
                    X, _, _ = load_dataset(4)  # images are the same for 4 and 7 classes
                proba, infer, _ = eval_cnn(model, k, seed, X, rows)
                pred = proba.argmax(1)
            m = evaluate(y[rows], pred, CLASS_NAMES[k])
            r = ranking_metrics(y[rows], proba, CLASS_NAMES[k])
            np.savez(out_dir / f"{run_id}.npz", y_true=y[rows], y_pred=pred, proba=proba, rows=rows)
            res = {"model": model, "n_classes": k, "seed": seed, "run_id": run_id, "source_run_id": src_id,
                   "macro_f1": m["macro_f1"], "balanced_accuracy": m["balanced_accuracy"],
                   "accuracy": float((pred == y[rows]).mean()),
                   "roc_auc_macro": r["roc_auc_macro"], "pr_auc_macro": r["pr_auc_macro"],
                   "infer_s_per_1000": infer, **extra}
            res.update({f"recall_{c}": v for c, v in m["recall"].items()})
            res.update({f"roc_auc_{c}": v for c, v in r["roc_auc"].items()})
            res.update({f"pr_auc_{c}": v for c, v in r["pr_auc"].items()})
            res.update({f"prevalence_{c}": v for c, v in r["prevalence"].items()})
            results.append(res)
            print(f"{run_id:45s} macro-F1 {m['macro_f1']:.4f}  PR-AUC macro {r['pr_auc_macro']:.3f}  "
                  f"{infer:.3f} s/1000 cells {extra if extra else ''}", flush=True)
            if split == "val" and not has_run(run_id):
                settings = {"run_tag": f"n4_finaltest_{split}", "source_run_id": src_id,
                            "extra_metrics": {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in res.items()
                                              if k2.startswith(("roc_auc", "pr_auc", "infer", "oob", "mtry", "accuracy"))}}
                log_experiment(model, k, split, seed, settings, m, extra.get("fit_time_s", 0.0),
                               run_id=run_id, device="cpu")
    df = pd.DataFrame(results)
    df.to_csv(config.RESULTS_DIR / f"table_{split}_summary_per_run.csv", index=False)
    if split == "test":
        df.to_csv(test_file, index=False)
    else:
        check_against_log(df)
    md = to_markdown(df, split)
    # val: "_finaltest" suffix, because results/table_val_summary.md is the run-1 table of make_results_table.py
    (config.RESULTS_DIR / f"table_{split}_summary{'_finaltest' if split == 'val' else ''}.md").write_text(md)
    print(md)


def check_against_log(df):
    """val only: macro-F1 here vs the logged val row of the same model/seed (CNN: n2_final run; RF: run-1 final
    run; chance: run-1 dummy rows). Writes results/final_test_val_check.csv."""
    log = read_log()
    rows = []
    for _, r in df.iterrows():
        if isinstance(r["source_run_id"], str):
            logged = log.loc[log["run_id"] == r["source_run_id"], "macro_f1"]
        else:  # chance rows have no run_id: match model, classes, seed and a val row without run tag
            logged = log.loc[(log["model"] == r["model"]) & (log["n_classes"] == r["n_classes"])
                             & (log["seed"] == r["seed"]) & (log["run_id"].isna()), "macro_f1"]
        logged = float(logged.iloc[0]) if len(logged) else np.nan
        rows.append({"model": r["model"], "n_classes": r["n_classes"], "seed": r["seed"],
                     "source_run_id": r["source_run_id"] if isinstance(r["source_run_id"], str) else "(dummy row, no run_id)",
                     "macro_f1_logged": logged, "macro_f1_now": round(r["macro_f1"], 4),
                     # "matches to 3 decimals" = differs by less than 0.0005 (rounding both to 3 decimals
                     # can split values such as 0.7385 vs 0.73848 across 0.738 / 0.739)
                     "abs_diff": round(abs(logged - r["macro_f1"]), 5),
                     "match_3dp": abs(logged - r["macro_f1"]) < 5e-4})
    check = pd.DataFrame(rows)
    check.to_csv(config.RESULTS_DIR / "final_test_val_check.csv", index=False)
    print(check.to_string(index=False))
    print(f"{int(check['match_3dp'].sum())}/{len(check)} match the logged val macro-F1 to 3 decimals")


def to_markdown(df, split):
    title = "Validation" if split == "val" else "TEST"
    lines = [f"# {title} summary (mean ± SD over seeds 0, 1, 2), made by scripts/final_test.py --split {split}", ""]
    if split == "val":
        lines += ["**Validation results — a dress rehearsal of the test-day script, not the final test numbers.**", ""]
    lines += ["Frozen models: CNN run-2 finals (`n2_final`); random forest re-fitted on train with the run-1 "
              "best configuration; chance = DummyClassifier. ROC-AUC/PR-AUC: one-vs-rest from class probabilities "
              "(chance: training class prior → ROC-AUC 0.5, PR-AUC = prevalence). Inference: CPU, 4 threads, "
              "s per 1,000 cells. OOB = out-of-bag accuracy of the random forest (eval accuracy next to it).", ""]
    for k in (4, 7):
        sub = df[df["n_classes"] == k]
        classes = CLASS_NAMES[k]
        lines += [f"## {k} classes", "",
                  f"| Model | Seeds | Macro-F1 ({split}) | Balanced acc. ({split}) | "
                  + " | ".join(f"Recall {c}" for c in classes)
                  + " | ROC-AUC macro | PR-AUC macro | Inference (s/1,000 cells) | Accuracy | OOB accuracy |",
                  "|---" * (9 + len(classes)) + "|"]
        for model, g in sub.groupby("model", sort=False):
            cell = lambda col, d=3: fmt(*mean_sd(g[col]), d) if col in g and g[col].notna().all() else "—"
            lines.append(f"| {MODEL_LABELS.get(model, model)} | {len(g)} | {cell('macro_f1')} | "
                         f"{cell('balanced_accuracy')} | " + " | ".join(cell(f'recall_{c}', 2) for c in classes)
                         + f" | {cell('roc_auc_macro')} | {cell('pr_auc_macro')} | {cell('infer_s_per_1000', 3)} | "
                         f"{cell('accuracy')} | {cell('oob_accuracy')} |")
        lines += ["", f"PR-AUC per class ({k} classes; the prevalence row is the PR-AUC of a chance model):", "",
                  "| Model | " + " | ".join(classes) + " |", "|---" * (1 + len(classes)) + "|"]
        prev = sub.iloc[0]
        lines.append("| Prevalence (chance PR-AUC) | " + " | ".join(f"{prev[f'prevalence_{c}']:.3f}" for c in classes) + " |")
        for model, g in sub.groupby("model", sort=False):
            if model.startswith("dummy"):
                continue
            lines.append(f"| {MODEL_LABELS.get(model, model)} | "
                         + " | ".join(fmt(*mean_sd(g[f'pr_auc_{c}'])) for c in classes) + " |")
        lines.append("")
    rf = df[df["model"] == "random_forest"]
    if len(rf):
        lines += [f"Random forest mtry: max_features {int(rf['mtry'].iloc[0])} of {int(rf['n_features'].iloc[0])} features "
                  f"per split (sklearn default sqrt would be {np.sqrt(rf['n_features'].iloc[0]):.1f}).", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main(ARGS)
