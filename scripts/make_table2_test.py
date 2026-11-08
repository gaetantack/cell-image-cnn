"""Table 2 for the paper (TEST, 4 classes) -> notes/paper_draft/table2_test.md.

Reads only files that already exist; evaluates nothing:
- results/table_test_summary.md (written once by scripts/final_test.py --split test): the 4-class table;
- results/table_val_summary_run2.md: train time per model (training is not repeated on test day, so train time
  comes from the run-2 final runs on the training set).
Adds an Eulenberg et al. 2017 reference row with [CHECK] placeholders (copy their numbers from the paper by hand).
Run: python -m scripts.make_table2_test
"""
from src import config

KEEP = ["Model", "Seeds", "Macro-F1 (test)", "Balanced acc. (test)", "Recall G1", "Recall S", "Recall G2",
        "Recall Mitosis", "ROC-AUC macro", "PR-AUC macro", "Inference (s/1,000 cells)"]


def first_table_after(text, heading):
    """Rows (list of cell lists) of the first markdown table after a heading line."""
    lines = text.split("\n")
    start = lines.index(heading)
    rows = []
    for line in lines[start + 1:]:
        if line.startswith("|"):
            if not line.startswith("|---"):
                rows.append([c.strip() for c in line.strip("|").split("|")])
        elif rows:
            break
    return rows


if __name__ == "__main__":
    test = first_table_after((config.RESULTS_DIR / "table_test_summary.md").read_text(), "## 4 classes")
    val = first_table_after((config.RESULTS_DIR / "table_val_summary_run2.md").read_text(), "## 4 classes")
    vh = val[0]
    train_time = {r[0]: r[vh.index("Train time (s)")] for r in val[1:]}
    th = test[0]
    idx = [th.index(c) for c in KEEP]
    header = KEEP + ["Train time (s, run 2)"]
    out = ["# Table 2 (TEST, 4 classes; mean ± SD over seeds 0–2)", "",
           "**Test numbers, one run on 2026-11-07 (tag `test-frozen`).** Source: `results/table_test_summary.md` "
           "(`scripts/final_test.py --split test`); train time from `results/table_val_summary_run2.md` (training on the "
           "training set, not repeated). Inference on CPU (4 threads). Made by `scripts/make_table2_test.py`.", "",
           "| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for r in test[1:]:
        out.append("| " + " | ".join([r[i] for i in idx] + [train_time.get(r[0], "—")]) + " |")
    out.append("| Eulenberg et al. 2017 (reference; different split and metric) | [CHECK] | [CHECK] | [CHECK] | "
               + " | ".join(["[CHECK]"] * (len(header) - 4)) + " |")
    out += ["", "Chance PR-AUC (macro) = 0.250; chance PR-AUC for mitosis = its test prevalence 0.022. "
            "Eulenberg et al. 2017: [CHECK reported metric (accuracy?), class set (7 phases or merged?), input "
            "(brightfield + darkfield?), split (theirs, not ours)]; not directly comparable."]
    path = config.RESULTS_DIR.parent / "notes" / "paper_draft" / "table2_test.md"
    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
