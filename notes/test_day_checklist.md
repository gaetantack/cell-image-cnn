# Test-day checklist (7 Nov 2026): the ONE test run

Drafted overnight 2026-11-07 by Claude (run 4). Read it fully before starting. The test split is evaluated **once**; nothing is re-run, re-tuned or re-selected afterwards, whatever the numbers are.

## Before (all must be true)
1. PROJECT.md §7a "Final configuration per model (frozen before the test)" is confirmed by you (remove "DRAFT").
2. The val dress rehearsal reproduces the logged val numbers: `results/final_test_val_check.csv` shows `match_3dp = True` for every row.
3. Every checkpoint exists: `ls checkpoints/*_n2_final_s*_best.pt` → 12 files (3 families × 3 seeds for 4 classes + SmallCNN 7-class × 3).
4. The working tree is clean and on the paper branch: `git status` (only untracked scratch files), `pytest` passes.
5. No GPU job is running (`pgrep -f src.train` prints nothing); inference runs on the CPU, so timings are comparable.

## The run (exact commands)
```bash
conda activate cell-image-cnn
# cd into the repository folder
git status && pytest -q
git log -1 --format=%H > results/test_code_commit.txt   # the code version that produced the test numbers
python -m scripts.final_test --split test --i-understand-this-is-the-one-test-run 2>&1 | tee logs/final_test_test.out
git add results/preds/finaltest_test results/test_results.csv results/table_test_summary.md results/table_test_summary_per_run.csv results/test_code_commit.txt
git commit -m "Final test: one run on the test split (7 Nov)"
git tag test-frozen
git push origin HEAD --tags
```
`final_test.py` refuses to run a second time if `results/test_results.csv` already exists. Never delete that file to re-run.
Predictions (with class probabilities) are written to `results/preds/finaltest_test/`; `results/preds/` is tracked by git, so add that folder to the commit too.

## After: what to remake with test numbers (scripts only, never by hand)
| Item | Source | Command / to do |
|---|---|---|
| Table 2 (test metrics, per-class recall, ROC-AUC/PR-AUC, inference time, train time) | `results/table_test_summary.md` | Copy the 4-class table into the paper. Add an **Eulenberg et al. 2017** reference row with [CHECK] placeholders: their reported accuracy [CHECK], their class set (7 phases, [CHECK] whether merged), their input (brightfield + darkfield [CHECK]), their split (not ours; not directly comparable). Train time: from `results/table_val_summary_run2.md` (training is not repeated) |
| Fig 2 (macro-F1 / balanced accuracy per model with chance line) | `results/table_test_summary_per_run.csv` | `python -m scripts.fig_final --split test` (val dress rehearsal: `--split val`) |
| Confusion matrices RF vs best CNN (4 and 7 classes) | `results/preds/finaltest_test/*.npz` | `python -m scripts.fig_final --split test` (same script; best CNN = the one chosen on VAL, ResNet18 ImageNet — do not re-choose on test) |
| Abstract / results text | — | Replace "val" numbers that are reported as the main result; keep val numbers for RQ2–RQ5 analyses (they are validation analyses) |

## Never on test
Re-training, re-tuning, choosing a model, threshold or seed on test numbers; running the masking, Grad-CAM, UMAP, border or learning-curve analyses on test (they stay validation analyses, and the paper says so).
