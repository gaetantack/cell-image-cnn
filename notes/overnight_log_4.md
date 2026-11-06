# Overnight log, run 4: 2026-11-06 → 2026-11-07

Branch `overnight-run-4` (from `overnight-run-2` @ 49db6bf). All numbers are VALIDATION numbers ("val"). The test split is never loaded. Every run tonight has a run tag starting with `n4_`; existing rows in `results/experiments.csv` are never changed. Ch4 is never used.

| Time | Step | Result | Next |
|---|---|---|---|
| 00:21 | Start-up (the session started after midnight, so the night is 7 Nov 00:21 → 07:00). Read CLAUDE.md, PROJECT.md, `notes/overnight_report_run2.md` (incl. follow-ups), LEARN.md headings, run-2 log, core code. `git status`: clean except untracked `try_mask.py` (not mine; left alone) | | branch |
| 00:22 | Branch `overnight-run-4` created from `overnight-run-2`; `pytest` | 42 passed | time guard |
| 00:23 | **Time guard fixed** in `scripts/queue_common.sh`: new `fits <minutes>` compares real clock times (epoch seconds) with a deadline fixed once when the queue starts: 06:30 today if started before 06:30, else 06:30 tomorrow; `QUEUE_DEADLINE="YYYY-MM-DD HH:MM"` overrides it for daytime runs. Deadline computed in Python because the conda env puts GNU `date` on the PATH (`date -j` failed). Tested: at 00:23, `fits 300` true, `fits 400` false; override 01:00 → `fits 10` true, `fits 60` false. The old run-2 queues keep their own (old) `fits`; only new queues use the shared one | works | budget |
| 00:25 | **Time budget** (worst case from run 2: SmallCNN ≤ 8 min/run, ResNet18 ≤ 33 min/run; masks/code ≈ 40 min first). Queue start ≈ 01:10. 3b.1 SmallCNN cell-only 3 × 8 = 24 min; 3b.2 SmallCNN bg-only 24 min; 3b.3 ResNet18-pre cell-only 3 × 33 = 99 min; 3b.4 ResNet18-pre bg-only s0 33 min; 6.3 label-shuffle SmallCNN 3 × 8 = 24 min. Total worst case ≈ 204 min → ends ≈ 04:35; realistic (early stopping) ≈ 03:45. All fits before 06:30 with ≈ 2 h slack. CPU work (inference masking check, border check seeds 1–2, final_test.py on val, figures, silhouette, Grad-CAM) runs alongside | fits | masks |
