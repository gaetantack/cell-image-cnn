# Overnight log, run 2: 2026-10-31 → 2026-11-01

Branch `overnight-run-2` (from `overnight-2026-10-25` @ 4ae09bc). All numbers are VALIDATION numbers. Test split never loaded. Every run tonight has a run tag starting with `n2_`, and run-1 rows in `results/experiments.csv` are untouched.

| Time | Step | Result | Next |
|---|---|---|---|
| 21:33 | Start-up: read CLAUDE.md, PROJECT.md, run-1 report, run-1 log. `git status`: clean except untracked `try_mask.py` (not mine; left alone, not committed) | | branch |
| 21:34 | Branch `overnight-run-2` created from `overnight-2026-10-25`; `pytest` | 37 passed | vflip |
| 21:35 | `src/train.py`: new `augment_vflip` (random up-down flip + the same intensity jitter as `basic`; no rotations, no left-right flips) and an `AUGMENTATIONS` table; `none` and `basic` unchanged. `tests/test_models.py`: `test_augment_vflip_never_flips_left_right`. `src/config.py`: `EARLY_STOP_PATIENCE` 5 → 6 (as the prompt asks; run-1 runs used 5), run-2 block (`RUN2_MAX_EPOCHS = 25`, `CNN_GRID_RUN2`, `RUN2_CHOICE_FILE`); run-1 values kept so run 1 stays reproducible | pytest 38 passed; smoke run with `--augment vflip` OK | speed |
| 21:36 | Speed re-measured on MPS (batch 64, train steps only, random data) | SmallCNN 1585 img/s ≈ 14 s/epoch; ResNet18 128 px 303 img/s ≈ 75 s/epoch: same as run 1. Real SmallCNN epoch incl. val ≈ 15.5 s | budget |
| 21:37 | Git: no user identity configured on this host, so the first commit failed. Commits use the same author as all previous commits (`Gaetan Tack <gaetantack@Gaetans-MacBook-Air.local>`), set per command with GIT_AUTHOR_*/GIT_COMMITTER_* environment variables; git config not changed | committed 12f25a3 | launch Phase A |
| 21:37 | Launched `scripts/queue_run2_A.sh` (18 SmallCNN runs, seed loop outermost) | running | budget |
| 21:40 | **Time budget** (stop starting runs so all ends by 06:30; wrap-up 06:30–07:00). Phase A: 18 SmallCNN runs × ≤ 25 epochs × 16 s ≤ 7 min, likely ≈ 4–5 min with early stopping → ≈ 1.5 h (worst 2.1 h) → ends ≈ 23:10 (worst 23:45). Phase B: SmallCNN 12 × 4 epochs ≈ 15 min, ResNet18-pre 12 × 4 × 80 s ≈ 65 min, scratch ≈ 65 min → ≈ 2.4 h → ≈ 01:35 (worst 02:10). Phase C: SmallCNN 3 × ≤ 7 min; ResNets 6 runs × ≤ 25 × 80 s ≤ 34 min each (≈ 3.4 h worst, ≈ 2.3 h if early stopping at ≈ 17 epochs) → ≈ 04:00 (worst 05:50). Phase D figures ≈ 20 min, mostly CPU, partly while Phase C runs. Phase E (7-class finals, then learning curve) only with what is left, guarded so nothing starts that cannot end by 06:30 | fits, with little slack in the worst case | choice script + B/C queue |
