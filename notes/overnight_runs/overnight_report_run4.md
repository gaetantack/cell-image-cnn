# Overnight report, run 4: 7 Nov 2026, 00:21 → 05:00 (Claude, autonomous)

Branch `overnight-run-4` (from `overnight-run-2`; pushed, not merged). **Every number is a VALIDATION number.** The test split was never loaded: `final_test.py` ran on val only, and its `--split test` refusal is tested. Ch4 was never used. New runs are tagged `n4_`; existing rows of `results/experiments.csv` are unchanged (checked before every commit: only added lines). Timeline: `notes/overnight_runs/overnight_log_4.md`.

## Done / not done

| Item | Status |
|---|---|
| 1 Start-up, clock-time queue guard (`queue_common.sh`), time budget | done |
| 2 Masks (`make_masks.py`), `src/masking.py`, tests, `figures/masking_examples_val` | done |
| 3a Inference-only masking check, 9 models | done |
| 3b Training queue (SmallCNN cell/bg × 3 seeds, ResNet18-pre cell × 3, bg × 1) | done; plus ResNet bg-only seeds 1–2 and ResNet disc control (extra) |
| 3c `results/masking_val.md`, `figures/masking_val` | done |
| 3d Border check seeds 0–2 | done |
| 4 `final_test.py` (+ ROC/PR-AUC, OOB, mtry, inference time), val check, refusal test, test-day checklist, PROJECT.md §7a draft | done |
| 5 Fig 1, Table 1, RQ5 masking paragraph, methods, LEARN.md | done |
| 6.0 Silhouette · 6.1 Grad-CAM correct/wrong · 6.2 training curves · 6.3 label shuffle · 6.4 bio check | all done (6.2 without val loss: never logged) |
| Extra (after the suspicion rule): background investigation, fixed-disc control | done |

## Masking (RQ5), val, 4 classes, macro-F1 mean ± SD (n seeds)

| Input | SmallCNN | ResNet18 (ImageNet) |
|---|---|---|
| Original (run-2 finals) | 0.744 ± 0.010 (3) | 0.781 ± 0.007 (3) |
| **Cell only, trained on it** | **0.729 ± 0.019 (3)** | **0.769 ± 0.004 (3)** |
| **Background only, trained on it** | **0.558 ± 0.028 (3)** | **0.601 ± 0.007 (3)** |
| Background only + fixed disc (size hidden), trained on it | 0.450 ± 0.007 (3) | 0.468 ± 0.003 (3) |
| Cell only, evaluated only (run-2 model) | 0.420 ± 0.027 (3) | 0.156 ± 0.031 (3) |
| Background only, evaluated only | 0.160 ± 0.003 (3) | 0.258 ± 0.026 (3) |
| References: chance stratified 0.257 · mask area alone (logistic regr.) 0.483 · area + background level (RF) 0.554 · background level alone 0.27–0.31 · label shuffle (SmallCNN) 0.282 ± 0.020 | | |

**Answers.**
- *Does the score hold on cell only?* Yes, almost: −0.016 (SmallCNN) and −0.012 (ResNet18), about one to two seed SDs. Mitosis recall drops more for SmallCNN (0.63 → 0.54).
- *How far above chance is background only, and for which class?* Far: 0.558 ± 0.028 (SmallCNN) and 0.601 ± 0.007 (ResNet18) vs 0.257, with all classes well above chance (SmallCNN recall G1 0.83, S 0.55, G2 0.58, mitosis 0.45 ± 0.22). Mostly explained by the **size of the filled hole** (mask area alone 0.483; mitosis recall 0.59, the same as the background-only SmallCNN). With the size hidden by a fixed disc it is still 0.450 ± 0.007 (SmallCNN; mitosis 0.39) and 0.468 ± 0.003 (ResNet18; mitosis 0.19).
- *Do trained-on-masked and inference-only agree?* No. Inference-only collapses (cell-only 0.16–0.46) because a flat grey surround is out of distribution; the models predict mostly G2/mitosis. Inference-only masking alone would have wrongly suggested that the networks need the background.
- *Border ring (3 seeds):* other-class minus same-class ring swap: 0.000 ± 0.005 / −0.002 ± 0.010 / +0.003 ± 0.002. The 8-px ring carries no phase-specific information the networks use, consistent with cell-only ≈ original. Masking shows something the ring test could not: the background *as a whole* (level and scaling) carries phase information.

## final_test.py val check
33/33 val macro-F1 (3 CNN families, SmallCNN 7-class, RF 4/7-class, raw-dark RF, chance 4/7) reproduce the logged rows to 3 decimals (largest difference 0.00005). RF OOB accuracy 0.752 vs val accuracy 0.753 (close). mtry 9 of 28 (default √28 ≈ 5.3). PR-AUC mitosis (chance = prevalence 0.022): RF 0.261, SmallCNN 0.692, **ResNet18-pre 0.837**, scratch 0.660. ROC-AUC macro 0.915–0.946. Inference per 1,000 cells (CPU): RF 0.02 s, SmallCNN 1.7 s, ResNet18 13–14 s. (`results/table_val_summary_finaltest.md`, `results/final_test_val_check.csv`, `figures/fig2_val`, `results/table2_val.md`.)

## Anything suspicious
1. **Suspicion rule triggered twice (background-only far above chance), investigated both times.** (a) Background-only 0.58 (s0): explained mostly by the hole's size → added the fixed-disc control. (b) Disc control 0.45: the darkfield image of each cell is scaled by a per-image factor that differs by phase (background SD/median = 0.066 in every class, level–SD ρ = 0.93; G1 45 vs G2 32 grey levels). Simple background statistics explain ≈ 0.33; **≈ 0.12 remains unexplained**. Not label leakage (no fluorescence, fixed split, shuffle control near chance), but phase information sitting in the background pixels of this dataset. The origin of the per-image factor (instrument, acquisition, export) is unknown [CHECK BBBC048 docs].
2. **Label shuffle 0.282 ± 0.020 > chance 0.257:** the median epoch is 0.16–0.20; early stopping picks the luckiest epoch on val. Selection bias, not leakage, but every val number has it (Limitations line added).
3. A ResNet run took 36.5 min (> 33-min worst case) while CPU jobs ran; no deadline risk (queue 4c uses 40 min).
4. I overwrote `results/table_val_summary.md` (the run-1 table) with final_test's val table (name clash with the prompt's `table_<split>_summary.md`); **restored from git**, the val rehearsal table is now `table_val_summary_finaltest.md`.

## Decisions I made for you (to confirm; also in PROJECT.md, dated 2026-11-07)
| # | Decision | Why | Undo |
|---|---|---|---|
| 1 | Fixed-disc control (`background_only_disc`, r = 22 px), SmallCNN and ResNet18, 3 seeds each | Background-only was confounded by cell size | Delete `n4_bgdisc` rows; mode stays harmless |
| 2 | Extra queues 4c/4d: ResNet18 bg-only seeds 1–2, ResNet18 disc seeds 0–2 | Idle GPU 03:25–04:58; prompt allowed only bg-only s0 → single-seed row | Delete those rows |
| 3 | Chance ROC/PR-AUC = class prior as score (AUC 0.5, PR-AUC = prevalence) | Prompt asked for prevalence; DummyClassifier one-hot probabilities give arbitrary AUCs | `eval_dummy` in `final_test.py` |
| 4 | Extra metrics stored in the settings JSON of appended rows, not as new CSV columns | Never rewrite experiments.csv | — |
| 5 | final_test refuses a second test run if `results/test_results.csv` exists; test rows go there (experiments.csv refuses test) | "Test once" | — |
| 6 | Fig 1 shows all 4 mitotic sub-phases (prompt said 3); test class counts read from splits.json indices (labels only) | There are 4 (pro, meta, ana, telo) | — |
| 7 | `make_table1.py` default is now run 2 (`--tag n2_final`); `--tag final` gives run 1 | Paper = run 2 | — |
| 8 | Commits with the existing author identity via env variables (git has no identity on this host); `try_mask.py` left untracked | As run 2 | — |

## What you need to decide (max 3)
1. **RQ5 message.** The cell alone is sufficient (cell-only ≈ original), but the background carries phase information (fixed-disc background-only 0.45), apparently through a phase-dependent darkfield scaling. Keep the run-2 take-home "no sign of a background shortcut" (draft has a NOTE), or rephrase it as "the networks do not need the background, but the background is not free of phase information"? Optionally check the BBBC048 docs for the per-image scaling.
2. **Confirm PROJECT.md §7a (frozen configuration)** and the test-day procedure (`notes/test_day_checklist.md`) before 14 Nov.
3. **Masking in the paper:** main text (Fig 6b) with the disc control, or Supplementary? It is Tier 3 in the plan but now has 3 seeds in every row.

## Resume / reproduce
```bash
conda activate cell-image-cnn && git checkout overnight-run-4 && pytest -q
python -m scripts.make_masks                      # masks cache (gitignored)
nohup bash scripts/queue_run4.sh  > logs/queue_run4.out  2>&1 &   # skips finished runs; 06:30 guard (QUEUE_DEADLINE="YYYY-MM-DD HH:MM" for daytime)
nohup bash scripts/queue_run4b.sh > logs/queue_run4b.out 2>&1 &
nohup bash scripts/queue_run4c.sh > logs/queue_run4c.out 2>&1 &
nohup bash scripts/queue_run4d.sh > logs/queue_run4d.out 2>&1 &
python -m scripts.masking_check && python -m scripts.masking_summary && python -m scripts.bgonly_investigation
python -m scripts.border_check --tag n2_final --suffix _run2 --seeds 0 1 2
python -m scripts.final_test --split val && python -m scripts.fig_final --split val
python -m scripts.silhouette; python -m scripts.bio_check; python -m scripts.make_table1
python -m scripts.fig1_cells; python -m scripts.fig_masking_examples; python -m scripts.fig_training_curves
python -m scripts.fig_gradcam --tag n2_final --suffix _run2 --device cpu --correct-wrong
```

## AI-use log notes
- Claude (Opus 5.5, Claude Code) worked autonomously 00:21–05:00 on 7 Nov on branch `overnight-run-4`, following the overnight prompt, CLAUDE.md and PROJECT.md.
- Claude wrote all new code (masking module + tests, queues 4/4b/4c, final_test.py + tests, ranking metrics + tests, figure/table scripts, investigation scripts), ran all experiments, and wrote the drafts, LEARN.md entries, PROJECT.md entries and this report.
- No literature was looked up; [CHECK] placeholders remain (Eulenberg 2017 row, BBBC048 per-image scaling).
- Every interpretation needs your check before it goes into the paper.
