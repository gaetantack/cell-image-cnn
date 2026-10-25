# Overnight log 2026-10-25 → 2026-10-26

Branch `overnight-2026-10-25`. All numbers are VALIDATION numbers. Test split never loaded.

| Time | Step | Result | Next |
|---|---|---|---|
| 22:18 | Start-up: branch `overnight-2026-10-25` created from main @ 9288be7; repo tree read | clean tree | pytest |
| 22:20 | `pytest` | 21 passed | device check |
| 22:20 | Device: Apple M2, 8 GB RAM, 8 cores; torch 2.14.1, MPS available, no CUDA. umap-learn not installed | device = mps | speed benchmark |
| 22:25 | Speed benchmark on MPS (random data, batch 128, train step only) | SmallCNN 66 px ≈ 1590 img/s (≈15 s/epoch on 22.6k train cells); ResNet18 128 px ≈ 390 img/s (≈60 s/epoch); ResNet18 66 px ≈ 780 img/s | write time budget |
| 22:30 | **Time budget** (latest end 07:00, wrap-up from 06:30). Full plan at 12 configs × 15 epochs for both ResNets ≈ 9 h of MPS time: does not fit. Plan: keep 12 configs for every family (rigour rule) but cap tuning runs at 4 epochs and final 3-seed runs at 12 epochs, same for all families. Estimate: RF ≈ 10 min (CPU, while CNN code is written); SmallCNN tuning ≈ 15 min + finals ≈ 15 min; ResNet18-pre tuning ≈ 55 min + finals ≈ 40 min; ResNet18-scratch same ≈ 95 min; 7-class finals ≈ 80 min. Total MPS ≈ 5 h → done ≈ 04:30 if queues start ≈ 23:30. Then learning curve + ablation (≈ 1 h) | RF |
| 22:21 | Phase 1 RF: `scripts/run_rf.py`, feature set `main` (28 features, not 32: PROJECT.md decision 2026-10-24 excludes background-dependent Ch6 features); 12 configs (max_features × min_samples_leaf × class_weight), 500 trees | best = max_features 0.33, min_samples_leaf 5, class_weight balanced. 4 cl. val macro-F1 0.632/0.632/0.631 (seeds 0–2); 7 cl. 0.555/0.559/0.562. Mitosis recall only 0.26 | CNN code |
| 22:22 | `src/experiments.py`: added columns run_id, device, git_commit (old rows empty); `has_run()` for skipping | committed 97290b3 | models |
| 22:24 | `src/models.py`, `src/train.py`; smoke tests for all 3 models pass on MPS | SmallCNN 389,636 params; ResNet18 11,175,428 | launch queue |
| 22:24 | Launched `scripts/queue_phase3.sh` (all of phase 3: tune 3 families, finals 4 cl., finals 7 cl.) | SmallCNN real speed ≈ 17 s/epoch (bs 64) | tests |
| 22:27 | `tests/test_models.py` (12 tests) | pytest 33 passed | RF importance figure |
| 22:27 | RF permutation importance (`scripts/fig_rf_importance.py`, `src/plotting.py`) + draft paragraph `notes/paper_draft/results_rf.md` | area dominates (−0.113 macro-F1), then Ch3 GLCM correlation, axis lengths, Ch6 GLCM energy | methods draft while queue runs |
| 22:29 | Figure/table scripts (make_results_table, fig_model_comparison, fig_confusion) and methods draft committed. `pip install umap-learn==0.5.12` (numba 0.68.0) works; added to environment.yml | | Phase 5 code (UMAP, Grad-CAM) |
| 22:33 | train.py: an epoch now always draws len(full train) images (repeats cells for label fractions < 1) so learning-curve runs get the same number of gradient steps; identical behaviour for full runs. Wrote queue_phase5.sh (waits for phase 3, then figures, UMAP, Grad-CAM, LC seed 0, ablation, LC seeds 1–2; no new run after 06:30), fig_learning_curve, fig_ablation, LEARN.md | committed | launch phase-5 queue (it waits) |
| 22:34 | Sanity check: SmallCNN tuning (4 epochs) reaches val macro-F1 0.60–0.64, about the RF level (0.632). Confusion (tune08): errors are G1↔S↔G2 and G2→mitosis (148 G2 cells predicted mitosis, mitosis precision 0.27). RF has the same pattern with fewer mitosis predictions. No sign of leakage or a bug: neither suspicion rule (val F1 > 0.98, CNN far below RF) is triggered | plausible | Table 1 script written (`scripts/make_table1.py`), to run at wrap-up (not added to the running phase-5 queue, since editing a running bash script is unsafe) |
