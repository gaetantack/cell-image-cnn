# Overnight report, run 2: 31 Oct → 1 Nov 2026 (Claude, autonomous)

Branch `overnight-run-2` (from `overnight-2026-10-25`; pushed, not merged). Every number here is a **VALIDATION** number. The test split was never loaded. All run-2 runs have run tags starting with `n2_`, and the run-1 rows in `results/experiments.csv` are byte-identical to before (checked). Timeline: `notes/overnight_log_2.md`.

## Done / not done

| Phase | Status |
|---|---|
| 1 Start-up: branch, pytest, speed (same as run 1), time budget | done |
| A Augmentation × balancing (SmallCNN, 3 aug × 2 bal × 3 seeds = 18 runs) | done → **AUG* = vflip, BAL* = weighted** |
| B Re-tuning, 12 configs × 3 families, new grid | done; all families picked lr 1e-3, wd 1e-4 |
| C Finals, 3 seeds × 3 families, max 25 epochs, patience 6 | done |
| D Tables + figures `_run2` (summary, run 1 vs run 2, model comparison, confusion, UMAP, pseudotime, Grad-CAM best CNN + SmallCNN), border swap test repeated | done |
| E1 7-class finals | SmallCNN only (3 seeds); ResNets skipped (would not end by 06:30) |
| E2 Learning curve 1/5/20 %, seed 0 | SmallCNN and ResNet18-pretrained done (pretrained started by hand, see decision 3); scratch not run (time) |
| Writing: methods.md, results_val_draft.md, LEARN.md (+3 examiner questions), PROJECT.md decisions | done |

## Phase A: augmentation × balancing (SmallCNN, val, mean ± SD over 3 seeds)

| Augment | Balance | Macro-F1 | Mitosis recall |
|---|---|---|---|
| **vflip** | **weighted** | **0.743 ± 0.013** | **0.66 ± 0.01** |
| vflip | oversample | 0.741 ± 0.010 | 0.57 ± 0.10 |
| none | weighted | 0.737 ± 0.015 | 0.59 ± 0.05 |
| basic | weighted | 0.723 ± 0.030 | 0.64 ± 0.12 |
| none | oversample | 0.716 ± 0.008 | 0.60 ± 0.08 |
| basic | oversample | 0.715 ± 0.053 | 0.54 ± 0.14 |

**Chosen: vflip + weighted.** The best two tied: they differ by 0.002, less than one seed SD (0.013), and their mitosis recalls also differ by less than one SD. The tie-break then gave weighted > oversample. All six settings lie within 0.03, so the augmentation choice matters much less than run 1's single seed suggested (gap none vs basic: 0.05 → 0.014). The orientation check on the chosen model shows almost no change on up-down flips (0.757 → 0.743) but a large drop on left-right flips (0.621; mitosis recall 0.67 → 0.19). So it still uses the darkfield direction, as intended. (`results/run2_phaseA_val.md`, `figures/phaseA_val_run2`, `results/orientation_check_val_run2.csv`)

## Run-2 results vs run 1 (4 classes, val macro-F1, mean ± SD over seeds 0–2)

| Model | Run 1 | Run 2 | Change | Mitosis recall run 2 (run 1) | Best epochs run 2 (epochs run) |
|---|---|---|---|---|---|
| Chance majority / stratified | 0.154 / 0.257 | same | | 0 / 0.03 | |
| Random forest | 0.632 ± 0.001 | not re-run | | 0.26 | |
| SmallCNN | 0.687 ± 0.006 | 0.744 ± 0.010 | +0.057 | 0.63 ± 0.04 (0.65) | 19/21/14 (25/25/21) |
| **ResNet18 (ImageNet)** | 0.761 ± 0.006 | **0.781 ± 0.007** | +0.021 | 0.75 ± 0.03 (0.83) | 19/6/11 (25/13/18) |
| ResNet18 (scratch) | 0.696 ± 0.012 | 0.745 ± 0.003 | +0.050 | 0.60 ± 0.02 (0.64) | 20/20/18 (25/25/25) |
| 7 classes: SmallCNN | 0.560 ± 0.007 | 0.593 ± 0.034 | +0.033 | | 11/3/4 |

Learning curve, run 2, seed 0, val macro-F1 at 1 / 5 / 20 / 100 %: SmallCNN 0.503 / 0.598 / 0.664 / 0.744; ResNet18-pretrained 0.540 / 0.639 / 0.721 / 0.781 (`figures/learning_curve_val_run2`). These are lower than run 1 at small fractions (e.g. SmallCNN 1 %: 0.538 in run 1), probably because small-subset runs stopped very early (best epoch 2 at 5 %). Single seed.

Full tables: `results/table_val_summary_run2.md`, `results/table_run1_vs_run2_val.md`. Run-2 configs: all three families lr 1e-3, wd 1e-4, batch 64, vflip, weighted. The ranking is unchanged (pretrained > scratch ≈ SmallCNN > RF). The pretrained − scratch gap shrank from 0.065 to 0.036. Scratch never stopped early, but its best epochs were 18–20, so 25 epochs is no longer a hard cap. Several settings changed at once (augmentation, lr, epochs, patience), so the gains cannot be assigned to one cause.

## Figures made tonight (all by scripts; run-1 files kept)

| File (`figures/`) | Script |
|---|---|
| `phaseA_val_run2` | `scripts/fig_phaseA_run2.py` |
| `model_comparison_val_run2` | `fig_model_comparison.py --tag n2_final --suffix _run2` |
| `confusion_val_4class_run2`, `confusion_val_7class_run2` | `fig_confusion.py --tag n2_final --suffix _run2` (7-class panel shows only the RF: the best CNN, pretrained ResNet18, has no run-2 7-class runs) |
| `umap_val_run2` | `fig_umap.py --tag n2_final --suffix _run2` |
| `pseudotime_val_run2` | `pseudotime.py --tag n2_final --suffix _run2` |
| `gradcam_val_run2`, `gradcam_cell30439_train_run2` | `fig_gradcam.py --tag n2_final --suffix _run2` |
| `gradcam_val_smallcnn_run2`, `gradcam_cell30439_train_smallcnn_run2` | `fig_gradcam.py --model smallcnn --tag n2_final --suffix _run2` |
| `learning_curve_val_run2` | `fig_learning_curve.py --prefix n2_ --suffix _run2` (SmallCNN only + 100 % points) |

The run-1 `model_comparison_val` was regenerated with the legend moved (same numbers).

## Anything suspicious, and what I checked

1. **No suspicion rule triggered.** The highest val macro-F1 is 0.787 (≪ 0.98), and every CNN is ≥ 0.11 above the RF.
2. **The pretrained ResNet18 gained macro-F1 but lost mitosis recall** (0.83 → 0.75). Checked: it predicts mitosis less often but more precisely (seed 0: 87/138 correct = 0.63 → 77/91 = 0.85). Macro-F1 rewards this. In the paper, report precision and recall, not only macro-F1.
3. **Grad-CAM "lower-right spot" (run 1) remains** in the pretrained ResNet18's G1/S/G2 maps (lower / lower-right cell edge). This model was trained with up-down flips, and the run-1 model with full rotations, so the spot is probably a property of the 4 × 4 Grad-CAM grid or the stride alignment, not of the data. Untested.
4. **The S-phase border pattern of the run-1 SmallCNN is gone.** S heat in the border is 0.44 → 0.16, in the mask 0.18 → 0.43. We can't tell which change removed it.
5. **Border swap test repeated on the run-2 models:** other-class vs same-class swap differs by −0.010 to +0.006. As in run 1, no evidence of a phase-specific background shortcut. The originals reproduce the logged scores.
6. **Seed noise is large in some settings** (basic + oversample: 0.659–0.765). Early stopping sometimes ends a run in a noisy phase: SmallCNN none/weighted s2 had best epoch 5 vs 22–23 for the other seeds, and two SmallCNN 7-class seeds had best epochs 3 and 4. Patience 6 was a fixed setting; reported, not changed.
7. **A best epoch near 25 is partly built in:** the cosine schedule reaches lr ≈ 0 at the maximum epoch. So "hit 25" alone does not prove under-training.

## Decisions I made for you (each to confirm; also in the PROJECT.md Decisions log, dated 2026-11-01)

| # | Decision | Reason | How to undo |
|---|---|---|---|
| 1 | "Within one seed SD" in the Phase A rule = the larger of the two settings' SDs; tie-break order vflip > none > basic (basic was not ranked in the prompt) | If either setting's noise covers the gap, the gap means nothing; basic removes the darkfield direction | `scripts/choose_aug_bal.py` (`choose`, `AUG_PREFERENCE`); re-run it |
| 2 | Kept the run-1 config values (`TUNE_EPOCHS`, `FINAL_EPOCHS`, `CNN_GRID`) and added a separate run-2 block (`RUN2_*`, `CNN_GRID_RUN2`) in `src/config.py` | Run 1 stays reproducible; one place for config | Delete the run-2 block |
| 3a | ResNet18-pretrained learning curve started by hand at 05:12 (`scripts/queue_run2_lc_pretrained.sh`), although the worst-case guard (3 × 33 min) said it would not fit: realistic ≈ 60 min, GPU otherwise idle, RQ2 is Tier 1. I stopped the script before the 20 % run when 1 % took 26 min, then started 20 % at 05:53 when the worst case (31 min) fitted again | Use the idle GPU for a Tier-1 item without risking the 06:30 end | Delete the `resnet18_pretrained_c4_n2_lf*` rows |
| 3 | Phase E started per item (all 3 seeds or fractions of one family must end before 06:30) instead of per run: new `scripts/queue_run2_CE.sh`; stopped the bash process of `queue_run2_BC.sh` at 02:40 (its running training continued) | A family with < 3 seeds can't be reported; the SmallCNN learning curve was more useful than one lone ResNet 7-class seed | Run `queue_run2_BC.sh` instead (it continues where CE stopped) |
| 4 | RF and chance baselines not re-run; run-2 tables use their run-1 rows | Their settings did not change | — |
| 5 | Orientation check run on the seed-0 Phase A models of all three augmentations (with BAL*), not only the chosen one | Gives the comparison none / vflip / basic in one table | — |
| 6 | Extra (cheap, CPU): border swap test repeated on the run-2 models; Phase A figure | Checks whether the run-1 RQ5 conclusion still holds | Delete `results/border_check_val_run2.csv`, `figures/phaseA_val_run2.*` |
| 7 | Git commits with the same author as before (`Gaetan Tack <gaetantack@Gaetans-MacBook-Air.local>`) via environment variables, because git has no identity configured on this host; git config not changed | The first commit failed without an identity | Amend authors if you prefer another identity |
| 8 | `try_mask.py` (untracked, not mine) left alone and not committed | Not part of the plan | — |

## What you need to decide (max 5)

1. **Confirm AUG*/BAL* = vflip + weighted** (and the run-2 settings: 25 epochs, patience 6, new grid) as the settings for the paper and the week-5 test. Alternatively, report Phase A as "the choice hardly matters" and keep the simpler "none".
2. **Using the darkfield direction:** the models now rely on the instrument's left/right geometry. That is OK for this dataset, but it should be stated as a limitation (transfer to other instruments). Do you agree, or do you prefer direction-invariant models (basic; ≈ 0.02 lower, less stable)?
3. **Which run goes in the paper:** run 2 everywhere (my suggestion; the run-1 numbers only as a note), or both?
4. **Missing pieces before the test:** the ResNet 7-class finals (≈ 3 × 33 min per family), the scratch-ResNet learning curve and learning-curve seeds 1–2 (run-2 settings) were not run. Run them, or drop the 7-class ResNets (val 7-class macro-F1 is unreliable anyway: 2 anaphase and 4 telophase cells)?
5. **Early stopping noise:** keep patience 6, or select the epoch on a smoothed val curve / use a fixed epoch count? (It affects a few runs, e.g. best epoch 3–5.)

## Resume / reproduce commands

```bash
conda activate cell-image-cnn
git checkout overnight-run-2
pytest
# Phase A (skips finished runs):
nohup bash scripts/queue_run2_A.sh > logs/queue_run2_A.out 2>&1 &
python -m scripts.choose_aug_bal          # table + choice JSON
# Phase B + C + E (skips finished runs; the CE queue does C and E per item):
nohup bash scripts/queue_run2_BC.sh > logs/queue_run2_BC.out 2>&1 &   # B (then C/E per run)
nohup bash scripts/queue_run2_CE.sh > logs/queue_run2_CE.out 2>&1 &   # C, then E per item (06:30 guard: edit fits() for daytime use)
# Missing E items, e.g. pretrained 7-class seed 0:
python -m src.train --model resnet18_pretrained --classes 7 --seed 0 --lr 0.001 --weight-decay 0.0001 --batch-size 64 --epochs 25 --augment vflip --balance weighted --run-tag n2_final --resume
# Tables and figures, run 2:
python -m scripts.make_results_table --tag n2_final --suffix _run2
python -m scripts.compare_runs
for S in fig_model_comparison fig_confusion fig_umap pseudotime fig_gradcam; do python -m scripts.$S --tag n2_final --suffix _run2; done
python -m scripts.fig_gradcam --model smallcnn --tag n2_final --suffix _run2
python -m scripts.fig_learning_curve --prefix n2_ --suffix _run2
python -m scripts.fig_phaseA_run2
python -m scripts.border_check --tag n2_final --suffix _run2
python -m scripts.orientation_check --runs smallcnn_c4_n2_A_none_weighted_s0 smallcnn_c4_n2_A_vflip_weighted_s0 smallcnn_c4_n2_A_basic_weighted_s0 --out orientation_check_val_run2.csv
```
Note: the queue's 06:30 guard counts minutes from noon, so in the daytime (before 12:00) it would refuse to start runs. For a daytime run, change `1110` in `fits()`.

## AI-use log notes

- Claude (Opus 5.5, Claude Code) worked autonomously from 21:33 on 31 Oct to about 06:30 on 1 Nov on branch `overnight-run-2`, following the user's overnight prompt, CLAUDE.md and PROJECT.md.
- Claude wrote the new code (`augment_vflip` + test, `scripts/choose_aug_bal.py` + tests, `scripts/fig_phaseA_run2.py`, `scripts/compare_runs.py`, queue scripts, `--tag/--suffix` options), ran all experiments, and wrote the drafts, LEARN.md entries and this report.
- All numbers come from `results/experiments.csv` and the scripts. No literature was looked up; [CHECK]/[CITE] placeholders remain (e.g. the ImageStream side-scatter geometry).
- Every interpretation in the drafts needs to be checked by the user before it goes into the paper.
