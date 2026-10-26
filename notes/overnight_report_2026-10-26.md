# Overnight report — 25/26 October 2026 (Claude, autonomous run)

Branch `overnight-2026-10-25` (pushed to origin; not merged). Every number here is a **VALIDATION** number. The test split was never loaded. Detailed time line: `notes/overnight_log.md`.

## Done / not done

| Phase | Status |
|---|---|
| 1 Random forest (12 configs, 3 seeds, 4 + 7 classes, permutation importance, `results_rf.md`) | done |
| 2 CNN code (`src/models.py`, `src/train.py`, tests, smoke tests) | done |
| 3 Tuning (12 configs per family) + 3-seed finals, 4 classes, all 3 CNNs; 7-class finals | done, with the epoch cap below (deviation, needs your OK) |
| 4 Results table, Table 1, model comparison, confusion matrices (4 + 7 classes) | done |
| 5a UMAP (RQ4) | done |
| 5b Grad-CAM (RQ5), incl. training cell 30439 | done (best CNN + SmallCNN) |
| 5c Learning curve | [LC_STATUS] |
| 5d RQ3 ablation (seed 0, preliminary) | done |
| Extra: PC1 pseudotime (Tier 3, RQ4) | done |
| Extra: border test with background swap (RQ5) | done |
| Extra: orientation check (RQ3) | done — **important, see below** |
| 8 Writing: `methods.md`, `table1_methods.md`, `results_val_draft.md` | done (drafts; no Introduction/Discussion) |
| 9 LEARN.md entries + 5 examiner questions | done |

## Validation results (mean ± SD over seeds 0, 1, 2)

[TABLE]

Figures (all in `figures/`, PNG + PDF, made by scripts in `scripts/`):

[FIGURES]

## Anything suspicious (all checked; none looks like leakage)

1. **The augmentation removes real information (most important).** In the darkfield images the left half is brighter than the right in 81% of training images (brightfield: 45%). The side-scatter light probably comes from one side of the instrument. SmallCNN trained *without* augmentation reached 0.742 val macro-F1, against 0.690 for our final setting (rotations/flips + weighted loss). On left-right-flipped val images the no-augmentation model drops to 0.608, on up-down flips only to 0.710. Augmentation plus oversampling also reached 0.746. All CNN numbers above therefore use a setting that is probably not the best. Seed 0 only. (`scripts/orientation_check.py`, `results/orientation_check_val.csv`, `results/ablation_val.md`)
2. **7-class: the scratch ResNet18 is 0.09 below the random forest** (suspicion rule). Checked: the gap comes from 6 validation cells (2 anaphase, 4 telophase). The forest got all 4 telophase cells with only 4 telophase predictions; the scratch ResNet18 made 9 wrong anaphase predictions. Not a bug, but 7-class val macro-F1 cannot rank models.
3. **Val macro-F1 jumps between epochs** (e.g. 0.62–0.77 within one ResNet run). Because the best epoch is chosen on validation, all CNN val numbers are optimistic. The test set will give the honest number.
4. **Grad-CAM, SmallCNN, S phase:** the heat lies in the image border, not on the cell. The border test speaks against a background shortcut: pasting the border of another cell of the *same* class lowers the score as much as pasting one from a *different* class (difference −0.014 to +0.012). For the best model (pretrained ResNet18) the heat is on the cell for all phases. Its G1/S/G2 maps often peak at the same lower-right spot. That may be the coarse 4 × 4 grid or the darkfield direction from point 1; untested.
5. **Grid edge:** all three CNN families picked the largest learning rate in the grid (2e-3).
6. **Pseudotime:** the 4-class and 7-class Spearman ρ are identical to 3 decimals. Checked: only 16 val cells differ between the two labellings.

## Decisions I made for you (each to confirm; all also in the PROJECT.md Decisions log)

| # | Decision | Reason | How to undo |
|---|---|---|---|
| 1 | **Epoch caps: tuning 4 epochs, final runs 12 epochs (deviation from §5, applied equally to all CNN families)** | 12 configs × 3 families at full length ≈ 9 h of GPU time; did not fit in one night | Change `TUNE_EPOCHS`/`FINAL_EPOCHS` in `src/config.py`, delete the CNN rows (by run_id) from `results/experiments.csv`, re-run `scripts/queue_phase3.sh` |
| 2 | RF feature set `main` (28 features), not the "32" in the prompt | PROJECT.md decision 2026-10-24 excludes background-dependent raw darkfield features; `raw_dark` run as a side experiment | `RF_FEATURE_SET` in `src/config.py` |
| 3 | RF grid: max_features {sqrt, 0.33} × min_samples_leaf {1, 5, 20} × class_weight {balanced, balanced_subsample}, 500 trees | 12 configs, the same budget as the CNNs | `RF_GRID` in `src/config.py` |
| 4 | CNN grid: lr {1e-4, 5e-4, 2e-3} × weight decay {1e-4, 1e-2} × batch size {64, 128}; augment = basic and balance = weighted fixed | As in the prompt; same grid for every family | `CNN_GRID` in `src/config.py` |
| 5 | Final runs re-train seed 0 (not reuse the 4-epoch tuning run) | Tuning runs are shorter (4 epochs) than final runs | — |
| 6 | 7-class runs reuse the 4-class configuration, no re-tuning | As in the prompt; time | Tune separately (see "What you need to decide") |
| 7 | ResNet18 input upscaled 66 → 128 px (bilinear) | 66 px leaves a 3 × 3 map after ResNet18's 32× downsampling; 128 px ≈ 2× compute | `RESNET_INPUT_SIZE` in `src/config.py` |
| 8 | Pretrained conv1 for 2 channels = mean of the R, G, B filters, copied to both channels | Keeps the pretrained grey-level filters; any other choice is arbitrary | `src/models.py`, `ResNet18.__init__` |
| 9 | Learning curve: an "epoch" always draws as many samples as the full training set (small subsets repeated), stratified subsets with ≥ 1 cell per class | Same number of gradient steps, so the curve measures labels, not training length | `src/train.py`, `epoch_order` / `subsample_train` |
| 10 | Learning curve for all three CNNs (seed 0 first), ablation before the learning curve, no run started after 06:12 (ResNet) / 06:25 (SmallCNN) | Prompt order; everything finished by 06:30 | `scripts/queue_phase5.sh` |
| 11 | Installed `umap-learn==0.5.12` (+ numba 0.68.0, llvmlite 0.50.0, pynndescent 0.6.0); added to `environment.yml` | UMAP for RQ4; no simple self-written alternative | `pip uninstall umap-learn`, remove from `environment.yml` |
| 12 | Grad-CAM self-written, computed for the TRUE 4-class label, last conv block; scored with "heat in mask" and "heat in 8-px border" | Explainable in the oral; the true label shows what the model uses for the right answer | `src/interpret.py`, `GRADCAM_*` in `src/config.py` |
| 13 | Extra analyses beyond the plan: `raw_dark` RF, PC1 pseudotime, border test, orientation check | Cheap (CPU) and directly about RQ4/RQ5/RQ3 shortcut questions | Delete the scripts and their `results/` files |
| 14 | `results/experiments.csv`: columns `run_id`, `device`, `git_commit` added (old rows left empty) | Queue can skip finished runs; traceability | `src/experiments.py` |
| 15 | MPS (Apple GPU) training accepted although not bit-for-bit reproducible | Only GPU available; seeds fixed everywhere | — |

## What you need to decide (max 5)

1. **Augmentation (most important).** Keep rotations/left-right flips, switch to up-down flips only, or drop augmentation? This changes every CNN result, so the tuning and finals need to be re-run (≈ 6 h of GPU) before the test run. My suggestion: test "up-down flips + jitter" against "none" with 3 seeds for SmallCNN first (≈ 30 min).
2. **Approve the epoch-cap deviation** (4 / 12 epochs), or plan a longer re-run. Two of three scratch-ResNet runs were still improving in their last epochs, so the cap hurts the scratch model most and inflates the pretrained-vs-scratch gap (RQ2).
3. **Learning-rate grid edge:** every family picked 2e-3, the largest value. Extend the grid upwards (e.g. 5e-3)?
4. **7-class task:** keep it as a secondary result without its own tuning, and report that with 2 anaphase and 4 telophase val cells (similar in test) the 7-class macro-F1 is unreliable? Or report per-phase results only for prophase?
5. **Balancing:** weighted loss (fixed so far) or oversampling? Oversampling with augmentation gave the best SmallCNN mitosis recall (0.69 vs 0.56, one seed).

## Resume / reproduce commands

```bash
conda activate cell-image-cnn
git checkout overnight-2026-10-25
pytest                                                     # all tests
nohup bash scripts/queue_phase3.sh > logs/queue_phase3.out 2>&1 &   # tuning + finals (skips finished runs)
nohup bash scripts/queue_phase5.sh > logs/queue_phase5.out 2>&1 &   # figures, ablation, learning curve
# single run, e.g.:
python -m src.train --model smallcnn --classes 4 --seed 0 --lr 0.002 --weight-decay 0.0001 --batch-size 64 --augment none --balance weighted --run-tag try_s0
# all tables and figures:
for S in make_results_table make_table1 fig_model_comparison fig_confusion fig_rf_importance fig_umap fig_gradcam pseudotime fig_ablation fig_learning_curve; do python -m scripts.$S; done
python -m scripts.fig_gradcam --model smallcnn
python -m scripts.border_check; python -m scripts.orientation_check
```
Checkpoints (best weights) are in `checkpoints/` (gitignored), per-run logs in `logs/`, validation predictions in `results/preds/`.

## AI-use log notes (for the course's AI-use statement)

- Claude (Opus 5.5, Claude Code) worked autonomously from 22:18 on 25 Oct to [END] on 26 Oct on branch `overnight-2026-10-25`, following the user's overnight prompt, CLAUDE.md and PROJECT.md.
- Claude wrote all new code in `src/`, `scripts/` and `tests/` (listed in LEARN.md), ran all experiments, wrote the drafts in `notes/paper_draft/` and this report.
- All numbers come from `results/experiments.csv` and the scripts. Literature references are left as [CHECK]/[CITE] placeholders and were not looked up.
- Every interpretation in the drafts needs to be checked by the user before it goes into the paper.
