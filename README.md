# Seeing cells: CNNs for cell-cycle phase from label-free images

Course project 19, *Introduction to AI for Scientists* (2026). Can we tell the cell-cycle phase of a single Jurkat cell from its **brightfield and darkfield** imaging flow cytometry images, without the DNA stain that was used to make the labels?

We compare a **random forest on 28 hand-crafted features** with three CNNs: a **small CNN** trained from scratch, a **ResNet18 trained from scratch** and a **ResNet18 pretrained on ImageNet**. We then look at what the networks learned (UMAP, pseudotime, Grad-CAM, masking).

- **Data:** BBBC048 (Eulenberg et al., 2017), 32,266 cells, 66 × 66 px. Not in this repo: put it in `data/raw/BBBC048v1/` (gitignored).
- **Main result:** 4 classes (G1, S, G2, mitosis). Secondary result: all 7 phases.
- **Full plan, rules and every decision:** [`PROJECT.md`](PROJECT.md). **Why each file exists (for the oral):** [`LEARN.md`](LEARN.md).

## Headline result (test set, used once on 7 Nov 2026, tag `test-frozen`)

4-class macro-F1, mean ± SD over 3 seeds (from [`results/table_test_summary.md`](results/table_test_summary.md)):

| Model | Macro-F1 | Mitosis recall |
|---|---|---|
| Chance (stratified) | 0.257 ± 0.006 | 0.03 |
| Random forest (hand-crafted features) | 0.659 ± 0.004 | 0.31 |
| SmallCNN (scratch) | 0.746 ± 0.003 | 0.64 |
| ResNet18 (scratch) | 0.735 ± 0.004 | 0.60 |
| **ResNet18 (ImageNet-pretrained)** | **0.776 ± 0.004** | **0.77** |

Every other number in the paper (learning curve, augmentation, UMAP, Grad-CAM, masking) is a **validation** number.

## What is in each folder

| Folder | What it is |
|---|---|
| `src/` | The machinery: loading images, features, models, training, metrics (`evaluate.py`), all settings (`config.py`). Not run directly. |
| `scripts/` | The entry points: one script per experiment, figure or table. `queue_*.sh` are the overnight run queues. |
| `tests/` | Unit tests (`pytest`). |
| `splits/` | `splits.json`: the fixed train/val/test cell indices (70/15/15, seed 42). Never changed. |
| `results/` | All numbers. `experiments.csv` = the log of every run. `preds/` = saved predictions. Test: `test_results.csv`, `table_test_summary.md`, `table2_test.md`. |
| `figures/` | Figures used in the paper and the supplementary (PNG + PDF of each). `figures/archive/` holds the pilot run and early checks. |
| `notes/` | `paper_draft/` (draft Methods/Results, tables), `test_day_checklist.md`, `brief.md`. `overnight_runs/` holds the reports and logs of the overnight runs. |
| `notebooks/` | Exploration only (EDA). Nothing in the paper depends on it. |

## Paper figures: which file, which script

| Paper item | File in `figures/` | Made by | Research question |
|---|---|---|---|
| Fig 1 | `fig1_cells_val` | `scripts/fig1_cells.py` | Setup: example cells, class counts |
| Fig 2 | `fig2_test` | `scripts/fig_final.py --split test` | RQ1: models compared (test) |
| Fig 3 | `learning_curve_val_run2` | `scripts/fig_learning_curve.py` | RQ2: few labelled cells, pretrained vs scratch |
| Fig 4 | `phaseA_val_run2` | `scripts/fig_phaseA_run2.py` | RQ3: augmentation × class balancing |
| Fig 5 | `umap_val_run2`, `pseudotime_val_run2` | `scripts/fig_umap.py`, `scripts/pseudotime.py` | RQ4: continuous cell cycle |
| Fig 6a | `gradcam_val_correct_wrong_run2` | `scripts/fig_gradcam.py --correct-wrong` | RQ5: where the network looks |
| Fig 6b | `fig6b_masking_val` | `scripts/fig_masking.py` | RQ5: cell only vs background only |
| Table 1 | `results/table1_methods.md` | `scripts/make_table1.py` | Methods overview |
| Table 2 | `notes/paper_draft/table2_test.md` | `scripts/make_table2_test.py` (from `results/table_test_summary.md`) | Test metrics + Eulenberg 2017 row |
| Table 3 | `results/run2_phaseA_val.md` | `scripts/choose_aug_bal.py` | Augmentation × balancing numbers |

**Supplementary** (also in `figures/`): `confusion_val_7class_run2` (7 classes), `confusion_val_4class_run2`, `masking_val` and `masking_examples_val` (masking details), `training_curves_val_run2`, `rf_importance` (random forest features), `gradcam_val_run2` and `gradcam_val_smallcnn_run2` (full Grad-CAM grids). `fig2_val` and `model_comparison_val_run2` are the validation versions of Fig 2.

**`figures/archive/`**: figures without `_run2` in the name are from **run 1, the pilot** that found the darkfield left/right problem; they were replaced by run 2. The `eda_*`, `channel_check`, `mask_check`, `tiny_masks` and `gradcam_cell30439_*` figures are early data checks. All are kept for the record and the oral, not for the paper.

## How to reproduce

```bash
conda env create -f environment.yml
conda activate cell-image-cnn
pytest -q                                   # checks the code
python -m scripts.compute_features          # hand-crafted features for the random forest
python -m scripts.run_rf                    # random forest (tuning + finals)
python -m src.train --help                  # CNN training; the exact runs are in scripts/queue_*.sh
```

Each figure script writes to `figures/` and each table script to `results/`. The test split is evaluated only by `scripts/final_test.py`, which refuses a second test run. Training was done on a Mac (Apple GPU, `mps`); results on other hardware can differ slightly.
