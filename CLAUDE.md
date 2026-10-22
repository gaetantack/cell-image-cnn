# CLAUDE.md

## Start of every session
Read `PROJECT.md` first. It holds the goal, research questions, rigour rules, figure plan and priority tiers.

## Folder layout
| Folder | Contents |
|---|---|
| `data/` | Raw and preprocessed images. Gitignored, never committed. |
| `splits/` | `splits.json`: the fixed train/val/test image indices. Committed; never regenerated or edited. |
| `src/` | Importable code: dataset loading, models, training, features, `evaluate.py`, `config.py`. |
| `scripts/` | Entry points: training runs, feature extraction, and one script per figure/table. |
| `tests/` | Unit tests for `src/` (pytest). |
| `results/` | `experiments.csv`, saved metrics, predictions, logs. |
| `figures/` | Figure files, written only by scripts in `scripts/`. |
| `notebooks/` | Exploration only. Nothing the paper depends on lives only here. |

## Code rules
- Python 3, PyTorch + timm, scikit-learn, scikit-image.
- Seed all randomness (Python, NumPy, PyTorch, data loaders, scikit-learn `random_state`).
- Config values (paths, split seed, hyperparameters, class mappings) live only in `src/config.py`, a plain Python file. Never hard-code them in scripts.
- All metrics are computed only through `src/evaluate.py`.

## Hard rules
- **Test split:** never load or evaluate it unless the user explicitly says "final test" (week 5). Use train/val only.
- **Fluorescence:** never use fluorescence channels as model input. Only brightfield + darkfield.
- **Experiment log:** every training run appends one row to `results/experiments.csv` with model, settings, seed, metrics, train time and date.
- **Figures:** every figure is made by a script in `scripts/`, never by hand.

## End of every session
Report:
1. What you changed.
2. Which decisions you made for the user.
3. What the user needs to decide.
