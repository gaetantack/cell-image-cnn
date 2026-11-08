"""RQ5 side experiment for the random forest: the `raw_dark` feature set (`main` + raw darkfield mean and
percentiles, which follow the phase-dependent darkfield background). Same configuration as the main
random forest (best of its 12 tuning runs, no extra tuning), seeds 0, 1, 2, 4 classes, validation only.
If raw_dark scores clearly higher than main, part of the gain probably comes from the background shortcut.
"""
from scripts.run_rf import best_params, run
from src import config

params, best_id = best_params()
print(f"configuration of {best_id}: {params}")
for seed in config.SEEDS:
    run(params, 4, seed, run_id=f"rf_raw_dark_c4_final_s{seed}", save_preds=True, feature_set="raw_dark",
        model_name="random_forest_raw_dark")
