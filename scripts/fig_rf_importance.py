"""Permutation importance of the hand-crafted features for the best random forest (4 classes, seed 0), on val.

Each feature column of the VALIDATION set is shuffled RF_PERMUTATION_REPEATS times; importance = mean drop
in val macro-F1 (computed through src.evaluate). Writes figures/rf_importance.png/.pdf and
results/rf_importance_val.csv.
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from scripts.run_rf import best_params, load_features, make_rf
from src import config
from src.evaluate import evaluate
from src.plotting import MODEL_COLORS, MUTED, ONE_COLUMN, save, set_style
from src.splits import load_split

import matplotlib.pyplot as plt


def macro_f1_scorer(estimator, X, y):
    return evaluate(y, estimator.predict(X), config.CLASSES_4)["macro_f1"]


params, best_id = best_params()
X, y = load_features(4)
train, val = load_split("train"), load_split("val")
rf = make_rf(params, seed=0).fit(X[train], y[train])
res = permutation_importance(rf, X[val], y[val], scoring=macro_f1_scorer,
                             n_repeats=config.RF_PERMUTATION_REPEATS, random_state=0, n_jobs=1)
names = list(config.FEATURE_SETS[config.RF_FEATURE_SET])
table = pd.DataFrame({"feature": names, "importance_mean": res.importances_mean,
                      "importance_sd": res.importances_std}).sort_values("importance_mean", ascending=False)
table.to_csv(config.RESULTS_DIR / "rf_importance_val.csv", index=False)
print(table.head(config.RF_TOP_FEATURES).to_string(index=False))

top = table.head(config.RF_TOP_FEATURES)[::-1]
set_style()
fig, ax = plt.subplots(figsize=(ONE_COLUMN, 7 * 0.3937))
ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], height=0.6,
        color=MODEL_COLORS["random_forest"], error_kw={"elinewidth": 0.6, "ecolor": MUTED})
ax.set_xlabel("Drop in validation macro-F1 when shuffled")
ax.set_title(f"Random forest, 4 classes: top {config.RF_TOP_FEATURES} features (validation)")
ax.axvline(0, color=MUTED, lw=0.6)
save(fig, "rf_importance")
