"""Chance baselines (majority + stratified DummyClassifier), fitted on train, scored on val, 4 and 7 classes, all seeds."""
import time

import numpy as np
from sklearn.dummy import DummyClassifier

from src import config
from src.data import load_dataset
from src.evaluate import evaluate
from src.experiments import log_experiment
from src.splits import load_split

STRATEGIES = {"majority": "most_frequent", "stratified": "stratified"}

train, val = load_split("train"), load_split("val")
for n_classes, class_names in ((4, config.CLASSES_4), (7, config.CLASSES_7)):
    _, y, _ = load_dataset(n_classes)
    # DummyClassifier ignores features; a zero column keeps the sklearn API happy without touching images
    X_train, X_val = np.zeros((len(train), 1)), np.zeros((len(val), 1))
    for name, strategy in STRATEGIES.items():
        for seed in config.SEEDS:
            t0 = time.perf_counter()
            clf = DummyClassifier(strategy=strategy, random_state=seed).fit(X_train, y[train])
            train_time = time.perf_counter() - t0
            m = evaluate(y[val], clf.predict(X_val), class_names)
            log_experiment(f"dummy_{name}", n_classes, "val", seed, {"strategy": strategy}, m, train_time)
            print(f"{n_classes} classes  {name:<10}  seed {seed}  macro-F1 {m['macro_f1']:.3f}  "
                  f"bal. acc {m['balanced_accuracy']:.3f}")
print(f"appended to {config.EXPERIMENTS_FILE.relative_to(config.ROOT)}")
