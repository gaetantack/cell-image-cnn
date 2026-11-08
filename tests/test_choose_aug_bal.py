"""Tests for the Phase A choice rule in scripts/choose_aug_bal.py (synthetic tables, no training)."""
import pandas as pd

from scripts.choose_aug_bal import choose


def table(rows):
    cols = ["augment", "balance", "macro_f1_mean", "macro_f1_sd", "mitosis_recall_mean", "mitosis_recall_sd"]
    return pd.DataFrame(rows, columns=cols).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def test_clear_winner_is_chosen():
    t = table([("none", "weighted", 0.75, 0.005, 0.6, 0.05), ("vflip", "weighted", 0.72, 0.005, 0.7, 0.05)])
    assert choose(t)[0]["augment"] == "none"


def test_tie_on_f1_goes_to_higher_mitosis_recall():
    t = table([("none", "weighted", 0.745, 0.01, 0.55, 0.02), ("basic", "oversample", 0.740, 0.01, 0.70, 0.02)])
    assert choose(t)[0]["augment"] == "basic"


def test_full_tie_prefers_vflip_then_weighted():
    t = table([("none", "weighted", 0.745, 0.01, 0.60, 0.05), ("vflip", "oversample", 0.742, 0.01, 0.62, 0.05)])
    assert choose(t)[0]["augment"] == "vflip"
    t = table([("vflip", "oversample", 0.745, 0.01, 0.60, 0.05), ("vflip", "weighted", 0.742, 0.01, 0.62, 0.05)])
    assert choose(t)[0]["balance"] == "weighted"
