"""Agreement-based label combination."""

from __future__ import annotations

import numpy as np
import pandas as pd

from combine_labels import (
    CombineWeights,
    combine,
    evaluate_by_provenance,
    provenance_summary,
)

COLS = ["ACL", "MCL"]


def _frame(d):
    return pd.DataFrame(d, index=["s1", "s2", "s3"], columns=COLS, dtype=float)


def test_agreement_conflict_and_single_source():
    a = _frame({"ACL": [1.0, 0.0, np.nan], "MCL": [1.0, np.nan, 1.0]})
    b = _frame({"ACL": [1.0, 1.0, 0.0], "MCL": [np.nan, 1.0, 1.0]})
    labels, weight, prov = combine(a, b, COLS)

    assert prov.loc["s1", "ACL"] == "agree" and labels.loc["s1", "ACL"] == 1.0
    assert prov.loc["s2", "ACL"] == "conflict"
    assert np.isnan(labels.loc["s2", "ACL"]), "conflicts must abstain by default"
    assert weight.loc["s2", "ACL"] == 0.0
    assert prov.loc["s3", "ACL"] == "only_b" and labels.loc["s3", "ACL"] == 0.0
    assert prov.loc["s1", "MCL"] == "only_a" and labels.loc["s1", "MCL"] == 1.0


def test_agreement_carries_the_highest_weight():
    a = _frame({"ACL": [1.0, 1.0, np.nan], "MCL": [np.nan] * 3})
    b = _frame({"ACL": [1.0, np.nan, np.nan], "MCL": [np.nan] * 3})
    _, weight, _ = combine(a, b, COLS, CombineWeights(agree=1.0, only_a=0.5))
    assert weight.loc["s1", "ACL"] == 1.0   # both agreed
    assert weight.loc["s2", "ACL"] == 0.5   # single source
    assert weight.loc["s3", "ACL"] == 0.0   # nobody


def test_prefer_on_conflict_breaks_ties():
    a = _frame({"ACL": [1.0, np.nan, np.nan], "MCL": [np.nan] * 3})
    b = _frame({"ACL": [0.0, np.nan, np.nan], "MCL": [np.nan] * 3})
    labels, _, _ = combine(a, b, COLS, prefer_on_conflict="a")
    assert labels.loc["s1", "ACL"] == 1.0
    labels, _, _ = combine(a, b, COLS, prefer_on_conflict="b")
    assert labels.loc["s1", "ACL"] == 0.0


def test_union_of_indexes_is_kept():
    a = pd.DataFrame({"ACL": [1.0]}, index=["x"], columns=COLS, dtype=float)
    b = pd.DataFrame({"ACL": [0.0]}, index=["y"], columns=COLS, dtype=float)
    labels, _, _ = combine(a, b, COLS)
    assert set(labels.index) == {"x", "y"}


def test_never_invents_a_label():
    a = _frame({"ACL": [np.nan] * 3, "MCL": [np.nan] * 3})
    b = _frame({"ACL": [np.nan] * 3, "MCL": [np.nan] * 3})
    labels, weight, prov = combine(a, b, COLS)
    assert labels.isna().to_numpy().all()
    assert (weight.to_numpy() == 0).all()
    assert (prov.to_numpy() == "none").all()


def test_provenance_summary_counts():
    a = _frame({"ACL": [1.0, 0.0, np.nan], "MCL": [np.nan] * 3})
    b = _frame({"ACL": [1.0, 1.0, 0.0], "MCL": [np.nan] * 3})
    s = provenance_summary(combine(a, b, COLS)[2]).set_index("column")
    assert s.loc["ACL", "agree"] == 1
    assert s.loc["ACL", "conflict"] == 1
    assert s.loc["ACL", "only_b"] == 1


def test_evaluate_by_provenance_separates_classes():
    """The whole scheme rests on 'agree' cells being more accurate."""
    gold = _frame({"ACL": [1.0, 1.0, 1.0], "MCL": [np.nan] * 3})
    a = _frame({"ACL": [1.0, 0.0, np.nan], "MCL": [np.nan] * 3})
    b = _frame({"ACL": [1.0, np.nan, np.nan], "MCL": [np.nan] * 3})
    labels, _, prov = combine(a, b, COLS)
    res = evaluate_by_provenance(gold, labels, prov, COLS).set_index("provenance")
    assert res.loc["agree", "accuracy"] == 1.0
    assert res.loc["only_a", "accuracy"] == 0.0
