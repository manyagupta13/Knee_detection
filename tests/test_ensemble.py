"""Rank-averaging: the metric is rank-based, so probabilities must not dominate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensemble import probability_average, rank_average, rank_normalize

COLS = ["ACL", "MCL"]
IDX = ["a", "b", "c", "d"]


def _f(acl, mcl):
    return pd.DataFrame({"ACL": acl, "MCL": mcl}, index=IDX, dtype=float)


def test_rank_normalize_spans_zero_to_one():
    out = rank_normalize(_f([0.1, 0.2, 0.3, 0.4], [4, 3, 2, 1]))
    assert out["ACL"].min() == 0.0 and out["ACL"].max() == 1.0
    assert list(out["ACL"]) == [0.0, 1 / 3, 2 / 3, 1.0]


def test_rank_average_ignores_scale_but_keeps_order():
    """A confident model and a timid one that agree on ORDER must contribute
    equally — this is the entire reason for rank-averaging."""
    confident = _f([0.99, 0.98, 0.02, 0.01], [1, 1, 0, 0])
    timid = _f([0.52, 0.51, 0.49, 0.48], [1, 1, 0, 0])
    out = rank_average([confident, timid])
    assert list(out["ACL"].rank()) == [4.0, 3.0, 2.0, 1.0]


def test_probability_average_is_dominated_by_the_confident_model():
    """Contrast case: this is what we are avoiding."""
    confident = _f([0.99, 0.01, 0.99, 0.01], [0, 0, 0, 0])
    timid = _f([0.50, 0.51, 0.50, 0.51], [0, 0, 0, 0])
    prob = probability_average([confident, timid])
    # the confident model's ordering wins outright
    assert prob.loc["a", "ACL"] > prob.loc["b", "ACL"]
    # rank-averaging gives the timid model an equal vote, producing a tie
    ranked = rank_average([confident, timid])
    assert ranked.loc["a", "ACL"] == pytest.approx(ranked.loc["b", "ACL"], abs=1e-9)


def test_single_frame_passes_through():
    f = _f([0.1, 0.2, 0.3, 0.4], [1, 2, 3, 4])
    assert rank_average([f]).equals(f)


def test_auc_is_preserved_by_rank_normalization():
    from sklearn.metrics import roc_auc_score

    y = [0, 0, 1, 1]
    f = _f([0.1, 0.4, 0.35, 0.8], [0, 0, 0, 0])
    before = roc_auc_score(y, f["ACL"])
    after = roc_auc_score(y, rank_normalize(f)["ACL"])
    assert before == pytest.approx(after)


def test_missing_studies_get_mid_rank_not_zero():
    """A model that failed on a study must not drag it to the bottom."""
    full = _f([0.9, 0.8, 0.7, 0.6], [1, 2, 3, 4])
    partial = full.copy()
    partial.loc["c"] = np.nan
    out = rank_average([full, partial])
    assert not out.isna().to_numpy().any()
    assert 0.0 < out.loc["c", "ACL"] < 1.0


def test_weights_are_honoured():
    a = _f([1.0, 2.0, 3.0, 4.0], [1, 2, 3, 4])
    b = _f([4.0, 3.0, 2.0, 1.0], [4, 3, 2, 1])
    heavy_a = rank_average([a, b], weights=[10.0, 1.0])
    assert heavy_a.loc["d", "ACL"] > heavy_a.loc["a", "ACL"]


def test_empty_input_raises():
    with pytest.raises(ValueError):
        rank_average([])
