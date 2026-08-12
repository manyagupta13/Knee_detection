"""Rank-averaging across models.

PLAN.md §4: ensemble by RANK average, not probability average. The metric is
AUC, which only sees ordering, and the prevalence shift between splits means
calibrated probabilities do not transfer while rankings do. Averaging raw
probabilities also lets an over-confident model dominate a better-ordered one.

Ranks are computed per column, independently per model, then averaged and
rescaled to (0, 1) - a monotone transform, so AUC is unchanged by the rescale
but the output still looks like a probability for the submission format.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def rank_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column ranks scaled to (0, 1). Ties get their average rank."""
    if len(frame) == 0:
        return frame.copy()
    if len(frame) == 1:
        return frame.rank(axis=0).astype(float) * 0.0 + 0.5
    ranks = frame.rank(axis=0, method="average")
    return (ranks - 1.0) / (len(frame) - 1.0)


def rank_average(
    frames: Sequence[pd.DataFrame], weights: Sequence[float] | None = None
) -> pd.DataFrame:
    """Rank-average aligned prediction frames (index = study, cols = findings)."""
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        raise ValueError("nothing to ensemble")
    if len(frames) == 1:
        return frames[0].copy()

    index = frames[0].index
    columns = list(frames[0].columns)
    if weights is None:
        weights = [1.0] * len(frames)
    if len(weights) != len(frames):
        raise ValueError("weights must match frames")

    total = np.zeros((len(index), len(columns)), dtype=float)
    denom = 0.0
    for frame, w in zip(frames, weights):
        aligned = frame.reindex(index=index, columns=columns).astype(float)
        # A model that failed on some study must not drag the ensemble to 0;
        # fill its gaps with the mid-rank, which is "no opinion".
        normed = rank_normalize(aligned).fillna(0.5)
        total += w * normed.to_numpy()
        denom += w
    return pd.DataFrame(total / denom, index=index, columns=columns)


def probability_average(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Provided only for comparison. Prefer rank_average for this metric."""
    frames = [f for f in frames if f is not None and len(f)]
    index, columns = frames[0].index, list(frames[0].columns)
    stack = np.stack(
        [f.reindex(index=index, columns=columns).to_numpy(dtype=float) for f in frames]
    )
    return pd.DataFrame(np.nanmean(stack, axis=0), index=index, columns=columns)
