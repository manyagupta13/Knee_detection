"""Combine independent labelers by agreement.

Measured on the 58 gold studies: the rule labeler reaches 0.805 weighted
agreement, the LLM 0.752 - but they fail on DIFFERENT columns. The rules win
ACL/MCL/Fracture/Effusion; the LLM wins Medial OA, Medial Meniscus, PF OA
precision and Contusion, i.e. exactly the columns three lexicon passes could not
fix. The LLM also labels ~44% more cells and reads the ~23% of the corpus
(Greek, Cyrillic, Slavic, French) the lexicon cannot touch at all.

Two independent labelers agreeing is far stronger evidence than either alone, so
provenance is tracked per cell:

    agree      both committed to the same value  -> highest confidence
    only_a     only the first had an opinion     -> that labeler's confidence
    only_b     only the second                   -> that labeler's confidence
    conflict   they disagree                     -> abstain (or heavy discount)

Nothing here assumes how much better agreement is. Run evaluate_labeler.py on
the combined frame and let gold say.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from config import TARGET_COLUMNS


@dataclass(frozen=True)
class CombineWeights:
    """Loss weight per provenance class."""

    agree: float = 1.0
    only_a: float = 0.6
    only_b: float = 0.6
    conflict: float = 0.0  # 0.0 = abstain entirely

    def as_dict(self) -> dict[str, float]:
        return {
            "agree": self.agree,
            "only_a": self.only_a,
            "only_b": self.only_b,
            "conflict": self.conflict,
        }


def combine(
    a: pd.DataFrame,
    b: pd.DataFrame,
    columns: Sequence[str] = TARGET_COLUMNS,
    weights: CombineWeights | None = None,
    prefer_on_conflict: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge two label frames.

    Returns ``(labels, weight, provenance)`` aligned on the union of both
    indexes. ``prefer_on_conflict`` may be "a" or "b" to break ties with the
    labeler measured stronger on that column instead of abstaining.
    """
    columns = list(columns)
    weights = weights or CombineWeights()
    index = a.index.union(b.index)

    A = a.reindex(index=index, columns=columns).astype(float)
    B = b.reindex(index=index, columns=columns).astype(float)

    has_a, has_b = A.notna(), B.notna()
    agree = has_a & has_b & (A == B)
    conflict = has_a & has_b & (A != B)
    only_a = has_a & ~has_b
    only_b = has_b & ~has_a

    labels = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    labels = labels.where(~agree, A)
    labels = labels.where(~only_a, A)
    labels = labels.where(~only_b, B)
    if prefer_on_conflict == "a":
        labels = labels.where(~conflict, A)
    elif prefer_on_conflict == "b":
        labels = labels.where(~conflict, B)

    weight = pd.DataFrame(0.0, index=index, columns=columns, dtype=float)
    weight = weight.mask(agree, weights.agree)
    weight = weight.mask(only_a, weights.only_a)
    weight = weight.mask(only_b, weights.only_b)
    weight = weight.mask(conflict, weights.conflict)
    weight = weight.where(labels.notna(), 0.0)

    provenance = pd.DataFrame("none", index=index, columns=columns, dtype=object)
    provenance = provenance.mask(agree, "agree")
    provenance = provenance.mask(only_a, "only_a")
    provenance = provenance.mask(only_b, "only_b")
    provenance = provenance.mask(conflict, "conflict")

    return labels, weight, provenance


def provenance_summary(provenance: pd.DataFrame) -> pd.DataFrame:
    """Per-column counts of how each label was decided."""
    rows = []
    for col in provenance.columns:
        counts = provenance[col].value_counts()
        rows.append(
            {
                "column": col,
                "agree": int(counts.get("agree", 0)),
                "only_a": int(counts.get("only_a", 0)),
                "only_b": int(counts.get("only_b", 0)),
                "conflict": int(counts.get("conflict", 0)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_by_provenance(
    gold: pd.DataFrame,
    labels: pd.DataFrame,
    provenance: pd.DataFrame,
    columns: Sequence[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Agreement with gold, split by provenance class.

    This is the number that decides the weights: if "agree" cells are ~0.95
    accurate and single-source cells ~0.75, the split is worth having. If they
    are the same, the whole scheme is pointless and one labeler should be used.
    """
    columns = [c for c in columns if c in labels.columns and c in gold.columns]
    rows = []
    for kind in ("agree", "only_a", "only_b", "conflict"):
        correct = total = 0
        for col in columns:
            sel = (provenance[col] == kind) & gold[col].notna() & labels[col].notna()
            idx = gold.index[gold.index.isin(provenance.index)]
            sel = sel.reindex(idx).fillna(False)
            if not sel.any():
                continue
            g = gold.loc[sel[sel].index, col].to_numpy(dtype=float)
            p = labels.loc[sel[sel].index, col].to_numpy(dtype=float)
            correct += int((g == p).sum())
            total += len(g)
        rows.append(
            {
                "provenance": kind,
                "n_cells": total,
                "accuracy": (correct / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows)
