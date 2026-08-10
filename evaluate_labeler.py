"""Measure a text labeler against gold. The Phase-1 instrument.

Phase 0 flew blind: the only feedback signal was downstream image-model AUC,
which conflates label quality with model capacity and takes ~20 minutes per
look. This scores the labeler directly against the 58 gold studies in seconds,
per column and per language, so lexicon changes can be iterated on quickly.

Exit criterion for Phase 1 (PLAN.md): high agreement with gold on common
findings, and a documented, honest lower number on the rare ones. This module
is what produces that document.

    python evaluate_labeler.py --data-dir <competition data>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config import ID_COLUMN, TARGET_COLUMNS
from data import load_tables
from langbucket import detect_language


def score_column(
    gold: pd.Series, pred: pd.Series
) -> dict:
    """Agreement stats on the studies where BOTH gold and prediction exist.

    Coverage and correctness are reported separately on purpose: a labeler that
    abstains on everything has perfect precision and is useless, and one that
    guesses on everything has full coverage and is worse than useless.
    """
    both = gold.notna() & pred.notna()
    n_gold = int(gold.notna().sum())
    n_pred = int(pred.notna().sum())
    n_both = int(both.sum())

    out = {
        "n_gold": n_gold,
        "n_pred": n_pred,
        "n_overlap": n_both,
        "agreement": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "n_pred_pos": int((pred == 1).sum()),
        "n_gold_pos": int((gold == 1).sum()),
    }
    if n_both == 0:
        return out

    g = gold[both].to_numpy(dtype=float)
    p = pred[both].to_numpy(dtype=float)
    out["agreement"] = float((g == p).mean())

    tp = float(((p == 1) & (g == 1)).sum())
    fp = float(((p == 1) & (g == 0)).sum())
    fn = float(((p == 0) & (g == 1)).sum())
    if tp + fp > 0:
        out["precision"] = tp / (tp + fp)
    if tp + fn > 0:
        out["recall"] = tp / (tp + fn)
    return out


def evaluate(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    columns: Sequence[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Per-column agreement table. Both frames indexed by StudyInstanceUID."""
    rows = []
    for col in columns:
        g = gold[col] if col in gold.columns else pd.Series(dtype=float)
        p = pred[col] if col in pred.columns else pd.Series(dtype=float)
        idx = gold.index
        stats = score_column(g.reindex(idx), p.reindex(idx))
        rows.append({"column": col, **stats})
    return pd.DataFrame(rows)


def evaluate_by_language(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    languages: pd.Series,
    columns: Sequence[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    """Same, split by report language - a labeler can be strong in English and
    useless in Turkish, and the macro metric would hide it."""
    rows = []
    for lang, uids in languages.groupby(languages).groups.items():
        idx = gold.index.intersection(pd.Index(uids))
        if len(idx) == 0:
            continue
        sub = evaluate(gold.loc[idx], pred.reindex(idx), columns)
        overlap = int(sub["n_overlap"].sum())
        rows.append(
            {
                "language": lang,
                "n_gold_studies": len(idx),
                "n_overlap_cells": overlap,
                "agreement": float(
                    np.nansum(sub["agreement"] * sub["n_overlap"]) / overlap
                )
                if overlap
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("n_gold_studies", ascending=False)


def coverage_summary(pred: pd.DataFrame, columns: Sequence[str] = TARGET_COLUMNS) -> pd.DataFrame:
    """How many studies got a label at all, per column, over the FULL corpus."""
    rows = []
    for col in columns:
        s = pred[col] if col in pred.columns else pd.Series(dtype=float)
        rows.append(
            {
                "column": col,
                "labeled": int(s.notna().sum()),
                "pct_of_corpus": round(100.0 * s.notna().sum() / max(len(pred), 1), 1),
                "positive": int((s == 1).sum()),
                "negative": int((s == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def run(data_dir: str | Path, columns: Sequence[str] | None = None) -> dict:
    from textlabel import label_reports

    tables = load_tables(data_dir, split="train")
    if tables.reports is None:
        raise SystemExit("no reports found — check the Report column in train.csv")
    columns = list(columns) if columns is not None else list(TARGET_COLUMNS)

    pred, _ = label_reports(tables.reports, columns=columns)

    langs = pd.Series(
        [detect_language(t) for t in tables.reports["report_text"]],
        index=pd.Index(tables.reports[ID_COLUMN].astype(str), name=ID_COLUMN),
        name="language",
    )

    print("=" * 72)
    print("COVERAGE over the full corpus (how much supervision this buys)")
    print("=" * 72)
    cov = coverage_summary(pred, columns)
    print(cov.to_string(index=False))

    print("\nlanguage mix:")
    print(langs.value_counts().head(12).to_string())

    gold = None
    if tables.gold is not None:
        g = tables.gold.set_index(tables.gold[ID_COLUMN].astype(str))
        keep = [c for c in columns if c in g.columns]
        g = g[keep]
        gold = g[g.notna().any(axis=1)]

    agree = by_lang = None
    if gold is not None and len(gold):
        print("\n" + "=" * 72)
        print(f"AGREEMENT vs {len(gold)} gold studies (the only truth available)")
        print("=" * 72)
        agree = evaluate(gold, pred, columns)
        print(agree.to_string(index=False))

        overlap = int(agree["n_overlap"].sum())
        if overlap:
            weighted = float(
                np.nansum(agree["agreement"] * agree["n_overlap"]) / overlap
            )
            print(f"\noverall agreement on {overlap} overlapping cells: {weighted:.3f}")

        print("\nby language:")
        by_lang = evaluate_by_language(gold, pred, langs, columns)
        print(by_lang.to_string(index=False))

        blind = agree[agree["n_overlap"] == 0]["column"].tolist()
        if blind:
            print(
                f"\nNOT MEASURABLE ({len(blind)} columns — no gold/prediction overlap): {blind}"
            )
    return {"coverage": cov, "agreement": agree, "by_language": by_lang, "languages": langs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    run(args.data_dir)


if __name__ == "__main__":
    main()
