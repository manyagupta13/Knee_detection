"""Cheap checks to run BEFORE an expensive training run.

A 5-fold run costs hours. Everything here takes minutes and answers the
questions that would otherwise surface partway through it, or - worse - silently
cost points on the leaderboard:

  * do TEST studies resolve to series? Any that do not get a hardcoded 0.5 in
    the submission, which is a guaranteed loss on that row, and we have never
    actually measured this on test;
  * is laterality determinable? Canonicalization is a no-op where it is not,
    so this is exactly how much variance reduction is available;
  * how many slices do series actually have? Sampling 12 from 40 means a
    stride of 3, and a focal fracture living on 2 slices can be skipped;
  * how long will one epoch take, extrapolated from a small sample.

    python preflight.py --data-dir <data>
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from config import MULTI_PLANE_PREFS, PreprocessConfig
from data import load_tables
from preprocess import (
    _list_dicoms,
    _peek_laterality,
    classify_series_row,
    preprocess_study,
    select_series,
)


def check_test_coverage(data_dir: str | Path) -> dict:
    """Every test study must resolve to at least one series."""
    print("=" * 70)
    print("TEST COVERAGE - studies that fail here get a hardcoded 0.5")
    print("=" * 70)
    data_dir = Path(data_dir)
    tables = load_tables(data_dir, split="test")
    sample_path = data_dir / "sample_submission.csv"
    required = (
        pd.read_csv(sample_path, dtype=str)["StudyInstanceUID"].astype(str).tolist()
        if sample_path.exists()
        else tables.study_uids
    )

    have_series = set(tables.study_uids)
    missing = [u for u in required if u not in have_series]
    resolved = 0
    for uid in required:
        if uid not in have_series:
            continue
        rows = tables.series[tables.series["StudyInstanceUID"].astype(str) == uid]
        if select_series(rows, MULTI_PLANE_PREFS, max_series=3):
            resolved += 1

    print(f"required studies      : {len(required)}")
    print(f"present in test_series: {len(required) - len(missing)}")
    print(f"resolve to >=1 series : {resolved}")
    if resolved < len(required):
        print(f"WARNING: {len(required) - resolved} studies will fall back to 0.5")
    else:
        print("OK - every test study resolves")
    return {"required": len(required), "resolved": resolved, "missing": len(missing)}


def check_laterality(data_dir: str | Path, sample: int = 200) -> dict:
    """How often can we determine the knee side? Canonicalization needs it."""
    print("\n" + "=" * 70)
    print("LATERALITY - canonicalization is a no-op where this is unknown")
    print("=" * 70)
    tables = load_tables(data_dir, split="train")
    rows = tables.series.head(sample)
    counts = Counter(_peek_laterality(r["series_dir"]) for _, r in rows.iterrows())
    known = sum(v for k, v in counts.items() if k in ("L", "R"))
    total = sum(counts.values())
    print(f"sampled series: {total}")
    for key in ("L", "R", None):
        print(f"  {str(key):>5}: {counts.get(key, 0)}")
    pct = 100.0 * known / total if total else 0.0
    print(f"determinable: {pct:.1f}%")
    if pct < 50:
        print("NOTE: canonicalization will do little; not worth the cache rebuild")
    return {"determinable_pct": pct, "counts": dict(counts)}


def check_slice_counts(data_dir: str | Path, sample: int = 200) -> dict:
    """If series carry many more slices than we sample, we are skipping tissue."""
    print("\n" + "=" * 70)
    print("SLICE COUNTS - sampling 12 of 40 can skip a focal fracture")
    print("=" * 70)
    tables = load_tables(data_dir, split="train")
    counts = []
    for _, r in tables.series.head(sample).iterrows():
        n = len(_list_dicoms(r["series_dir"]))
        if n:
            counts.append(n)
    if not counts:
        print("no series readable")
        return {}
    arr = np.array(counts)
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3}: {np.percentile(arr, q):.0f} slices")
    print(f"  mean: {arr.mean():.1f}")
    for n_slices in (12, 16, 20, 24, 32):
        frac = float((arr <= n_slices).mean())
        print(f"  n_slices={n_slices:<3} covers every slice in {100*frac:.0f}% of series")
    return {"median": float(np.median(arr)), "mean": float(arr.mean())}


def time_preprocess(data_dir: str | Path, n_slices: int, size: int,
                    max_series: int, sample: int = 25) -> dict:
    """Extrapolate the one-time cache build cost."""
    print("\n" + "=" * 70)
    print("DECODE TIMING - the cache pays this once, not once per epoch")
    print("=" * 70)
    tables = load_tables(data_dir, split="train")
    cfg = PreprocessConfig(n_slices=n_slices, size=size, max_series=max_series,
                           plane_prefs=MULTI_PLANE_PREFS)
    uids = tables.study_uids[:sample]
    t0 = time.time()
    for uid in uids:
        preprocess_study(uid, tables.series, cfg.plane_prefs, cfg.n_slices,
                         cfg.size, cfg.max_series)
    per = (time.time() - t0) / max(len(uids), 1)
    total = per * len(tables.study_uids)
    print(f"{per:.2f}s/study  ->  {total/60:.0f} min for {len(tables.study_uids)} studies")
    print(f"(with 4 workers: ~{total/60/4:.0f} min, once, reused by every fold)")
    return {"seconds_per_study": per, "estimated_cache_minutes": total / 60}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-slices", type=int, default=20)
    ap.add_argument("--size", type=int, default=288)
    ap.add_argument("--max-series", type=int, default=3)
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()

    check_test_coverage(args.data_dir)
    check_laterality(args.data_dir, args.sample)
    check_slice_counts(args.data_dir, args.sample)
    time_preprocess(args.data_dir, args.n_slices, args.size, args.max_series)


if __name__ == "__main__":
    main()
