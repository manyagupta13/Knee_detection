"""Show the report text behind labeler misses and false positives.

Twice now, guessing at how reports phrase things has cost a round trip. This
prints the actual evidence for the cases that matter:

  * MISSED   - gold says positive, the labeler abstained. What phrasing did we
               fail to recognise?
  * WRONG    - the labeler said positive, gold says negative. What tripped it?

For findings with near-zero coverage it also runs a deliberately broad
morphological probe (e.g. "arthr|artr|osteo|chondr|kondr|degener|gonar") and
prints the matching sentences, since the whole problem is not knowing which
words to look for yet.

    python inspect_labeler.py --data-dir <competition data> --columns "Medial OA,PF OA"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config import ID_COLUMN, TARGET_COLUMNS
from data import load_tables
from langbucket import detect_language
from textlabel import normalize, split_clauses

# Broad, language-agnostic stems for the concepts we are currently blind to.
PROBES: dict[str, str] = {
    "Medial OA": r"arthr|artr|osteo|chondr|kondr|degener|gonar|kikirdak|knorpel|kraakbeen|medial|interno|innen",
    "Lateral OA": r"arthr|artr|osteo|chondr|kondr|degener|gonar|kikirdak|knorpel|kraakbeen|lateral|externo|aussen",
    "PF OA": r"patell|rotulian|femoropat|retropat|chondromalac|condromalac|kondromalaz",
    "Synovitis": r"synov|sinov",
    "Contusion": r"contus|kontus|bruise|edema|odem|oedeem",
    "Baker's": r"baker|popliteal|poplite",
    "MCL": r"collateral|colateral|kollateral|innenband|binnenband|yan bag|mcl|lcm",
}
DEFAULT_PROBE = r"."


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;:\n])\s+", str(text)) if s.strip()]


def _matching_sentences(text: str, probe: str, limit: int = 3) -> list[str]:
    rx = re.compile(probe)
    hits = [s for s in _sentences(text) if rx.search(normalize(s))]
    return hits[:limit]


def inspect_column(
    column: str,
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    reports: pd.DataFrame,
    max_examples: int = 6,
    snippet: int = 300,
) -> None:
    text_by_uid = dict(
        zip(reports[ID_COLUMN].astype(str), reports["report_text"].astype(str))
    )
    probe = PROBES.get(column, DEFAULT_PROBE)

    g = gold[column] if column in gold.columns else pd.Series(dtype=float)
    p = pred[column].reindex(g.index) if column in pred.columns else pd.Series(dtype=float)

    missed = [u for u in g.index if g.get(u) == 1 and (u not in p.index or pd.isna(p.get(u)))]
    wrong = [u for u in g.index if g.get(u) == 0 and p.get(u) == 1]
    missed_neg = [u for u in g.index if g.get(u) == 1 and p.get(u) == 0]

    print("\n" + "=" * 74)
    print(f"{column}   gold+={int((g == 1).sum())}  missed={len(missed)}  "
          f"called-positive-but-gold-negative={len(wrong)}  called-negative-but-gold-positive={len(missed_neg)}")
    print("=" * 74)

    for tag, uids in (("MISSED (gold=1, labeler abstained)", missed),
                      ("WRONG (labeler=1, gold=0)", wrong),
                      ("FLIPPED (labeler=0, gold=1)", missed_neg)):
        if not uids:
            continue
        print(f"\n--- {tag} ---")
        for uid in uids[:max_examples]:
            text = text_by_uid.get(uid, "")
            lang = detect_language(text)
            hits = _matching_sentences(text, probe)
            print(f"\n[{lang}] {uid[-12:]}")
            if hits:
                for h in hits:
                    print(f"    > {h[:snippet]}")
            else:
                print(f"    (no probe hit) {text[:snippet]}")


def run(
    data_dir: str | Path,
    columns: Sequence[str] | None = None,
    max_examples: int = 6,
) -> None:
    from textlabel import label_reports

    tables = load_tables(data_dir, split="train")
    if tables.reports is None or tables.gold is None:
        raise SystemExit("need both reports and gold labels")

    columns = list(columns) if columns else list(TARGET_COLUMNS)
    pred, _ = label_reports(tables.reports, columns=list(TARGET_COLUMNS))

    g = tables.gold.set_index(tables.gold[ID_COLUMN].astype(str))
    keep = [c for c in TARGET_COLUMNS if c in g.columns]
    g = g[keep]
    g = g[g.notna().any(axis=1)]

    for column in columns:
        inspect_column(column, g, pred, tables.reports, max_examples=max_examples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--columns", default="Medial OA,Lateral OA,PF OA")
    ap.add_argument("--max-examples", type=int, default=6)
    args = ap.parse_args()
    run(args.data_dir, [c.strip() for c in args.columns.split(",") if c.strip()],
        max_examples=args.max_examples)


if __name__ == "__main__":
    main()
