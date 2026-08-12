"""How much of the gold label is recoverable from the report AT ALL?

Before spending more on labeling, establish whether 0.805 agreement is our
failure or the data's ceiling. A gold study marked PF OA = 1 whose report says
"Cartilago rotuliano sin alteraciones" (patellar cartilage unremarkable) is not
a labeler bug - it is a finding the radiologist did not write down, and no
extractor can recover it.

This decomposes labeler error into two very different things:

  MENTION RATE      of gold-positive cells, how many reports mention the finding
                    at all (using a deliberately HIGH-RECALL probe - any hint
                    counts). This is the hard ceiling on any text labeler.

  ACCURACY | MENTION  where the finding IS mentioned, how often we get it right.
                      This is the part that is actually ours to fix.

If mention rate is ~95%, the ceiling is high and better extraction pays. If it
is ~65%, then 0.805 is close to the maximum and further labeling work is wasted
effort - the remaining headroom is entirely in the image model.

    python ceiling.py --data-dir <data>
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
from lexicon import SPEC_BY_COLUMN
from textlabel import normalize

# Deliberately high-recall: ANY hint that the finding was discussed counts.
# Over-matching is fine here - the point is an upper bound, so a false "mention"
# only makes the ceiling look better than it is, which is the conservative
# direction for the conclusion "stop investing in labeling".
EXTRA_PROBES: dict[str, tuple[str, ...]] = {
    "ACL": (r"cruciat", r"cruzado", r"kreuzband", r"kruisband", r"capraz", r"\bacl\b",
            r"\blca\b", r"\bvkb\b", r"σταυρ", r"крест"),
    "MCL": (r"collateral", r"colateral", r"kollateral", r"innenband", r"binnenband",
            r"yan bag", r"\bmcl\b", r"\blcm\b", r"πλαγ", r"колла"),
    "Medial Meniscus": (r"menisc", r"menisk", r"μηνισκ", r"мениск"),
    "Lateral Meniscus": (r"menisc", r"menisk", r"μηνισκ", r"мениск"),
    "Medial OA": (r"arthr", r"artr", r"osteo", r"chondr", r"kondr", r"knorpel",
                  r"kraakbeen", r"kikirdak", r"degener", r"αρθρ", r"артро", r"хондр"),
    "Lateral OA": (r"arthr", r"artr", r"osteo", r"chondr", r"kondr", r"knorpel",
                   r"kraakbeen", r"kikirdak", r"degener", r"αρθρ", r"артро", r"хондр"),
    "PF OA": (r"patell", r"rotulian", r"femoropat", r"retropat", r"chondromalac",
              r"condromalac", r"kondromalaz", r"trochlea", r"επιγονατ", r"надколен"),
    "Effusion": (r"effusion", r"derrame", r"efuzyon", r"erguss", r"hydrops", r"vocht",
                 r"fluid", r"liquid", r"sivi", r"υγρ", r"выпот", r"izliv"),
    "Synovitis": (r"synov", r"sinov", r"υμεν", r"синов"),
    "Baker's": (r"baker", r"popliteal", r"poplite", r"cyst", r"quiste", r"kist",
                r"zyste", r"cyste", r"κύστ", r"киста"),
    "Contusion": (r"contus", r"kontus", r"bruise", r"edema", r"odem", r"oedeem",
                  r"ödem", r"οίδημα", r"отек", r"medul"),
    "Fracture": (r"fractur", r"fraktur", r"kirik", r"breuk", r"bruch", r"κάταγμα",
                 r"перелом", r"prijelom"),
}


def _mention_regex(column: str) -> re.Pattern:
    spec = SPEC_BY_COLUMN.get(column)
    parts: list[str] = list(EXTRA_PROBES.get(column, ()))
    if spec is not None:
        parts += list(spec.self_sufficient) + list(spec.structures)
    return re.compile("|".join(parts) if parts else r"(?!x)x")


def mention_table(
    gold: pd.DataFrame,
    reports: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    columns: Sequence[str] = TARGET_COLUMNS,
) -> pd.DataFrame:
    text_by_uid = {
        str(u): normalize(t)
        for u, t in zip(reports[ID_COLUMN].astype(str), reports["report_text"])
    }
    rows = []
    for col in columns:
        if col not in gold.columns:
            continue
        rx = _mention_regex(col)
        labeled = gold.index[gold[col].notna()]
        if len(labeled) == 0:
            continue

        mentioned = np.array([bool(rx.search(text_by_uid.get(u, ""))) for u in labeled])
        truth = gold.loc[labeled, col].to_numpy(dtype=float)
        pos = truth == 1

        row = {
            "column": col,
            "n_gold": int(len(labeled)),
            "n_pos": int(pos.sum()),
            "mention_rate_pos": float(mentioned[pos].mean()) if pos.any() else np.nan,
            "mention_rate_all": float(mentioned.mean()),
        }

        if predictions is not None and col in predictions.columns:
            pred = predictions[col].reindex(labeled).to_numpy(dtype=float)
            committed = ~np.isnan(pred)
            correct = committed & (pred == truth)
            row["labeler_coverage"] = float(committed.mean())
            with np.errstate(invalid="ignore"):
                row["acc_given_mention"] = (
                    float(correct[mentioned & committed].sum())
                    / max(int((mentioned & committed).sum()), 1)
                    if (mentioned & committed).any()
                    else np.nan
                )
                # Gold positives the report never discusses - irrecoverable.
                row["missed_unmentionable"] = int((pos & ~mentioned).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def run(data_dir: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    from textlabel import label_reports

    columns = list(columns) if columns else list(TARGET_COLUMNS)
    tables = load_tables(data_dir, split="train")
    gold = tables.gold.set_index(tables.gold[ID_COLUMN].astype(str))
    keep = [c for c in columns if c in gold.columns]
    gold = gold[keep]
    gold = gold[gold.notna().any(axis=1)]

    preds, _ = label_reports(tables.reports, columns=columns)
    table = mention_table(gold, tables.reports, preds.reindex(gold.index), keep)

    print("=" * 78)
    print("IS THE LABEL EVEN IN THE REPORT?")
    print("=" * 78)
    print(table.to_string(index=False))

    pos_rate = np.nansum(table["mention_rate_pos"] * table["n_pos"]) / max(
        table["n_pos"].sum(), 1
    )
    unmentionable = int(table.get("missed_unmentionable", pd.Series([0])).sum())
    total_pos = int(table["n_pos"].sum())
    print(f"\nweighted mention rate on gold POSITIVES: {pos_rate:.3f}")
    print(f"gold positives the report never discusses: {unmentionable}/{total_pos}")
    print(
        "\nINTERPRETATION\n"
        "  mention rate ~0.95 -> reports carry the signal; better extraction pays.\n"
        "  mention rate ~0.65 -> gold was largely read off IMAGES, not reports.\n"
        "                        0.805 is then near the ceiling and further\n"
        "                        labeling work is wasted - spend it on the image model."
    )

    langs = pd.Series(
        [detect_language(t) for t in tables.reports["report_text"]],
        index=tables.reports[ID_COLUMN].astype(str),
    )
    gold_langs = langs.reindex(gold.index).value_counts()
    print(f"\ngold studies by language:\n{gold_langs.to_string()}")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    run(args.data_dir)


if __name__ == "__main__":
    main()
