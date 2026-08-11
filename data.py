"""Table loading + the torch Dataset. Shared verbatim by train.py and the
inference notebook - there is no test-only preprocessing path.

torch is imported LAZILY: the text-labeling tools (evaluate_labeler, textlabel)
need load_tables but have no use for a GPU stack, and a broken or missing torch
in the environment should not stop you from measuring label quality.
``StudyDataset`` is resolved on first attribute access via PEP-562.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config import ID_COLUMN, PreprocessConfig, TARGET_COLUMNS
from preprocess import attach_series_dirs, preprocess_study, to_model_input


@dataclass
class Tables:
    series: pd.DataFrame  # includes series_dir
    gold: pd.DataFrame | None  # StudyInstanceUID + 12 label columns
    reports: pd.DataFrame | None
    study_uids: list[str]


REPORT_COLUMN = "Report"


def load_tables(data_dir: str | Path, split: str = "train") -> Tables:
    """Load the CSVs for a split.

    Competition layout (which tools/make_fixture.py reproduces)::

        data_dir/
          train.csv          # StudyInstanceUID, Report, + the 12 label columns
                             #   one row per study; labels are NaN for all but ~58
          train_series.csv   # StudyInstanceUID, SeriesInstanceUID,
                             #   Fluid_Sensitive, Fat_Suppression, Anatomical_Plane
          train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
          test.csv, test_series.csv, test_series/, sample_submission.csv

    Note the pixel directory is ``<split>_series/``, the same stem as the CSV.
    The reports are a *column of train.csv*, not a separate file, and the test
    split has none — which is the whole reason text is supervision-only.
    """
    data_dir = Path(data_dir)
    series = pd.read_csv(data_dir / f"{split}_series.csv", dtype=str)

    images_root = data_dir / f"{split}_series"
    if not images_root.is_dir():  # tolerate an <split>_images/ layout
        alt = data_dir / f"{split}_images"
        if alt.is_dir():
            images_root = alt
    series = attach_series_dirs(series, images_root)

    gold = None
    reports = None
    table_path = data_dir / f"{split}.csv"
    if table_path.exists():
        table = pd.read_csv(table_path)
        table[ID_COLUMN] = table[ID_COLUMN].astype(str)

        if REPORT_COLUMN in table.columns:
            reports = table[[ID_COLUMN, REPORT_COLUMN]].rename(
                columns={REPORT_COLUMN: "report_text"}
            )
            table = table.drop(columns=[REPORT_COLUMN])

        # test.csv carries only the id column - that is not a label table.
        if len(table.columns) > 1:
            gold = table

    study_uids = list(dict.fromkeys(series[ID_COLUMN].astype(str)))
    return Tables(series=series, gold=gold, reports=reports, study_uids=study_uids)


def build_targets(
    study_uids: Sequence[str],
    gold: pd.DataFrame | None,
    pseudo: pd.DataFrame | None,
    columns: Sequence[str] = TARGET_COLUMNS,
    pseudo_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gold UNION pseudo-labels. Gold always wins on conflict.

    Returns ``(targets, weights)`` aligned to ``study_uids``. ``weights`` is 0
    where there is no label, 1.0 for gold, and - when ``pseudo_weights`` is
    supplied - the per-column confidence for text-derived labels.

    The masked BCE multiplies by this and normalizes by its sum, so a binary
    mask and a weight matrix are the same object; passing measured per-column
    precision here is what turns "which columns do we trust" from an
    include/exclude decision into a graded one.
    """
    columns = list(columns)
    index = pd.Index([str(u) for u in study_uids], name=ID_COLUMN)
    targets = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    from_gold = pd.DataFrame(False, index=index, columns=columns)

    if pseudo is not None and len(pseudo):
        p = pseudo.reindex(index=index, columns=columns)
        targets = targets.where(p.isna(), p)

    if gold is not None and len(gold):
        g = gold.set_index(gold[ID_COLUMN].astype(str)).reindex(
            index=index, columns=columns
        )
        targets = targets.where(g.isna(), g)
        from_gold = g.notna()

    labeled = targets.notna()
    if pseudo_weights is None:
        return targets, labeled

    weights = pd.DataFrame(0.0, index=index, columns=columns, dtype=float)
    for col in columns:
        w = float(pseudo_weights.get(col, 1.0))
        weights[col] = np.where(labeled[col], w, 0.0)
        weights[col] = np.where(from_gold[col], 1.0, weights[col])
    return targets, weights


def __getattr__(name: str):
    """Lazily re-export StudyDataset so ``from data import StudyDataset`` keeps
    working, without making torch a hard import for pandas-only callers."""
    if name == "StudyDataset":
        from dataset import StudyDataset

        return StudyDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
