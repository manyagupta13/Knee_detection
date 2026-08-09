"""Table loading + the torch Dataset. Shared verbatim by train.py and the
inference notebook - there is no test-only preprocessing path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import ID_COLUMN, PreprocessConfig, TARGET_COLUMNS
from preprocess import attach_series_dirs, preprocess_study, to_model_input


@dataclass
class Tables:
    series: pd.DataFrame  # includes series_dir
    gold: pd.DataFrame | None  # StudyInstanceUID + 12 label columns
    reports: pd.DataFrame | None
    study_uids: list[str]


def load_tables(data_dir: str | Path, split: str = "train") -> Tables:
    """Load the CSVs for a split.

    Expected layout (mirrors the competition, and what tools/make_fixture.py
    emits)::

        data_dir/
          train.csv              # gold labels, only for labeled studies
          train_reports.csv      # StudyInstanceUID, report_text  (train only)
          train_series.csv       # StudyInstanceUID, SeriesInstanceUID, SeriesDescription
          train_images/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
          test_series.csv, test_images/, sample_submission.csv
    """
    data_dir = Path(data_dir)
    series = pd.read_csv(data_dir / f"{split}_series.csv", dtype=str)
    series = attach_series_dirs(series, data_dir / f"{split}_images")

    gold = None
    gold_path = data_dir / f"{split}.csv"
    if gold_path.exists():
        gold = pd.read_csv(gold_path)
        gold[ID_COLUMN] = gold[ID_COLUMN].astype(str)

    reports = None
    reports_path = data_dir / f"{split}_reports.csv"
    if reports_path.exists():
        reports = pd.read_csv(reports_path, dtype=str)

    study_uids = list(dict.fromkeys(series[ID_COLUMN].astype(str)))
    return Tables(series=series, gold=gold, reports=reports, study_uids=study_uids)


def build_targets(
    study_uids: Sequence[str],
    gold: pd.DataFrame | None,
    pseudo: pd.DataFrame | None,
    columns: Sequence[str] = TARGET_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gold UNION pseudo-labels. Gold always wins on conflict.

    Returns ``(targets, mask)`` aligned to ``study_uids``; mask is True exactly
    where a label exists, which is what the masked BCE keys off.
    """
    columns = list(columns)
    index = pd.Index([str(u) for u in study_uids], name=ID_COLUMN)
    targets = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)

    if pseudo is not None and len(pseudo):
        p = pseudo.reindex(index=index, columns=columns)
        targets = targets.where(p.isna(), p)

    if gold is not None and len(gold):
        g = gold.set_index(gold[ID_COLUMN].astype(str)).reindex(
            index=index, columns=columns
        )
        targets = targets.where(g.isna(), g)

    return targets, targets.notna()


class StudyDataset(Dataset):
    """One item per study. Yields the model input, targets and mask."""

    def __init__(
        self,
        study_uids: Sequence[str],
        series_df: pd.DataFrame,
        config: PreprocessConfig,
        targets: pd.DataFrame | None = None,
        mask: pd.DataFrame | None = None,
        series_index: int = 0,
    ):
        self.study_uids = [str(u) for u in study_uids]
        self.series_df = series_df
        self.config = config
        self.targets = targets
        self.mask = mask
        # Phase 0 feeds one series to the model; the stacked form is still what
        # preprocess_study returns. TODO(phase-2): consume all series slots.
        self.series_index = series_index

    def __len__(self) -> int:
        return len(self.study_uids)

    def volume(self, uid: str) -> np.ndarray:
        study = preprocess_study(
            uid,
            self.series_df,
            self.config.plane_prefs,
            self.config.n_slices,
            self.config.size,
            self.config.max_series,
        )
        assert study.pixels.ndim == 4, "preprocess_study must return the stacked form"
        return study.pixels[self.series_index]

    def __getitem__(self, i: int):
        uid = self.study_uids[i]
        x = torch.from_numpy(to_model_input(self.volume(uid)))
        n_targets = len(self.targets.columns) if self.targets is not None else 0
        if self.targets is None:
            y = torch.zeros(0)
            m = torch.zeros(0)
        else:
            y = torch.from_numpy(
                np.nan_to_num(self.targets.loc[uid].to_numpy(dtype=np.float32), nan=0.0)
            )
            m = torch.from_numpy(
                self.mask.loc[uid].to_numpy(dtype=np.float32)
                if self.mask is not None
                else np.ones(n_targets, dtype=np.float32)
            )
        return {"x": x, "y": y, "mask": m, "study_uid": uid}
