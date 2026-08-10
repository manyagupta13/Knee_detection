"""The torch Dataset. Split out of data.py so that table loading and the text
labeler stay importable without a working torch install."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import PreprocessConfig
from preprocess import preprocess_study, to_model_input


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
