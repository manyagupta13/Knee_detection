"""The torch Dataset. Split out of data.py so that table loading and the text
labeler stay importable without a working torch install."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from augment import AugmentConfig, augment_study
from cache import StudyCache
from config import PreprocessConfig
from model import PLANE_IDS
from preprocess import preprocess_study, slice_indices, to_model_input


class StudyDataset(Dataset):
    """One item per study.

    Yields the FULL series stack ``(S, T, 3, H, W)`` plus the series mask and
    plane ids, so the same Dataset serves single-series (S=1) and multi-series
    runs with no branching.
    """

    def __init__(
        self,
        study_uids: Sequence[str],
        series_df: pd.DataFrame,
        config: PreprocessConfig,
        targets: pd.DataFrame | None = None,
        mask: pd.DataFrame | None = None,
        augment: AugmentConfig | None = None,
        seed: int = 0,
        cache: StudyCache | None = None,
        input_mode: str = "2.5d",
        slice_jitter: float = 0.0,
        slice_offset: float = 0.0,
    ):
        self.study_uids = [str(u) for u in study_uids]
        self.series_df = series_df
        self.config = config
        self.targets = targets
        self.mask = mask
        self.augment = augment
        self.seed = seed
        # Read-through decode cache. Augmentation runs AFTER this, on the uint8
        # volume, so caching cannot reduce augmentation diversity.
        self.cache = cache
        self.input_mode = input_mode
        # Sampling n_slices out of the cached pool. Jitter (training) varies the
        # window per epoch so the same slices are not skipped every time; offset
        # (TTA) shifts it deterministically so several passes see different
        # tissue and can be averaged.
        self.slice_jitter = slice_jitter
        self.slice_offset = slice_offset

    def __len__(self) -> int:
        return len(self.study_uids)

    def study(self, uid: str):
        if self.cache is not None and self.cache.enabled:
            return self.cache.get(uid, self.series_df)
        out = preprocess_study(
            uid,
            self.series_df,
            self.config.plane_prefs,
            self.config.pool_slices,
            self.config.size,
            self.config.max_series,
            canonicalize=self.config.canonicalize,
        )
        assert out.pixels.ndim == 4, "preprocess_study must return the stacked form"
        return out

    def __getitem__(self, i: int):
        uid = self.study_uids[i]
        study = self.study(uid)
        pixels = study.pixels

        # Pick n_slices out of the cached pool.
        pool_len = pixels.shape[1]
        if pool_len != self.config.n_slices or self.slice_jitter or self.slice_offset:
            rng_idx = (
                np.random.default_rng((self.seed * 7_919 + i * 104_729) % (2**32))
                if self.slice_jitter
                else None
            )
            idx = slice_indices(pool_len, self.config.n_slices,
                                offset_frac=self.slice_offset,
                                jitter=self.slice_jitter, rng=rng_idx)
            pixels = pixels[:, idx]

        if self.augment is not None and self.augment.enabled:
            # Seeded per (epoch-agnostic) item so a worker restart is harmless,
            # but varied across items.
            rng = np.random.default_rng((self.seed * 1_000_003 + i) % (2**32))
            pixels = augment_study(pixels, rng, self.augment)

        x = torch.from_numpy(
            np.stack([to_model_input(pixels[s], self.input_mode) for s in range(pixels.shape[0])])
        )  # (S, T, 3, H, W)
        series_mask = torch.from_numpy(study.series_mask.astype(np.float32))
        plane_ids = torch.tensor(
            [PLANE_IDS.get(p, 0) for p in study.planes], dtype=torch.long
        )

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
        return {
            "x": x,
            "y": y,
            "mask": m,
            "series_mask": series_mask,
            "plane_ids": plane_ids,
            "study_uid": uid,
        }
