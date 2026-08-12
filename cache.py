"""Decode-once cache for preprocessed studies.

Measured: a 3-series/12-slice run over 3,324 studies spent ~46 min PER EPOCH
decoding DICOMs (3324 x 3 x 0.28 s). Across 8 epochs that is ~6 hours of wall
clock with the GPU mostly idle, and it is pure waste - the decoded tensor is
identical every epoch because augmentation happens after this stage, on the
uint8 volume.

Caching the decoded uint8 volumes turns training from decode-bound into
GPU-bound and makes k-fold ensembles and larger backbones affordable.

The cache key includes every preprocessing parameter that changes the pixels,
so a config change can never silently serve stale tensors - the one bug that
would be catastrophic here and invisible in the metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import PreprocessConfig
from preprocess import StudyVolume, preprocess_study

PLANE_CODES = {"unknown": 0, "sagittal": 1, "coronal": 2, "axial": 3}
CODE_PLANES = {v: k for k, v in PLANE_CODES.items()}


def cache_key(config: PreprocessConfig) -> str:
    """Short hash of everything that affects the decoded pixels."""
    payload = json.dumps(config.to_dict(), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


class StudyCache:
    """One .npz per study: pixels + series_mask + plane codes."""

    def __init__(self, root: str | Path | None, config: PreprocessConfig):
        self.config = config
        self.key = cache_key(config)
        self.root = Path(root) / self.key if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "config.json").write_text(
                json.dumps(config.to_dict(), indent=2), encoding="utf-8"
            )

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _path(self, study_uid: str) -> Path:
        # Shard by prefix so one directory does not hold 4,400 entries.
        digest = hashlib.sha1(str(study_uid).encode("utf-8")).hexdigest()[:2]
        return self.root / digest / f"{study_uid}.npz"

    def load(self, study_uid: str) -> StudyVolume | None:
        if not self.enabled:
            return None
        path = self._path(study_uid)
        if not path.exists():
            return None
        try:
            with np.load(path) as z:
                pixels = z["pixels"]
                series_mask = z["series_mask"].astype(bool)
                planes = [CODE_PLANES.get(int(c), "unknown") for c in z["planes"]]
                lat = [None if v == "" else str(v) for v in z["laterality"]]
        except Exception:
            return None  # a truncated file must not kill the run
        expected = (
            self.config.max_series,
            self.config.n_slices,
            self.config.size,
            self.config.size,
        )
        if pixels.shape != expected:
            return None
        return StudyVolume(
            pixels=pixels,
            series_mask=series_mask,
            series_uids=[None] * len(series_mask),
            planes=planes,
            laterality=lat,
        )

    def store(self, study_uid: str, study: StudyVolume) -> None:
        if not self.enabled:
            return
        path = self._path(study_uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            pixels=study.pixels,
            series_mask=study.series_mask,
            planes=np.array([PLANE_CODES.get(p, 0) for p in study.planes], dtype=np.int8),
            laterality=np.array(
                [("" if v is None else str(v)) for v in study.laterality], dtype="<U1"
            ),
        )
        tmp.replace(path)  # atomic, so a killed session leaves no half-file

    def get(self, study_uid: str, series_df: pd.DataFrame) -> StudyVolume:
        """Cached read-through to preprocess_study()."""
        hit = self.load(study_uid)
        if hit is not None:
            return hit
        study = preprocess_study(
            study_uid,
            series_df,
            self.config.plane_prefs,
            self.config.n_slices,
            self.config.size,
            self.config.max_series,
            canonicalize=self.config.canonicalize,
        )
        self.store(study_uid, study)
        return study


def build_cache(
    study_uids,
    series_df: pd.DataFrame,
    config: PreprocessConfig,
    root: str | Path,
    num_workers: int = 4,
    verbose: bool = True,
) -> StudyCache:
    """Populate the cache up front, in parallel. Decode is CPU-bound and
    embarrassingly parallel, so this is where multiprocessing actually pays."""
    import time
    from concurrent.futures import ProcessPoolExecutor

    cache = StudyCache(root, config)
    uids = [str(u) for u in study_uids]
    todo = [u for u in uids if cache.load(u) is None]
    if verbose:
        print(f"cache {cache.key}: {len(uids) - len(todo)} hit, {len(todo)} to decode")
    if not todo:
        return cache

    t0 = time.time()
    if num_workers and num_workers > 1:
        cols = ["StudyInstanceUID", "SeriesInstanceUID", "series_dir"]
        for extra in ("Anatomical_Plane", "Fluid_Sensitive", "SeriesDescription"):
            if extra in series_df.columns:
                cols.append(extra)
        slim = series_df[[c for c in cols if c in series_df.columns]]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(
                    _decode_one,
                    uid,
                    slim[slim["StudyInstanceUID"].astype(str) == uid],
                    config,
                ): uid
                for uid in todo
            }
            for i, fut in enumerate(futures):
                uid = futures[fut]
                try:
                    cache.store(uid, fut.result())
                except Exception as exc:  # a bad study must not sink the run
                    print(f"  cache miss for {uid}: {type(exc).__name__}: {exc}")
                if verbose and (i + 1) % 250 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"  {i + 1}/{len(todo)} @ {rate:.1f}/s")
    else:
        for i, uid in enumerate(todo):
            cache.get(uid, series_df)
            if verbose and (i + 1) % 250 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(todo)} @ {rate:.1f}/s")

    if verbose:
        print(f"cache built in {time.time() - t0:.0f}s -> {root}")
    return cache


def _decode_one(uid: str, study_series: pd.DataFrame, config: PreprocessConfig):
    return preprocess_study(
        uid,
        study_series,
        config.plane_prefs,
        config.n_slices,
        config.size,
        config.max_series,
        canonicalize=config.canonicalize,
    )
