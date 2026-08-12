"""The decode cache must be transparent: same tensors, never stale."""

from __future__ import annotations

import numpy as np
import pytest

from cache import StudyCache, build_cache, cache_key
from config import PreprocessConfig
from data import load_tables
from dataset import StudyDataset

CFG = PreprocessConfig(
    n_slices=4, size=32, max_series=3,
    plane_prefs=("sagittal_fluid", "coronal_fluid", "axial_fluid"),
)


@pytest.fixture(scope="module")
def tables(fixture_dir):
    return load_tables(fixture_dir, split="train")


def test_cached_tensors_are_identical_to_uncached(tables, tmp_path):
    cache = build_cache(tables.study_uids, tables.series, CFG, tmp_path, num_workers=1,
                        verbose=False)
    cold = StudyDataset(tables.study_uids, tables.series, CFG)
    warm = StudyDataset(tables.study_uids, tables.series, CFG, cache=cache)
    for i in range(len(cold)):
        a, b = cold[i], warm[i]
        assert np.array_equal(a["x"].numpy(), b["x"].numpy()), i
        assert np.array_equal(a["series_mask"].numpy(), b["series_mask"].numpy())
        assert np.array_equal(a["plane_ids"].numpy(), b["plane_ids"].numpy())


def test_key_changes_with_every_pixel_affecting_setting():
    """A config change must never silently serve stale tensors — that bug would
    be catastrophic and invisible in the metrics."""
    base = cache_key(CFG)
    from dataclasses import replace

    assert cache_key(replace(CFG, n_slices=8)) != base
    assert cache_key(replace(CFG, size=64)) != base
    assert cache_key(replace(CFG, max_series=1)) != base
    assert cache_key(replace(CFG, plane_prefs=("axial_fluid",))) != base
    assert cache_key(replace(CFG, clip_percentiles=(2.0, 98.0))) != base
    assert cache_key(CFG) == base  # stable across calls


def test_different_configs_do_not_collide(tables, tmp_path):
    small = PreprocessConfig(n_slices=2, size=16, max_series=1)
    build_cache(tables.study_uids[:2], tables.series, CFG, tmp_path, num_workers=1, verbose=False)
    build_cache(tables.study_uids[:2], tables.series, small, tmp_path, num_workers=1, verbose=False)

    a = StudyCache(tmp_path, CFG).load(tables.study_uids[0])
    b = StudyCache(tmp_path, small).load(tables.study_uids[0])
    assert a.pixels.shape == (3, 4, 32, 32)
    assert b.pixels.shape == (1, 2, 16, 16)


def test_miss_returns_none_and_read_through_populates(tables, tmp_path):
    cache = StudyCache(tmp_path, CFG)
    uid = tables.study_uids[0]
    assert cache.load(uid) is None
    got = cache.get(uid, tables.series)
    assert got.pixels.shape == (3, 4, 32, 32)
    assert cache.load(uid) is not None, "read-through should have stored it"


def test_corrupt_entry_falls_back_instead_of_crashing(tables, tmp_path):
    cache = StudyCache(tmp_path, CFG)
    uid = tables.study_uids[0]
    cache.get(uid, tables.series)
    cache._path(uid).write_bytes(b"not an npz")
    assert cache.load(uid) is None  # degrades to a miss, not an exception
    assert cache.get(uid, tables.series).pixels.shape == (3, 4, 32, 32)


def test_disabled_cache_is_a_noop(tables):
    cache = StudyCache(None, CFG)
    assert not cache.enabled
    assert cache.load(tables.study_uids[0]) is None


def test_parallel_build_matches_serial(tables, tmp_path):
    serial = build_cache(tables.study_uids[:4], tables.series, CFG,
                         tmp_path / "s", num_workers=1, verbose=False)
    parallel = build_cache(tables.study_uids[:4], tables.series, CFG,
                           tmp_path / "p", num_workers=2, verbose=False)
    for uid in tables.study_uids[:4]:
        assert np.array_equal(serial.load(uid).pixels, parallel.load(uid).pixels), uid
