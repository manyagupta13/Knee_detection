"""Preprocessing invariants. Most of these guard design invariant #1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydicom
import pytest

from config import DEFAULT_PLANE_PREFS
from data import load_tables
from preprocess import (
    _list_dicoms,
    classify_plane_from_iop,
    classify_series,
    classify_series_row,
    preprocess_series,
    preprocess_study,
    select_series,
    to_model_input,
)


@pytest.fixture(scope="module")
def tables(fixture_dir):
    return load_tables(fixture_dir, split="train")


def test_returns_stacked_form_even_with_max_series_1(tables):
    uid = tables.study_uids[0]
    out = preprocess_study(uid, tables.series, DEFAULT_PLANE_PREFS, 4, 64, 1)
    assert out.pixels.ndim == 4, "must keep the multi-series axis at max_series=1"
    assert out.pixels.shape == (1, 4, 64, 64)
    assert out.pixels.dtype == np.uint8
    assert out.series_mask.shape == (1,)
    assert len(out.series_uids) == 1


@pytest.mark.parametrize("max_series", [1, 2, 3])
def test_shape_scales_with_max_series(tables, max_series):
    uid = tables.study_uids[0]
    out = preprocess_study(uid, tables.series, DEFAULT_PLANE_PREFS, 5, 32, max_series)
    assert out.pixels.shape == (max_series, 5, 32, 32)
    assert out.series_mask.shape == (max_series,)
    assert out.series_mask[0], "the canonical series must always be filled first"


def test_every_fixture_study_yields_a_series(tables):
    for uid in tables.study_uids:
        out = preprocess_study(uid, tables.series, DEFAULT_PLANE_PREFS, 4, 32, 1)
        assert out.series_mask[0], f"no series selected for {uid}"
        assert out.pixels[0].max() > 0


def test_missing_study_returns_zeros_and_false_mask(tables):
    out = preprocess_study("no-such-study", tables.series, DEFAULT_PLANE_PREFS, 4, 32, 1)
    assert out.pixels.shape == (1, 4, 32, 32)
    assert not out.series_mask.any()
    assert out.pixels.sum() == 0
    assert out.series_uids == [None]


def test_deterministic(tables):
    uid = tables.study_uids[1]
    a = preprocess_study(uid, tables.series, DEFAULT_PLANE_PREFS, 4, 32, 1)
    b = preprocess_study(uid, tables.series, DEFAULT_PLANE_PREFS, 4, 32, 1)
    assert np.array_equal(a.pixels, b.pixels)
    assert a.series_uids == b.series_uids


def test_prefers_sagittal_fluid_then_falls_back_to_sagittal_any(tables):
    """Every study must land on a sagittal series; the ones without a
    fluid-sensitive sagittal exercise the fallback."""
    used_fallback = 0
    for uid in tables.study_uids:
        study = tables.series[tables.series["StudyInstanceUID"] == uid]
        chosen = select_series(study, DEFAULT_PLANE_PREFS, max_series=1)
        assert chosen, uid
        plane, contrast = classify_series_row(chosen[0])
        assert plane == "sagittal", (uid, chosen[0]["SeriesDescription"])
        if contrast != "fluid":
            used_fallback += 1
    assert used_fallback > 0, "fixture should contain studies without sagittal-fluid"


def test_slices_sorted_by_position_not_instance_number(tables):
    """The fixture shuffles InstanceNumber on purpose. Sorting must follow
    ImagePositionPatient projected on the slice normal."""
    row = tables.series.iloc[0]
    files = _list_dicoms(row["series_dir"])
    headers = [pydicom.dcmread(str(f), stop_before_pixels=True) for f in files]

    instance_order = [
        int(h.InstanceNumber)
        for h in sorted(headers, key=lambda h: int(h.InstanceNumber))
    ]
    geometric = sorted(
        headers,
        key=lambda h: float(
            np.dot(
                np.asarray([float(v) for v in h.ImagePositionPatient]),
                np.cross(
                    [float(v) for v in h.ImageOrientationPatient[:3]],
                    [float(v) for v in h.ImageOrientationPatient[3:]],
                ),
            )
        ),
    )
    geometric_instance_order = [int(h.InstanceNumber) for h in geometric]
    assert geometric_instance_order != instance_order, "fixture is not exercising the trap"

    # A short series read at full length must come back in geometric order:
    # the phantom's bright blob sits at the centre, so intensity is unimodal
    # along the true axis and multi-modal if we sorted wrong.
    vol = preprocess_series(row["series_dir"], n_slices=len(files), size=32)
    means = vol.reshape(len(files), -1).mean(axis=1)
    assert np.argmax(means) not in (0, len(files) - 1)


def test_intensity_is_clipped_per_series_to_uint8(tables):
    row = tables.series.iloc[0]
    vol = preprocess_series(row["series_dir"], n_slices=6, size=32)
    assert vol.dtype == np.uint8
    assert vol.min() >= 0 and vol.max() <= 255
    # Per-series (not per-slice) normalization: individual slices must be free
    # to occupy only part of the range.
    per_slice_max = vol.reshape(vol.shape[0], -1).max(axis=1)
    assert per_slice_max.min() < 255


def test_short_series_is_padded_by_repetition(tables):
    row = tables.series.iloc[0]
    n_files = len(_list_dicoms(row["series_dir"]))
    vol = preprocess_series(row["series_dir"], n_slices=n_files + 10, size=16)
    assert vol.shape == (n_files + 10, 16, 16)


def test_classify_series_multilingual():
    assert classify_series("SAG PD FS") == ("sagittal", "fluid")
    assert classify_series("Sagital DP SPAIR") == ("sagittal", "fluid")
    assert classify_series("KORONAL TIRM") == ("coronal", "fluid")
    assert classify_series("TRA T2 FS") == ("axial", "fluid")
    assert classify_series("SAG T1 TSE") == ("sagittal", "other")
    assert classify_series("localizer")[0] == "unknown"


def test_classify_plane_from_iop():
    assert classify_plane_from_iop([0, 1, 0, 0, 0, -1]) == "sagittal"
    assert classify_plane_from_iop([1, 0, 0, 0, 0, -1]) == "coronal"
    assert classify_plane_from_iop([1, 0, 0, 0, 1, 0]) == "axial"
    assert classify_plane_from_iop(None) == "unknown"


def test_iop_fallback_when_description_is_useless(tables):
    """A series whose description says nothing must still be planed by IOP."""
    row = tables.series.iloc[0].to_dict()
    row["SeriesDescription"] = "series 3"
    df = pd.DataFrame([row])
    chosen = select_series(df, ("sagittal_fluid", "sagittal_any"), max_series=1)
    expected = classify_plane_from_iop(
        [float(v) for v in pydicom.dcmread(str(_list_dicoms(row["series_dir"])[0])).ImageOrientationPatient]
    )
    assert bool(chosen) == (expected == "sagittal")


def test_to_model_input_shape_and_range(tables):
    out = preprocess_study(
        tables.study_uids[0], tables.series, DEFAULT_PLANE_PREFS, 4, 32, 1
    )
    x = to_model_input(out.pixels[0])
    assert x.shape == (4, 3, 32, 32)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_train_and_test_use_the_identical_function(fixture_dir):
    """The same call, the same shapes, on both splits."""
    train_tables = load_tables(fixture_dir, split="train")
    test_tables = load_tables(fixture_dir, split="test")
    shared = [u for u in test_tables.study_uids if u in train_tables.study_uids]
    assert shared
    for uid in shared[:3]:
        a = preprocess_study(uid, train_tables.series, DEFAULT_PLANE_PREFS, 4, 32, 1)
        b = preprocess_study(uid, test_tables.series, DEFAULT_PLANE_PREFS, 4, 32, 1)
        assert np.array_equal(a.pixels, b.pixels)
