"""Augmentation must vary intensity without corrupting laterality or geometry."""

from __future__ import annotations

import numpy as np
import pytest

from augment import AugmentConfig, augment_series, augment_study


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    # asymmetric so a horizontal flip would be detectable
    v = rng.integers(0, 200, size=(6, 32, 32), dtype=np.uint16).astype(np.uint8)
    v[:, :, :8] = 240
    return v


def test_shape_and_dtype_preserved(volume):
    out = augment_series(volume, np.random.default_rng(1), AugmentConfig())
    assert out.shape == volume.shape
    assert out.dtype == np.uint8


def test_disabled_is_identity(volume):
    out = augment_series(volume, np.random.default_rng(1), AugmentConfig(enabled=False))
    assert np.array_equal(out, volume)


def test_actually_changes_something(volume):
    out = augment_series(volume, np.random.default_rng(3), AugmentConfig(p=1.0))
    assert not np.array_equal(out, volume)


def test_never_horizontally_flips(volume):
    """Medial and lateral are separate scored columns; mirroring the knee swaps
    them and silently corrupts four labels. No augmentation may produce the
    mirror image."""
    flipped = volume[:, :, ::-1]
    for seed in range(25):
        out = augment_series(volume, np.random.default_rng(seed), AugmentConfig(p=1.0))
        # bright band must stay on the left half
        assert out[:, :, :16].mean() > out[:, :, 16:].mean(), seed
        assert not np.array_equal(out, flipped)


def test_geometry_is_shared_across_slices():
    """A volume whose slices are each warped differently is not a volume."""
    base = np.zeros((4, 32, 32), dtype=np.uint8)
    base[:, 8:24, 8:24] = 255  # identical square on every slice
    out = augment_series(
        base,
        np.random.default_rng(5),
        AugmentConfig(p=1.0, noise_std=0.0, bias_field=0.0, slice_dropout=0.0,
                      gamma=(1.0, 1.0), brightness=0.0, contrast=0.0),
    )
    for i in range(1, out.shape[0]):
        assert np.array_equal(out[0], out[i]), "slices warped inconsistently"


def test_values_stay_in_range(volume):
    for seed in range(10):
        out = augment_series(volume, np.random.default_rng(seed), AugmentConfig(p=1.0))
        assert out.min() >= 0 and out.max() <= 255


def test_augment_study_handles_series_axis(volume):
    study = np.stack([volume, volume, volume])
    out = augment_study(study, np.random.default_rng(2), AugmentConfig(p=1.0))
    assert out.shape == study.shape
    assert not np.array_equal(out[0], out[1]), "series should vary independently"


def test_slice_dropout_never_blanks_everything(volume):
    out = augment_series(
        volume, np.random.default_rng(7), AugmentConfig(p=1.0, slice_dropout=1.0)
    )
    assert out.reshape(out.shape[0], -1).max(axis=1).max() > 0
