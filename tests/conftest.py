from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.make_fixture import make_fixture  # noqa: E402

# Tiny but structurally identical to the production settings.
FIXTURE_SLICES = 4
FIXTURE_SIZE = 64


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    """~12 synthetic studies of real DICOMs in the competition schema."""
    out = tmp_path_factory.mktemp("knee_fixture")
    return make_fixture(out, n_studies=12, seed=0)


@pytest.fixture(scope="session")
def trained_artifacts(fixture_dir, tmp_path_factory) -> Path:
    """Train one tiny fold on the fixture; returns the artifacts dir."""
    from train import train

    out = tmp_path_factory.mktemp("knee_artifacts")
    train(
        data_dir=fixture_dir,
        out_dir=out,
        epochs=1,
        batch_size=2,
        n_slices=FIXTURE_SLICES,
        size=FIXTURE_SIZE,
        max_series=1,
        pretrained=False,  # tests must run with no internet
        device="cpu",
    )
    return out
