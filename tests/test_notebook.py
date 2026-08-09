"""Execute the real submission notebook offline and validate submission.csv.

This is the Phase-0 definition of done: the notebook that runs on Kaggle is the
notebook that runs here, against the synthetic fixture.
"""

from __future__ import annotations

import os
import shutil

import nbformat
import pandas as pd
import pytest
from nbclient import NotebookClient

from config import ID_COLUMN


@pytest.fixture(scope="module")
def submission(repo_root, fixture_dir, trained_artifacts, tmp_path_factory):
    work = tmp_path_factory.mktemp("nb_run")
    nb_path = repo_root / "notebooks" / "infer_notebook.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    env = dict(os.environ)
    env.update(
        {
            "KNEE_CODE_DIR": str(repo_root),
            "KNEE_WEIGHTS_DIR": str(trained_artifacts),
            "KNEE_DATA_DIR": str(fixture_dir),
            "KNEE_SPLIT": "test",
            "KNEE_OUT": str(work / "submission.csv"),
            "KNEE_NUM_WORKERS": "0",
            "HF_HUB_OFFLINE": "1",  # prove it never reaches the network
            "TRANSFORMERS_OFFLINE": "1",
            "no_proxy": "*",
        }
    )
    old = dict(os.environ)
    os.environ.update(env)
    try:
        client = NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(work)}})
        client.execute()
    finally:
        os.environ.clear()
        os.environ.update(old)

    out = work / "submission.csv"
    assert out.exists(), "notebook did not write submission.csv"
    return {"path": out, "nb": nb, "fixture": fixture_dir}


def test_submission_schema(submission):
    sub = pd.read_csv(submission["path"], dtype={ID_COLUMN: str})
    sample = pd.read_csv(submission["fixture"] / "sample_submission.csv", dtype={ID_COLUMN: str})

    assert list(sub.columns) == list(sample.columns), "column order must match exactly"
    assert len(sub.columns) == 13  # id + 12 findings


def test_every_test_study_is_present_exactly_once(submission):
    sub = pd.read_csv(submission["path"], dtype={ID_COLUMN: str})
    sample = pd.read_csv(submission["fixture"] / "sample_submission.csv", dtype={ID_COLUMN: str})
    assert sub[ID_COLUMN].is_unique
    assert set(sub[ID_COLUMN]) == set(sample[ID_COLUMN])
    assert len(sub) == len(sample)


def test_probabilities_are_valid(submission):
    sub = pd.read_csv(submission["path"], dtype={ID_COLUMN: str})
    probs = sub.drop(columns=[ID_COLUMN])
    assert not probs.isna().to_numpy().any()
    assert probs.to_numpy().min() >= 0.0
    assert probs.to_numpy().max() <= 1.0
    # A constant-0.5 sheet means every study silently failed to decode.
    assert probs.to_numpy().std() > 0


def test_notebook_prints_wall_clock(submission):
    text = "\n".join(
        out.get("text", "")
        for cell in submission["nb"].cells
        for out in cell.get("outputs", [])
    )
    assert "TOTAL WALL CLOCK" in text
