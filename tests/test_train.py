"""End-to-end training on the fixture + the label-merge rules."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from config import ID_COLUMN, PreprocessConfig, RunConfig, TARGET_COLUMNS
from data import build_targets, load_tables
from model import build_model


def test_build_targets_gold_beats_pseudo():
    uids = ["a", "b"]
    pseudo = pd.DataFrame(
        {"Effusion": [1.0, 0.0], "Fracture": [np.nan, 1.0]}, index=uids
    )
    gold = pd.DataFrame(
        {ID_COLUMN: ["a"], **{c: [np.nan] for c in TARGET_COLUMNS}}
    )
    gold.loc[0, "Effusion"] = 0.0  # gold disagrees with the pseudo-label

    targets, mask = build_targets(uids, gold, pseudo)
    assert targets.loc["a", "Effusion"] == 0.0, "gold must win"
    assert targets.loc["b", "Effusion"] == 0.0
    assert targets.loc["b", "Fracture"] == 1.0
    assert np.isnan(targets.loc["a", "Fracture"])
    assert not mask.loc["a", "Fracture"]
    assert mask.loc["b", "Fracture"]
    assert list(targets.columns) == list(TARGET_COLUMNS)


def test_build_targets_union_is_wider_than_gold_alone(fixture_dir):
    from textlabel import label_reports

    tables = load_tables(fixture_dir, split="train")
    pseudo, _ = label_reports(tables.reports)
    gold_only, gold_mask = build_targets(tables.study_uids, tables.gold, None)
    union, union_mask = build_targets(tables.study_uids, tables.gold, pseudo)
    assert union_mask.to_numpy().sum() > gold_mask.to_numpy().sum()


def test_training_writes_weights_and_exact_preprocess_config(trained_artifacts):
    assert (trained_artifacts / "model.pt").exists()
    cfg = RunConfig.load(trained_artifacts / "run_config.json")
    assert cfg.preprocess.n_slices == 4
    assert cfg.preprocess.size == 64
    assert cfg.preprocess.max_series == 1
    assert list(cfg.target_columns) == list(TARGET_COLUMNS)

    model = build_model(backbone=cfg.backbone, pretrained=False)
    model.load_state_dict(torch.load(trained_artifacts / "model.pt", map_location="cpu"))


def test_metrics_are_gold_only_and_report_every_column(trained_artifacts):
    metrics = json.loads((trained_artifacts / "metrics.json").read_text())
    assert metrics["n_val_gold_studies"] > 0
    assert len(metrics["per_column_auc"]) == len(TARGET_COLUMNS)
    assert metrics["labeled_cells"] > 0
    assert np.isfinite(metrics["history"][0]["train_loss"])


def test_val_studies_are_gold_and_excluded_from_training(fixture_dir, tmp_path):
    """Validation must be gold-only, and never seen in training."""
    from train import train

    metrics = train(
        data_dir=fixture_dir,
        out_dir=tmp_path / "art",
        epochs=1,
        batch_size=2,
        n_slices=2,
        size=32,
        pretrained=False,
        device="cpu",
    )
    tables = load_tables(fixture_dir, split="train")
    n_gold = len(tables.gold)
    assert 0 < metrics["n_val_gold_studies"] <= n_gold
    assert metrics["n_train_studies"] <= len(tables.study_uids) - metrics["n_val_gold_studies"]


def test_preprocess_config_round_trips(tmp_path):
    cfg = PreprocessConfig(n_slices=8, size=128, max_series=3)
    path = tmp_path / "pre.json"
    cfg.save(path)
    assert PreprocessConfig.load(path) == cfg
