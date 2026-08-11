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


def test_reports_come_from_the_train_csv_report_column(fixture_dir):
    """Reports are a column of train.csv, not a separate file. When this broke,
    the pseudo-labeler silently no-opped and training fell back to gold-only."""
    tables = load_tables(fixture_dir, split="train")
    assert tables.reports is not None, "no reports loaded -> pseudo-labeler is a no-op"
    assert "report_text" in tables.reports.columns
    assert tables.reports["report_text"].notna().all()
    # ...and the Report column must not leak into the label table.
    assert "Report" not in tables.gold.columns


def test_pseudo_labels_widen_supervision_beyond_gold(fixture_dir):
    """The entire point of Phase 0: text turns a handful of gold studies into
    meaningfully more supervised rows."""
    from textlabel import label_reports

    tables = load_tables(fixture_dir, split="train")
    pseudo, _ = label_reports(tables.reports)
    _, gold_mask = build_targets(tables.study_uids, tables.gold, None)
    _, union_mask = build_targets(tables.study_uids, tables.gold, pseudo)

    gold_studies = int(gold_mask.any(axis=1).sum())
    union_studies = int(union_mask.any(axis=1).sum())
    assert union_studies > gold_studies, (gold_studies, union_studies)


def test_pixels_actually_load_from_the_series_directory(fixture_dir):
    """Guards the train_images/ vs train_series/ path bug: a wrong image root
    yields all-zero inputs and a model that predicts a constant."""
    from config import PreprocessConfig
    from data import StudyDataset

    tables = load_tables(fixture_dir, split="train")
    ds = StudyDataset(tables.study_uids[:4], tables.series, PreprocessConfig(n_slices=2, size=32))
    for i in range(len(ds)):
        x = ds[i]["x"]
        assert float(x.max()) > 0.0, "zero-filled input — image root is wrong"


def test_columns_override_is_honoured(fixture_dir, tmp_path):
    """train() must take the header from its argument, not a stale import-time
    binding of config.TARGET_COLUMNS."""
    from train import train

    reordered = list(reversed(TARGET_COLUMNS))
    metrics = train(
        data_dir=fixture_dir,
        out_dir=tmp_path / "cols",
        epochs=1,
        n_slices=2,
        size=32,
        pretrained=False,
        device="cpu",
        columns=reordered,
    )
    assert [r["column"] for r in metrics["per_column_auc"]] == reordered
    saved = RunConfig.load(tmp_path / "cols" / "run_config.json")
    assert list(saved.target_columns) == reordered


def test_confidence_weights_downweight_untrusted_columns():
    """Measured labeler precision becomes a per-column loss weight. Gold always
    weighs 1.0; a coin-flip column (precision 0.5) weighs 0."""
    from config import confidence_weight

    assert confidence_weight(1.0) == 1.0
    assert confidence_weight(0.5) == 0.0
    assert confidence_weight(0.0) == 0.0
    assert 0.4 < confidence_weight(0.75) < 0.6

    uids = ["a", "b"]
    pseudo = pd.DataFrame(
        {"Effusion": [1.0, 0.0], "Contusion": [1.0, 1.0]}, index=uids
    )
    gold = pd.DataFrame({ID_COLUMN: ["a"], **{c: [np.nan] for c in TARGET_COLUMNS}})
    gold.loc[0, "Contusion"] = 1.0

    _, w = build_targets(
        uids, gold, pseudo, pseudo_weights={"Effusion": 0.5, "Contusion": 0.1}
    )
    assert w.loc["a", "Contusion"] == 1.0, "gold must override the pseudo weight"
    assert w.loc["b", "Contusion"] == 0.1
    assert w.loc["a", "Effusion"] == 0.5
    assert w.loc["a", "ACL"] == 0.0, "unlabeled cells carry no weight"


def test_weighted_loss_ignores_zero_weight_cells():
    """A weight matrix and a binary mask are the same object to the loss."""
    import torch

    from model import masked_bce_with_logits

    logits = torch.zeros(1, 3, requires_grad=True)
    targets = torch.tensor([[1.0, 1.0, 1.0]])
    weights = torch.tensor([[1.0, 0.0, 0.5]])
    loss = masked_bce_with_logits(logits, targets, weights)
    loss.backward()
    g = logits.grad[0]
    assert g[1] == 0.0, "zero-weight cell must not contribute a gradient"
    assert abs(float(g[2])) < abs(float(g[0])), "half-weight cell contributes less"


def test_preprocess_config_round_trips(tmp_path):
    cfg = PreprocessConfig(n_slices=8, size=128, max_series=3)
    path = tmp_path / "pre.json"
    cfg.save(path)
    assert PreprocessConfig.load(path) == cfg
