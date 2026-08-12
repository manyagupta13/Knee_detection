"""Frozen-feature extraction, caching, and the head that consumes it."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config import MULTI_PLANE_PREFS, N_TARGETS, PreprocessConfig
from data import load_tables
from features import FeatureStore, extract_features, feature_key
from model import FeatureHead

CFG = PreprocessConfig(n_slices=4, size=64, max_series=3,
                       plane_prefs=MULTI_PLANE_PREFS)
BACKBONE = "efficientnet_b0"  # stand-in: no weight download in tests


@pytest.fixture(scope="module")
def tables(fixture_dir):
    return load_tables(fixture_dir, split="train")


@pytest.fixture(scope="module")
def store(tables, tmp_path_factory):
    out = tmp_path_factory.mktemp("feats")
    return extract_features(tables.study_uids, tables.series, CFG, BACKBONE, out,
                            pretrained=False, verbose=False)


def test_features_have_series_slice_dim_layout(store, tables):
    feats, mask, planes = store.load(tables.study_uids[0])
    assert feats.ndim == 3                      # (S, T, D)
    assert feats.shape[:2] == (CFG.max_series, CFG.n_slices)
    assert mask.shape == (CFG.max_series,)
    assert len(planes) == CFG.max_series
    assert planes[:3] == ["sagittal", "coronal", "axial"]


def test_every_fixture_study_is_cached(store, tables):
    for uid in tables.study_uids:
        assert store.has(uid), uid


def test_key_separates_backbone_and_preprocessing():
    """Embeddings depend on the backbone AND the pixels, so both must key the
    cache or a config change silently serves the wrong vectors."""
    from dataclasses import replace

    base = feature_key(CFG, BACKBONE, "2.5d")
    assert feature_key(CFG, "resnet18", "2.5d") != base
    assert feature_key(CFG, BACKBONE, "grey") != base
    assert feature_key(replace(CFG, n_slices=8), BACKBONE, "2.5d") != base
    assert feature_key(replace(CFG, canonicalize=False), BACKBONE, "2.5d") != base


def test_extraction_is_idempotent(store, tables, tmp_path):
    before = store.load(tables.study_uids[0])[0]
    again = extract_features(tables.study_uids[:2], tables.series, CFG, BACKBONE,
                             store.root.parent, pretrained=False, verbose=False)
    assert np.allclose(before, again.load(tables.study_uids[0])[0])


def test_head_shapes_and_series_masking():
    head = FeatureHead(feat_dim=384).eval()
    feats = torch.rand(2, 3, 8, 384)
    sm = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    pid = torch.tensor([[1, 2, 3], [1, 2, 0]])
    with torch.no_grad():
        logits, attn = head(feats, series_mask=sm, plane_ids=pid, return_attention=True)
    assert logits.shape == (2, N_TARGETS)
    assert attn["series"].shape == (2, 3)
    assert float(attn["series"][1, 2]) == pytest.approx(0.0, abs=1e-6)


def test_head_accepts_single_series_3d_input():
    head = FeatureHead(feat_dim=64).eval()
    with torch.no_grad():
        assert head(torch.rand(2, 5, 64)).shape == (2, N_TARGETS)


def test_head_rejects_wrong_rank():
    head = FeatureHead(feat_dim=64)
    with pytest.raises(ValueError):
        head(torch.rand(2, 64))


def test_feature_augmentation_only_applies_in_training():
    """Eval must be deterministic; training must actually perturb."""
    head = FeatureHead(feat_dim=64, slice_dropout=0.5, feature_noise=0.5)
    feats = torch.rand(2, 2, 6, 64)
    head.eval()
    with torch.no_grad():
        assert torch.allclose(head(feats), head(feats))
    head.train()
    torch.manual_seed(0)
    a = head(feats)
    b = head(feats)
    assert not torch.allclose(a, b)


def test_slice_dropout_never_empties_a_series():
    head = FeatureHead(feat_dim=32, slice_dropout=1.0, feature_noise=0.0).train()
    out = head(torch.rand(2, 2, 5, 32))
    assert torch.isfinite(out).all(), "dropping every slice would produce NaN"


def test_head_can_overfit_a_batch():
    torch.manual_seed(0)
    from model import masked_bce_with_logits

    head = FeatureHead(feat_dim=32, dropout=0.0, slice_dropout=0.0, feature_noise=0.0)
    feats = torch.rand(4, 2, 6, 32)
    y = (torch.rand(4, N_TARGETS) > 0.5).float()
    w = torch.ones_like(y)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2)
    first = last = None
    for step in range(30):
        opt.zero_grad()
        loss = masked_bce_with_logits(head(feats), y, w)
        loss.backward()
        opt.step()
        first = float(loss.detach()) if step == 0 else first
        last = float(loss.detach())
    assert last < first * 0.8
