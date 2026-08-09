"""Model shapes, attention pooling, and the masked-BCE contract (invariant #2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config import N_TARGETS
from model import GatedAttentionPool, build_model, masked_bce_with_logits


@pytest.fixture(scope="module")
def model():
    return build_model(pretrained=False).eval()


def test_forward_shape(model):
    x = torch.rand(2, 4, 3, 64, 64)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, N_TARGETS)
    assert torch.isfinite(logits).all()


def test_rejects_wrong_rank(model):
    with pytest.raises(ValueError):
        model(torch.rand(2, 3, 64, 64))


def test_attention_is_a_distribution_over_slices(model):
    x = torch.rand(2, 5, 3, 64, 64)
    with torch.no_grad():
        _, attn = model(x, return_attention=True)
    assert attn.shape == (2, 5)
    assert torch.allclose(attn.sum(1), torch.ones(2), atol=1e-5)
    assert (attn >= 0).all()


def test_attention_pool_respects_mask():
    pool = GatedAttentionPool(8)
    x = torch.rand(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    with torch.no_grad():
        pooled, attn = pool(x, mask=mask)
    assert pytest.approx(float(attn[0, 2:].sum()), abs=1e-6) == 0.0
    assert pooled.shape == (1, 8)


def test_permuting_slices_leaves_prediction_unchanged(model):
    """Attention pooling is order-invariant; slice order matters via content,
    not position. Guards against accidentally baking in a positional prior."""
    x = torch.rand(1, 6, 3, 64, 64)
    with torch.no_grad():
        a = model(x)
        b = model(x[:, torch.randperm(6)])
    assert torch.allclose(a, b, atol=1e-4)


def test_masked_bce_ignores_unlabeled_columns():
    logits = torch.zeros(2, N_TARGETS, requires_grad=True)
    targets = torch.full((2, N_TARGETS), float("nan"))
    mask = torch.zeros(2, N_TARGETS)
    targets[0, 0], mask[0, 0] = 1.0, 1.0

    loss = masked_bce_with_logits(logits, targets, mask)
    assert torch.isfinite(loss)
    loss.backward()

    grad = logits.grad
    assert grad[0, 0] != 0
    assert torch.count_nonzero(grad) == 1, "gradient leaked into unlabeled cells"


def test_masked_bce_equals_plain_bce_when_all_labeled():
    torch.manual_seed(0)
    logits = torch.randn(4, N_TARGETS)
    targets = (torch.rand(4, N_TARGETS) > 0.5).float()
    mask = torch.ones_like(targets)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(masked_bce_with_logits(logits, targets, mask), expected)


def test_masked_bce_is_nan_safe():
    logits = torch.zeros(1, N_TARGETS, requires_grad=True)
    targets = torch.full((1, N_TARGETS), float("nan"))
    mask = torch.zeros(1, N_TARGETS)
    loss = masked_bce_with_logits(logits, targets, mask)
    assert torch.isfinite(loss) and float(loss.detach()) == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_model_can_overfit_a_single_batch():
    """Sanity: the head + pool actually learn something."""
    torch.manual_seed(0)
    m = build_model(pretrained=False)
    x = torch.rand(2, 3, 3, 32, 32)
    y = torch.tensor([[1.0] * N_TARGETS, [0.0] * N_TARGETS])
    mask = torch.ones_like(y)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = last = None
    for step in range(8):
        opt.zero_grad()
        loss = masked_bce_with_logits(m(x), y, mask)
        loss.backward()
        opt.step()
        first = float(loss.detach()) if step == 0 else first
        last = float(loss.detach())
    assert last < first
