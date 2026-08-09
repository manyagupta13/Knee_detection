"""EfficientNet-B0 shared over slices -> gated attention pool -> 12 sigmoid heads.

Input is (B, n_slices, 3, H, W). Findings are focal (a fracture lives on two
slices), so mean-pooling over slices would drown the signal - hence ABMIL-style
gated attention over the slice axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_TARGETS


class GatedAttentionPool(nn.Module):
    """Ilse et al. (ABMIL) gated attention over a variable-length token axis."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.v = nn.Linear(in_dim, hidden_dim)
        self.u = nn.Linear(in_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D) -> pooled (B, D), attention (B, T)."""
        scores = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bt,btd->bd", attn, x)
        return pooled, attn


class KneeModel(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        n_targets: int = N_TARGETS,
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        import timm

        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.slice_pool = GatedAttentionPool(feat_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, n_targets)
        # TODO(phase-3): a second GatedAttentionPool over the series axis, plus
        # plane/contrast embeddings. Phase 0 consumes one series.

    def forward(
        self,
        x: torch.Tensor,
        slice_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """x: (B, n_slices, 3, H, W) -> logits (B, n_targets)."""
        if x.ndim != 5:
            raise ValueError(f"expected (B, S, 3, H, W), got {tuple(x.shape)}")
        b, s = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1))  # (B*S, D)
        feats = feats.view(b, s, -1)
        pooled, attn = self.slice_pool(feats, mask=slice_mask)
        logits = self.head(self.dropout(pooled))
        if return_attention:
            return logits, attn
        return logits


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """BCE that backprops only through labeled (study, column) cells.

    Design invariant #2. This is what lets gold, high-precision pseudo-labels and
    (later) low-confidence soft labels coexist in a single run. ``targets`` may
    contain NaN wherever ``mask`` is False - those entries are neutralized before
    the loss so no NaN can reach the gradient.
    """
    mask = mask.to(dtype=logits.dtype)
    safe_targets = torch.nan_to_num(targets, nan=0.0).to(dtype=logits.dtype) * mask
    per_cell = F.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none"
    )
    per_cell = per_cell * mask
    if reduction == "none":
        return per_cell
    denom = mask.sum().clamp(min=1.0)
    total = per_cell.sum()
    return total / denom if reduction == "mean" else total


def build_model(backbone: str = "efficientnet_b0", pretrained: bool = True) -> KneeModel:
    return KneeModel(backbone=backbone, pretrained=pretrained)
