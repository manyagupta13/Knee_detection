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


# Plane ids for the series embedding. 0 is reserved for "unknown/missing".
PLANE_IDS: dict[str, int] = {"unknown": 0, "sagittal": 1, "coronal": 2, "axial": 3}
N_PLANES = 4


class KneeModel(nn.Module):
    """Slice encoder -> attention over slices -> attention over series -> 12 heads.

    Findings live in different planes: MCL and the femorotibial compartments are
    coronal calls, patellofemoral cartilage is best seen axially, the cruciates
    and meniscal horns sagittally. A sagittal-only model is structurally unable
    to see a third of what it is scored on, so series are pooled with their own
    gated attention and tagged with a plane embedding.
    """

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
        self.feat_dim = feat_dim
        self.slice_pool = GatedAttentionPool(feat_dim)
        self.series_pool = GatedAttentionPool(feat_dim)
        self.plane_emb = nn.Embedding(N_PLANES, feat_dim)
        nn.init.zeros_(self.plane_emb.weight)  # start as a no-op
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, n_targets)

    def set_grad_checkpointing(self, enable: bool = True) -> bool:
        """Trade ~30% speed for a large memory saving.

        Activation memory scales with B*S*T images per step - at 3 series x 24
        slices that is 72 images per study, which OOMs a 16GB card long before
        the batch size is interesting. Checkpointing is what makes more slices
        affordable, and slice count matters: sampling 12 of 40 slices can skip a
        focal fracture entirely.
        """
        try:
            self.backbone.set_grad_checkpointing(enable)
            return True
        except (AttributeError, NotImplementedError):
            return False

    def forward(
        self,
        x: torch.Tensor,
        series_mask: torch.Tensor | None = None,
        plane_ids: torch.Tensor | None = None,
        slice_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """(B, S, T, 3, H, W) -> logits (B, n_targets).

        A 5-D input (B, T, 3, H, W) is accepted as the single-series case and
        promoted to S=1, so the Phase-0 call signature keeps working.
        """
        if x.ndim == 5:
            x = x.unsqueeze(1)
        if x.ndim != 6:
            raise ValueError(f"expected (B, S, T, 3, H, W), got {tuple(x.shape)}")

        b, s, t = x.shape[:3]
        feats = self.backbone(x.flatten(0, 2))            # (B*S*T, D)
        feats = feats.view(b * s, t, -1)
        series_feat, slice_attn = self.slice_pool(feats, mask=slice_mask)
        series_feat = series_feat.view(b, s, -1)          # (B, S, D)

        if plane_ids is not None:
            series_feat = series_feat + self.plane_emb(plane_ids.long())

        pooled, series_attn = self.series_pool(series_feat, mask=series_mask)
        logits = self.head(self.dropout(pooled))
        if return_attention:
            return logits, {
                "slice": slice_attn.view(b, s, t),
                "series": series_attn,
            }
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
