"""Training augmentation for uint8 slice stacks.

Scanner variation is the stated domain shift, so intensity augmentation matters
more than geometry here. Applied per SERIES with shared parameters across its
slices, because a volume whose slices were each warped differently is not a
volume any more.

NO HORIZONTAL FLIP. Medial and lateral are separate scored columns; mirroring a
knee swaps them and silently corrupts four of the twelve labels. It only becomes
safe after laterality canonicalization via ImageOrientationPatient.
TODO(phase-2): canonicalize orientation, then enable h-flip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AugmentConfig:
    enabled: bool = True
    p: float = 0.8                     # chance of applying the whole pipeline
    gamma: tuple[float, float] = (0.7, 1.45)
    brightness: float = 0.15           # additive, fraction of full scale
    contrast: float = 0.20             # multiplicative around the mean
    noise_std: float = 0.025           # gaussian, fraction of full scale
    bias_field: float = 0.20           # smooth multiplicative field strength
    max_rotate_deg: float = 8.0
    max_scale: float = 0.10
    max_translate: float = 0.06        # fraction of side length
    slice_dropout: float = 0.10        # chance of blanking an individual slice

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled, "p": self.p, "gamma": list(self.gamma),
            "brightness": self.brightness, "contrast": self.contrast,
            "noise_std": self.noise_std, "bias_field": self.bias_field,
            "max_rotate_deg": self.max_rotate_deg, "max_scale": self.max_scale,
            "max_translate": self.max_translate, "slice_dropout": self.slice_dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AugmentConfig":
        return cls(
            enabled=bool(d.get("enabled", True)),
            p=float(d.get("p", 0.8)),
            gamma=tuple(d.get("gamma", (0.7, 1.45))),
            brightness=float(d.get("brightness", 0.15)),
            contrast=float(d.get("contrast", 0.20)),
            noise_std=float(d.get("noise_std", 0.025)),
            bias_field=float(d.get("bias_field", 0.20)),
            max_rotate_deg=float(d.get("max_rotate_deg", 8.0)),
            max_scale=float(d.get("max_scale", 0.10)),
            max_translate=float(d.get("max_translate", 0.06)),
            slice_dropout=float(d.get("slice_dropout", 0.10)),
        )


def _bias_field(shape: tuple[int, int], strength: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth low-frequency multiplicative field — mimics coil inhomogeneity."""
    h, w = shape
    coarse = rng.normal(0.0, 1.0, size=(4, 4)).astype(np.float32)
    field = np.asarray(
        Image.fromarray(coarse, mode="F").resize((w, h), Image.BICUBIC), dtype=np.float32
    )
    m = np.abs(field).max()
    if m > 0:
        field = field / m
    return 1.0 + strength * field


def _affine_matrix(
    size: int, angle_deg: float, scale: float, tx: float, ty: float
) -> tuple[float, float, float, float, float, float]:
    """PIL AFFINE matrix mapping OUTPUT coords -> INPUT coords, about the centre."""
    angle = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle) / scale, math.sin(angle) / scale
    c = size / 2.0
    a, b = cos_a, sin_a
    d, e = -sin_a, cos_a
    cx = c - a * c - b * c + tx
    cy = c - d * c - e * c + ty
    return (a, b, cx, d, e, cy)


def augment_series(
    volume: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig
) -> np.ndarray:
    """(T, H, W) uint8 -> augmented (T, H, W) uint8. Parameters shared across T."""
    if not cfg.enabled or rng.random() > cfg.p:
        return volume

    t, h, w = volume.shape
    x = volume.astype(np.float32) / 255.0

    # --- geometry: one transform for the whole series ---
    if h == w and (cfg.max_rotate_deg > 0 or cfg.max_scale > 0 or cfg.max_translate > 0):
        angle = rng.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg)
        scale = 1.0 + rng.uniform(-cfg.max_scale, cfg.max_scale)
        tx = rng.uniform(-cfg.max_translate, cfg.max_translate) * w
        ty = rng.uniform(-cfg.max_translate, cfg.max_translate) * h
        matrix = _affine_matrix(h, angle, scale, tx, ty)
        out = np.empty_like(x)
        for i in range(t):
            img = Image.fromarray((x[i] * 255).astype(np.uint8))
            img = img.transform((w, h), Image.AFFINE, matrix, resample=Image.BILINEAR)
            out[i] = np.asarray(img, dtype=np.float32) / 255.0
        x = out

    # --- intensity: the augmentation that actually matters for scanner shift ---
    gamma = rng.uniform(*cfg.gamma)
    x = np.clip(x, 0.0, 1.0) ** gamma

    if cfg.contrast > 0:
        factor = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
        mean = float(x.mean())
        x = (x - mean) * factor + mean

    if cfg.brightness > 0:
        x = x + rng.uniform(-cfg.brightness, cfg.brightness)

    if cfg.bias_field > 0:
        x = x * _bias_field((h, w), cfg.bias_field, rng)[None, :, :]

    if cfg.noise_std > 0:
        x = x + rng.normal(0.0, cfg.noise_std, size=x.shape).astype(np.float32)

    if cfg.slice_dropout > 0 and t > 2:
        drop = rng.random(t) < cfg.slice_dropout
        if drop.all():
            drop[rng.integers(t)] = False
        x[drop] = 0.0

    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


def augment_study(
    pixels: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig
) -> np.ndarray:
    """(S, T, H, W) uint8 -> augmented, each series drawn independently."""
    if not cfg.enabled:
        return pixels
    return np.stack([augment_series(pixels[s], rng, cfg) for s in range(pixels.shape[0])])
