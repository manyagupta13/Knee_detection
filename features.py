"""Frozen-backbone slice embeddings, extracted once and cached.

Why not just swap the backbone: ViT-S/14 is roughly 11x the FLOPs of
EfficientNet-B0 per image, and at 3 series x 32 slices we push 96 images per
study. Training through it would turn a 6-hour run into 60+.

DINOv2's entire design point is that its FROZEN features are strong - that is
what linear-probe benchmarks measure. So the backbone runs ONCE over every
slice, the 384-d embeddings are cached, and training touches only the attention
pooling and heads. One forward pass over ~320k slices costs ~20 minutes on a
T4; after that an epoch is seconds, which finally makes k-fold, long schedules
and hyper-parameter search affordable.

The trade is pixel-space augmentation: a frozen embedding cannot be re-augmented
per epoch. Feature-space slice dropout and noise survive, and extract_features()
can be run for several augmented views if that proves worth the extra passes.

    python features.py --data-dir <data> --out-dir /kaggle/working/feats \\
        --backbone vit_small_patch14_dinov2.lvd142m
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from cache import PLANE_CODES, CODE_PLANES, StudyCache
from config import MULTI_PLANE_PREFS, PreprocessConfig
from preprocess import to_model_input


def feature_key(config: PreprocessConfig, backbone: str, input_mode: str) -> str:
    """Hash of everything that changes the embeddings."""
    payload = json.dumps(
        {"pre": config.to_dict(), "backbone": backbone, "input_mode": input_mode},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


class FeatureStore:
    """One .npz per study: features (S, T, D) float16 + mask + plane codes."""

    def __init__(self, root: str | Path, config: PreprocessConfig,
                 backbone: str, input_mode: str = "2.5d"):
        self.config = config
        self.backbone = backbone
        self.input_mode = input_mode
        self.key = feature_key(config, backbone, input_mode)
        self.root = Path(root) / self.key
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta.json").write_text(
            json.dumps(
                {"preprocess": config.to_dict(), "backbone": backbone,
                 "input_mode": input_mode},
                indent=2,
            ),
            encoding="utf-8",
        )

    def _path(self, study_uid: str) -> Path:
        digest = hashlib.sha1(str(study_uid).encode("utf-8")).hexdigest()[:2]
        return self.root / digest / f"{study_uid}.npz"

    def load(self, study_uid: str):
        path = self._path(study_uid)
        if not path.exists():
            return None
        try:
            with np.load(path) as z:
                return (
                    z["features"].astype(np.float32),
                    z["series_mask"].astype(bool),
                    [CODE_PLANES.get(int(c), "unknown") for c in z["planes"]],
                )
        except Exception:
            return None

    def store(self, study_uid: str, features: np.ndarray,
              series_mask: np.ndarray, planes: Sequence[str]) -> None:
        path = self._path(study_uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            features=features.astype(np.float16),  # half precision is plenty
            series_mask=np.asarray(series_mask, dtype=bool),
            planes=np.array([PLANE_CODES.get(p, 0) for p in planes], dtype=np.int8),
        )
        tmp.replace(path)

    def has(self, study_uid: str) -> bool:
        return self._path(study_uid).exists()


def build_backbone(name: str, size: int, device: str, pretrained: bool = True):
    import timm
    import torch

    kwargs = dict(pretrained=pretrained, num_classes=0)
    if "patch14" in name or "vit_" in name:
        # patch14 needs img_size divisible by 14 (224 = 16*14) and the DINOv2
        # checkpoints default to 518, so the position embedding must be resized.
        kwargs.update(img_size=size, dynamic_img_size=True)
    model = timm.create_model(name, **kwargs)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_features(
    study_uids: Sequence[str],
    series_df: pd.DataFrame,
    config: PreprocessConfig,
    backbone: str,
    out_dir: str | Path,
    pixel_cache_dir: str | Path | None = None,
    input_mode: str = "2.5d",
    batch_slices: int = 64,
    device: str | None = None,
    pretrained: bool = True,
    amp: bool = True,
    verbose: bool = True,
) -> FeatureStore:
    """Run the frozen backbone over every slice of every study, once."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    store = FeatureStore(out_dir, config, backbone, input_mode)
    pixels_cache = StudyCache(pixel_cache_dir, config) if pixel_cache_dir else None

    uids = [str(u) for u in study_uids]
    todo = [u for u in uids if not store.has(u)]
    if verbose:
        print(f"features {store.key}: {len(uids) - len(todo)} hit, {len(todo)} to extract")
    if not todo:
        return store

    model = build_backbone(backbone, config.size, device, pretrained)
    use_amp = amp and device == "cuda"
    t0 = time.time()

    for i, uid in enumerate(todo):
        if pixels_cache is not None and pixels_cache.enabled:
            study = pixels_cache.get(uid, series_df)
        else:
            from preprocess import preprocess_study

            study = preprocess_study(
                uid, series_df, config.plane_prefs, config.n_slices,
                config.size, config.max_series, canonicalize=config.canonicalize,
            )

        s, t = study.pixels.shape[:2]
        x = np.stack([to_model_input(study.pixels[k], input_mode) for k in range(s)])
        flat = torch.from_numpy(x.reshape(s * t, *x.shape[2:]))

        outs = []
        with torch.no_grad():
            for start in range(0, flat.shape[0], batch_slices):
                chunk = flat[start : start + batch_slices].to(device)
                with torch.autocast("cuda", enabled=use_amp):
                    outs.append(model(chunk).float().cpu())
        feats = torch.cat(outs, 0).numpy().reshape(s, t, -1)
        store.store(uid, feats, study.series_mask, study.planes)

        if verbose and (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(todo) - i - 1) / max(rate, 1e-9)
            print(f"  {i + 1}/{len(todo)} @ {rate:.1f} studies/s  eta {eta/60:.0f} min")

    if verbose:
        print(f"features extracted in {(time.time() - t0)/60:.1f} min -> {store.root}")
    return store


def main() -> None:
    from data import load_tables

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pixel-cache-dir", default=None)
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-slices", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--max-series", type=int, default=3)
    args = ap.parse_args()

    tables = load_tables(args.data_dir, split=args.split)
    cfg = PreprocessConfig(
        n_slices=args.n_slices, size=args.size, max_series=args.max_series,
        plane_prefs=MULTI_PLANE_PREFS,
    )
    extract_features(
        tables.study_uids, tables.series, cfg, args.backbone, args.out_dir,
        pixel_cache_dir=args.pixel_cache_dir,
    )


if __name__ == "__main__":
    main()
