"""K-fold training over cached frozen-backbone features.

The whole point of caching embeddings: an epoch is seconds, not tens of minutes,
so this runs many folds x many epochs in the time the end-to-end path needed for
one fold. Everything else is unchanged from train.py - gold UNION pseudo-labels,
masked BCE weighted by measured labeler precision, out-of-fold validation on the
58 gold studies, rank-averaged ensembling.

    python train_features.py --data-dir <data> --feature-dir <feats> \\
        --out-dir artifacts --n-folds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config
from config import ID_COLUMN, MULTI_PLANE_PREFS, PreprocessConfig, RunConfig
from data import build_targets, load_tables
from ensemble import rank_average
from features import FeatureStore
from model import PLANE_IDS, FeatureHead, masked_bce_with_logits
from textlabel import coverage_report, label_reports
from train import per_column_auc


class FeatureDataset(Dataset):
    """Loads cached (S, T, D) embeddings. Small enough to hold in RAM."""

    def __init__(
        self,
        study_uids: Sequence[str],
        store: FeatureStore,
        targets: pd.DataFrame | None = None,
        mask: pd.DataFrame | None = None,
        preload: bool = True,
    ):
        self.store = store
        self.targets = targets
        self.mask = mask
        self.items: list[tuple[str, np.ndarray, np.ndarray, list[str]]] = []
        for uid in (str(u) for u in study_uids):
            got = store.load(uid)
            if got is None:
                continue
            feats, series_mask, planes = got
            self.items.append((uid, feats, series_mask, planes))
        if not self.items:
            raise RuntimeError("no cached features found - run features.py first")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        uid, feats, series_mask, planes = self.items[i]
        out = {
            "feats": torch.from_numpy(feats),
            "series_mask": torch.from_numpy(series_mask.astype(np.float32)),
            "plane_ids": torch.tensor([PLANE_IDS.get(p, 0) for p in planes],
                                      dtype=torch.long),
            "study_uid": uid,
        }
        if self.targets is None:
            out["y"] = torch.zeros(0)
            out["mask"] = torch.zeros(0)
        else:
            out["y"] = torch.from_numpy(
                np.nan_to_num(self.targets.loc[uid].to_numpy(dtype=np.float32), nan=0.0)
            )
            out["mask"] = torch.from_numpy(
                self.mask.loc[uid].to_numpy(dtype=np.float32)
            )
        return out


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, uids = [], []
    for b in loader:
        logits = model(
            b["feats"].to(device),
            series_mask=b["series_mask"].to(device),
            plane_ids=b["plane_ids"].to(device),
        )
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        uids.extend(b["study_uid"])
    if not probs:
        return np.zeros((0, 12)), []
    return np.concatenate(probs, 0), uids


def train_kfold_features(
    data_dir: str | Path,
    feature_dir: str | Path,
    out_dir: str | Path,
    backbone: str = "vit_small_patch14_dinov2.lvd142m",
    n_slices: int = 32,
    size: int = 224,
    max_series: int = 3,
    input_mode: str = "2.5d",
    n_folds: int = 5,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    slice_dropout: float = 0.1,
    feature_noise: float = 0.05,
    seed: int = 0,
    device: str | None = None,
    columns: Sequence[str] | None = None,
    use_confidence_weights: bool = True,
) -> dict:
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = list(columns) if columns is not None else list(config.TARGET_COLUMNS)

    pre = PreprocessConfig(n_slices=n_slices, size=size, max_series=max_series,
                           plane_prefs=MULTI_PLANE_PREFS)
    store = FeatureStore(feature_dir, pre, backbone, input_mode)

    tables = load_tables(data_dir, split="train")
    pseudo = None
    if tables.reports is not None:
        pseudo, _ = label_reports(tables.reports, columns=columns)
        print("pseudo-label coverage:\n", coverage_report(pseudo).to_string(index=False))

    weights = None
    if use_confidence_weights:
        weights = dict(config.PSEUDO_LABEL_WEIGHTS)
        if pseudo is not None:
            for col in columns:
                if col in pseudo.columns and pseudo[col].notna().sum():
                    rate = float((pseudo[col] == 1).sum()) / float(pseudo[col].notna().sum())
                    weights[col] = weights.get(col, 1.0) * config.balance_factor(rate)

    targets, mask = build_targets(tables.study_uids, tables.gold, pseudo, columns,
                                  pseudo_weights=weights)

    gold_uids = []
    if tables.gold is not None:
        g = tables.gold.set_index(tables.gold[ID_COLUMN].astype(str))
        label_cols = [c for c in columns if c in g.columns]
        really = set(g.index[g[label_cols].notna().any(axis=1)])
        gold_uids = [u for u in tables.study_uids if u in really]
    print(f"gold studies: {len(gold_uids)} | labeled cells: {int((mask.to_numpy() > 0).sum())}")

    rng = np.random.default_rng(seed)
    shuffled = list(gold_uids)
    rng.shuffle(shuffled)
    blocks = np.array_split(np.arange(len(shuffled)), n_folds)

    feat_dim = store.load(next(iter(tables.study_uids)))
    feat_dim = feat_dim[0].shape[-1] if feat_dim is not None else 384

    oof_probs, oof_uids, fold_metrics = [], [], []
    for fold in range(n_folds):
        val_idx = set(blocks[fold].tolist())
        val_uids = [u for i, u in enumerate(shuffled) if i in val_idx]
        train_uids = [u for u in tables.study_uids
                      if u not in set(val_uids) and bool((mask.loc[u] > 0).any())]

        train_ds = FeatureDataset(train_uids, store, targets, mask)
        val_ds = FeatureDataset(val_uids, store, targets, mask)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        torch.manual_seed(seed + fold)
        model = FeatureHead(feat_dim, n_targets=len(columns), hidden_dim=hidden_dim,
                            dropout=dropout, slice_dropout=slice_dropout,
                            feature_noise=feature_noise).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=max(1, len(train_loader)) * epochs, pct_start=0.2
        )

        for epoch in range(epochs):
            model.train()
            losses = []
            for b in train_loader:
                opt.zero_grad(set_to_none=True)
                logits = model(b["feats"].to(device),
                               series_mask=b["series_mask"].to(device),
                               plane_ids=b["plane_ids"].to(device))
                loss = masked_bce_with_logits(logits, b["y"].to(device),
                                              b["mask"].to(device))
                loss.backward()
                opt.step()
                sched.step()
                losses.append(float(loss.detach()))
            if (epoch + 1) % max(1, epochs // 4) == 0:
                print(f"  fold {fold} epoch {epoch+1}/{epochs}: {np.mean(losses):.4f}")

        probs, uids = predict(model, val_loader, device)
        torch.save(model.state_dict(), out_dir / f"head_fold{fold}.pt")
        oof_probs.append(probs)
        oof_uids.extend(uids)
        fold_metrics.append({"fold": fold, "n_train": len(train_uids), "n_val": len(val_uids)})
        print(f"fold {fold}: trained on {len(train_uids)}, validated on {len(val_uids)}")

    summary = {"folds": fold_metrics, "seconds": round(time.time() - t0, 1),
               "backbone": backbone, "feat_dim": int(feat_dim)}

    if oof_probs:
        probs = np.concatenate(oof_probs, 0)
        g = tables.gold.set_index(tables.gold[ID_COLUMN].astype(str))
        keep = [c for c in columns if c in g.columns]
        y = g.reindex(index=oof_uids, columns=keep)
        table = per_column_auc(np.nan_to_num(y.to_numpy(dtype=float)), probs,
                               y.notna().to_numpy(), keep)
        aucs = table["auc"].to_numpy(dtype=float)
        macro = float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan")
        print("\n" + "=" * 70)
        print(f"OUT-OF-FOLD over {len(oof_uids)} gold studies")
        print("=" * 70)
        print(table.to_string(index=False))
        print(f"OOF macro-AUC: {macro:.4f}")
        summary["oof_macro_auc"] = macro
        summary["oof_per_column"] = table.to_dict("records")
        pd.DataFrame(probs, index=oof_uids, columns=keep).to_csv(
            out_dir / "oof_predictions.csv"
        )

    RunConfig(preprocess=pre, backbone=backbone, target_columns=columns,
              input_mode=input_mode).save(out_dir / "run_config.json")
    (out_dir / "kfold_summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")
    print(f"\n{n_folds} heads -> {out_dir} in {summary['seconds']:.0f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--feature-dir", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-slices", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--max-series", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train_kfold_features(
        data_dir=args.data_dir, feature_dir=args.feature_dir, out_dir=args.out_dir,
        backbone=args.backbone, n_folds=args.n_folds, epochs=args.epochs,
        n_slices=args.n_slices, size=args.size, max_series=args.max_series, lr=args.lr,
    )


if __name__ == "__main__":
    main()
