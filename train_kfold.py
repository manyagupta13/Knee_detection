"""K-fold training with out-of-fold validation and a rank-averaged ensemble.

Two problems this solves at once.

Validation. A single split validated on 17 gold studies, which is noise - it
reported 0.594 when the LB said 0.602 and 0.787 when the LB said 0.764. With
K folds every gold study is validated exactly once, so out-of-fold AUC spans all
58. Still small, but it is the difference between a signal and a coin flip, and
it is the only honest basis for choosing between backbones.

Ensembling. The folds also give K models for free. They are correlated - the
pseudo-labeled bulk is shared - so the diversity comes mostly from seed and
augmentation. Rank-average them (PLAN.md §4: never probability-average, the
metric is rank-based and prevalence shifts between splits).

    python train_kfold.py --data-dir <data> --out-dir artifacts --n-folds 5 \\
        --cache-dir /kaggle/working/cache --backbone tf_efficientnetv2_s --size 288
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import config
from config import ID_COLUMN, RunConfig
from data import load_tables
from train import per_column_auc, train


def train_kfold(
    data_dir: str | Path,
    out_dir: str | Path,
    n_folds: int = 5,
    columns: Sequence[str] | None = None,
    **train_kwargs,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = list(columns) if columns is not None else list(config.TARGET_COLUMNS)

    t0 = time.time()
    oof_probs: list[np.ndarray] = []
    oof_uids: list[str] = []
    fold_metrics = []

    for fold in range(n_folds):
        print("\n" + "=" * 70)
        print(f"FOLD {fold + 1}/{n_folds}")
        print("=" * 70)
        m = train(
            data_dir=data_dir,
            out_dir=out_dir,
            fold=fold,
            n_folds=n_folds,
            columns=columns,
            # Vary the seed per fold so the ensemble members differ by more than
            # their validation slice.
            seed=train_kwargs.pop("seed", 0) + fold if fold == 0 else fold,
            **train_kwargs,
        )
        fold_metrics.append(m)
        if m.get("val_uids"):
            oof_uids.extend(m["val_uids"])
            oof_probs.append(np.asarray(m["val_probs"], dtype=float))

    # ---- out-of-fold score over ALL gold studies ---------------------------
    summary = {"folds": fold_metrics, "seconds": round(time.time() - t0, 1)}
    if oof_probs:
        probs = np.concatenate(oof_probs, axis=0)
        tables = load_tables(data_dir, split="train")
        gold = tables.gold.set_index(tables.gold[ID_COLUMN].astype(str))
        keep = [c for c in columns if c in gold.columns]
        y = gold.reindex(index=oof_uids, columns=keep)
        mask = y.notna().to_numpy()
        table = per_column_auc(
            np.nan_to_num(y.to_numpy(dtype=float)), probs, mask, keep
        )
        aucs = table["auc"].to_numpy(dtype=float)
        macro = float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan")

        print("\n" + "=" * 70)
        print(f"OUT-OF-FOLD over {len(oof_uids)} gold studies (vs 17 for a single split)")
        print("=" * 70)
        print(table.to_string(index=False))
        print(f"OOF macro-AUC: {macro:.4f}")

        summary["oof_macro_auc"] = macro
        summary["oof_per_column"] = table.to_dict("records")
        summary["n_oof_studies"] = len(oof_uids)
        pd.DataFrame(probs, index=oof_uids, columns=keep).to_csv(out_dir / "oof_predictions.csv")

    (out_dir / "kfold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {n_folds} fold models to {out_dir} in {summary['seconds']}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-slices", type=int, default=12)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--max-series", type=int, default=3)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--finetune-gold-epochs", type=int, default=2)
    ap.add_argument("--slice-jitter", type=float, default=0.5)
    ap.add_argument("--n-slices-pool", type=int, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    train_kfold(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_slices=args.n_slices,
        size=args.size,
        max_series=args.max_series,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        finetune_gold_epochs=args.finetune_gold_epochs,
        slice_jitter=args.slice_jitter,
        n_slices_pool=args.n_slices_pool,
    )


if __name__ == "__main__":
    main()
