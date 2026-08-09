"""Phase-0 training: gold UNION high-precision pseudo-labels, masked BCE, 1 fold.

Saves the weights *and* the exact preprocess config, so the offline inference
notebook rebuilds the identical pipeline instead of a look-alike.

Validation is on GOLD ONLY (design invariant #3). Pseudo-label AUC would measure
agreement with the text labeler, not truth, and must never be reported as a
score.
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
from torch.utils.data import DataLoader

from config import ID_COLUMN, PreprocessConfig, RunConfig, TARGET_COLUMNS
from data import StudyDataset, build_targets, load_tables
from model import build_model, masked_bce_with_logits
from textlabel import coverage_report, label_reports


def per_column_auc(
    y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray, columns: Sequence[str]
) -> pd.DataFrame:
    """AUC per column on labeled cells only. NaN where a column has one class.

    Macro-averaging hides rare-column collapse, so always log the whole table.
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    for j, col in enumerate(columns):
        sel = mask[:, j].astype(bool)
        yt, ys = y_true[sel, j], y_score[sel, j]
        auc = np.nan
        if sel.sum() >= 2 and len(np.unique(yt)) == 2:
            auc = float(roc_auc_score(yt, ys))
        rows.append({"column": col, "n": int(sel.sum()), "auc": auc})
    return pd.DataFrame(rows)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, list[str]]:
    model.eval()
    probs, uids = [], []
    for batch in loader:
        logits = model(batch["x"].to(device))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        uids.extend(batch["study_uid"])
    if not probs:
        return np.zeros((0, len(TARGET_COLUMNS))), []
    return np.concatenate(probs, axis=0), uids


def train(
    data_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 3e-4,
    n_slices: int = 16,
    size: int = 224,
    max_series: int = 1,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    val_fraction: float = 0.3,
    seed: int = 0,
    device: str | None = None,
    num_workers: int = 0,
) -> dict:
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(data_dir, split="train")
    columns = list(TARGET_COLUMNS)

    # ---- labels: gold UNION high-precision pseudo-labels --------------------
    pseudo = None
    if tables.reports is not None:
        pseudo, _ = label_reports(tables.reports, columns=columns)
        print("pseudo-label coverage:\n", coverage_report(pseudo).to_string(index=False))
    targets, mask = build_targets(tables.study_uids, tables.gold, pseudo, columns)
    print(f"labeled cells: {int(mask.to_numpy().sum())} / {mask.size}")

    # ---- split: validate on gold only --------------------------------------
    # "Gold" means at least one non-null label in train.csv, NOT mere presence
    # of the StudyInstanceUID: some releases of train.csv carry one row per
    # study with every column NaN except for the ~1.3% that are truly labeled,
    # and a presence-only check would silently swallow the whole training set
    # into "validation".
    gold_uids = []
    if tables.gold is not None:
        g = tables.gold.set_index(tables.gold[ID_COLUMN].astype(str))
        label_cols = [c for c in columns if c in g.columns]
        really_labeled = set(g.index[g[label_cols].notna().any(axis=1)])
        gold_uids = [u for u in tables.study_uids if u in really_labeled]
        print(
            f"gold table: {len(g)} rows, {len(really_labeled)} with >=1 non-null label"
        )
    rng = np.random.default_rng(seed)
    gold_shuffled = list(gold_uids)
    rng.shuffle(gold_shuffled)
    n_val = max(1, int(round(len(gold_shuffled) * val_fraction))) if gold_shuffled else 0
    val_uids = gold_shuffled[:n_val]
    train_uids = [u for u in tables.study_uids if u not in set(val_uids)]
    # Keep only studies that actually carry at least one label.
    train_uids = [u for u in train_uids if bool(mask.loc[u].any())]
    print(f"train studies: {len(train_uids)} | gold val studies: {len(val_uids)}")

    pre = PreprocessConfig(
        n_slices=n_slices, size=size, max_series=max_series
    )
    train_ds = StudyDataset(train_uids, tables.series, pre, targets, mask)
    val_ds = StudyDataset(val_uids, tables.series, pre, targets, mask)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = build_model(backbone=backbone, pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(batch["x"].to(device))
            loss = masked_bce_with_logits(
                logits, batch["y"].to(device), batch["mask"].to(device)
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(f"epoch {epoch}: masked BCE {mean_loss:.4f}")

    # ---- gold-only validation ----------------------------------------------
    auc_table = pd.DataFrame(columns=["column", "n", "auc"])
    macro_auc = float("nan")
    if val_uids:
        probs, uids = predict(model, val_loader, device)
        yt = targets.loc[uids, columns].to_numpy(dtype=float)
        mk = mask.loc[uids, columns].to_numpy(dtype=bool)
        auc_table = per_column_auc(np.nan_to_num(yt), probs, mk, columns)
        aucs = auc_table["auc"].to_numpy(dtype=float)
        # With 58 gold studies most columns will be single-class in the fold and
        # score NaN. That is information, not an error - report it honestly.
        macro_auc = float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan")
        print("gold-only per-column AUC:\n", auc_table.to_string(index=False))
        print(f"gold macro-AUC (n={len(val_uids)} studies, treat as noise): {macro_auc:.4f}")

    run_cfg = RunConfig(preprocess=pre, backbone=backbone, target_columns=columns)
    run_cfg.save(out_dir / "run_config.json")
    torch.save(model.state_dict(), out_dir / "model.pt")
    metrics = {
        "history": history,
        "gold_macro_auc": macro_auc,
        "per_column_auc": auc_table.to_dict("records"),
        "n_train_studies": len(train_uids),
        "n_val_gold_studies": len(val_uids),
        "labeled_cells": int(mask.to_numpy().sum()),
        "seconds": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved weights + run_config.json to {out_dir} in {metrics['seconds']}s")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-slices", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--max-series", type=int, default=1)
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--no-pretrained", action="store_true", help="offline / tests")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_slices=args.n_slices,
        size=args.size,
        max_series=args.max_series,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
