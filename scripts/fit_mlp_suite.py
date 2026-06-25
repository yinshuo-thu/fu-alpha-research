#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from fu_alpha_research.config import load_config
from fu_alpha_research.feature_matrix import read_feature_list
from fu_alpha_research.mlp import MLPConfig, make_mlp, recency_weights, scrub_matrix, weighted_loss, weighted_mean_scale


def resolve_output_path(output_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return output_dir / path


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = {"set", "path"} - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    return manifest


def fit_one(train: pd.DataFrame, features: list[str], cfg: MLPConfig, device: torch.device, fit_month: str):
    y_all = train[cfg.target_col].to_numpy(np.float32, copy=False)
    mask = np.isfinite(y_all)
    work = train.loc[mask].copy()
    y = y_all[mask]
    w = recency_weights(train, fit_month, cfg.half_life_months)
    if w is None:
        w = np.ones(len(y_all), dtype=np.float32)
    w = np.nan_to_num(w[mask].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 0.0)
    w = w / max(float(w.mean()), 1e-8)

    x = scrub_matrix(work[features].to_numpy(np.float32, copy=False))
    if cfg.standardize == "weighted":
        mean, scale = weighted_mean_scale(x, w)
    elif cfg.standardize == "unweighted":
        mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    else:
        raise ValueError(f"bad standardize={cfg.standardize!r}")
    x = ((x - mean) / scale).astype(np.float32)
    y = np.clip(y, -8, 8).astype(np.float32)

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    model = make_mlp(x.shape[1], cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w.astype(np.float32)))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=device.type == "cuda")
    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        batches = 0
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = weighted_loss(pred, yb, wb, cfg.loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        print(f"[fit-mlp] epoch={epoch + 1}/{cfg.epochs} loss={total_loss / max(batches, 1):.6f}", flush=True)
    return model, mean, scale, int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample-dir", default="mlp_samples/is")
    parser.add_argument("--fit-month", default="2020-01")
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--half-life-months", type=float, default=12.0)
    parser.add_argument("--target-col", default="label_xsz")
    parser.add_argument("--loss", choices=["mse", "huber", "corr_mse"], default="mse")
    parser.add_argument("--standardize", choices=["unweighted", "weighted"], default="unweighted")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = resolve_output_path(cfg.output_dir, args.manifest)
    sample_dir = resolve_output_path(cfg.output_dir, args.sample_dir)
    manifest = load_manifest(manifest_path)
    set_features = {row.set: read_feature_list(row.path) for row in manifest.itertuples(index=False)}
    union_features = list(dict.fromkeys(x for features in set_features.values() for x in features))
    sample_files = sorted(sample_dir.glob("*.parquet"))
    if not sample_files:
        raise FileNotFoundError(f"no sample files in {sample_dir}")
    columns = ["symbol", "datetime", "label", args.target_col] + union_features
    train = pd.concat([pd.read_parquet(path, columns=columns) for path in sample_files], ignore_index=True)
    train["datetime"] = pd.to_datetime(train["datetime"])
    print(f"[fit-mlp] loaded rows={len(train)} union_features={len(union_features)} device={'cuda' if torch.cuda.is_available() else 'cpu'}")

    model_cfg = MLPConfig(
        hidden=args.hidden,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        half_life_months=args.half_life_months,
        target_col=args.target_col,
        loss=args.loss,
        standardize=args.standardize,
        seed=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = cfg.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for set_name, features in set_features.items():
        name = f"mlp_{set_name}"
        model_path = model_dir / f"{name}.pt"
        meta_path = model_dir / f"{name}.json"
        if meta_path.exists() and not args.force:
            print(f"[fit-mlp] exists {name}")
            continue
        model, mean, scale, train_rows = fit_one(train[["datetime", args.target_col] + features], features, model_cfg, device, args.fit_month)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mean": mean,
                "scale": scale,
                "features": features,
                "config": asdict(model_cfg),
            },
            model_path,
        )
        meta = {
            "name": name,
            "set": set_name,
            "model": "mlp",
            "model_file": str(model_path),
            "features": features,
            "train_rows": train_rows,
            "union_train_rows": int(len(train)),
            "config": asdict(model_cfg),
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[fit-mlp] wrote {name} features={len(features)} train_rows={train_rows}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
