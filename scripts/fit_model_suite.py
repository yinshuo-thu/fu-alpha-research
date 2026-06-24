#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.feature_matrix import read_feature_list
from fu_alpha_research.modeling import fit_lightgbm, fit_ridge


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"set", "path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--model", choices=["ridge", "lightgbm"], required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = Path(args.manifest) if args.manifest else cfg.output_dir / "model_feature_sets" / "manifest.csv"
    if not manifest_path.is_absolute():
        manifest_path = cfg.output_dir / manifest_path
    sample_dir = cfg.output_dir / "extended_samples" / "is" if args.sample_dir is None else Path(args.sample_dir)
    if not sample_dir.is_absolute():
        sample_dir = cfg.output_dir / sample_dir

    manifest = load_manifest(manifest_path)
    set_features = {row.set: read_feature_list(row.path) for row in manifest.itertuples(index=False)}
    union_features = list(dict.fromkeys(x for features in set_features.values() for x in features))
    sample_files = sorted(sample_dir.glob("*.parquet"))
    if not sample_files:
        raise FileNotFoundError(f"no sample files in {sample_dir}")
    frames = [pd.read_parquet(path, columns=["label"] + union_features) for path in sample_files]
    train = pd.concat(frames, ignore_index=True)
    print(f"[fit-suite] loaded rows={len(train)} union_features={len(union_features)}")

    model_dir = cfg.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for set_name, features in set_features.items():
        name = f"{'lgbm' if args.model == 'lightgbm' else 'ridge'}_{set_name}"
        meta_file = model_dir / f"{name}.json"
        if meta_file.exists() and not args.force:
            print(f"[fit-suite] exists {name}")
            continue
        if args.model == "ridge":
            model = fit_ridge(train[["label"] + features], features, alpha=args.alpha)
            np.savez(
                model_dir / f"{name}.npz",
                mean=model.mean,
                scale=model.scale,
                weight=model.weight,
                y_mean=np.array([model.y_mean], dtype=np.float64),
                alpha=np.array([model.alpha], dtype=np.float64),
            )
            meta = {
                "name": name,
                "set": set_name,
                "model": "ridge",
                "features": features,
                "train_rows": int(len(train)),
                "alpha": args.alpha,
            }
        else:
            params = {"n_estimators": args.n_estimators}
            model = fit_lightgbm(train[["label"] + features], features, seed=args.seed, params=params)
            model_path = model_dir / f"{name}.txt"
            model.booster_.save_model(str(model_path))
            meta = {
                "name": name,
                "set": set_name,
                "model": "lightgbm",
                "features": features,
                "train_rows": int(len(train)),
                "seed": args.seed,
                "params": params,
                "model_file": str(model_path),
            }
        meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[fit-suite] wrote {name} features={len(features)}")


if __name__ == "__main__":
    main()
