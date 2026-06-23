#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.factor_store import FactorStore
from fu_alpha_research.modeling import fit_ridge


def load_features(path: str | None, cfg) -> list[str]:
    if path:
        return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return FactorStore(cfg).selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--feature-file")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    features = load_features(args.feature_file, cfg)
    sample_dir = cfg.output_dir / "samples" / "is"
    sample_files = sorted(sample_dir.glob("*.parquet"))
    if not sample_files:
        raise FileNotFoundError(f"no sample files in {sample_dir}")
    frames = [pd.read_parquet(path, columns=["label"] + features) for path in sample_files]
    train = pd.concat(frames, ignore_index=True)
    model = fit_ridge(train, features, alpha=args.alpha)

    model_dir = cfg.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        model_dir / f"{args.name}.npz",
        mean=model.mean,
        scale=model.scale,
        weight=model.weight,
        y_mean=np.array([model.y_mean], dtype=np.float64),
        alpha=np.array([model.alpha], dtype=np.float64),
    )
    meta = {"name": args.name, "model": "ridge", "features": features, "train_rows": int(len(train))}
    (model_dir / f"{args.name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[fit-ridge] wrote {model_dir / (args.name + '.npz')} rows={len(train)} features={len(features)}")


if __name__ == "__main__":
    main()
