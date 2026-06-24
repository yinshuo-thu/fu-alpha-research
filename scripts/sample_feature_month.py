#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from fu_alpha_research.config import load_config
from fu_alpha_research.feature_matrix import FeatureMatrix, read_feature_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--month", required=True)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--expression-file", default=None)
    parser.add_argument("--sample-dir", default=None)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.output_dir / "extended_samples" / "is" if args.sample_dir is None else cfg.output_dir / args.sample_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.month}.parquet"
    if out_file.exists() and not args.force:
        print(f"[sample-feature-month] exists {out_file}")
        return

    features = read_feature_list(args.feature_file)
    matrix = FeatureMatrix(cfg, None if args.expression_file is None else cfg.output_dir / args.expression_file)
    df = matrix.read_month(args.month, features).dropna(subset=["label"])
    if len(df) > args.rows:
        rng = np.random.default_rng(args.seed + int(args.month.replace("-", "")))
        idx = np.sort(rng.choice(len(df), size=args.rows, replace=False))
        df = df.iloc[idx].copy()
    df.to_parquet(out_file, index=False)
    print(f"[sample-feature-month] wrote {out_file} rows={len(df)} features={len(features)}")


if __name__ == "__main__":
    main()
