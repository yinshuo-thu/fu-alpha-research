#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from fu_alpha_research.config import load_config
from fu_alpha_research.feature_matrix import FeatureMatrix, read_feature_list
from fu_alpha_research.mlp import add_event_sampling_cols, sample_rows


def resolve_output_path(output_dir, value: str | None):
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else output_dir / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--month", required=True)
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--expression-file", default=None)
    parser.add_argument("--sample-dir", default="mlp_samples/is")
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--sample-mode", default="soft_event")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.output_dir / args.sample_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.month}.parquet"
    if out_file.exists() and not args.force:
        print(f"[sample-mlp-month] exists {out_file}")
        return

    feature_file = resolve_output_path(cfg.output_dir, args.feature_file)
    features = read_feature_list(feature_file)
    expr_path = resolve_output_path(cfg.output_dir, args.expression_file)
    matrix = FeatureMatrix(cfg, expr_path)
    df = matrix.read_month(args.month, features).dropna(subset=["label"])
    df = add_event_sampling_cols(df)
    keep = [
        "symbol",
        "datetime",
        "label",
        "label_xsz",
        "label_xrank",
        "event_score",
        "_bars_to_month_end",
    ] + features
    sampled = sample_rows(df[keep], args.rows, args.sample_mode, args.seed + int(args.month.replace("-", "")))
    sampled.to_parquet(out_file, index=False)
    print(f"[sample-mlp-month] wrote {out_file} rows={len(sampled)} features={len(features)}")


if __name__ == "__main__":
    main()
