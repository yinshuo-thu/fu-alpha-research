#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.expressions import compute_expression_block, load_expression_table
from fu_alpha_research.factor_store import FactorStore
from fu_alpha_research.mining import month_sufficient_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--prefix", required=True, choices=["is", "oos"])
    parser.add_argument("--month", required=True)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cand_path = Path(args.candidates) if args.candidates else cfg.reports_dir / "new_factor_candidates.csv"
    candidates = load_expression_table(cand_path)
    out_dir = cfg.reports_dir / "new_factor_ic_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.prefix}_{args.month}.parquet"
    if out_file.exists() and not args.force:
        print(f"[expr-ic-part] exists {out_file}")
        return

    store = FactorStore(cfg)
    deps = sorted(set(candidates["left"]).union(candidates["right"]))
    base = store.read_month(args.month, columns=deps)
    meta = base[["label", "datetime"]].copy()
    parts = []
    for start in range(0, len(candidates), args.block_size):
        block = candidates.iloc[start : start + args.block_size].reset_index(drop=True)
        vals = compute_expression_block(base, block)
        frame = pd.concat([meta[["label"]], vals], axis=1)
        stats = month_sufficient_stats(frame, block["name"].tolist(), block_size=args.block_size)
        parts.append(stats)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(out_file, index=False)
    print(f"[expr-ic-part] wrote {out_file} rows={len(base)} candidates={len(out)}")


if __name__ == "__main__":
    main()
