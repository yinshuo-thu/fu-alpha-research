#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.expressions import load_expression_table
from fu_alpha_research.mining import aggregate_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--min-is-ic", type=float, default=0.002)
    parser.add_argument("--min-oos-ic", type=float, default=0.001)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cand_path = Path(args.candidates) if args.candidates else cfg.reports_dir / "new_factor_candidates.csv"
    candidates = load_expression_table(cand_path)
    part_dir = cfg.reports_dir / "new_factor_ic_parts"
    is_parts = [pd.read_parquet(p) for p in sorted(part_dir.glob("is_*.parquet"))]
    oos_parts = [pd.read_parquet(p) for p in sorted(part_dir.glob("oos_*.parquet"))]
    if not is_parts or not oos_parts:
        raise FileNotFoundError(f"missing expression IC parts in {part_dir}")
    is_df = aggregate_stats(is_parts, "is")
    oos_df = aggregate_stats(oos_parts, "oos")
    out = is_df.merge(oos_df, on="factor", how="inner")
    out = out.rename(columns={"factor": "name"}).merge(candidates, on="name", how="left")
    out["same_sign"] = np.sign(out["is_ic"]) == np.sign(out["oos_ic"])
    out["abs_is_ic"] = out["is_ic"].abs()
    out["abs_oos_ic"] = out["oos_ic"].abs()
    out["effective"] = (
        out["same_sign"]
        & (out["abs_is_ic"] >= args.min_is_ic)
        & (out["abs_oos_ic"] >= args.min_oos_ic)
        & (out["is_coverage_proxy"] >= 0.5)
        & (out["oos_coverage_proxy"] >= 0.5)
    )
    out = out.sort_values(["effective", "abs_oos_ic", "abs_is_ic"], ascending=[False, False, False])
    out.to_csv(cfg.reports_dir / "new_factor_scores.csv", index=False)
    selected = out[out["effective"]].head(args.top_n).copy()
    selected.to_csv(cfg.reports_dir / "new_effective_factors_100.csv", index=False)
    expr_dir = cfg.output_dir / "expression_sets"
    expr_dir.mkdir(parents=True, exist_ok=True)
    selected[["name", "op", "left", "right", "formula"]].to_csv(expr_dir / "new100.csv", index=False)
    summary = {
        "method": "ic_prefilter_legacy_not_final_acceptance",
        "candidates": int(len(out)),
        "effective": int(out["effective"].sum()),
        "legacy_selected": int(len(selected)),
        "min_is_ic": args.min_is_ic,
        "min_oos_ic": args.min_oos_ic,
        "top_oos_ic": float(selected["oos_ic"].abs().max()) if len(selected) else math.nan,
        "note": (
            "This is only a fast IC prefilter. Final new-factor acceptance should be produced by "
            "scripts/evaluate_expression_scorecard.py, which adds data-quality, bucket, regime, "
            "correlation, trading, and robustness gates."
        ),
    }
    (cfg.reports_dir / "new_factor_mining_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(selected[["name", "formula", "is_ic", "oos_ic"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
