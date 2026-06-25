#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.expressions import load_expression_table, precompute_expression_inputs
from fu_alpha_research.factor_store import FactorStore


def expression_array_from_inputs(ranks: pd.DataFrame, zscores: pd.DataFrame, exprs: pd.DataFrame) -> np.ndarray:
    out = np.empty((len(ranks), len(exprs)), dtype=np.float32)
    for col_idx, row in enumerate(exprs.itertuples(index=False)):
        if row.op.startswith("rank_"):
            left = ranks[row.left].to_numpy(np.float32, copy=False)
            right = ranks[row.right].to_numpy(np.float32, copy=False)
        else:
            left = zscores[row.left].to_numpy(np.float32, copy=False)
            right = zscores[row.right].to_numpy(np.float32, copy=False)
        if row.op.endswith("_add"):
            np.add(left, right, out=out[:, col_idx])
        elif row.op.endswith("_spread"):
            np.subtract(left, right, out=out[:, col_idx])
        elif row.op.endswith("_product"):
            np.multiply(left, right, out=out[:, col_idx])
        elif row.op == "rank_gate_pos":
            out[:, col_idx] = np.where(right > 0, left, 0.0)
        elif row.op == "rank_gate_neg":
            out[:, col_idx] = np.where(right < 0, left, 0.0)
        else:
            raise ValueError(f"unknown expression op: {row.op}")
    return out


def matrix_sufficient_stats(values: np.ndarray, label: np.ndarray, names: list[str]) -> pd.DataFrame:
    y = np.asarray(label, dtype=np.float64)
    y_ok = np.isfinite(y)
    y0 = np.where(y_ok, y, 0.0)
    y2 = y0 * y0
    valid = np.isfinite(values) & y_ok[:, None]
    x0 = np.where(valid, values, 0.0).astype(np.float64, copy=False)
    xy = x0.T @ y0
    xx = np.sum(x0 * x0, axis=0)
    yy = valid.astype(np.float64).T @ y2
    count = np.sum(valid, axis=0)
    return pd.DataFrame({"factor": names, "xy": xy, "xx": xx, "yy": yy, "count": count.astype(np.int64)})


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
    label = base["label"].to_numpy(np.float64, copy=False)
    ranks, zscores = precompute_expression_inputs(base, candidates)
    parts = []
    for start in range(0, len(candidates), args.block_size):
        block = candidates.iloc[start : start + args.block_size].reset_index(drop=True)
        vals = expression_array_from_inputs(ranks, zscores, block)
        stats = matrix_sufficient_stats(vals, label, block["name"].tolist())
        parts.append(stats)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(out_file, index=False)
    print(f"[expr-ic-part] wrote {out_file} rows={len(base)} candidates={len(out)}")


if __name__ == "__main__":
    main()
