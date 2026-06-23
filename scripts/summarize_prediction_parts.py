#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.metrics import (
    add_prediction_views,
    compute_ic,
    long_short_backtest,
    summarize_backtest,
)


def prediction_stats(df: pd.DataFrame, pred_col: str) -> dict[str, float | int]:
    clean = df[[pred_col, "label"]].dropna()
    p = clean[pred_col].astype("float64")
    y = clean["label"].astype("float64")
    return {
        "xy": float((p * y).sum()),
        "xx": float((p * p).sum()),
        "yy": float((y * y).sum()),
        "count": int(len(clean)),
    }


def ic_from_stats(stats: dict[str, float | int]) -> float:
    import math

    denom = math.sqrt(max(float(stats["xx"]) * float(stats["yy"]), 1e-30))
    return float(stats["xy"]) / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    part_dir = cfg.output_dir / "prediction_parts" / args.name
    parts = sorted(part_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no prediction parts in {part_dir}")
    pred_cols = ["pred", "pred_xsz", "pred_xrank"]
    totals = {col: {"xy": 0.0, "xx": 0.0, "yy": 0.0, "count": 0} for col in pred_cols}
    monthly_rows = []
    bt_parts = []
    rows_total = 0
    label_rows = 0
    for path in parts:
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = add_prediction_views(df, "pred")
        rows_total += len(df)
        label_rows += int(df["label"].notna().sum())
        month = path.stem
        row = {"month": month}
        for col in pred_cols:
            stats = prediction_stats(df, col)
            for key in totals[col]:
                totals[col][key] += stats[key]
            row[col] = compute_ic(df[col].to_numpy(), df["label"].to_numpy())
        monthly_rows.append(row)
        bt_parts.append(long_short_backtest(df, pred_col="pred_xrank"))
        print(f"[summarize] {args.name} {month} rows={len(df)}", flush=True)

    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(cfg.reports_dir / f"monthly_ic_{args.name}.csv", index=False)
    summary_rows = []
    for col in pred_cols:
        vals = monthly[col]
        total = totals[col]
        summary_rows.append(
            {
                "pred_col": col,
                "rows": rows_total,
                "label_rows": label_rows,
                "coverage": float(total["count"]) / max(float(rows_total), 1.0),
                "total_ic": ic_from_stats(total),
                "monthly_mean": float(vals.mean()),
                "monthly_std": float(vals.std()),
                "monthly_ir": float(vals.mean() / vals.std()) if float(vals.std()) > 0 else float("nan"),
                "ic_2020": ic_from_stats(total),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.insert(0, "name", args.name)
    summary.to_csv(cfg.reports_dir / f"baseline_summary_{args.name}.csv", index=False)
    bt = pd.concat(bt_parts, ignore_index=True)
    bt.to_csv(cfg.output_dir / f"backtest_{args.name}.csv", index=False)
    (cfg.reports_dir / f"backtest_{args.name}.json").write_text(
        json.dumps(summarize_backtest(bt), indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
