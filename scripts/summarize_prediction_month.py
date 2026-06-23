#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.metrics import add_prediction_views, compute_ic, long_short_backtest


def stats(df: pd.DataFrame, pred_col: str) -> dict[str, float | int]:
    clean = df[[pred_col, "label"]].dropna()
    p = clean[pred_col].astype("float64")
    y = clean["label"].astype("float64")
    return {
        "xy": float((p * y).sum()),
        "xx": float((p * p).sum()),
        "yy": float((y * y).sum()),
        "count": int(len(clean)),
        "ic": compute_ic(df[pred_col].to_numpy(), df["label"].to_numpy()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--month", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    part = cfg.output_dir / "prediction_parts" / args.name / f"{args.month}.parquet"
    df = pd.read_parquet(part)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = add_prediction_views(df, "pred")
    pred_cols = ["pred", "pred_xsz", "pred_xrank"]
    out = {
        "name": args.name,
        "month": args.month,
        "rows": int(len(df)),
        "label_rows": int(df["label"].notna().sum()),
        "pred_cols": {col: stats(df, col) for col in pred_cols},
    }
    stat_dir = cfg.reports_dir / "prediction_stats"
    stat_dir.mkdir(parents=True, exist_ok=True)
    (stat_dir / f"{args.name}_{args.month}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    bt = long_short_backtest(df, pred_col="pred_xrank")
    bt_dir = cfg.output_dir / "backtest_parts" / args.name
    bt_dir.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(bt_dir / f"{args.month}.parquet", index=False)
    print(f"[summarize-month] {args.name} {args.month} rows={len(df)}")


if __name__ == "__main__":
    main()
