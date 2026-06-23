#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.metrics import summarize_backtest


def ic_from_totals(total: dict[str, float | int]) -> float:
    denom = math.sqrt(max(float(total["xx"]) * float(total["yy"]), 1e-30))
    return float(total["xy"]) / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    stat_dir = cfg.reports_dir / "prediction_stats"
    files = sorted(stat_dir.glob(f"{args.name}_*.json"))
    if not files:
        raise FileNotFoundError(f"no stats files for {args.name}")

    pred_cols = ["pred", "pred_xsz", "pred_xrank"]
    totals = {col: {"xy": 0.0, "xx": 0.0, "yy": 0.0, "count": 0} for col in pred_cols}
    monthly_rows = []
    rows_total = 0
    label_rows = 0
    for path in files:
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows_total += int(obj["rows"])
        label_rows += int(obj["label_rows"])
        row = {"month": obj["month"]}
        for col in pred_cols:
            s = obj["pred_cols"][col]
            row[col] = float(s["ic"])
            for key in totals[col]:
                totals[col][key] += s[key]
        monthly_rows.append(row)

    monthly = pd.DataFrame(monthly_rows).sort_values("month")
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
                "total_ic": ic_from_totals(total),
                "monthly_mean": float(vals.mean()),
                "monthly_std": float(vals.std()),
                "monthly_ir": float(vals.mean() / vals.std()) if float(vals.std()) > 0 else float("nan"),
                "ic_2020": ic_from_totals(total),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.insert(0, "name", args.name)
    summary.to_csv(cfg.reports_dir / f"baseline_summary_{args.name}.csv", index=False)

    bt_dir = cfg.output_dir / "backtest_parts" / args.name
    bt_files = sorted(bt_dir.glob("*.parquet"))
    bt = pd.concat([pd.read_parquet(path) for path in bt_files], ignore_index=True)
    bt.to_csv(cfg.output_dir / f"backtest_{args.name}.csv", index=False)
    (cfg.reports_dir / f"backtest_{args.name}.json").write_text(
        json.dumps(summarize_backtest(bt), indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
