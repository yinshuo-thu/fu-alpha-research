#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.factor_store import FactorStore
from fu_alpha_research.metrics import add_prediction_views, compute_ic


def pred_stats(df: pd.DataFrame, pred_col: str) -> dict[str, float | int]:
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


def ic_from_totals(total: dict[str, float | int]) -> float:
    denom = math.sqrt(max(float(total["xx"]) * float(total["yy"]), 1e-30))
    return float(total["xy"]) / denom


def summarize_model(cfg, name: str, months: list[str], force: bool) -> pd.DataFrame:
    pred_cols = ["pred", "pred_xsz", "pred_xrank"]
    stat_dir = cfg.reports_dir / "prediction_stats_ic"
    stat_dir.mkdir(parents=True, exist_ok=True)
    totals = {col: {"xy": 0.0, "xx": 0.0, "yy": 0.0, "count": 0} for col in pred_cols}
    monthly_rows = []
    rows_total = 0
    label_rows = 0

    for month in months:
        stat_file = stat_dir / f"{name}_{month}.json"
        if stat_file.exists() and not force:
            obj = json.loads(stat_file.read_text(encoding="utf-8"))
        else:
            part = cfg.output_dir / "prediction_parts" / name / f"{month}.parquet"
            if not part.exists():
                raise FileNotFoundError(part)
            df = pd.read_parquet(part)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = add_prediction_views(df, "pred")
            obj = {
                "name": name,
                "month": month,
                "rows": int(len(df)),
                "label_rows": int(df["label"].notna().sum()),
                "pred_cols": {col: pred_stats(df, col) for col in pred_cols},
            }
            stat_file.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        rows_total += int(obj["rows"])
        label_rows += int(obj["label_rows"])
        row = {"name": name, "month": obj["month"]}
        for col in pred_cols:
            stats = obj["pred_cols"][col]
            row[col] = float(stats["ic"])
            for key in totals[col]:
                totals[col][key] += stats[key]
        monthly_rows.append(row)
        print(f"[summarize-ic] {name} {month}", flush=True)

    monthly = pd.DataFrame(monthly_rows).sort_values("month")
    monthly.to_csv(cfg.reports_dir / f"monthly_ic_{name}.csv", index=False)
    summary_rows = []
    for col in pred_cols:
        vals = monthly[col]
        total = totals[col]
        std = float(vals.std())
        summary_rows.append(
            {
                "name": name,
                "pred_col": col,
                "rows": rows_total,
                "label_rows": label_rows,
                "coverage": float(total["count"]) / max(float(rows_total), 1.0),
                "total_ic": ic_from_totals(total),
                "monthly_mean": float(vals.mean()),
                "monthly_std": std,
                "monthly_ir": float(vals.mean() / std) if std > 0 else float("nan"),
                "ic_2020": ic_from_totals(total),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(cfg.reports_dir / f"baseline_summary_{name}.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--models", default="ridge,lgbm")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = Path(args.manifest) if args.manifest else cfg.output_dir / "model_feature_sets" / "manifest.csv"
    if not manifest_path.is_absolute():
        manifest_path = cfg.output_dir / manifest_path
    manifest = pd.read_csv(manifest_path)
    months = FactorStore(cfg).available_months(args.start or cfg.oos_start, args.end or cfg.oos_end)
    prefixes = ["lgbm" if x.strip() in {"lgbm", "lightgbm"} else x.strip() for x in args.models.split(",") if x.strip()]
    summaries = []
    for prefix in prefixes:
        for set_name in manifest["set"]:
            summaries.append(summarize_model(cfg, f"{prefix}_{set_name}", months, args.force))
    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(cfg.reports_dir / "model_ic_summary.csv", index=False)
    print(f"[summarize-ic] wrote {cfg.reports_dir / 'model_ic_summary.csv'} rows={len(all_summary)}")


if __name__ == "__main__":
    main()
