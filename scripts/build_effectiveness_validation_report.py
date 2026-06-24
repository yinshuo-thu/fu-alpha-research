#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.feature_matrix import read_feature_list, write_feature_list


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(rows)


def fmt(x: float) -> str:
    return f"{float(x):.8f}"


def load_lgbm(validation_dir: Path, month: str, num_shards: int) -> pd.DataFrame:
    full_path = validation_dir / f"lgbm_shuffle_{month}.csv"
    shards = sorted(validation_dir.glob(f"lgbm_shuffle_{month}_shard??of{num_shards:02d}.csv"))
    if shards:
        df = pd.concat([pd.read_csv(path) for path in shards], ignore_index=True)
        df.to_csv(full_path, index=False)
        return df
    if full_path.exists():
        return pd.read_csv(full_path)
    raise FileNotFoundError(f"missing LightGBM validation outputs in {validation_dir}")


def summarize_result(df: pd.DataFrame, model: str, new_names: set[str]) -> dict[str, object]:
    retained = df[df["retained"].astype(bool)].copy()
    removed = df[~df["retained"].astype(bool)].copy()
    retained_new = sorted(set(retained["factor"]) & new_names)
    removed_new = sorted(set(removed["factor"]) & new_names)
    return {
        "model": model,
        "base_ic": float(df["base_ic"].iloc[0]),
        "features": int(len(df)),
        "retained": int(len(retained)),
        "removed": int(len(removed)),
        "new_factor_pool": int(len(new_names)),
        "retained_new_factors": int(len(retained_new)),
        "removed_new_factors": int(len(removed_new)),
        "retained_new_factor_names": retained_new,
        "removed_new_factor_names": removed_new,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--month", default="2020-01")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--feature-file", default="model_feature_sets/new_all1244.txt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    validation_dir = cfg.reports_dir / "effectiveness_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    report_dir = cfg.reports_dir.parent

    feature_file = Path(args.feature_file)
    if not feature_file.is_absolute():
        feature_file = cfg.output_dir / feature_file
    feature_order = read_feature_list(feature_file)
    order = {name: i for i, name in enumerate(feature_order)}
    new_df = pd.read_csv(cfg.output_dir / "expression_sets" / "new100.csv")
    new_names = set(new_df["name"])

    ridge = pd.read_csv(validation_dir / f"ridge_leave_one_{args.month}.csv")
    lgbm = load_lgbm(validation_dir, args.month, args.num_shards)
    lgbm["__order"] = lgbm["factor"].map(order)
    lgbm = lgbm.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    lgbm.to_csv(validation_dir / f"lgbm_shuffle_{args.month}.csv", index=False)

    ridge_summary = summarize_result(ridge, "ridge", new_names)
    lgbm_summary = summarize_result(lgbm, "lightgbm", new_names)
    overlap_new = sorted(set(ridge_summary["retained_new_factor_names"]) & set(lgbm_summary["retained_new_factor_names"]))

    write_feature_list(
        validation_dir / f"lgbm_retained_{args.month}.txt",
        lgbm[lgbm["retained"].astype(bool)]["factor"].tolist(),
    )
    write_feature_list(
        validation_dir / f"lgbm_removed_{args.month}.txt",
        lgbm[~lgbm["retained"].astype(bool)]["factor"].tolist(),
    )

    combined = {
        "month": args.month,
        "feature_universe": "new_all1244",
        "ridge": ridge_summary,
        "lightgbm": lgbm_summary,
        "retained_new_overlap_count": len(overlap_new),
        "retained_new_overlap_names": overlap_new,
    }
    (validation_dir / f"factor_effectiveness_{args.month}_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary_table = pd.DataFrame(
        [
            {
                "model": "Ridge",
                "validation": "leave-one-factor-out retrain, remove if IC does not decline",
                "base_ic_2020_01": fmt(ridge_summary["base_ic"]),
                "retained": ridge_summary["retained"],
                "removed": ridge_summary["removed"],
                "retained_from_new100": ridge_summary["retained_new_factors"],
                "removed_from_new100": ridge_summary["removed_new_factors"],
            },
            {
                "model": "LightGBM",
                "validation": "single-factor standard-normal replacement, remove if IC does not decline",
                "base_ic_2020_01": fmt(lgbm_summary["base_ic"]),
                "retained": lgbm_summary["retained"],
                "removed": lgbm_summary["removed"],
                "retained_from_new100": lgbm_summary["retained_new_factors"],
                "removed_from_new100": lgbm_summary["removed_new_factors"],
            },
        ]
    )

    new_overlap_table = pd.DataFrame({"factor": overlap_new})
    if new_overlap_table.empty:
        new_overlap_table = pd.DataFrame({"factor": ["<none>"]})

    ridge_top_new = (
        ridge[ridge["factor"].isin(new_names)]
        .sort_values("delta_ic", ascending=False)
        .head(15)[["factor", "drop_ic", "delta_ic", "retained"]]
        .copy()
    )
    lgbm_top_new = (
        lgbm[lgbm["factor"].isin(new_names)]
        .sort_values("delta_ic", ascending=False)
        .head(15)[["factor", "shuffled_ic", "delta_ic", "retained", "split_importance"]]
        .copy()
    )
    for frame in [ridge_top_new, lgbm_top_new]:
        for col in ["drop_ic", "shuffled_ic", "delta_ic"]:
            if col in frame:
                frame[col] = frame[col].map(fmt)

    report = f"""# Factor Effectiveness Validation

This report validates the `new_all1244` universe on `{args.month}` using the two model-specific tests requested:

- Ridge: exact leave-one-factor-out retraining. A factor is retained only when removing it lowers `{args.month}` OOS `pred_xsz` IC.
- LightGBM: single-factor standard-normal replacement on `{args.month}`. A factor is retained only when replacing it lowers OOS `pred_xsz` IC.

## Summary

{md_table(summary_table)}

Among the 100 newly mined expression factors, Ridge retains {ridge_summary["retained_new_factors"]} and LightGBM retains {lgbm_summary["retained_new_factors"]}. Their retained-new-factor overlap is {len(overlap_new)}.

## Retained New-Factor Overlap

{md_table(new_overlap_table)}

## Strongest New Factors Under Ridge Test

{md_table(ridge_top_new)}

## Strongest New Factors Under LightGBM Shuffle Test

{md_table(lgbm_top_new)}

## Artifacts

- Ridge full validation CSV: `reports/generated/effectiveness_validation/ridge_leave_one_{args.month}.csv`
- LightGBM full validation CSV: `reports/generated/effectiveness_validation/lgbm_shuffle_{args.month}.csv`
- Combined summary JSON: `reports/generated/effectiveness_validation/factor_effectiveness_{args.month}_summary.json`
- Ridge retained list: `reports/generated/effectiveness_validation/ridge_retained_{args.month}.txt`
- LightGBM retained list: `reports/generated/effectiveness_validation/lgbm_retained_{args.month}.txt`
"""
    out = report_dir / "factor_effectiveness_validation.md"
    out.write_text(report, encoding="utf-8")
    print(f"[effectiveness-report] wrote {out}")
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
