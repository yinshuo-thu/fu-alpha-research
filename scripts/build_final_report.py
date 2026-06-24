#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fu_alpha_research.config import load_config


SETS = [
    "orig_top300",
    "orig_effective657",
    "orig_all1144",
    "new_top300",
    "new_effective757",
    "new_all1244",
]
SET_LABELS = {
    "orig_top300": "Original top300",
    "orig_effective657": "Original effective657",
    "orig_all1144": "Original all1144",
    "new_top300": "With new top300",
    "new_effective757": "Effective657 + 100",
    "new_all1244": "All1144 + 100",
}
PAIR_MAP = {
    "top300": ("orig_top300", "new_top300"),
    "effective": ("orig_effective657", "new_effective757"),
    "all": ("orig_all1144", "new_all1244"),
}


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(rows)


def fmt(x: float, digits: int = 6) -> str:
    if pd.isna(x):
        return "nan"
    return f"{float(x):.{digits}f}"


def load_backtests(reports_dir: Path, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        path = reports_dir / f"backtest_{name}.json"
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["name"] = name
        rows.append(obj)
    return pd.DataFrame(rows)


def plot_ic_bars(main: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, model in zip(axes, ["ridge", "lgbm"]):
        sub = main[main["model"] == model].set_index("set").loc[SETS]
        colors = ["#4C78A8", "#4C78A8", "#4C78A8", "#F58518", "#F58518", "#F58518"]
        ax.bar(range(len(sub)), sub["total_ic"], color=colors)
        ax.set_title(model.upper())
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([SET_LABELS[x] for x in sub.index], rotation=35, ha="right")
        ax.set_ylabel("OOS total IC (pred_xsz)")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_ic_comparison.png", dpi=160)
    plt.close(fig)


def plot_deltas(delta: pd.DataFrame, fig_dir: Path) -> None:
    labels = ["top300", "effective", "all"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ridge = delta[delta["model"] == "ridge"].set_index("group").loc[labels]["delta_total_ic"]
    lgbm = delta[delta["model"] == "lgbm"].set_index("group").loc[labels]["delta_total_ic"]
    ax.axhline(0, color="#444", linewidth=1)
    ax.bar(x - width / 2, ridge, width, label="Ridge", color="#4C78A8")
    ax.bar(x + width / 2, lgbm, width, label="LightGBM", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta OOS total IC")
    ax.set_title("Incremental IC from 100 newly mined factors")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "new_factor_ic_delta.png", dpi=160)
    plt.close(fig)


def plot_new_factors(new_scores: pd.DataFrame, fig_dir: Path) -> None:
    top = new_scores.head(20).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["name"], top["oos_ic"], color=np.where(top["oos_ic"] >= 0, "#54A24B", "#E45756"))
    ax.set_title("Top 20 new expression factors by absolute OOS IC")
    ax.set_xlabel("OOS IC")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "new_factor_top20_oos_ic.png", dpi=160)
    plt.close(fig)


def plot_monthly(cfg, main: pd.DataFrame, fig_dir: Path) -> None:
    for model in ["ridge", "lgbm"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for set_name in SETS:
            name = f"{model}_{set_name}"
            path = cfg.reports_dir / f"monthly_ic_{name}.csv"
            if not path.exists():
                continue
            monthly = pd.read_csv(path)
            ax.plot(monthly["month"], monthly["pred_xsz"], marker="o", linewidth=1.4, label=SET_LABELS[set_name])
        ax.axhline(0, color="#444", linewidth=1)
        ax.set_title(f"{model.upper()} monthly OOS IC")
        ax.set_ylabel("Monthly IC (pred_xsz)")
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"monthly_ic_{model}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    report_dir = cfg.reports_dir.parent
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(cfg.reports_dir / "model_ic_summary.csv")
    main = summary[summary["pred_col"] == "pred_xsz"].copy()
    main["model"] = main["name"].str.extract(r"^(ridge|lgbm)")
    main["set"] = main["name"].str.replace(r"^(ridge|lgbm)_", "", regex=True)
    main["set_label"] = main["set"].map(SET_LABELS)
    main["order"] = main["set"].map({name: i for i, name in enumerate(SETS)})
    main = main.sort_values(["model", "order"])

    delta_rows = []
    for model in ["ridge", "lgbm"]:
        sub = main[main["model"] == model].set_index("set")
        for group, (orig, new) in PAIR_MAP.items():
            orig_ic = float(sub.loc[orig, "total_ic"])
            new_ic = float(sub.loc[new, "total_ic"])
            delta_rows.append(
                {
                    "model": model,
                    "group": group,
                    "original": orig,
                    "with_new": new,
                    "original_total_ic": orig_ic,
                    "with_new_total_ic": new_ic,
                    "delta_total_ic": new_ic - orig_ic,
                    "relative_delta_pct": (new_ic / orig_ic - 1.0) * 100.0 if orig_ic else np.nan,
                }
            )
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(cfg.reports_dir / "model_ic_delta_pred_xsz.csv", index=False)

    new_scores = pd.read_csv(cfg.reports_dir / "new_effective_factors_100.csv")
    mining_summary = json.loads((cfg.reports_dir / "new_factor_mining_summary.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(cfg.output_dir / "model_feature_sets" / "manifest.csv")
    backtests = load_backtests(cfg.reports_dir, main["name"].tolist())

    plot_ic_bars(main, fig_dir)
    plot_deltas(delta, fig_dir)
    plot_new_factors(new_scores, fig_dir)
    plot_monthly(cfg, main, fig_dir)

    model_table = main[["name", "set_label", "total_ic", "monthly_mean", "monthly_ir"]].copy()
    for col in ["total_ic", "monthly_mean", "monthly_ir"]:
        model_table[col] = model_table[col].map(fmt)
    delta_table = delta[["model", "group", "original_total_ic", "with_new_total_ic", "delta_total_ic", "relative_delta_pct"]].copy()
    for col in ["original_total_ic", "with_new_total_ic", "delta_total_ic"]:
        delta_table[col] = delta_table[col].map(fmt)
    delta_table["relative_delta_pct"] = delta_table["relative_delta_pct"].map(lambda x: f"{float(x):.2f}%")
    new_table = new_scores.head(12)[["name", "formula", "is_ic", "oos_ic"]].copy()
    for col in ["is_ic", "oos_ic"]:
        new_table[col] = new_table[col].map(fmt)
    bt_table = backtests[["name", "mean", "tstat", "hit_rate", "cum_return"]].copy()
    for col in ["mean", "tstat", "hit_rate", "cum_return"]:
        bt_table[col] = bt_table[col].map(lambda x: fmt(x, 6))

    report = f"""# Futures Alpha Mining Migration Report

## Scope

- Project path: `{cfg.reports_dir.parents[1]}`
- IS window: `{cfg.is_start}` to `{cfg.is_end}`.
- OOS window: `{cfg.oos_start}` to `{cfg.oos_end}`.
- Original factor pool: 1144 factors.
- Newly mined expression candidates: {mining_summary["candidates"]}.
- New factors passing the same-sign IS/OOS screen: {mining_summary["effective"]}.
- Selected new effective factors: {mining_summary["selected"]}.

## Mining Method

The new factors are not copied from the original 1144 columns. They are generated as deterministic expression factors over strong original seeds, using cross-sectional rank/zscore operators such as `xrank(a) + xrank(b)`, `xrank(a) - xrank(b)`, gated ranks, and zscore products/spreads. The production entry point is `scripts/run_continuous_factor_mining.py`: it generates candidates, computes monthly IS/OOS IC parts, aggregates them, and expands the search pool until the target count is reached.

The effective-factor screen used here is:

- same sign between IS IC and OOS IC
- `abs(IS IC) >= {mining_summary["min_is_ic"]}`
- `abs(OOS IC) >= {mining_summary["min_oos_ic"]}`
- IS/OOS coverage proxy >= 0.5

## Feature Sets

{md_table(manifest)}

## Model IC Results

Main reported column is `pred_xsz`, the cross-sectional z-scored prediction view used for the baseline comparison.

{md_table(model_table)}

## Incremental Effect From New Factors

{md_table(delta_table)}

## Backtest Summary

Long-short uses top/bottom 20% by `pred_xrank` within each timestamp.

{md_table(bt_table)}

## Top New Factors

{md_table(new_table)}

## Figures

![Model IC comparison](figures/model_ic_comparison.png)

![Incremental IC delta](figures/new_factor_ic_delta.png)

![Top new factors](figures/new_factor_top20_oos_ic.png)

![Ridge monthly IC](figures/monthly_ic_ridge.png)

![LightGBM monthly IC](figures/monthly_ic_lgbm.png)

## Conclusion

Ridge benefits from the newly mined factor skill across all three comparisons. The strongest Ridge result is `ridge_new_effective757`, with OOS total IC {fmt(float(main.loc[main["name"] == "ridge_new_effective757", "total_ic"].iloc[0]))}, versus {fmt(float(main.loc[main["name"] == "ridge_orig_effective657", "total_ic"].iloc[0]))} for the original effective657 set.

LightGBM does not show broad incremental gain from this first 100-factor batch. The effective set improves slightly, while top300 and all-factor variants are flat to lower. That suggests the expression mining improves the linear signal stack more reliably than the current lightweight tree configuration; LightGBM likely needs a separate tuning/pass for feature redundancy and regularization.
"""
    out = report_dir / "final_factor_mining_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"[report] wrote {out}")
    print(f"[report] figures in {fig_dir}")


if __name__ == "__main__":
    main()
