#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.expressions import compute_expression_array_from_inputs, load_expression_table, precompute_expression_inputs
from fu_alpha_research.factor_store import FactorStore


def sample_months(store: FactorStore, start: str, end: str, rows_per_month: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    months = store.available_months(start, end)
    out: list[tuple[str, np.ndarray]] = []
    for month in months:
        meta = store.read_month(month, columns=[], sort=False)[["datetime", "label"]]
        valid = meta["label"].notna()
        counts = meta.loc[valid].groupby("datetime", sort=False).size()
        if rows_per_month and int(counts.sum()) > rows_per_month:
            avg_cross_section = max(float(counts.median()), 1.0)
            n_times = max(1, int(np.ceil(rows_per_month / avg_cross_section)))
            n_times = min(n_times, len(counts))
            chosen_times = set(rng.choice(counts.index.to_numpy(), size=n_times, replace=False))
            chosen = np.flatnonzero(valid.to_numpy() & meta["datetime"].isin(chosen_times).to_numpy())
        else:
            chosen = np.flatnonzero(valid.to_numpy())
        out.append((month, np.sort(chosen)))
    return out


def read_sample_frame(store: FactorStore, months_and_idx: list[tuple[str, np.ndarray]], columns: list[str]) -> pd.DataFrame:
    frames = []
    for month, idx in months_and_idx:
        if len(idx) == 0:
            continue
        df = store.read_month(month, columns=columns, sort=False).iloc[idx].copy()
        df["month"] = month
        frames.append(df)
        print(f"[prefilter] sampled {month} rows={len(df)} columns={len(df.columns)}", flush=True)
    if not frames:
        raise RuntimeError("empty sample")
    return pd.concat(frames, ignore_index=True)


def block_stats(values: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(label, dtype=np.float64)
    y_ok = np.isfinite(y)
    y0 = np.where(y_ok, y, 0.0)
    y2 = y0 * y0
    valid = np.isfinite(values) & y_ok[:, None]
    x0 = np.where(valid, values, 0.0).astype(np.float64, copy=False)
    xy = x0.T @ y0
    xx = np.sum(x0 * x0, axis=0)
    yy = valid.astype(np.float64).T @ y2
    count = np.sum(valid, axis=0).astype(np.int64)
    mean = np.divide(np.sum(x0, axis=0), np.maximum(count, 1), where=count > 0)
    var = np.divide(xx, np.maximum(count, 1), where=count > 0) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    return xy, xx, yy, count, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--sample-start", default=None)
    parser.add_argument("--sample-end", default=None)
    parser.add_argument("--rows-per-month", type=int, default=1200)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--top-n", type=int, default=30000)
    parser.add_argument("--min-coverage", type=float, default=0.70)
    parser.add_argument("--min-std", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-candidates", default="new_factor_candidates_prefilter.csv")
    parser.add_argument("--output-scores", default="new_factor_scores_prefilter.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cand_path = Path(args.candidates) if args.candidates else cfg.reports_dir / "new_factor_candidates.csv"
    if not cand_path.is_absolute():
        cand_path = cfg.reports_dir / cand_path
    candidates = load_expression_table(cand_path)
    store = FactorStore(cfg)
    deps = sorted(set(candidates["left"]).union(candidates["right"]))
    months_and_idx = sample_months(
        store,
        args.sample_start or cfg.is_start,
        args.sample_end or cfg.is_end,
        args.rows_per_month,
        args.seed,
    )
    sample = read_sample_frame(store, months_and_idx, deps)
    label = sample["label"].to_numpy(np.float64, copy=False)
    ranks, zscores = precompute_expression_inputs(sample, candidates)

    rows = []
    for start in range(0, len(candidates), args.block_size):
        end = min(start + args.block_size, len(candidates))
        block = candidates.iloc[start:end].reset_index(drop=True)
        values = compute_expression_array_from_inputs(ranks, zscores, block)
        xy, xx, yy, count, std = block_stats(values, label)
        denom = np.sqrt(np.maximum(xx * yy, 1e-30))
        ic = xy / denom
        ic[count < 2] = np.nan
        rows.append(
            pd.DataFrame(
                {
                    "name": block["name"].tolist(),
                    "sample_ic": ic,
                    "sample_n": count,
                    "sample_coverage": count / max(float(len(label)), 1.0),
                    "sample_std": std,
                }
            )
        )
        print(f"[prefilter] evaluated {end}/{len(candidates)}", flush=True)

    scores = pd.concat(rows, ignore_index=True)
    scores["abs_sample_ic"] = scores["sample_ic"].abs()
    eligible = scores[(scores["sample_coverage"] >= args.min_coverage) & (scores["sample_std"] > args.min_std)].copy()
    eligible = eligible.sort_values(["abs_sample_ic", "sample_coverage"], ascending=False)
    selected_scores = eligible.head(args.top_n).copy()
    selected = candidates.merge(selected_scores[["name"]], on="name", how="inner")
    selected = selected.merge(selected_scores[["name", "abs_sample_ic"]], on="name", how="left")
    selected = selected.sort_values("abs_sample_ic", ascending=False).drop(columns="abs_sample_ic")

    out_candidates = cfg.reports_dir / args.output_candidates
    out_scores = cfg.reports_dir / args.output_scores
    selected[["name", "op", "left", "right", "formula"]].to_csv(out_candidates, index=False)
    full_scores = scores.rename(
        columns={
            "sample_ic": "is_ic",
            "sample_n": "is_n",
            "sample_coverage": "is_coverage_proxy",
        }
    )
    full_scores["oos_ic"] = full_scores["is_ic"]
    full_scores["oos_n"] = full_scores["is_n"]
    full_scores["oos_coverage_proxy"] = full_scores["is_coverage_proxy"]
    full_scores["same_sign"] = True
    full_scores["abs_is_ic"] = full_scores["is_ic"].abs()
    full_scores["abs_oos_ic"] = full_scores["oos_ic"].abs()
    full_scores["effective"] = full_scores["abs_is_ic"].ge(0.001)
    full_scores.to_csv(out_scores, index=False)
    summary = {
        "candidates": int(len(candidates)),
        "eligible": int(len(eligible)),
        "selected": int(len(selected)),
        "top_n": int(args.top_n),
        "min_coverage": float(args.min_coverage),
        "min_std": float(args.min_std),
        "rows": int(len(sample)),
        "sample_start": args.sample_start or cfg.is_start,
        "sample_end": args.sample_end or cfg.is_end,
        "output_candidates": str(out_candidates),
        "output_scores": str(out_scores),
        "max_abs_sample_ic": float(selected_scores["abs_sample_ic"].max()) if len(selected_scores) else float("nan"),
        "min_abs_sample_ic": float(selected_scores["abs_sample_ic"].min()) if len(selected_scores) else float("nan"),
    }
    (cfg.reports_dir / "new_factor_prefilter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
