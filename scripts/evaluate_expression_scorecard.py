#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.expressions import compute_expression_block, load_expression_table
from fu_alpha_research.factor_scorecard import (
    ScorecardThresholds,
    composite_score,
    decision_flags,
    final_grade,
    product_code,
    rank_corr,
    residual_ic,
    safe_corr,
    signed_bucket_stats,
    standardize_matrix,
    turnover_proxy,
)
from fu_alpha_research.factor_store import FactorStore
from fu_alpha_research.feature_matrix import read_feature_list


def resolve_output_path(base: Path, value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_ic_scores(cfg, path_arg: str | None) -> pd.DataFrame:
    path = resolve_output_path(cfg.reports_dir, path_arg, cfg.reports_dir / "new_factor_scores.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run scripts/aggregate_expression_factors.py first to build the IC prefilter table"
        )
    scores = pd.read_csv(path)
    if "factor" in scores.columns and "name" not in scores.columns:
        scores = scores.rename(columns={"factor": "name"})
    for col in ("is_ic", "oos_ic", "same_sign", "abs_is_ic", "abs_oos_ic"):
        if col not in scores.columns:
            if col == "same_sign":
                scores[col] = np.sign(scores["is_ic"]) == np.sign(scores["oos_ic"])
            elif col == "abs_is_ic":
                scores[col] = scores["is_ic"].abs()
            elif col == "abs_oos_ic":
                scores[col] = scores["oos_ic"].abs()
            else:
                scores[col] = np.nan
    return scores


def sample_months(store: FactorStore, start: str, end: str, rows_per_month: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    months = store.available_months(start, end)
    out: list[tuple[str, np.ndarray]] = []
    for month in months:
        # Sample whole timestamps so cross-sectional bucket/rank tests remain meaningful.
        meta = store.read_month(month, columns=[], sort=False)[["symbol", "datetime", "label"]]
        valid = meta["label"].notna()
        valid_meta = meta.loc[valid, ["datetime"]].copy()
        if valid_meta.empty:
            out.append((month, np.array([], dtype=np.int64)))
            continue
        counts = valid_meta.groupby("datetime", sort=False).size()
        if rows_per_month and int(counts.sum()) > rows_per_month:
            avg_cross_section = max(float(counts.median()), 1.0)
            n_times = max(1, int(np.ceil(rows_per_month / avg_cross_section)))
            n_times = min(n_times, len(counts))
            chosen_times = set(rng.choice(counts.index.to_numpy(), size=n_times, replace=False))
            chosen = np.flatnonzero(valid.to_numpy() & meta["datetime"].isin(chosen_times).to_numpy())
        else:
            chosen = np.flatnonzero(valid.to_numpy())
        chosen = np.sort(chosen)
        out.append((month, chosen))
    return out


def read_sample_frame(
    store: FactorStore,
    months_and_idx: list[tuple[str, np.ndarray]],
    columns: list[str],
) -> pd.DataFrame:
    frames = []
    for month, idx in months_and_idx:
        df = store.read_month(month, columns=columns, sort=False)
        df = df.iloc[idx].copy()
        df["month"] = month
        frames.append(df)
        print(f"[scorecard] sampled {month} rows={len(df)} columns={len(columns)}", flush=True)
    if not frames:
        raise RuntimeError("no sample rows available for scorecard evaluation")
    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["product"] = out["symbol"].map(product_code)
    return out


def monthly_ic(values: np.ndarray, label: np.ndarray, months: pd.Series) -> dict[str, float]:
    rows = []
    for month, idx in pd.Series(np.arange(len(months))).groupby(months, sort=True):
        loc = idx.to_numpy()
        ic = safe_corr(values[loc], label[loc])
        if np.isfinite(ic):
            rows.append((str(month), ic))
    if not rows:
        return {
            "monthly_ic_mean": np.nan,
            "monthly_ic_std": np.nan,
            "monthly_ic_hit_rate": 0.0,
            "monthly_ic_worst": np.nan,
            "monthly_ic_count": 0,
        }
    vals = np.array([x[1] for x in rows], dtype=np.float64)
    sign = np.sign(np.nanmean(vals)) or 1.0
    return {
        "monthly_ic_mean": float(np.nanmean(vals)),
        "monthly_ic_std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "monthly_ic_hit_rate": float(np.mean(np.sign(vals) == sign)),
        "monthly_ic_worst": float(vals[np.argmin(np.abs(vals))]),
        "monthly_ic_count": int(len(vals)),
    }


def grouped_ic(values: np.ndarray, label: np.ndarray, groups: pd.Series, prefix: str) -> dict[str, float]:
    vals = []
    for _group, idx in pd.Series(np.arange(len(groups))).groupby(groups, sort=True):
        loc = idx.to_numpy()
        ic = safe_corr(values[loc], label[loc])
        if np.isfinite(ic):
            vals.append(ic)
    if not vals:
        return {
            f"{prefix}_ic_mean": np.nan,
            f"{prefix}_ic_min_abs": np.nan,
            f"{prefix}_ic_hit_rate": 0.0,
            f"{prefix}_count": 0,
        }
    arr = np.asarray(vals, dtype=np.float64)
    sign = np.sign(np.nanmean(arr)) or 1.0
    return {
        f"{prefix}_ic_mean": float(np.nanmean(arr)),
        f"{prefix}_ic_min_abs": float(np.nanmin(np.abs(arr))),
        f"{prefix}_ic_hit_rate": float(np.mean(np.sign(arr) == sign)),
        f"{prefix}_count": int(len(arr)),
    }


def regime_labels(sample: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    volume = sample["volume"] if "volume" in sample.columns else pd.Series(np.nan, index=sample.index)
    try:
        liquidity = pd.qcut(volume.rank(method="first"), q=3, labels=["low_liq", "mid_liq", "high_liq"], duplicates="drop")
    except ValueError:
        liquidity = pd.Series("unknown_liq", index=sample.index)
    if {"high", "low", "close"}.issubset(sample.columns):
        vol_proxy = (sample["high"] - sample["low"]).abs() / sample["close"].abs().replace(0, np.nan)
    else:
        vol_proxy = sample["label"].abs()
    try:
        volatility = pd.qcut(vol_proxy.rank(method="first"), q=3, labels=["low_vol", "mid_vol", "high_vol"], duplicates="drop")
    except ValueError:
        volatility = pd.Series("unknown_vol", index=sample.index)
    return liquidity.astype(str), volatility.astype(str)


def block_library_correlations(
    values: pd.DataFrame,
    library_z: np.ndarray,
    library_names: list[str],
) -> dict[str, dict[str, object]]:
    if library_z.size == 0 or not library_names:
        return {
            name: {"max_abs_corr": 0.0, "max_corr": 0.0, "closest_factor": "", "compared": 0}
            for name in values.columns
        }
    block_z, block_names = standardize_matrix(values)
    corr = (block_z.astype(np.float64).T @ library_z.astype(np.float64)) / max(len(block_z), 1)
    corr[~np.isfinite(corr)] = 0.0
    out: dict[str, dict[str, object]] = {}
    for row_idx, name in enumerate(block_names):
        col = corr[row_idx]
        best = int(np.argmax(np.abs(col)))
        out[name] = {
            "max_abs_corr": float(abs(col[best])),
            "max_corr": float(col[best]),
            "closest_factor": library_names[best],
            "compared": int(len(library_names)),
        }
    return out


def evaluate_candidate_values(
    *,
    name: str,
    values: np.ndarray,
    sample: pd.DataFrame,
    label: np.ndarray,
    library_corr: dict[str, object],
    library_raw: pd.DataFrame,
) -> dict[str, object]:
    finite = np.isfinite(values)
    coverage = float(finite.mean()) if len(values) else 0.0
    clean = values[finite]
    std = float(np.nanstd(clean)) if len(clean) else np.nan
    mean = float(np.nanmean(clean)) if len(clean) else np.nan
    if len(clean) and np.isfinite(std) and std > 1e-12:
        z = (clean - mean) / std
        outlier_ratio = float(np.mean(np.abs(z) > 8.0))
    else:
        outlier_ratio = 1.0

    closest_name = str(library_corr.get("closest_factor", ""))
    closest = library_raw[closest_name].to_numpy(np.float32, copy=False) if closest_name in library_raw.columns else None
    bucket = signed_bucket_stats(values, label, sample["datetime"], buckets=5)
    liquidity, volatility = regime_labels(sample)
    row: dict[str, object] = {
        "name": name,
        "coverage": coverage,
        "mean": mean,
        "std": std,
        "outlier_ratio": outlier_ratio,
        "pearson_ic": safe_corr(values, label),
        "rank_ic": rank_corr(values, label),
        "turnover_proxy": turnover_proxy(values, sample["symbol"], sample["datetime"]),
        "max_abs_corr_to_library": float(library_corr["max_abs_corr"]),
        "max_corr_to_library": float(library_corr["max_corr"]),
        "closest_library_factor": closest_name,
        "library_factors_compared": int(library_corr["compared"]),
        "residual_ic": residual_ic(values, label, closest),
    }
    row.update(monthly_ic(values, label, sample["month"]))
    row.update(grouped_ic(values, label, sample["product"], "product"))
    row.update(grouped_ic(values, label, liquidity, "liquidity_regime"))
    row.update(grouped_ic(values, label, volatility, "volatility_regime"))
    row.update(bucket)
    return row


def select_with_candidate_correlation(
    scorecard: pd.DataFrame,
    candidates: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    target: int,
    thresholds: ScorecardThresholds,
    pool_size: int,
    max_per_op: int,
) -> pd.DataFrame:
    selected_names: list[str] = []
    selected_vectors: list[tuple[str, np.ndarray]] = []
    op_counts: dict[str, int] = {}
    ranked = scorecard[scorecard["decision_reason"].eq("pass_all")].copy()
    ranked = ranked.sort_values(["composite_score", "abs_selection_ic", "abs_is_ic"], ascending=False).head(pool_size)
    candidate_by_name = candidates.set_index("name", drop=False)

    for row in ranked.itertuples(index=False):
        if len(selected_names) >= target:
            break
        name = row.name
        if name not in candidate_by_name.index:
            continue
        expr = candidate_by_name.loc[[name]].reset_index(drop=True)
        op = str(expr.iloc[0]["op"])
        if max_per_op and op_counts.get(op, 0) >= max_per_op:
            continue
        values = compute_expression_block(sample, expr)[name].to_numpy(np.float32, copy=False)
        z, _names = standardize_matrix(values.reshape(-1, 1))
        max_peer_corr = 0.0
        for _prior_name, prior in selected_vectors:
            corr = float((z[:, 0].astype(np.float64) @ prior.astype(np.float64)) / max(len(z), 1))
            if np.isfinite(corr):
                max_peer_corr = max(max_peer_corr, abs(corr))
        if max_peer_corr > thresholds.max_abs_corr:
            scorecard.loc[scorecard["name"] == name, "candidate_peer_max_abs_corr"] = max_peer_corr
            continue
        selected_names.append(name)
        selected_vectors.append((name, z[:, 0].copy()))
        op_counts[op] = op_counts.get(op, 0) + 1
        scorecard.loc[scorecard["name"] == name, "candidate_peer_max_abs_corr"] = max_peer_corr

    scorecard["selected_multilayer"] = scorecard["name"].isin(selected_names)
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--ic-scores", default=None)
    parser.add_argument("--library-feature-file", default=None)
    parser.add_argument("--sample-start", default=None)
    parser.add_argument("--sample-end", default=None)
    parser.add_argument("--rows-per-month", type=int, default=3000)
    parser.add_argument("--block-size", type=int, default=48)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--selection-pool", type=int, default=800)
    parser.add_argument("--max-per-op", type=int, default=20)
    parser.add_argument("--write-partial", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.70)
    parser.add_argument("--min-abs-ic", type=float, default=0.001)
    parser.add_argument("--min-abs-rank-ic", type=float, default=0.0005)
    parser.add_argument("--min-monthly-hit-rate", type=float, default=0.50)
    parser.add_argument("--min-monthly-ic-count", type=int, default=6)
    parser.add_argument("--min-product-hit-rate", type=float, default=0.45)
    parser.add_argument("--min-regime-hit-rate", type=float, default=0.40)
    parser.add_argument("--min-bucket-monotonicity", type=float, default=0.25)
    parser.add_argument("--max-corr", type=float, default=0.90)
    parser.add_argument("--max-outlier-ratio", type=float, default=0.02)
    parser.add_argument("--max-turnover-proxy", type=float, default=0.85)
    parser.add_argument("--min-abs-residual-ic", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cand_path = resolve_output_path(cfg.reports_dir, args.candidates, cfg.reports_dir / "new_factor_candidates.csv")
    candidates = load_expression_table(cand_path)
    scores = load_ic_scores(cfg, args.ic_scores)
    candidates = candidates.merge(scores.drop(columns=[c for c in candidates.columns if c in scores.columns and c != "name"]), on="name", how="left")
    candidates["same_sign"] = np.sign(candidates["is_ic"]) == np.sign(candidates["oos_ic"])
    candidates["abs_is_ic"] = candidates["is_ic"].abs()
    candidates["abs_oos_ic"] = candidates["oos_ic"].abs()
    candidates = candidates.sort_values(["same_sign", "abs_oos_ic", "abs_is_ic"], ascending=False)
    if args.max_candidates:
        candidates = candidates.head(args.max_candidates).copy()

    store = FactorStore(cfg)
    library_file = Path(args.library_feature_file) if args.library_feature_file else cfg.selected_factors_path
    if not library_file.is_absolute():
        library_file = Path.cwd() / library_file
    library_features = read_feature_list(library_file)
    deps = sorted(set(candidates["left"]).union(candidates["right"]))
    read_cols = list(dict.fromkeys(deps + library_features + ["volume", "high", "low", "close"]))
    months_and_idx = sample_months(
        store,
        args.sample_start or cfg.is_start,
        args.sample_end or cfg.is_end,
        args.rows_per_month,
        args.seed,
    )
    sample = read_sample_frame(store, months_and_idx, read_cols)
    library_features = [col for col in library_features if col in sample.columns]
    library_raw = sample[library_features].copy().astype(np.float32) if library_features else pd.DataFrame(index=sample.index)
    library_z, library_names = standardize_matrix(library_raw) if library_features else (np.empty((len(sample), 0)), [])
    label = sample["label"].to_numpy(np.float64, copy=False)

    rows = []
    for start in range(0, len(candidates), args.block_size):
        end = min(start + args.block_size, len(candidates))
        block = candidates.iloc[start:end].reset_index(drop=True)
        values = compute_expression_block(sample, block)
        library_corrs = block_library_correlations(values, library_z, library_names)
        for name in block["name"]:
            metric = evaluate_candidate_values(
                name=name,
                values=values[name].to_numpy(np.float32, copy=False),
                sample=sample,
                label=label,
                library_corr=library_corrs[name],
                library_raw=library_raw,
            )
            rows.append(metric)
        print(f"[scorecard] evaluated {end}/{len(candidates)}", flush=True)

    detail = pd.DataFrame(rows)
    scorecard = candidates.merge(detail, on="name", how="left")
    scorecard["selection_ic"] = scorecard["is_ic"].where(scorecard["is_ic"].notna(), scorecard["pearson_ic"])
    scorecard["abs_selection_ic"] = scorecard["selection_ic"].abs()
    scorecard["sample_same_sign"] = np.sign(scorecard["selection_ic"]) == np.sign(scorecard["pearson_ic"])
    thresholds = ScorecardThresholds(
        min_coverage=args.min_coverage,
        min_abs_ic=args.min_abs_ic,
        min_abs_rank_ic=args.min_abs_rank_ic,
        min_monthly_hit_rate=args.min_monthly_hit_rate,
        min_monthly_ic_count=args.min_monthly_ic_count,
        min_product_hit_rate=args.min_product_hit_rate,
        min_regime_hit_rate=args.min_regime_hit_rate,
        min_bucket_monotonicity=args.min_bucket_monotonicity,
        max_abs_corr=args.max_corr,
        max_outlier_ratio=args.max_outlier_ratio,
        max_turnover_proxy=args.max_turnover_proxy,
        min_abs_residual_ic=args.min_abs_residual_ic,
    )
    flag_frame = scorecard.apply(lambda row: decision_flags(row, thresholds), axis=1, result_type="expand")
    scorecard = pd.concat([scorecard, flag_frame], axis=1)
    scorecard["final_grade"] = scorecard.apply(lambda row: final_grade(row, thresholds), axis=1)
    scorecard["composite_score"] = scorecard.apply(lambda row: composite_score(row, thresholds), axis=1)
    scorecard["candidate_peer_max_abs_corr"] = np.nan
    scorecard = select_with_candidate_correlation(
        scorecard,
        candidates,
        sample,
        target=args.target,
        thresholds=thresholds,
        pool_size=args.selection_pool,
        max_per_op=args.max_per_op,
    )
    flag_cols = [
        "pass_data_quality",
        "pass_ic",
        "pass_bucket",
        "pass_regime",
        "pass_trading",
        "pass_incremental",
        "pass_robustness",
        "decision_reason",
    ]
    scorecard = scorecard.drop(columns=[col for col in flag_cols if col in scorecard.columns])
    flag_frame = scorecard.apply(lambda row: decision_flags(row, thresholds), axis=1, result_type="expand")
    scorecard = pd.concat([scorecard, flag_frame], axis=1)
    scorecard["final_grade"] = scorecard.apply(lambda row: final_grade(row, thresholds), axis=1)
    scorecard["composite_score"] = scorecard.apply(lambda row: composite_score(row, thresholds), axis=1)
    scorecard = scorecard.sort_values(["selected_multilayer", "final_grade", "composite_score"], ascending=[False, True, False])

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    scorecard_file = cfg.reports_dir / "new_factor_scorecard.csv"
    selected_file = cfg.reports_dir / "new_effective_factors_scorecard.csv"
    legacy_alias_file = cfg.reports_dir / "new_effective_factors_100.csv"
    scorecard.to_csv(scorecard_file, index=False)
    selected = scorecard[scorecard["selected_multilayer"]].head(args.target).copy()
    selected.to_csv(selected_file, index=False)
    final_artifacts_written = False
    if len(selected) >= args.target or args.write_partial:
        selected.to_csv(legacy_alias_file, index=False)
        expr_dir = cfg.output_dir / "expression_sets"
        expr_dir.mkdir(parents=True, exist_ok=True)
        selected[["name", "op", "left", "right", "formula"]].to_csv(expr_dir / "new100.csv", index=False)
        final_artifacts_written = True

    summary = {
        "method": "multi_layer_scorecard_low_correlation",
        "candidates_evaluated": int(len(scorecard)),
        "selected": int(len(selected)),
        "target": args.target,
        "sample_rows": int(len(sample)),
        "sample_start": args.sample_start or cfg.is_start,
        "sample_end": args.sample_end or cfg.is_end,
        "library_feature_file": str(library_file),
        "library_features_compared": int(len(library_names)),
        "thresholds": thresholds.__dict__,
        "grade_counts": {str(k): int(v) for k, v in scorecard["final_grade"].value_counts().to_dict().items()},
        "gate_pass_counts": {
            col: int(scorecard[col].sum())
            for col in [
                "pass_data_quality",
                "pass_ic",
                "pass_bucket",
                "pass_regime",
                "pass_trading",
                "pass_incremental",
                "pass_robustness",
            ]
            if col in scorecard.columns
        },
        "selected_op_counts": {str(k): int(v) for k, v in selected["op"].value_counts().to_dict().items()},
        "pass_all_candidates": int(scorecard["decision_reason"].eq("pass_all").sum()),
        "final_artifacts_written": final_artifacts_written,
        "write_partial": bool(args.write_partial),
    }
    (cfg.reports_dir / "new_factor_scorecard_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(selected):
        cols = [
            "name",
            "final_grade",
            "composite_score",
            "selection_ic",
            "oos_ic",
            "rank_ic",
            "bucket_monotonicity",
            "max_abs_corr_to_library",
        ]
        print(selected[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
