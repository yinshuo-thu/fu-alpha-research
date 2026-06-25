from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


def product_code(symbol: object) -> str:
    text = str(symbol)
    match = re.match(r"([A-Za-z]+)", text)
    return match.group(1).upper() if match else text.upper()


def safe_corr(left: np.ndarray | pd.Series, right: np.ndarray | pd.Series) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask] - float(np.nanmean(x[mask]))
    y = y[mask] - float(np.nanmean(y[mask]))
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom <= 1e-18:
        return float("nan")
    return float(np.dot(x, y) / denom)


def rank_corr(left: np.ndarray | pd.Series, right: np.ndarray | pd.Series) -> float:
    frame = pd.DataFrame({"x": left, "y": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return float("nan")
    return safe_corr(frame["x"].rank(pct=True).to_numpy(), frame["y"].rank(pct=True).to_numpy())


def clamp01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def signed_bucket_stats(
    values: np.ndarray,
    label: np.ndarray,
    datetimes: pd.Series,
    buckets: int = 5,
) -> dict[str, float]:
    frame = pd.DataFrame({"value": values, "label": label, "datetime": datetimes})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["value", "label", "datetime"])
    if len(frame) < buckets * 20:
        return {
            "bucket_top_bottom_spread": float("nan"),
            "bucket_monotonicity": float("nan"),
            "bucket_extreme_contribution": float("nan"),
        }

    pct = frame.groupby("datetime", sort=False)["value"].rank(pct=True)
    bucket = np.floor(pct.to_numpy() * buckets).astype(int)
    bucket = np.clip(bucket, 0, buckets - 1)
    frame["_bucket"] = bucket
    means = frame.groupby("_bucket", sort=True)["label"].mean().reindex(range(buckets))
    if means.isna().any():
        return {
            "bucket_top_bottom_spread": float("nan"),
            "bucket_monotonicity": float("nan"),
            "bucket_extreme_contribution": float("nan"),
        }
    bucket_values = means.to_numpy(dtype=np.float64)
    spread = float(bucket_values[-1] - bucket_values[0])
    monotonicity = safe_corr(np.arange(buckets, dtype=np.float64), bucket_values)
    denom = float(np.sum(np.abs(bucket_values))) + 1e-18
    extreme = float(max(abs(bucket_values[0]), abs(bucket_values[-1])) / denom)
    return {
        "bucket_top_bottom_spread": spread,
        "bucket_monotonicity": monotonicity,
        "bucket_extreme_contribution": extreme,
    }


def turnover_proxy(values: np.ndarray, symbols: pd.Series, datetimes: pd.Series, quantile: float = 0.2) -> float:
    frame = pd.DataFrame({"value": values, "symbol": symbols, "datetime": datetimes})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["value", "symbol", "datetime"])
    if len(frame) < 100:
        return float("nan")
    pct = frame.groupby("datetime", sort=False)["value"].rank(pct=True)
    signal = np.where(pct >= 1.0 - quantile, 1.0, np.where(pct <= quantile, -1.0, 0.0))
    frame["_signal"] = signal.astype(np.float32)
    frame = frame.sort_values(["symbol", "datetime"])
    changed = frame.groupby("symbol", sort=False)["_signal"].diff().abs()
    return float(changed.replace([np.inf, -np.inf], np.nan).mean())


def standardize_matrix(frame: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
    if isinstance(frame, pd.DataFrame):
        names = list(frame.columns)
        arr = frame.to_numpy(np.float32, copy=True)
    else:
        names = []
        arr = np.asarray(frame, dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = np.nan
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, np.nan)
    z = (arr - mean) / std
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32, copy=False), names


def max_library_correlation(values: np.ndarray, library_z: np.ndarray, library_names: list[str]) -> dict[str, float | str]:
    if library_z.size == 0 or not library_names:
        return {"max_abs_corr": 0.0, "max_corr": 0.0, "closest_factor": "", "compared": 0}
    xz, _names = standardize_matrix(values.reshape(-1, 1))
    corr = (xz[:, 0].astype(np.float64) @ library_z.astype(np.float64)) / max(len(xz), 1)
    corr[~np.isfinite(corr)] = 0.0
    idx = int(np.argmax(np.abs(corr)))
    return {
        "max_abs_corr": float(abs(corr[idx])),
        "max_corr": float(corr[idx]),
        "closest_factor": library_names[idx],
        "compared": int(len(library_names)),
    }


def residual_ic(values: np.ndarray, label: np.ndarray, closest: np.ndarray | None) -> float:
    if closest is None:
        return safe_corr(values, label)
    x = np.asarray(values, dtype=np.float64)
    z = np.asarray(closest, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(z) & np.isfinite(y)
    if int(mask.sum()) < 10:
        return float("nan")
    x0 = x[mask] - float(np.mean(x[mask]))
    z0 = z[mask] - float(np.mean(z[mask]))
    denom = float(np.dot(z0, z0))
    if denom <= 1e-18:
        return safe_corr(x, y)
    beta = float(np.dot(x0, z0) / denom)
    resid = x0 - beta * z0
    return safe_corr(resid, y[mask])


@dataclass(frozen=True)
class ScorecardThresholds:
    min_coverage: float = 0.70
    min_abs_ic: float = 0.001
    min_abs_rank_ic: float = 0.0005
    min_monthly_hit_rate: float = 0.50
    min_monthly_ic_count: int = 6
    min_product_hit_rate: float = 0.45
    min_regime_hit_rate: float = 0.40
    min_bucket_monotonicity: float = 0.25
    max_abs_corr: float = 0.90
    max_outlier_ratio: float = 0.02
    max_turnover_proxy: float = 0.85
    min_abs_residual_ic: float = 0.001


def decision_flags(row: pd.Series, thresholds: ScorecardThresholds) -> dict[str, bool | str]:
    selection_ic = float(row.get("selection_ic", row.get("oos_ic", 0.0)) or 0.0)
    abs_selection_ic = abs(selection_ic)
    abs_rank_ic = abs(float(row.get("rank_ic", 0.0) or 0.0))
    same_sign = bool(row.get("sample_same_sign", row.get("same_sign", False)))
    spread = float(row.get("bucket_top_bottom_spread", 0.0) or 0.0)
    monotonicity = abs(float(row.get("bucket_monotonicity", 0.0) or 0.0))
    turnover = float(row.get("turnover_proxy", 0.0) or 0.0)
    max_corr = float(row.get("max_abs_corr_to_library", 1.0) or 1.0)
    peer_corr = float(row.get("candidate_peer_max_abs_corr", 0.0) or 0.0)
    if not np.isfinite(peer_corr):
        peer_corr = 0.0
    residual_abs_ic = abs(float(row.get("residual_ic", 0.0) or 0.0))

    pass_data = (
        float(row.get("coverage", 0.0) or 0.0) >= thresholds.min_coverage
        and float(row.get("outlier_ratio", 1.0) or 1.0) <= thresholds.max_outlier_ratio
        and float(row.get("std", 0.0) or 0.0) > 1e-8
    )
    pass_ic = (
        same_sign
        and abs_selection_ic >= thresholds.min_abs_ic
        and abs_rank_ic >= thresholds.min_abs_rank_ic
        and float(row.get("monthly_ic_hit_rate", 0.0) or 0.0) >= thresholds.min_monthly_hit_rate
        and int(row.get("monthly_ic_count", 0) or 0) >= thresholds.min_monthly_ic_count
    )
    pass_bucket = (
        np.isfinite(spread)
        and np.sign(spread) == np.sign(selection_ic)
        and monotonicity >= thresholds.min_bucket_monotonicity
    )
    pass_regime = (
        float(row.get("product_ic_hit_rate", 0.0) or 0.0) >= thresholds.min_product_hit_rate
        and float(row.get("liquidity_regime_ic_hit_rate", 0.0) or 0.0) >= thresholds.min_regime_hit_rate
        and float(row.get("volatility_regime_ic_hit_rate", 0.0) or 0.0) >= thresholds.min_regime_hit_rate
    )
    pass_trading = (not np.isfinite(turnover)) or turnover <= thresholds.max_turnover_proxy
    pass_incremental = (
        (max_corr <= thresholds.max_abs_corr and peer_corr <= thresholds.max_abs_corr)
        or residual_abs_ic >= thresholds.min_abs_residual_ic
    )
    pass_robustness = pass_ic and pass_incremental
    reasons = []
    for key, value in {
        "data": pass_data,
        "ic": pass_ic,
        "bucket": pass_bucket,
        "regime": pass_regime,
        "trading": pass_trading,
        "incremental": pass_incremental,
        "robustness": pass_robustness,
    }.items():
        if not value:
            reasons.append(f"fail_{key}")
    return {
        "pass_data_quality": bool(pass_data),
        "pass_ic": bool(pass_ic),
        "pass_bucket": bool(pass_bucket),
        "pass_regime": bool(pass_regime),
        "pass_trading": bool(pass_trading),
        "pass_incremental": bool(pass_incremental),
        "pass_robustness": bool(pass_robustness),
        "decision_reason": "pass_all" if not reasons else ",".join(reasons),
    }


def final_grade(row: pd.Series, thresholds: ScorecardThresholds) -> str:
    flags = decision_flags(row, thresholds)
    if not flags["pass_data_quality"]:
        return "E"

    selection_ic = float(row.get("selection_ic", row.get("oos_ic", 0.0)) or 0.0)
    abs_selection_ic = abs(selection_ic)
    monthly_hit = float(row.get("monthly_ic_hit_rate", 0.0) or 0.0)

    if all(
        bool(flags[key])
        for key in (
            "pass_ic",
            "pass_bucket",
            "pass_regime",
            "pass_trading",
            "pass_incremental",
            "pass_robustness",
        )
    ):
        return "A" if abs_selection_ic >= thresholds.min_abs_ic * 2 and monthly_hit >= 0.58 else "B"
    if flags["pass_ic"] and flags["pass_incremental"] and flags["pass_robustness"] and (
        flags["pass_bucket"] or flags["pass_regime"]
    ):
        return "B"
    if flags["pass_ic"] and flags["pass_incremental"]:
        return "C"
    if bool(row.get("sample_same_sign", row.get("same_sign", False))) and abs_selection_ic >= thresholds.min_abs_ic * 0.5:
        return "D"
    return "E"


def composite_score(row: pd.Series, thresholds: ScorecardThresholds) -> float:
    coverage_score = clamp01((float(row.get("coverage", 0.0) or 0.0) - thresholds.min_coverage) / 0.30)
    outlier_score = clamp01(1.0 - float(row.get("outlier_ratio", 1.0) or 1.0) / max(thresholds.max_outlier_ratio, 1e-9))
    ic_score = clamp01(
        abs(float(row.get("selection_ic", row.get("oos_ic", 0.0)) or 0.0))
        / max(thresholds.min_abs_ic * 5.0, 1e-9)
    )
    rank_score = clamp01(abs(float(row.get("rank_ic", 0.0) or 0.0)) / max(thresholds.min_abs_ic * 5.0, 1e-9))
    monthly_score = clamp01(float(row.get("monthly_ic_hit_rate", 0.0) or 0.0))
    product_score = clamp01(float(row.get("product_ic_hit_rate", 0.0) or 0.0))
    bucket_score = clamp01(abs(float(row.get("bucket_monotonicity", 0.0) or 0.0)))
    corr_score = clamp01(1.0 - float(row.get("max_abs_corr_to_library", 1.0) or 1.0) / max(thresholds.max_abs_corr, 1e-9))
    residual_score = clamp01(abs(float(row.get("residual_ic", 0.0) or 0.0)) / max(thresholds.min_abs_ic * 5.0, 1e-9))
    turnover = float(row.get("turnover_proxy", 0.0) or 0.0)
    turnover_score = 0.5 if not np.isfinite(turnover) else clamp01(1.0 - turnover)
    same_sign_bonus = 0.05 if bool(row.get("sample_same_sign", row.get("same_sign", False))) else -0.10

    score = (
        0.10 * coverage_score
        + 0.05 * outlier_score
        + 0.16 * ic_score
        + 0.10 * rank_score
        + 0.14 * monthly_score
        + 0.12 * product_score
        + 0.12 * bucket_score
        + 0.13 * max(corr_score, residual_score)
        + 0.08 * turnover_score
        + same_sign_bonus
    )
    if final_grade(row, thresholds) == "E":
        score *= 0.25
    return float(score)
