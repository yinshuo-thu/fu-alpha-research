from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ic(pred: np.ndarray | pd.Series, label: np.ndarray | pd.Series) -> float:
    """Pooled cosine IC: E[p*y] / sqrt(E[p^2] E[y^2])."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    mask = np.isfinite(p) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return float("nan")
    p = p[mask]
    y = y[mask]
    denom = np.sqrt(np.mean(p * p) * np.mean(y * y))
    if denom <= 1e-18:
        return float("nan")
    return float(np.mean(p * y) / denom)


def ic_by_period(
    df: pd.DataFrame,
    pred_col: str = "pred",
    label_col: str = "label",
    period: str = "M",
) -> pd.Series:
    frame = df.dropna(subset=[pred_col, label_col]).copy()
    if frame.empty:
        return pd.Series(dtype=float)
    if period == "Y":
        frame["_period"] = frame["datetime"].dt.year.astype(str)
    else:
        frame["_period"] = frame["datetime"].dt.to_period(period).astype(str)
    values = {
        period_key: compute_ic(grp[pred_col].to_numpy(), grp[label_col].to_numpy())
        for period_key, grp in frame.groupby("_period", sort=True)
    }
    return pd.Series(values, dtype=float)


def add_prediction_views(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("datetime", sort=False)[pred_col]
    out[f"{pred_col}_xsz"] = (out[pred_col] - g.transform("mean")) / (g.transform("std") + 1e-9)
    out[f"{pred_col}_xrank"] = g.rank(pct=True) - 0.5
    return out


def summarize_predictions(df: pd.DataFrame, pred_cols: list[str] | None = None) -> pd.DataFrame:
    if pred_cols is None:
        pred_cols = [c for c in ("pred", "pred_xsz", "pred_xrank") if c in df.columns]
    rows: list[dict[str, float | str | int]] = []
    for pred_col in pred_cols:
        monthly = ic_by_period(df, pred_col, "label", "M")
        yearly = ic_by_period(df, pred_col, "label", "Y")
        row: dict[str, float | str | int] = {
            "pred_col": pred_col,
            "rows": int(len(df)),
            "label_rows": int(df["label"].notna().sum()),
            "coverage": float((df[pred_col].notna() & df["label"].notna()).mean()),
            "total_ic": compute_ic(df[pred_col].to_numpy(), df["label"].to_numpy()),
            "monthly_mean": float(monthly.mean()) if len(monthly) else float("nan"),
            "monthly_std": float(monthly.std()) if len(monthly) else float("nan"),
        }
        row["monthly_ir"] = (
            float(row["monthly_mean"]) / float(row["monthly_std"])
            if np.isfinite(row["monthly_std"]) and float(row["monthly_std"]) > 0
            else float("nan")
        )
        for key, value in yearly.items():
            row[f"ic_{key}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def long_short_backtest(
    df: pd.DataFrame,
    pred_col: str = "pred_xrank",
    label_col: str = "label",
    quantile: float = 0.2,
) -> pd.DataFrame:
    rows = []
    for dt, grp in df.dropna(subset=[pred_col, label_col]).groupby("datetime", sort=True):
        if len(grp) < 10:
            continue
        lo = grp[pred_col].quantile(quantile)
        hi = grp[pred_col].quantile(1.0 - quantile)
        long_ret = grp.loc[grp[pred_col] >= hi, label_col].mean()
        short_ret = grp.loc[grp[pred_col] <= lo, label_col].mean()
        rows.append({"datetime": dt, "ls_return": float(long_ret - short_ret), "n": int(len(grp))})
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame) -> dict[str, float | int]:
    if bt.empty:
        return {"rows": 0, "mean": float("nan"), "std": float("nan"), "tstat": float("nan")}
    ret = bt["ls_return"].to_numpy(np.float64)
    std = float(np.nanstd(ret, ddof=1))
    mean = float(np.nanmean(ret))
    return {
        "rows": int(len(bt)),
        "mean": mean,
        "std": std,
        "tstat": float(mean / (std / np.sqrt(len(ret)))) if std > 0 else float("nan"),
        "hit_rate": float(np.mean(ret > 0)),
        "cum_return": float(np.nansum(ret)),
    }
