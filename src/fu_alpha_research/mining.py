from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .factor_store import FactorStore


class ICAccumulator:
    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        n = len(feature_cols)
        self.xy = np.zeros(n, dtype=np.float64)
        self.xx = np.zeros(n, dtype=np.float64)
        self.yy = np.zeros(n, dtype=np.float64)
        self.count = np.zeros(n, dtype=np.int64)

    def update(self, df: pd.DataFrame, block_size: int = 128) -> None:
        y = df["label"].to_numpy(np.float64, copy=False)
        y_ok = np.isfinite(y)
        y0 = np.where(y_ok, y, 0.0)
        y2 = y0 * y0
        for start in range(0, len(self.feature_cols), block_size):
            end = min(start + block_size, len(self.feature_cols))
            cols = self.feature_cols[start:end]
            x = df[cols].to_numpy(np.float32, copy=False)
            valid = np.isfinite(x) & y_ok[:, None]
            x0 = np.where(valid, x, 0.0).astype(np.float64, copy=False)
            self.xy[start:end] += x0.T @ y0
            self.xx[start:end] += np.sum(x0 * x0, axis=0)
            self.yy[start:end] += valid.astype(np.float64).T @ y2
            self.count[start:end] += np.sum(valid, axis=0)

    def frame(self, prefix: str) -> pd.DataFrame:
        denom = np.sqrt(np.maximum(self.xx * self.yy, 1e-30))
        ic = self.xy / denom
        ic[self.count < 2] = np.nan
        return pd.DataFrame(
            {
                "factor": self.feature_cols,
                f"{prefix}_ic": ic,
                f"{prefix}_n": self.count,
                f"{prefix}_coverage_proxy": self.count / max(float(self.count.max()), 1.0),
            }
        )


def month_sufficient_stats(df: pd.DataFrame, feature_cols: list[str], block_size: int = 64) -> pd.DataFrame:
    y = df["label"].to_numpy(np.float64, copy=False)
    y_ok = np.isfinite(y)
    y0 = np.where(y_ok, y, 0.0)
    y2 = y0 * y0
    n = len(feature_cols)
    xy = np.zeros(n, dtype=np.float64)
    xx = np.zeros(n, dtype=np.float64)
    yy = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.int64)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        cols = feature_cols[start:end]
        x = df[cols].to_numpy(np.float32, copy=False)
        valid = np.isfinite(x) & y_ok[:, None]
        x0 = np.where(valid, x, 0.0).astype(np.float64, copy=False)
        xy[start:end] = x0.T @ y0
        xx[start:end] = np.sum(x0 * x0, axis=0)
        yy[start:end] = valid.astype(np.float64).T @ y2
        count[start:end] = np.sum(valid, axis=0)
    return pd.DataFrame({"factor": feature_cols, "xy": xy, "xx": xx, "yy": yy, "count": count})


def aggregate_stats(parts: list[pd.DataFrame], prefix: str) -> pd.DataFrame:
    stats = pd.concat(parts, ignore_index=True).groupby("factor", as_index=False, sort=False).sum(numeric_only=True)
    denom = np.sqrt(np.maximum(stats["xx"].to_numpy() * stats["yy"].to_numpy(), 1e-30))
    ic = stats["xy"].to_numpy() / denom
    count = stats["count"].to_numpy()
    ic[count < 2] = np.nan
    max_count = max(float(np.nanmax(count)) if len(count) else 0.0, 1.0)
    return pd.DataFrame(
        {
            "factor": stats["factor"],
            f"{prefix}_ic": ic,
            f"{prefix}_n": count.astype(np.int64),
            f"{prefix}_coverage_proxy": count / max_count,
        }
    )


def compute_factor_ic(
    store: FactorStore,
    start: str,
    end: str,
    feature_cols: list[str] | None = None,
    prefix: str = "is",
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    cols = feature_cols or store.selected
    parts: list[pd.DataFrame] = []
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
    for month in store.available_months(start, end):
        part_file = cache_path / f"{prefix}_{month}.parquet" if cache_path is not None else None
        if part_file is not None and part_file.exists() and not force:
            stats = pd.read_parquet(part_file)
            print(f"  [factor-ic][{prefix}] {month} cached factors={len(stats)}", flush=True)
        else:
            df = store.read_month(month, columns=cols)
            stats = month_sufficient_stats(df, cols)
            if part_file is not None:
                stats.to_parquet(part_file, index=False)
            print(f"  [factor-ic][{prefix}] {month} rows={len(df)} cached={part_file is not None}", flush=True)
            del df
            gc.collect()
        parts.append(stats)
    return aggregate_stats(parts, prefix)


def mine_effective_factors(
    store: FactorStore,
    is_start: str,
    is_end: str,
    oos_start: str,
    oos_end: str,
    min_is_ic: float = 0.002,
    min_oos_ic: float = 0.001,
    min_coverage_proxy: float = 0.5,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    features = store.selected
    is_df = compute_factor_ic(store, is_start, is_end, features, "is", cache_dir=cache_dir)
    oos_df = compute_factor_ic(store, oos_start, oos_end, features, "oos", cache_dir=cache_dir)
    out = is_df.merge(oos_df, on="factor", how="inner")
    out["same_sign"] = np.sign(out["is_ic"]) == np.sign(out["oos_ic"])
    out["abs_is_ic"] = out["is_ic"].abs()
    out["abs_oos_ic"] = out["oos_ic"].abs()
    out["effective"] = (
        out["same_sign"]
        & (out["abs_is_ic"] >= min_is_ic)
        & (out["abs_oos_ic"] >= min_oos_ic)
        & (out["is_coverage_proxy"] >= min_coverage_proxy)
        & (out["oos_coverage_proxy"] >= min_coverage_proxy)
    )
    out = out.sort_values(["effective", "abs_oos_ic", "abs_is_ic"], ascending=[False, False, False])
    return out


def attach_catalog(scores: pd.DataFrame, catalog_path: str) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path)
    return scores.merge(catalog, on="factor", how="left")
