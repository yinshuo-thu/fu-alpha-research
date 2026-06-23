from __future__ import annotations

import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Config
from .factor_lib import compute_symbol_factors
from .factor_spec import load_selected
from .factor_store import META_COLS, FactorStore
from .labels import build_labels
from .raw_data import get_symbols, load_symbol
from .sessions import detect_sessions


TSZ_WINDOW = 120
TSZ_MINP = 30


def tsz_frame(values: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=values.index)
    vals = values[cols].to_numpy(dtype=np.float32, copy=True)
    finite = np.isfinite(vals)
    clean = np.where(finite, vals, 0.0).astype(np.float64, copy=False)
    cnt = np.cumsum(finite.astype(np.float64), axis=0)
    s1 = np.cumsum(clean, axis=0)
    s2 = np.cumsum(clean * clean, axis=0)
    zeros = np.zeros((1, len(cols)), dtype=np.float64)
    cnt = np.vstack([zeros, cnt])
    s1 = np.vstack([zeros, s1])
    s2 = np.vstack([zeros, s2])
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - TSZ_WINDOW)
    nobs = cnt[end] - cnt[start]
    sums = s1[end] - s1[start]
    sumsq = s2[end] - s2[start]
    mean = sums / np.maximum(nobs, 1.0)
    var = (sumsq - sums * sums / np.maximum(nobs, 1.0)) / np.maximum(nobs - 1.0, 1.0)
    z = (clean - mean) / (np.sqrt(np.maximum(var, 0.0)) + 1e-8)
    z[(nobs < TSZ_MINP) | ~finite] = np.nan
    return pd.DataFrame(z.astype(np.float32), index=values.index, columns=[f"tsz_{c}" for c in cols])


def make_symbol_frame(sym: str, cfg: Config, start: str, end_exclusive: str) -> pd.DataFrame:
    spec = load_selected(cfg.selected_factors_path)
    raw = load_symbol(sym, cfg)
    raw = detect_sessions(raw)
    raw = build_labels(raw, horizon=cfg.label_horizon)
    factors = compute_symbol_factors(raw)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end_exclusive)
    keep = (factors["datetime"] >= start_ts) & (factors["datetime"] < end_ts)
    factors = factors.loc[keep].reset_index(drop=True)
    meta = raw.loc[(raw["datetime"] >= start_ts) & (raw["datetime"] < end_ts), META_COLS].reset_index(drop=True)
    if len(meta) != len(factors):
        meta = factors[["symbol", "datetime", "label"]].merge(
            raw[[c for c in META_COLS if c != "label"]],
            on=["symbol", "datetime"],
            how="left",
        )
    for col in spec.needed_base:
        if col not in factors.columns:
            factors[col] = np.nan
        factors[col] = factors[col].astype(np.float32)
    raw_cols = list(dict.fromkeys(spec.by_view["raw"]))
    cs_bases = sorted(set(spec.by_view["csz"]) | set(spec.by_view["csr"]))
    base_cols = list(dict.fromkeys(raw_cols + cs_bases))
    frames = [meta.copy(), factors[base_cols].astype(np.float32).reset_index(drop=True)]
    if spec.by_view["tsz"]:
        frames.append(tsz_frame(factors, spec.by_view["tsz"]).reset_index(drop=True))
    out = pd.concat(frames, axis=1)
    return out


def write_symbol_month_parts(frame: pd.DataFrame, sym: str, partition_dir: Path) -> None:
    frame = frame.copy()
    frame["_month"] = frame["datetime"].dt.to_period("M").astype(str)
    for month, chunk in frame.groupby("_month", sort=True):
        month_dir = partition_dir / f"month={month}"
        month_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(chunk.drop(columns=["_month"]), preserve_index=False), month_dir / f"{sym}.parquet", compression="zstd")


def build_intermediate_partitions(
    cfg: Config,
    start: str = "2017-01-01",
    end_exclusive: str = "2021-01-01",
    overwrite: bool = False,
) -> None:
    if overwrite and cfg.partition_dir.exists():
        shutil.rmtree(cfg.partition_dir)
    cfg.partition_dir.mkdir(parents=True, exist_ok=True)
    symbols = get_symbols(cfg)
    for i, sym in enumerate(symbols, 1):
        frame = make_symbol_frame(sym, cfg, start, end_exclusive)
        write_symbol_month_parts(frame, sym, cfg.partition_dir)
        print(f"  [build-partitions] {i:02d}/{len(symbols)} {sym} rows={len(frame)}", flush=True)
        del frame
        gc.collect()


def materialize_final_parquet(cfg: Config, final_path: Path, start: str, end: str) -> None:
    store = FactorStore(cfg)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_suffix(final_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer = None
    rows = 0
    for month, df in store.iter_months(start, end):
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table)
        rows += len(df)
        print(f"  [materialize] {month} rows={len(df)} total={rows}", flush=True)
        del df, table
        gc.collect()
    if writer is not None:
        writer.close()
        tmp.replace(final_path)
