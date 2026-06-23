from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config


def get_symbols(cfg: Config) -> list[str]:
    excluded = set(cfg.excluded_symbols)
    return [p.stem for p in sorted(cfg.raw_data_dir.glob("*.csv")) if p.stem not in excluded]


def load_symbol(symbol: str, cfg: Config) -> pd.DataFrame:
    path = cfg.raw_data_dir / f"{symbol}.csv"
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    dt_cols = [c for c in df.columns if "time" in c or "date" in c or c in {"datetime", "dt"}]
    dt_col = dt_cols[0] if dt_cols else df.columns[0]
    df = df.rename(columns={dt_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"].astype(str).str.slice(0, 16), format="%Y-%m-%d %H:%M")
    df["symbol"] = symbol
    df = df.rename(columns={"open interest": "oi", "openinterest": "oi", "open_interest": "oi"})
    keep = ["symbol", "datetime", "open", "high", "low", "close", "volume", "amount", "oi"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.drop_duplicates(subset=["symbol", "datetime"]).sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "amount", "oi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
