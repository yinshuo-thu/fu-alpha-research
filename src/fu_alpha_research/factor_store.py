from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import Config
from .factor_spec import FactorSpec, load_selected, selected_subset


META_COLS = [
    "symbol",
    "datetime",
    "label",
    "is_long_break_before",
    "session_id",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "amount",
    "oi",
]


class FactorStore:
    """Read intermediate month partitions and materialize selected factors on demand."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.partition_dir
        self.panel_path = cfg.factor_panel_path
        self.spec = load_selected(cfg.selected_factors_path)
        self._panel_months: list[str] | None = None
        self._panel_row_groups: dict[str, list[int]] | None = None

    @property
    def selected(self) -> list[str]:
        return self.spec.selected

    def month_dirs(self) -> list[Path]:
        return sorted(p for p in self.root.glob("month=*") if p.is_dir())

    def available_months(self, start: str | None = None, end: str | None = None) -> list[str]:
        if self._use_final_panel():
            months = self._final_panel_months()
            return self._filter_months(months, start, end)
        months = [path.name.split("=", 1)[1] for path in self.month_dirs()]
        return self._filter_months(months, start, end)

    @staticmethod
    def _filter_months(months: list[str], start: str | None, end: str | None) -> list[str]:
        start_ts = pd.Timestamp(start) if start else None
        end_ts = pd.Timestamp(end) if end else None
        out = []
        for month in months:
            month_start = pd.Period(month, freq="M").to_timestamp()
            month_next = month_start + pd.DateOffset(months=1)
            if start_ts is not None and month_next <= start_ts:
                continue
            if end_ts is not None and month_start > end_ts:
                continue
            out.append(month)
        return out

    def audit(self) -> dict[str, object]:
        months = self.available_months()
        symbols: set[str] = set()
        schemas: set[int] = set()
        for month in months[:2] + months[-2:]:
            if self._use_final_panel():
                schemas.add(len(pq.ParquetFile(self.panel_path).schema_arrow.names))
            else:
                for file in (self.root / f"month={month}").glob("*.parquet"):
                    symbols.add(file.stem)
                    schemas.add(len(pq.ParquetFile(file).schema_arrow.names))
        if self._use_final_panel():
            all_symbols = set(pd.read_parquet(self.panel_path, columns=["symbol"])["symbol"].unique())
        else:
            all_symbols = {file.stem for month_dir in self.month_dirs() for file in month_dir.glob("*.parquet")}
        return {
            "partition_dir": str(self.root),
            "factor_panel_path": str(self.panel_path),
            "storage_mode": "final_panel" if self._use_final_panel() else "month_partitions",
            "months": months,
            "month_count": len(months),
            "symbol_count": len(all_symbols),
            "sample_schema_widths": sorted(schemas),
            "selected_factor_count": len(self.spec.selected),
            "raw_count": len(self.spec.by_view["raw"]),
            "tsz_count": len(self.spec.by_view["tsz"]),
            "csz_count": len(self.spec.by_view["csz"]),
            "csr_count": len(self.spec.by_view["csr"]),
        }

    def read_month(
        self,
        month: str,
        columns: list[str] | None = None,
        symbols: list[str] | None = None,
        sort: bool = True,
    ) -> pd.DataFrame:
        month_dir = self.root / f"month={month}"
        if self._use_final_panel():
            return self._read_final_month(month, columns=columns, symbols=symbols, sort=sort)
        if not month_dir.exists():
            raise FileNotFoundError(f"missing month partition: {month_dir}")
        spec = selected_subset(self.spec, columns)
        read_cols = self._read_columns(spec)
        panel = pd.read_parquet(month_dir, columns=read_cols)
        panel["datetime"] = pd.to_datetime(panel["datetime"])
        if symbols is not None:
            panel = panel[panel["symbol"].isin(symbols)].copy()

        frames = [panel[[c for c in META_COLS if c in panel.columns]].reset_index(drop=True)]
        if spec.by_view["raw"]:
            frames.append(panel[spec.by_view["raw"]].astype(np.float32).reset_index(drop=True))
        if spec.by_view["tsz"]:
            tsz_cols = [f"tsz_{base}" for base in spec.by_view["tsz"]]
            frames.append(panel[tsz_cols].astype(np.float32).reset_index(drop=True))
        if spec.by_view["csz"]:
            frames.append(self._csz(panel, spec.by_view["csz"]).reset_index(drop=True))
        if spec.by_view["csr"]:
            frames.append(self._csr(panel, spec.by_view["csr"]).reset_index(drop=True))

        out = pd.concat(frames, axis=1)
        final_cols = [c for c in META_COLS if c in out.columns] + spec.selected
        out = out[final_cols]
        if sort:
            out = out.sort_values(["symbol", "datetime"]).reset_index(drop=True)
        return out

    def iter_months(
        self,
        start: str,
        end: str,
        columns: list[str] | None = None,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        for month in self.available_months(start, end):
            yield month, self.read_month(month, columns=columns)

    def sample_rows(
        self,
        start: str,
        end: str,
        max_rows: int,
        columns: list[str] | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        months = self.available_months(start, end)
        if not months:
            raise ValueError(f"no partitions between {start} and {end}")
        rng = np.random.default_rng(seed)
        per_month = int(math.ceil(max_rows / len(months))) if max_rows else 0
        parts = []
        for month in months:
            df = self.read_month(month, columns=columns)
            df = df.dropna(subset=["label"])
            if per_month and len(df) > per_month:
                idx = rng.choice(len(df), size=per_month, replace=False)
                df = df.iloc[np.sort(idx)].copy()
            parts.append(df)
            print(f"  [sample] {month} rows={len(df)}", flush=True)
        out = pd.concat(parts, ignore_index=True)
        if max_rows and len(out) > max_rows:
            idx = rng.choice(len(out), size=max_rows, replace=False)
            out = out.iloc[np.sort(idx)].reset_index(drop=True)
        return out

    def _read_columns(self, spec: FactorSpec) -> list[str]:
        bases = list(dict.fromkeys(spec.by_view["raw"] + spec.by_view["csz"] + spec.by_view["csr"]))
        tsz_cols = [f"tsz_{base}" for base in spec.by_view["tsz"]]
        return list(dict.fromkeys(META_COLS + bases + tsz_cols))

    def _use_final_panel(self) -> bool:
        return self.panel_path.exists() and self.panel_path.is_file()

    def _final_panel_months(self) -> list[str]:
        if self._panel_months is not None:
            return self._panel_months
        self._panel_months = sorted(self._final_panel_row_groups().keys())
        return self._panel_months

    def _final_panel_row_groups(self) -> dict[str, list[int]]:
        if self._panel_row_groups is not None:
            return self._panel_row_groups
        pf = pq.ParquetFile(self.panel_path)
        dt_idx = pf.schema_arrow.names.index("datetime")
        out: dict[str, list[int]] = {}
        for i in range(pf.num_row_groups):
            stats = pf.metadata.row_group(i).column(dt_idx).statistics
            if stats is None or stats.min is None or stats.max is None:
                continue
            start_month = pd.Timestamp(stats.min).to_period("M")
            end_month = pd.Timestamp(stats.max).to_period("M")
            for period in pd.period_range(start_month, end_month, freq="M"):
                out.setdefault(str(period), []).append(i)
        self._panel_row_groups = out
        return out

    def _read_final_month(
        self,
        month: str,
        columns: list[str] | None = None,
        symbols: list[str] | None = None,
        sort: bool = True,
    ) -> pd.DataFrame:
        spec = selected_subset(self.spec, columns)
        month_start = pd.Period(month, freq="M").to_timestamp()
        month_next = month_start + pd.DateOffset(months=1)
        pf = pq.ParquetFile(self.panel_path)
        schema = set(pf.schema_arrow.names)
        read_cols = [c for c in META_COLS + spec.selected if c in schema]
        row_groups = self._final_panel_row_groups().get(month, [])
        if not row_groups:
            return pd.DataFrame(columns=read_cols)
        table = pf.read_row_groups(row_groups, columns=read_cols)
        df = table.to_pandas(self_destruct=True)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[(df["datetime"] >= month_start) & (df["datetime"] < month_next)].copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        if symbols is not None:
            df = df[df["symbol"].isin(symbols)].copy()
        if sort:
            df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
        return df

    @staticmethod
    def _csz(panel: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
        out = {}
        grouped = panel.groupby("datetime", sort=False)
        for base in bases:
            mu = grouped[base].transform("mean")
            sd = grouped[base].transform("std")
            out[f"csz_{base}"] = ((panel[base] - mu) / (sd + 1e-8)).astype(np.float32).to_numpy()
        return pd.DataFrame(out, index=panel.index)

    @staticmethod
    def _csr(panel: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
        out = {}
        grouped = panel.groupby("datetime", sort=False)
        for base in bases:
            out[f"csr_{base}"] = (grouped[base].rank(pct=True) - 0.5).astype(np.float32).to_numpy()
        return pd.DataFrame(out, index=panel.index)
