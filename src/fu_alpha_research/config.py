from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


@dataclass(frozen=True)
class Config:
    raw_data_dir: Path
    partition_dir: Path
    factor_panel_path: Path
    selected_factors_path: Path
    factor_catalog_path: Path
    output_dir: Path
    reports_dir: Path
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    label_horizon: int
    seed: int
    excluded_symbols: tuple[str, ...]
    ridge_train_rows: int
    lgb_train_rows: int
    baseline_models: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "Config":
        return cls(
            raw_data_dir=resolve_path(raw["raw_data_dir"]),
            partition_dir=resolve_path(raw["partition_dir"]),
            factor_panel_path=resolve_path(raw.get("factor_panel_path", raw["partition_dir"])),
            selected_factors_path=resolve_path(raw["selected_factors_path"]),
            factor_catalog_path=resolve_path(raw["factor_catalog_path"]),
            output_dir=resolve_path(raw["output_dir"]),
            reports_dir=resolve_path(raw["reports_dir"]),
            is_start=str(raw.get("is_start", "2018-01-01")),
            is_end=str(raw.get("is_end", "2019-12-31")),
            oos_start=str(raw.get("oos_start", "2020-01-01")),
            oos_end=str(raw.get("oos_end", "2020-12-31")),
            label_horizon=int(raw.get("label_horizon", 30)),
            seed=int(raw.get("seed", 42)),
            excluded_symbols=tuple(raw.get("excluded_symbols", [])),
            ridge_train_rows=int(raw.get("ridge_train_rows", 600000)),
            lgb_train_rows=int(raw.get("lgb_train_rows", 500000)),
            baseline_models=tuple(raw.get("baseline_models", ["ridge"])),
        )


def load_config(path: str | Path = "configs/futures.yaml") -> Config:
    cfg_path = resolve_path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config.from_mapping(raw)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    return cfg
