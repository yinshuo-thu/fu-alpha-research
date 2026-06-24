from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config
from .expressions import compute_expression_block, load_expression_table
from .factor_store import FactorStore, META_COLS


def read_feature_list(path: str | Path) -> list[str]:
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_feature_list(path: str | Path, features: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(features) + "\n", encoding="utf-8")


@dataclass
class FeatureMatrix:
    cfg: Config
    expression_path: Path | None = None

    def __post_init__(self) -> None:
        self.store = FactorStore(self.cfg)
        if self.expression_path is None:
            self.expression_path = self.cfg.output_dir / "expression_sets" / "new100.csv"
        if self.expression_path.exists():
            exprs = load_expression_table(self.expression_path)
        else:
            exprs = pd.DataFrame(columns=["name", "op", "left", "right", "formula"])
        self.exprs = exprs
        self.expr_by_name = {row.name: row for row in exprs.itertuples(index=False)}

    def expression_features(self, features: list[str]) -> list[str]:
        return [name for name in features if name in self.expr_by_name]

    def original_features(self, features: list[str]) -> list[str]:
        return [name for name in features if name not in self.expr_by_name]

    def dependencies(self, features: list[str]) -> list[str]:
        deps = list(self.original_features(features))
        for name in self.expression_features(features):
            row = self.expr_by_name[name]
            deps.extend([row.left, row.right])
        return list(dict.fromkeys(deps))

    def read_month(self, month: str, features: list[str], sort: bool = True) -> pd.DataFrame:
        expr_names = self.expression_features(features)
        deps = self.dependencies(features)
        base = self.store.read_month(month, columns=deps, sort=sort)
        meta_cols = [col for col in META_COLS if col in base.columns]

        frames = [base[meta_cols].reset_index(drop=True)]
        original = [col for col in self.original_features(features) if col in base.columns]
        if original:
            frames.append(base[original].reset_index(drop=True))
        if expr_names:
            expr_df = self.exprs[self.exprs["name"].isin(expr_names)].copy()
            expr_df["__order"] = expr_df["name"].map({name: i for i, name in enumerate(expr_names)})
            expr_df = expr_df.sort_values("__order").drop(columns="__order")
            values = compute_expression_block(base, expr_df)
            frames.append(values[expr_names].reset_index(drop=True))

        out = pd.concat(frames, axis=1)
        final_cols = meta_cols + features
        return out[final_cols]
