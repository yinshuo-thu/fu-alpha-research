from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpressionFactor:
    name: str
    op: str
    left: str
    right: str
    formula: str

    @property
    def inputs(self) -> tuple[str, str]:
        return (self.left, self.right)


def safe_name(op: str, left: str, right: str) -> str:
    raw = f"{op}:{left}:{right}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"nf_{op}_{digest}"


def make_expression(op: str, left: str, right: str) -> ExpressionFactor:
    formulas = {
        "rank_add": f"xrank({left}) + xrank({right})",
        "rank_spread": f"xrank({left}) - xrank({right})",
        "rank_product": f"xrank({left}) * xrank({right})",
        "rank_gate_pos": f"xrank({left}) * I[xrank({right}) > 0]",
        "rank_gate_neg": f"xrank({left}) * I[xrank({right}) < 0]",
        "z_add": f"xzscore({left}) + xzscore({right})",
        "z_spread": f"xzscore({left}) - xzscore({right})",
        "z_product": f"xzscore({left}) * xzscore({right})",
    }
    if op not in formulas:
        raise ValueError(f"unknown expression op: {op}")
    return ExpressionFactor(safe_name(op, left, right), op, left, right, formulas[op])


def _dedup_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out = []
    for left, right in pairs:
        if left == right:
            continue
        key = (left, right)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _seed_group(row: pd.Series) -> str:
    family = str(row.get("family", "") or "")
    view = str(row.get("view", "") or "")
    if family or view:
        return f"{family}:{view}"
    name = str(row.get("factor", "") or "")
    for prefix in ("tsz_", "csz_", "csr_"):
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return "raw"


def _select_diverse_seeds(scores: pd.DataFrame, max_seeds: int) -> pd.DataFrame:
    """Select seeds by strength, then diversify by catalog family/view.

    This avoids building every expression from a small cluster of highly similar
    legacy factors. The cap relaxes gradually until the requested seed count is
    reached, which makes the process deterministic and open-ended for continuous
    mining rounds.
    """
    if scores.empty or max_seeds <= 0:
        return scores.iloc[0:0].copy()
    scores = scores.copy()
    scores["_seed_group"] = scores.apply(_seed_group, axis=1)
    selected_idx: list[int] = []
    selected_set: set[int] = set()
    counts: Counter[str] = Counter()
    max_cap = max(2, max_seeds)
    for cap in range(1, max_cap + 1):
        for idx, row in scores.iterrows():
            if idx in selected_set:
                continue
            group = str(row["_seed_group"])
            if counts[group] >= cap:
                continue
            selected_idx.append(idx)
            selected_set.add(idx)
            counts[group] += 1
            if len(selected_idx) >= max_seeds:
                return scores.loc[selected_idx].drop(columns="_seed_group")
    return scores.loc[selected_idx].drop(columns="_seed_group")


def generate_candidate_expressions(
    factor_scores: pd.DataFrame,
    max_seeds: int = 80,
    max_pairs: int = 500,
) -> pd.DataFrame:
    """Generate reproducible expression candidates from prior factor scores.

    Seeds are chosen by absolute IS IC, with OOS same-sign factors preferred.
    Pairing is deterministic and mixes high-ranked seeds with lower-ranked,
    different-family/view factors when catalog metadata is available.
    """
    scores = factor_scores.copy()
    if "abs_is_ic" not in scores:
        scores["abs_is_ic"] = scores["is_ic"].abs()
    scores["same_sign"] = np.sign(scores["is_ic"]) == np.sign(scores["oos_ic"])
    scores = scores.sort_values(["same_sign", "abs_is_ic"], ascending=[False, False]).reset_index(drop=True)
    seeds = _select_diverse_seeds(scores, max_seeds)
    names = seeds["factor"].tolist()

    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(names):
        left_row = seeds.iloc[i]
        for j in range(i + 1, min(len(names), i + 18)):
            right = names[j]
            pairs.append((left, right))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
        # Add deterministic cross-family/view pairings from deeper seeds.
        for j in range(len(names) - 1, max(i, len(names) - 20), -1):
            right_row = seeds.iloc[j]
            if left_row.get("family") != right_row.get("family") or left_row.get("view") != right_row.get("view"):
                pairs.append((left, names[j]))
                break
        if len(pairs) >= max_pairs:
            break

    pairs = _dedup_pairs(pairs)[:max_pairs]
    ops = [
        "rank_add",
        "rank_spread",
        "rank_product",
        "rank_gate_pos",
        "rank_gate_neg",
        "z_add",
        "z_spread",
        "z_product",
    ]
    rows = []
    for left, right in pairs:
        for op in ops:
            expr = make_expression(op, left, right)
            rows.append(
                {
                    "name": expr.name,
                    "op": expr.op,
                    "left": expr.left,
                    "right": expr.right,
                    "formula": expr.formula,
                }
            )
    return pd.DataFrame(rows).drop_duplicates("name").reset_index(drop=True)


def load_expression_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"name", "op", "left", "right", "formula"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"expression table missing columns: {sorted(missing)}")
    return df


def xrank_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    return (df.groupby("datetime", sort=False)[columns].rank(pct=True) - 0.5).astype(np.float32)


def xzscore_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=df.index)
    g = df.groupby("datetime", sort=False)
    out = {}
    for col in columns:
        mu = g[col].transform("mean")
        sd = g[col].transform("std")
        out[col] = ((df[col] - mu) / (sd + 1e-8)).astype(np.float32).to_numpy()
    return pd.DataFrame(out, index=df.index)


def compute_expression_block(base: pd.DataFrame, exprs: pd.DataFrame) -> pd.DataFrame:
    inputs = sorted(set(exprs["left"]).union(exprs["right"]))
    ranks = xrank_frame(base, inputs)
    zscores = xzscore_frame(base, inputs)
    out: dict[str, np.ndarray] = {}
    for row in exprs.itertuples(index=False):
        if row.op.startswith("rank_"):
            left = ranks[row.left].to_numpy(np.float32, copy=False)
            right = ranks[row.right].to_numpy(np.float32, copy=False)
        else:
            left = zscores[row.left].to_numpy(np.float32, copy=False)
            right = zscores[row.right].to_numpy(np.float32, copy=False)

        if row.op.endswith("_add"):
            val = left + right
        elif row.op.endswith("_spread"):
            val = left - right
        elif row.op.endswith("_product"):
            val = left * right
        elif row.op == "rank_gate_pos":
            val = np.where(right > 0, left, 0.0)
        elif row.op == "rank_gate_neg":
            val = np.where(right < 0, left, 0.0)
        else:
            raise ValueError(f"unknown expression op: {row.op}")
        out[row.name] = np.asarray(val, dtype=np.float32)
    return pd.DataFrame(out, index=base.index)
