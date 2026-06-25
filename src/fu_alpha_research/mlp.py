from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def add_label_views(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("datetime", sort=False)["label"]
    mu = g.transform("mean")
    sd = g.transform("std")
    rank = g.rank(pct=True).astype(np.float32)
    out["label_xsz"] = ((out["label"] - mu) / (sd + 1e-9)).clip(-8, 8).astype(np.float32)
    out["label_xrank"] = (rank - 0.5).astype(np.float32)
    return out


def add_event_sampling_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = add_label_views(df)
    close = out["close"].astype(np.float64).abs().clip(lower=1e-12)
    open_ = out["open"].astype(np.float64).abs().clip(lower=1e-12)
    high = out["high"].astype(np.float64)
    low = out["low"].astype(np.float64)
    intrabar = np.log(close / open_).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    range_rel = ((high - low) / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    log_amount = np.log1p(out["amount"].clip(lower=0)).astype(np.float64)

    def xsec_abs(s: pd.Series) -> pd.Series:
        sg = s.groupby(out["datetime"], sort=False)
        z = (s - sg.transform("mean")) / (sg.transform("std") + 1e-9)
        return z.abs().clip(0, 8)

    out["event_score"] = (xsec_abs(intrabar) + xsec_abs(range_rel) + 0.5 * xsec_abs(log_amount)).astype(np.float32)
    pos = out.groupby("symbol", sort=False).cumcount()
    size = out.groupby("symbol", sort=False)["datetime"].transform("size")
    out["_bars_to_month_end"] = (size - pos - 1).astype(np.int32)
    return out


def stratified_pick(data: pd.DataFrame, pool: np.ndarray, need: int, rng: np.random.Generator) -> np.ndarray:
    if need <= 0 or len(pool) == 0:
        return np.empty(0, dtype=np.int64)
    if len(pool) <= need:
        return pool.astype(np.int64, copy=False)
    ranks = np.nan_to_num(data["label_xrank"].to_numpy(np.float32)[pool], nan=0.0)
    bins = np.floor(np.clip((ranks + 0.5) * 8.0, 0, 7)).astype(np.int16)
    pieces: list[np.ndarray] = []
    per = max(1, need // 8)
    for b in range(8):
        loc = pool[bins == b]
        if len(loc):
            pieces.append(rng.choice(loc, min(len(loc), per), replace=False))
    used = sum(len(x) for x in pieces)
    if used < need:
        already = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
        taken = np.zeros(len(data), dtype=bool)
        taken[already] = True
        rest = pool[~taken[pool]]
        if len(rest):
            pieces.append(rng.choice(rest, min(need - used, len(rest)), replace=False))
    out = np.concatenate(pieces) if pieces else pool
    if len(out) > need:
        out = rng.choice(out, need, replace=False)
    return np.sort(out.astype(np.int64, copy=False))


def sample_rows(data: pd.DataFrame, cap: int, mode: str, seed: int) -> pd.DataFrame:
    pool = np.flatnonzero(data["label"].notna().to_numpy() & data["label_xrank"].notna().to_numpy())
    if cap <= 0 or len(pool) <= cap:
        return data.iloc[pool].copy()
    rng = np.random.default_rng(seed)
    if mode == "random":
        idx = rng.choice(pool, cap, replace=False)
    elif mode == "stratified":
        idx = stratified_pick(data, pool, cap, rng)
    else:
        frac = {"soft_event": 0.25, "event35": 0.35, "event50": 0.50}.get(mode)
        if frac is None:
            raise ValueError(f"bad sample mode: {mode}")
        scores = np.nan_to_num(data["event_score"].to_numpy(np.float32)[pool], nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.sqrt(np.maximum(scores, 0.0) + 0.05)
        weights = weights / weights.sum()
        event_need = min(len(pool), int(cap * frac))
        event_pick = rng.choice(pool, event_need, replace=False, p=weights)
        used = np.zeros(len(data), dtype=bool)
        used[event_pick] = True
        rest = pool[~used[pool]]
        rest_pick = stratified_pick(data, rest, cap - len(event_pick), rng)
        idx = np.concatenate([event_pick, rest_pick])
    return data.iloc[np.sort(idx)].copy()


def scrub_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.array(x, dtype=np.float32, copy=True)
    return np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def weighted_mean_scale(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w64 = w.astype(np.float64)
    w64 = w64 / max(float(w64.sum()), 1e-12)
    mean = (w64[:, None] * x.astype(np.float64)).sum(axis=0).astype(np.float32)
    var = (w64[:, None] * (x.astype(np.float64) - mean) ** 2).sum(axis=0)
    scale = np.maximum(np.sqrt(var), 1e-6).astype(np.float32)
    return mean, scale


def recency_weights(train: pd.DataFrame, test_month: str, half_life_months: float) -> np.ndarray | None:
    if half_life_months <= 0:
        return None
    test_period = pd.Period(test_month, freq="M")
    periods = train["datetime"].dt.to_period("M")
    age = np.array([test_period.ordinal - p.ordinal for p in periods], dtype=np.float64)
    return np.exp(-np.log(2.0) * np.maximum(age, 0.0) / half_life_months).astype(np.float64)


@dataclass(frozen=True)
class MLPConfig:
    hidden: int = 192
    dropout: float = 0.12
    epochs: int = 5
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-4
    half_life_months: float = 12.0
    target_col: str = "label_xsz"
    loss: str = "mse"
    standardize: str = "unweighted"
    seed: int = 20260624


def make_mlp(n_features: int, hidden: int, dropout: float):
    import torch
    from torch import nn

    return nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.LayerNorm(hidden),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2),
        nn.LayerNorm(hidden // 2),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden // 2, 1),
    )


def weighted_loss(pred, y, w, loss_name: str):
    import torch
    from torch import nn

    if loss_name == "mse":
        return ((pred - y) ** 2 * w).mean()
    if loss_name == "huber":
        loss = nn.functional.smooth_l1_loss(pred, y, reduction="none", beta=1.0)
        return (loss * w).mean()
    if loss_name == "corr_mse":
        mse = ((pred - y) ** 2 * w).mean()
        ws = w / torch.clamp(w.sum(), min=1e-8)
        px = pred - (ws * pred).sum()
        yx = y - (ws * y).sum()
        cov = (ws * px * yx).sum()
        var_p = torch.clamp((ws * px * px).sum(), min=1e-8)
        var_y = torch.clamp((ws * yx * yx).sum(), min=1e-8)
        corr_loss = 1.0 - cov / torch.sqrt(var_p * var_y)
        return 0.65 * mse + 0.35 * corr_loss
    raise ValueError(f"bad loss: {loss_name}")
