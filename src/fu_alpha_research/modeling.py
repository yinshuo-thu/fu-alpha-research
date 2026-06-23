from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def scrub_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.array(x, dtype=np.float32, copy=True)
    return np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class RidgeModel:
    feature_cols: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    y_mean: float
    alpha: float

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = scrub_matrix(df[self.feature_cols].to_numpy(np.float32, copy=False))
        return ((x - self.mean) / self.scale) @ self.weight + self.y_mean


def fit_ridge(df: pd.DataFrame, feature_cols: list[str], alpha: float = 1.0, top_k: int = 0) -> RidgeModel:
    clean = df.dropna(subset=["label"])
    x = scrub_matrix(clean[feature_cols].to_numpy(np.float32, copy=False))
    y = clean["label"].to_numpy(np.float64, copy=False)
    y_mask = np.isfinite(y)
    x = x[y_mask]
    y = y[y_mask]
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    xz = ((x - mean) / scale).astype(np.float32)
    y_mean = float(y.mean())
    y0 = y - y_mean
    gram = (xz.T @ xz).astype(np.float64) / max(len(xz), 1)
    cov = (xz.T @ y0).astype(np.float64) / max(len(xz), 1)
    if top_k and top_k < len(feature_cols):
        keep = np.argpartition(np.abs(cov), -top_k)[-top_k:]
        w_sub = np.linalg.solve(gram[np.ix_(keep, keep)] + alpha * np.eye(len(keep)), cov[keep])
        weight = np.zeros(len(feature_cols), dtype=np.float64)
        weight[keep] = w_sub
    else:
        weight = np.linalg.solve(gram + alpha * np.eye(len(feature_cols)), cov)
    return RidgeModel(feature_cols, mean, scale, weight.astype(np.float32), y_mean, alpha)


def fit_lightgbm(
    df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 42,
    params: dict[str, Any] | None = None,
):
    import lightgbm as lgb

    clean = df.dropna(subset=["label"])
    x = scrub_matrix(clean[feature_cols].to_numpy(np.float32, copy=False))
    y = clean["label"].to_numpy(np.float32, copy=False)
    base_params = dict(
        n_estimators=260,
        learning_rate=0.035,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.65,
        min_child_samples=120,
        reg_lambda=4.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        base_params.update(params)
    model = lgb.LGBMRegressor(**base_params)
    model.fit(x, y)
    return model


def predict_lightgbm(model: Any, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = scrub_matrix(df[feature_cols].to_numpy(np.float32, copy=False))
    return model.predict(x)
