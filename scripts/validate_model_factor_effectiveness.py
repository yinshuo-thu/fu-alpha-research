#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.feature_matrix import FeatureMatrix, read_feature_list, write_feature_list
from fu_alpha_research.modeling import scrub_matrix


def group_bounds(datetimes: pd.Series) -> list[tuple[int, int]]:
    values = pd.to_datetime(datetimes).to_numpy()
    if len(values) == 0:
        return []
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    edges = np.r_[0, change, len(values)]
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]


def pred_xsz_ic(pred: np.ndarray, label: np.ndarray, bounds: list[tuple[int, int]]) -> float:
    xy = 0.0
    xx = 0.0
    yy = 0.0
    for start, end in bounds:
        p = pred[start:end].astype(np.float64, copy=False)
        y = label[start:end]
        if len(p) < 2:
            continue
        sd = float(np.nanstd(p, ddof=1))
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        z = (p - float(np.nanmean(p))) / (sd + 1e-9)
        mask = np.isfinite(y)
        if not mask.any():
            continue
        zv = z[mask]
        yv = y[mask]
        xy += float(np.dot(zv, yv))
        xx += float(np.dot(zv, zv))
        yy += float(np.dot(yv, yv))
    denom = math.sqrt(max(xx * yy, 1e-30))
    return xy / denom


def matrix_pred_xsz_ic(preds: np.ndarray, label: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64)
    out_len = preds.shape[1]
    xy = np.zeros(out_len, dtype=np.float64)
    xx = np.zeros(out_len, dtype=np.float64)
    yy = 0.0
    for start, end in bounds:
        p = preds[start:end]
        y = label[start:end]
        if len(p) < 2:
            continue
        mu = np.nanmean(p, axis=0)
        sd = np.nanstd(p, axis=0, ddof=1)
        ok = np.isfinite(sd) & (sd > 1e-12)
        if not ok.any():
            continue
        z = (p[:, ok] - mu[ok]) / (sd[ok] + 1e-9)
        mask = np.isfinite(y)
        if not mask.any():
            continue
        zv = z[mask]
        yv = y[mask]
        xy[ok] += zv.T @ yv
        xx[ok] += np.sum(zv * zv, axis=0)
        yy += float(np.dot(yv, yv))
    denom = np.sqrt(np.maximum(xx * yy, 1e-30))
    return xy / denom


def load_train_sample(cfg, features: list[str], sample_dir: Path) -> pd.DataFrame:
    files = sorted(sample_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no sample parquet files in {sample_dir}")
    frames = [pd.read_parquet(path, columns=["label"] + features) for path in files]
    return pd.concat(frames, ignore_index=True)


def resolve_output_path(cfg, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = cfg.output_dir / path
    return path


def load_eval_month(
    cfg,
    month: str,
    features: list[str],
    expression_path: Path | None,
) -> tuple[pd.DataFrame, np.ndarray, list[tuple[int, int]]]:
    df = FeatureMatrix(cfg, expression_path).read_month(month, features)
    df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    label = df["label"].to_numpy(np.float64, copy=False)
    bounds = group_bounds(df["datetime"])
    return df, label, bounds


def new_factor_set(cfg, path_arg: str | None) -> set[str]:
    path = resolve_output_path(cfg, path_arg) if path_arg else cfg.output_dir / "expression_sets" / "new100.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["name"].tolist())


def target_factor_indices(features: list[str], path_arg: str | None) -> list[int]:
    if not path_arg:
        return list(range(len(features)))
    target_path = Path(path_arg)
    target = set(read_feature_list(target_path))
    return [idx for idx, name in enumerate(features) if name in target]


def run_ridge(args) -> None:
    cfg = load_config(args.config)
    feature_file = Path(args.feature_file)
    if not feature_file.is_absolute():
        feature_file = cfg.output_dir / feature_file
    features = read_feature_list(feature_file)
    new_names = new_factor_set(cfg, args.new_factor_file)
    target_indices = target_factor_indices(features, args.target_factor_file)
    expression_path = resolve_output_path(cfg, args.expression_file)

    out_dir = cfg.reports_dir / "effectiveness_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    out_file = out_dir / f"ridge_leave_one_{args.month}{tag}.csv"
    if out_file.exists() and not args.force:
        print(f"[ridge-validate] exists {out_file}")
        return

    sample_dir = cfg.output_dir / args.sample_dir
    train = load_train_sample(cfg, features, sample_dir).dropna(subset=["label"])
    x_train = scrub_matrix(train[features].to_numpy(np.float32, copy=False))
    y = train["label"].to_numpy(np.float64, copy=False)
    mask = np.isfinite(y)
    x_train = x_train[mask]
    y = y[mask]
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x_train.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    xz = ((x_train - mean) / scale).astype(np.float32)
    y_mean = float(y.mean())
    y0 = y - y_mean
    print(f"[ridge-validate] train rows={len(xz)} features={len(features)}", flush=True)

    gram = (xz.T @ xz).astype(np.float64) / max(len(xz), 1)
    cov = (xz.T @ y0).astype(np.float64) / max(len(xz), 1)
    del x_train, xz, train
    gc.collect()

    k_mat = gram + args.alpha * np.eye(len(features), dtype=np.float64)
    k_inv = np.linalg.inv(k_mat)
    weight = k_inv @ cov
    diag = np.diag(k_inv)
    print("[ridge-validate] solved full ridge and inverse", flush=True)
    del gram, k_mat, cov
    gc.collect()

    eval_df, label, bounds = load_eval_month(cfg, args.month, features, expression_path)
    x_eval = scrub_matrix(eval_df[features].to_numpy(np.float32, copy=False))
    xz_eval = ((x_eval - mean) / scale).astype(np.float32)
    del x_eval, eval_df
    gc.collect()

    full_pred = xz_eval @ weight + y_mean
    base_ic = pred_xsz_ic(full_pred, label, bounds)
    print(f"[ridge-validate] base pred_xsz_ic={base_ic:.8f}", flush=True)

    rows = []
    for start in range(0, len(target_indices), args.block_size):
        end = min(start + args.block_size, len(target_indices))
        idx = np.asarray(target_indices[start:end], dtype=np.int64)
        h = xz_eval @ k_inv[:, idx]
        adjust = weight[idx] / diag[idx]
        drop_preds = full_pred[:, None] - h * adjust[None, :]
        drop_ics = matrix_pred_xsz_ic(drop_preds, label, bounds)
        for local, factor_idx in enumerate(idx):
            drop_ic = float(drop_ics[local])
            delta = float(base_ic - drop_ic)
            rows.append(
                {
                    "factor": features[factor_idx],
                    "is_new_factor": features[factor_idx] in new_names,
                    "base_ic": base_ic,
                    "drop_ic": drop_ic,
                    "delta_ic": delta,
                    "retained": bool(drop_ic < base_ic - args.tolerance),
                    "method": "ridge_leave_one_retrain_closed_form",
                    "month": args.month,
                }
            )
        print(f"[ridge-validate] evaluated {end}/{len(target_indices)}", flush=True)

    result = pd.DataFrame(rows).sort_values("delta_ic", ascending=False)
    result.to_csv(out_file, index=False)
    retained = result[result["retained"]]["factor"].tolist()
    removed = result[~result["retained"]]["factor"].tolist()
    write_feature_list(out_dir / f"ridge_retained_{args.month}{tag}.txt", retained)
    write_feature_list(out_dir / f"ridge_removed_{args.month}{tag}.txt", removed)
    summary = {
        "model": "ridge",
        "month": args.month,
        "base_ic": base_ic,
        "features": len(features),
        "target_features": len(target_indices),
        "retained": len(retained),
        "removed": len(removed),
        "new_factor_pool": len(new_names),
        "retained_new_factors": int(result[result["retained"]]["is_new_factor"].sum()),
        "removed_new_factors": int(result[~result["retained"]]["is_new_factor"].sum()),
        "tolerance": args.tolerance,
    }
    (out_dir / f"ridge_leave_one_{args.month}{tag}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def predict_lgbm_chunks(booster, x: np.ndarray, chunk_rows: int, num_threads: int) -> np.ndarray:
    parts = []
    for start in range(0, len(x), chunk_rows):
        parts.append(booster.predict(x[start : start + chunk_rows], num_threads=num_threads))
    return np.concatenate(parts)


def run_lgbm(args) -> None:
    import lightgbm as lgb

    cfg = load_config(args.config)
    feature_file = Path(args.feature_file)
    if not feature_file.is_absolute():
        feature_file = cfg.output_dir / feature_file
    features = read_feature_list(feature_file)
    new_names = new_factor_set(cfg, args.new_factor_file)
    expression_path = resolve_output_path(cfg, args.expression_file)

    out_dir = cfg.reports_dir / "effectiveness_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    shard_suffix = ""
    if args.num_shards > 1:
        shard_suffix = f"_shard{args.shard_index:02d}of{args.num_shards:02d}"
    out_file = out_dir / f"lgbm_shuffle_{args.month}{tag}{shard_suffix}.csv"
    partial_file = out_dir / f"lgbm_shuffle_{args.month}{tag}{shard_suffix}.partial.csv"
    if out_file.exists() and not args.force:
        print(f"[lgbm-validate] exists {out_file}")
        return

    model_file = Path(args.model_file)
    if not model_file.is_absolute():
        model_file = cfg.output_dir / model_file
    booster = lgb.Booster(model_file=str(model_file))
    split = booster.feature_importance("split")
    gain = booster.feature_importance("gain")

    eval_df, label, bounds = load_eval_month(cfg, args.month, features, expression_path)
    x_eval = scrub_matrix(eval_df[features].to_numpy(np.float32, copy=False))
    del eval_df
    gc.collect()
    base_pred = predict_lgbm_chunks(booster, x_eval, args.chunk_rows, args.num_threads)
    base_ic = pred_xsz_ic(base_pred, label, bounds)
    print(
        f"[lgbm-validate] eval rows={len(x_eval)} features={len(features)} "
        f"used_split={int((split > 0).sum())} base_ic={base_ic:.8f}",
        flush=True,
    )

    done: dict[str, dict[str, object]] = {}
    if partial_file.exists() and not args.force:
        partial = pd.read_csv(partial_file)
        done = {row.factor: row._asdict() for row in partial.itertuples(index=False)}
        print(f"[lgbm-validate] resuming done={len(done)}", flush=True)
    elif partial_file.exists() and args.force:
        partial_file.unlink()

    rows: list[dict[str, object]] = list(done.values())
    rng = np.random.default_rng(args.seed)
    normal_cache: dict[int, np.ndarray] = {}

    base_target_indices = target_factor_indices(features, args.target_factor_file)
    target_indices = [
        idx for idx in base_target_indices if args.num_shards <= 1 or idx % args.num_shards == args.shard_index
    ]
    if args.max_features:
        target_indices = target_indices[: args.max_features]

    for idx in target_indices:
        factor = features[idx]
        if factor in done:
            continue

        if split[idx] <= 0:
            shuffled_ic = base_ic
            delta = 0.0
            retained = False
        else:
            original = x_eval[:, idx].copy()
            if args.random_mode == "standard_normal":
                replacement = rng.standard_normal(len(x_eval)).astype(np.float32)
            else:
                mu = float(np.nanmean(original))
                sd = float(np.nanstd(original))
                replacement = rng.normal(mu, sd if sd > 1e-8 else 1.0, len(x_eval)).astype(np.float32)
            normal_cache.clear()
            x_eval[:, idx] = replacement
            pred = predict_lgbm_chunks(booster, x_eval, args.chunk_rows, args.num_threads)
            x_eval[:, idx] = original
            shuffled_ic = pred_xsz_ic(pred, label, bounds)
            delta = float(base_ic - shuffled_ic)
            retained = bool(shuffled_ic < base_ic - args.tolerance)

        rows.append(
            {
                "factor": factor,
                "is_new_factor": factor in new_names,
                "base_ic": base_ic,
                "shuffled_ic": float(shuffled_ic),
                "delta_ic": float(delta),
                "retained": retained,
                "split_importance": int(split[idx]),
                "gain_importance": float(gain[idx]),
                "random_mode": args.random_mode,
                "method": "lightgbm_single_factor_normal_replacement",
                "month": args.month,
            }
        )
        if (len(rows) % args.flush_every == 0) or len(rows) == len(target_indices):
            pd.DataFrame(rows).to_csv(partial_file, index=False)
            print(f"[lgbm-validate] evaluated {len(rows)}/{len(target_indices)}", flush=True)

    result = pd.DataFrame(rows).sort_values("delta_ic", ascending=False)
    result.to_csv(out_file, index=False)
    pd.DataFrame(rows).to_csv(partial_file, index=False)
    retained_factors = result[result["retained"]]["factor"].tolist()
    removed_factors = result[~result["retained"]]["factor"].tolist()
    write_feature_list(out_dir / f"lgbm_retained_{args.month}{tag}{shard_suffix}.txt", retained_factors)
    write_feature_list(out_dir / f"lgbm_removed_{args.month}{tag}{shard_suffix}.txt", removed_factors)
    summary = {
        "model": "lightgbm",
        "month": args.month,
        "base_ic": base_ic,
        "features": len(features),
        "target_features": len(target_indices),
        "evaluated": int(len(result)),
        "retained": len(retained_factors),
        "removed": len(removed_factors),
        "new_factor_pool": len(new_names),
        "retained_new_factors": int(result[result["retained"]]["is_new_factor"].sum()),
        "removed_new_factors": int(result[~result["retained"]]["is_new_factor"].sum()),
        "zero_split_features": int((split == 0).sum()),
        "tolerance": args.tolerance,
        "random_mode": args.random_mode,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    (out_dir / f"lgbm_shuffle_{args.month}{tag}{shard_suffix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--mode", choices=["ridge", "lgbm"], required=True)
    parser.add_argument("--month", default="2020-01")
    parser.add_argument("--feature-file", default="model_feature_sets/new_all1244.txt")
    parser.add_argument("--expression-file", default=None)
    parser.add_argument("--new-factor-file", default=None)
    parser.add_argument("--target-factor-file", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--sample-dir", default="extended_samples/is")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=32)

    parser.add_argument("--model-file", default="models/lgbm_new_all1244.txt")
    parser.add_argument("--seed", type=int, default=202601)
    parser.add_argument("--chunk-rows", type=int, default=int(os.environ.get("PREDICT_CHUNK_ROWS", "50000")))
    parser.add_argument("--num-threads", type=int, default=int(os.environ.get("LIGHTGBM_NUM_THREADS", "4")))
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=0)
    parser.add_argument("--random-mode", choices=["standard_normal", "match_feature"], default="standard_normal")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    if args.mode == "ridge":
        run_ridge(args)
    else:
        run_lgbm(args)


if __name__ == "__main__":
    main()
