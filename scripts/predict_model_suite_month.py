#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.feature_matrix import FeatureMatrix, read_feature_list
from fu_alpha_research.modeling import scrub_matrix


def ridge_predict(df: pd.DataFrame, features: list[str], model_file: Path) -> np.ndarray:
    params = np.load(model_file)
    x = scrub_matrix(df[features].to_numpy(np.float32, copy=False))
    return ((x - params["mean"]) / params["scale"]) @ params["weight"] + float(params["y_mean"][0])


def lgbm_predict(df: pd.DataFrame, features: list[str], model_file: Path) -> np.ndarray:
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(model_file))
    num_threads = int(os.environ.get("LIGHTGBM_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "4")))
    chunk_size = int(os.environ.get("PREDICT_CHUNK_ROWS", "50000"))
    parts = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        x = scrub_matrix(chunk[features].to_numpy(np.float32, copy=False))
        parts.append(booster.predict(x, num_threads=num_threads))
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--month", required=True)
    parser.add_argument("--models", default="ridge,lgbm")
    parser.add_argument("--sets", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = Path(args.manifest) if args.manifest else cfg.output_dir / "model_feature_sets" / "manifest.csv"
    if not manifest_path.is_absolute():
        manifest_path = cfg.output_dir / manifest_path
    manifest = pd.read_csv(manifest_path)
    if args.sets:
        wanted = {x.strip() for x in args.sets.split(",") if x.strip()}
        manifest = manifest[manifest["set"].isin(wanted)].copy()
        if manifest.empty:
            raise ValueError(f"no matching sets for {sorted(wanted)}")
    set_features = {row.set: read_feature_list(row.path) for row in manifest.itertuples(index=False)}
    union_features = list(dict.fromkeys(x for features in set_features.values() for x in features))

    model_kinds = [x.strip() for x in args.models.split(",") if x.strip()]
    model_dir = cfg.output_dir / "models"
    todo: list[tuple[str, str, Path]] = []
    for kind in model_kinds:
        prefix = "lgbm" if kind in {"lgbm", "lightgbm"} else kind
        for set_name in set_features:
            name = f"{prefix}_{set_name}"
            out_file = cfg.output_dir / "prediction_parts" / name / f"{args.month}.parquet"
            meta_file = model_dir / f"{name}.json"
            if not meta_file.exists():
                print(f"[predict-suite] missing model {name}, skip", flush=True)
                continue
            if out_file.exists() and not args.force:
                print(f"[predict-suite] exists {out_file}", flush=True)
                continue
            todo.append((name, set_name, meta_file))
    if not todo:
        print(f"[predict-suite] nothing to do for {args.month}", flush=True)
        return

    matrix = FeatureMatrix(cfg)
    df = matrix.read_month(args.month, union_features)
    base_out = df[["symbol", "datetime", "label"]].copy()
    print(
        f"[predict-suite] loaded {args.month} rows={len(df)} union_features={len(union_features)} models={len(todo)}",
        flush=True,
    )

    for name, _set_name, meta_file in todo:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        features = meta["features"]
        if meta["model"] == "ridge":
            pred = ridge_predict(df, features, model_dir / f"{name}.npz")
        elif meta["model"] == "lightgbm":
            pred = lgbm_predict(df, features, Path(meta["model_file"]))
        else:
            raise ValueError(f"unknown model type: {meta['model']}")
        out_dir = cfg.output_dir / "prediction_parts" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = base_out.copy()
        out["pred"] = np.asarray(pred, dtype=np.float32)
        out_file = out_dir / f"{args.month}.parquet"
        out.to_parquet(out_file, index=False)
        print(f"[predict-suite] wrote {out_file} rows={len(out)}", flush=True)


if __name__ == "__main__":
    main()
