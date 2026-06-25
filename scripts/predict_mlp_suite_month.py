#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fu_alpha_research.config import load_config
from fu_alpha_research.feature_matrix import FeatureMatrix, read_feature_list
from fu_alpha_research.mlp import make_mlp, scrub_matrix


def resolve_output_path(output_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else output_dir / path


def predict(model, x: np.ndarray, device: torch.device, chunk_rows: int) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), chunk_rows):
            end = min(start + chunk_rows, len(x))
            xb = torch.from_numpy(x[start:end]).to(device, non_blocking=True)
            out[start:end] = model(xb).squeeze(-1).detach().cpu().numpy().astype(np.float32)
    return out


def load_mlp(model_file: Path, device: torch.device):
    obj = torch.load(model_file, map_location=device, weights_only=False)
    config = obj["config"]
    features = list(obj["features"])
    model = make_mlp(len(features), int(config["hidden"]), float(config["dropout"])).to(device)
    model.load_state_dict(obj["state_dict"])
    return model, features, obj["mean"].astype(np.float32), obj["scale"].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--sets", default=None)
    parser.add_argument("--expression-file", default=None)
    parser.add_argument("--chunk-rows", type=int, default=200000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = resolve_output_path(cfg.output_dir, args.manifest)
    manifest = pd.read_csv(manifest_path)
    if args.sets:
        wanted = {x.strip() for x in args.sets.split(",") if x.strip()}
        manifest = manifest[manifest["set"].isin(wanted)].copy()
    set_features = {row.set: read_feature_list(row.path) for row in manifest.itertuples(index=False)}
    union_features = list(dict.fromkeys(x for features in set_features.values() for x in features))
    todo = []
    model_dir = cfg.output_dir / "models"
    for set_name in set_features:
        name = f"mlp_{set_name}"
        meta_path = model_dir / f"{name}.json"
        out_file = cfg.output_dir / "prediction_parts" / name / f"{args.month}.parquet"
        if not meta_path.exists():
            print(f"[predict-mlp] missing model {name}, skip")
            continue
        if out_file.exists() and not args.force:
            print(f"[predict-mlp] exists {out_file}")
            continue
        todo.append((name, meta_path, out_file))
    if not todo:
        print(f"[predict-mlp] nothing to do for {args.month}")
        return

    expr_path = resolve_output_path(cfg.output_dir, args.expression_file)
    df = FeatureMatrix(cfg, expr_path).read_month(args.month, union_features)
    base_out = df[["symbol", "datetime", "label"]].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict-mlp] loaded {args.month} rows={len(df)} union_features={len(union_features)} models={len(todo)}")

    for name, meta_path, out_file in todo:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model, features, mean, scale = load_mlp(Path(meta["model_file"]), device)
        x = scrub_matrix(df[features].to_numpy(np.float32, copy=False))
        x = ((x - mean) / scale).astype(np.float32)
        pred = predict(model, x, device, args.chunk_rows)
        out = base_out.copy()
        out["pred"] = pred
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_file, index=False)
        print(f"[predict-mlp] wrote {out_file} rows={len(out)}")
        del model, x, out, pred
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
