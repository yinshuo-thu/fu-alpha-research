#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fu_alpha_research.config import load_config
from fu_alpha_research.feature_matrix import FeatureMatrix, read_feature_list, write_feature_list
from fu_alpha_research.metrics import add_prediction_views, compute_ic
from fu_alpha_research.mlp import make_mlp, scrub_matrix


def resolve_output_path(output_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else output_dir / path


def load_mlp(model_file: Path, device: torch.device):
    obj = torch.load(model_file, map_location=device, weights_only=False)
    config = obj["config"]
    features = list(obj["features"])
    model = make_mlp(len(features), int(config["hidden"]), float(config["dropout"])).to(device)
    model.load_state_dict(obj["state_dict"])
    model.eval()
    return model, features, obj["mean"].astype(np.float32), obj["scale"].astype(np.float32)


def pred_xsz_ic(pred: np.ndarray, meta: pd.DataFrame) -> float:
    df = meta.copy()
    df["pred"] = pred.astype(np.float32)
    df = add_prediction_views(df, "pred")
    return compute_ic(df["pred_xsz"].to_numpy(), df["label"].to_numpy())


def predict_tensor(model, x_tensor: torch.Tensor, chunk_rows: int) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], chunk_rows):
            end = min(start + chunk_rows, x_tensor.shape[0])
            parts.append(model(x_tensor[start:end]).squeeze(-1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--month", default="2020-01")
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--expression-file", default=None)
    parser.add_argument("--new-factor-file", default=None)
    parser.add_argument("--target-factor-file", default=None)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    feature_file = resolve_output_path(cfg.output_dir, args.feature_file)
    all_features = read_feature_list(feature_file)
    target_file = resolve_output_path(cfg.output_dir, args.target_factor_file)
    target = all_features if target_file is None else read_feature_list(target_file)
    target_set = set(target)
    target_indices = [idx for idx, name in enumerate(all_features) if name in target_set]
    target_indices = [idx for idx in target_indices if args.num_shards <= 1 or idx % args.num_shards == args.shard_index]
    new_names: set[str] = set()
    if args.new_factor_file:
        new_path = resolve_output_path(cfg.output_dir, args.new_factor_file)
        if new_path.exists():
            new_names = set(pd.read_csv(new_path)["name"].tolist())

    out_dir = cfg.reports_dir / "effectiveness_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    shard_suffix = f"_shard{args.shard_index:02d}of{args.num_shards:02d}" if args.num_shards > 1 else ""
    out_file = out_dir / f"mlp_shuffle_{args.month}{tag}{shard_suffix}.csv"
    partial_file = out_dir / f"mlp_shuffle_{args.month}{tag}{shard_suffix}.partial.csv"
    if out_file.exists() and not args.force:
        print(f"[mlp-validate] exists {out_file}")
        return
    if partial_file.exists() and args.force:
        partial_file.unlink()

    expr_path = resolve_output_path(cfg.output_dir, args.expression_file)
    df = FeatureMatrix(cfg, expr_path).read_month(args.month, all_features)
    meta = df[["symbol", "datetime", "label"]].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file = resolve_output_path(cfg.output_dir, args.model_file)
    model, model_features, mean, scale = load_mlp(model_file, device)
    if model_features != all_features:
        raise ValueError("model features do not match --feature-file order")
    x = scrub_matrix(df[all_features].to_numpy(np.float32, copy=False))
    x = ((x - mean) / scale).astype(np.float32)
    x_tensor = torch.from_numpy(x).to(device)
    del df, x
    gc.collect()

    base_pred = predict_tensor(model, x_tensor, args.chunk_rows)
    base_ic = pred_xsz_ic(base_pred, meta)
    print(f"[mlp-validate] rows={len(meta)} features={len(all_features)} targets={len(target_indices)} base_ic={base_ic:.8f}", flush=True)

    rng = np.random.default_rng(args.seed + args.shard_index)
    rows: list[dict[str, object]] = []
    if partial_file.exists():
        partial = pd.read_csv(partial_file)
        rows = partial.to_dict("records")
        done = {str(row["factor"]) for row in rows}
    else:
        done = set()

    for idx in target_indices:
        factor = all_features[idx]
        if factor in done:
            continue
        original = x_tensor[:, idx].clone()
        replacement = torch.from_numpy(rng.standard_normal(x_tensor.shape[0]).astype(np.float32)).to(device)
        x_tensor[:, idx] = replacement
        pred = predict_tensor(model, x_tensor, args.chunk_rows)
        x_tensor[:, idx] = original
        shuffled_ic = pred_xsz_ic(pred, meta)
        delta = float(base_ic - shuffled_ic)
        rows.append(
            {
                "factor": factor,
                "is_new_factor": factor in new_names,
                "base_ic": base_ic,
                "shuffled_ic": float(shuffled_ic),
                "delta_ic": delta,
                "retained": bool(shuffled_ic < base_ic - args.tolerance),
                "method": "mlp_single_factor_standard_normal_replacement",
                "month": args.month,
            }
        )
        if len(rows) % args.flush_every == 0 or len(rows) == len(target_indices):
            pd.DataFrame(rows).to_csv(partial_file, index=False)
            print(f"[mlp-validate] evaluated {len(rows)}/{len(target_indices)}", flush=True)

    result = pd.DataFrame(rows).sort_values("delta_ic", ascending=False)
    result.to_csv(out_file, index=False)
    result.to_csv(partial_file, index=False)
    retained = result[result["retained"]]["factor"].tolist()
    removed = result[~result["retained"]]["factor"].tolist()
    write_feature_list(out_dir / f"mlp_retained_{args.month}{tag}{shard_suffix}.txt", retained)
    write_feature_list(out_dir / f"mlp_removed_{args.month}{tag}{shard_suffix}.txt", removed)
    summary = {
        "model": "mlp",
        "month": args.month,
        "base_ic": base_ic,
        "features": len(all_features),
        "target_features": len(target_indices),
        "evaluated": int(len(result)),
        "retained": len(retained),
        "removed": len(removed),
        "new_factor_pool": len(new_names),
        "retained_new_factors": int(result[result["retained"]]["is_new_factor"].sum()) if len(result) else 0,
        "removed_new_factors": int(result[~result["retained"]]["is_new_factor"].sum()) if len(result) else 0,
        "tolerance": args.tolerance,
        "random_mode": "standard_normal",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    (out_dir / f"mlp_shuffle_{args.month}{tag}{shard_suffix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
