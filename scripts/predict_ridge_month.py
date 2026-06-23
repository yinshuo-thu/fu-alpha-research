#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np

from fu_alpha_research.config import load_config
from fu_alpha_research.factor_store import FactorStore
from fu_alpha_research.modeling import scrub_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--month", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.output_dir / "prediction_parts" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.month}.parquet"
    if out_file.exists() and not args.force:
        print(f"[predict-ridge] exists {out_file}")
        return

    model_dir = cfg.output_dir / "models"
    meta = json.loads((model_dir / f"{args.name}.json").read_text(encoding="utf-8"))
    features = meta["features"]
    params = np.load(model_dir / f"{args.name}.npz")
    store = FactorStore(cfg)
    df = store.read_month(args.month, columns=features)
    x = scrub_matrix(df[features].to_numpy(np.float32, copy=False))
    pred = ((x - params["mean"]) / params["scale"]) @ params["weight"] + float(params["y_mean"][0])
    out = df[["symbol", "datetime", "label"]].copy()
    out["pred"] = pred.astype(np.float32)
    out.to_parquet(out_file, index=False)
    print(f"[predict-ridge] wrote {out_file} rows={len(out)}")


if __name__ == "__main__":
    main()
