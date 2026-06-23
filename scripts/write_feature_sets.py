#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.factor_store import FactorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--sets", default="100,300,all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.output_dir / "feature_sets"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(cfg.reports_dir / "factor_scores.csv")
    ranked = scores.sort_values("abs_is_ic", ascending=False)["factor"].tolist()
    all_features = FactorStore(cfg).selected
    for item in args.sets.split(","):
        item = item.strip().lower()
        if item == "all":
            name = "all"
            features = all_features
        else:
            k = int(item)
            name = f"top_{k}"
            features = ranked[:k]
        path = out_dir / f"{name}.txt"
        path.write_text("\n".join(features) + "\n", encoding="utf-8")
        print(f"[feature-set] wrote {path} n={len(features)}")


if __name__ == "__main__":
    main()
