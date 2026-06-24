#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.factor_store import FactorStore
from fu_alpha_research.feature_matrix import write_feature_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--new-count", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.output_dir / "model_feature_sets"
    out_dir.mkdir(parents=True, exist_ok=True)

    original_scores = pd.read_csv(cfg.reports_dir / "factor_scores.csv")
    original_ranked = original_scores.sort_values("abs_is_ic", ascending=False)["factor"].tolist()
    original_effective = pd.read_csv(cfg.reports_dir / "effective_factors.csv")["factor"].tolist()
    original_all = FactorStore(cfg).selected

    new_scores = pd.read_csv(cfg.reports_dir / "new_effective_factors_100.csv").head(args.new_count).copy()
    new_names = new_scores["name"].tolist()
    new_as_scores = new_scores.rename(columns={"name": "factor"})
    combined_scores = pd.concat(
        [original_scores[["factor", "abs_is_ic"]], new_as_scores[["factor", "abs_is_ic"]]],
        ignore_index=True,
    )
    combined_ranked = combined_scores.sort_values("abs_is_ic", ascending=False)["factor"].tolist()

    sets = {
        "orig_top300": original_ranked[:300],
        "orig_effective657": original_effective,
        "orig_all1144": original_all,
        "new_top300": combined_ranked[:300],
        "new_effective757": original_effective + new_names,
        "new_all1244": original_all + new_names,
    }

    manifest_rows = []
    for name, features in sets.items():
        features = list(dict.fromkeys(features))
        path = out_dir / f"{name}.txt"
        write_feature_list(path, features)
        new_count = sum(x in set(new_names) for x in features)
        manifest_rows.append(
            {
                "set": name,
                "path": str(path),
                "features": len(features),
                "original_features": len(features) - new_count,
                "new_features": new_count,
            }
        )
        print(f"[feature-set] wrote {path} n={len(features)} new={new_count}")

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"[feature-set] wrote {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
