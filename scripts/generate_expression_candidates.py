#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.expressions import generate_candidate_expressions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--max-seeds", type=int, default=80)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    scores = pd.read_csv(cfg.reports_dir / "factor_scores.csv")
    candidates = generate_candidate_expressions(scores, max_seeds=args.max_seeds, max_pairs=args.max_pairs)
    out = cfg.reports_dir / "new_factor_candidates.csv" if args.output is None else args.output
    candidates.to_csv(out, index=False)
    print(f"[generate-expressions] wrote {out} candidates={len(candidates)}")


if __name__ == "__main__":
    main()
