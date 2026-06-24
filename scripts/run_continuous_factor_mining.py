#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

from fu_alpha_research.config import load_config
from fu_alpha_research.factor_store import FactorStore


def run(cmd: list[str]) -> None:
    print("[continuous-mining] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--seed-step", type=int, default=20)
    parser.add_argument("--pair-step", type=int, default=250)
    parser.add_argument("--block-size", type=int, default=80)
    parser.add_argument("--scorecard-rows-per-month", type=int, default=3000)
    parser.add_argument("--max-corr", type=float, default=0.90)
    parser.add_argument("--selection-pool", type=int, default=800)
    args = parser.parse_args()

    cfg = load_config(args.config)
    store = FactorStore(cfg)
    is_months = store.available_months(cfg.is_start, cfg.is_end)
    oos_months = store.available_months(cfg.oos_start, cfg.oos_end)

    for round_id in range(1, args.max_rounds + 1):
        max_seeds = 80 + (round_id - 1) * args.seed_step
        max_pairs = 500 + (round_id - 1) * args.pair_step
        print(
            f"[continuous-mining] round={round_id} target={args.target} "
            f"max_seeds={max_seeds} max_pairs={max_pairs}",
            flush=True,
        )
        run(
            [
                sys.executable,
                "scripts/generate_expression_candidates.py",
                "--config",
                args.config,
                "--max-seeds",
                str(max_seeds),
                "--max-pairs",
                str(max_pairs),
            ]
        )
        for month in is_months:
            run(
                [
                    sys.executable,
                    "scripts/compute_expression_ic_part.py",
                    "--config",
                    args.config,
                    "--prefix",
                    "is",
                    "--month",
                    month,
                    "--block-size",
                    str(args.block_size),
                ]
            )
        for month in oos_months:
            run(
                [
                    sys.executable,
                    "scripts/compute_expression_ic_part.py",
                    "--config",
                    args.config,
                    "--prefix",
                    "oos",
                    "--month",
                    month,
                    "--block-size",
                    str(args.block_size),
                ]
            )
        run(
            [
                sys.executable,
                "scripts/aggregate_expression_factors.py",
                "--config",
                args.config,
                "--top-n",
                str(args.target),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/evaluate_expression_scorecard.py",
                "--config",
                args.config,
                "--target",
                str(args.target),
                "--rows-per-month",
                str(args.scorecard_rows_per_month),
                "--max-corr",
                str(args.max_corr),
                "--selection-pool",
                str(args.selection_pool),
            ]
        )
        selected = pd.read_csv(cfg.reports_dir / "new_effective_factors_scorecard.csv")
        if len(selected) >= args.target:
            print(f"[continuous-mining] target reached selected={len(selected)}", flush=True)
            return
        print(f"[continuous-mining] selected={len(selected)} < target={args.target}; expanding next round", flush=True)

    raise RuntimeError(f"target not reached after {args.max_rounds} rounds")


if __name__ == "__main__":
    main()
