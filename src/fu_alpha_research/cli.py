from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .build_panel import build_intermediate_partitions, materialize_final_parquet
from .config import load_config
from .experiments import (
    audit_data,
    run_baseline,
    run_factor_mining,
    run_incremental_models,
    write_markdown_report,
)
from .metrics import long_short_backtest, summarize_backtest


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/futures.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(prog="fu-alpha")
    add_common(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("audit-data")

    p_mine = sub.add_parser("mine-factors")
    p_mine.add_argument("--min-is-ic", type=float, default=0.002)
    p_mine.add_argument("--min-oos-ic", type=float, default=0.001)

    p_base = sub.add_parser("baseline")
    p_base.add_argument("--models", default=None, help="Comma-separated models: ridge,lightgbm")
    p_base.add_argument("--feature-file", default=None, help="Optional newline factor list")
    p_base.add_argument("--name", default="all")

    p_inc = sub.add_parser("incremental")
    p_inc.add_argument("--sets", default="100,300,all")

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--predictions", required=True)
    p_bt.add_argument("--pred-col", default="pred_xrank")
    p_bt.add_argument("--quantile", type=float, default=0.2)

    p_build = sub.add_parser("build-partitions")
    p_build.add_argument("--start", default="2017-01-01")
    p_build.add_argument("--end-exclusive", default="2021-01-01")
    p_build.add_argument("--overwrite", action="store_true")

    p_mat = sub.add_parser("materialize-final")
    p_mat.add_argument("--path", required=True)
    p_mat.add_argument("--start", default="2018-01-01")
    p_mat.add_argument("--end", default="2020-12-31")

    sub.add_parser("report")

    p_all = sub.add_parser("run-all")
    p_all.add_argument("--skip-lightgbm", action="store_true")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "audit-data":
        print(json.dumps(audit_data(cfg), indent=2, ensure_ascii=False, default=str))
    elif args.cmd == "mine-factors":
        df = run_factor_mining(cfg, min_is_ic=args.min_is_ic, min_oos_ic=args.min_oos_ic)
        print(df.head(30).to_string(index=False))
    elif args.cmd == "baseline":
        models = args.models.split(",") if args.models else None
        features = None
        if args.feature_file:
            features = [x.strip() for x in Path(args.feature_file).read_text().splitlines() if x.strip()]
        print(run_baseline(cfg, model_names=models, feature_cols=features, name_suffix=args.name).to_string(index=False))
    elif args.cmd == "incremental":
        sets: list[int | str] = []
        for item in args.sets.split(","):
            item = item.strip()
            sets.append(item if item.lower() == "all" else int(item))
        print(run_incremental_models(cfg, sets).to_string(index=False))
    elif args.cmd == "backtest":
        pred = pd.read_parquet(args.predictions)
        bt = long_short_backtest(pred, pred_col=args.pred_col, quantile=args.quantile)
        out = summarize_backtest(bt)
        print(json.dumps(out, indent=2))
    elif args.cmd == "build-partitions":
        build_intermediate_partitions(cfg, start=args.start, end_exclusive=args.end_exclusive, overwrite=args.overwrite)
    elif args.cmd == "materialize-final":
        materialize_final_parquet(cfg, Path(args.path), args.start, args.end)
    elif args.cmd == "report":
        print(write_markdown_report(cfg))
    elif args.cmd == "run-all":
        audit_data(cfg)
        run_factor_mining(cfg)
        models = ["ridge"] if args.skip_lightgbm else ["ridge", "lightgbm"]
        run_baseline(cfg, model_names=models)
        run_incremental_models(cfg, [100, 300, "all"])
        print(write_markdown_report(cfg))


if __name__ == "__main__":
    main()
