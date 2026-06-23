#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from fu_alpha_research.config import load_config
from fu_alpha_research.factor_store import FactorStore
from fu_alpha_research.mining import month_sufficient_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--prefix", required=True, choices=["is", "oos"])
    parser.add_argument("--month", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.reports_dir / "ic_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.prefix}_{args.month}.parquet"
    if out_file.exists() and not args.force:
        print(f"[ic-part] exists {out_file}")
        return
    store = FactorStore(cfg)
    df = store.read_month(args.month, columns=store.selected)
    stats = month_sufficient_stats(df, store.selected)
    stats.to_parquet(out_file, index=False)
    print(f"[ic-part] wrote {out_file} rows={len(df)} factors={len(stats)}")


if __name__ == "__main__":
    main()
