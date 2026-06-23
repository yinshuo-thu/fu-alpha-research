# FU Alpha Research

Local China futures alpha mining migrated from the lightweight
`wq-alpha-research` playbook into a runnable futures research project.

The project is designed for the AutoDL workspace layout:

- raw 1-minute futures bars live outside git at `/root/autodl-tmp/quant/data/raw`;
- intermediate month partitions live outside git at
  `/root/autodl-tmp/shared-nvme/feature_model/selected_month_parts`;
- the final rebuilt factor panel, if present, lives outside git at
  `/root/autodl-tmp/shared-nvme/feature_model/data_factors_big.parquet`;
- only source code and small reference metadata are committed.

## What This Runs

- 51 China futures, excluding financial futures by default.
- 30-bar forward return label.
- 1,144 selected factors:
  402 raw, 164 time-series z-score, 299 cross-sectional z-score, and
  279 cross-sectional rank factors.
- In-sample window defaults to 2018-2019.
- OOS window defaults to 2020.
- Baselines: Ridge and LightGBM.
- Factor mining: IS/OOS single-factor IC with same-sign effective-factor filter.
- Backtest: simple timestamp-level long-short spread on prediction ranks.

## Quick Start

```bash
cd /root/autodl-tmp/wq-alpha-research
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml audit-data
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml mine-factors
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml baseline --models ridge,lightgbm
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml incremental --sets 100,300,all
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml report
```

The same CLI is exposed as `fu-alpha` after installing the package:

```bash
pip install -e .
fu-alpha --config configs/futures.yaml run-all
```

## Rebuild Factors

If the final panel is missing, the loader can materialize 1,144 factors on the
fly from intermediate month partitions. To rebuild those partitions from raw
CSV files:

```bash
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml \
  build-partitions --start 2017-01-01 --end-exclusive 2021-01-01 --overwrite
```

To write a final monolithic parquet from month partitions:

```bash
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml \
  materialize-final --path /root/autodl-tmp/shared-nvme/feature_model/data_factors_big.parquet \
  --start 2018-01-01 --end 2020-12-31
```

## Git Hygiene

The repository intentionally ignores raw data, rebuilt parquet panels, model
outputs, generated reports, and logs. The committed `references/futures`
directory contains only small metadata:

- `selected_factors.txt`
- `factor_catalog.csv`

Do not commit `/root/autodl-tmp/quant/data/raw`, `selected_month_parts`,
`data_factors_big.parquet`, or prediction parquet files.
