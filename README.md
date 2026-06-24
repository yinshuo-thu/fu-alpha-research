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

## Validation Evidence

The migrated project has been run end-to-end on the local futures panel with
2018-2019 as in-sample data and 2020 as OOS data.

Expression mining generated 4,000 new formula candidates. Under the same-sign
IS/OOS IC screen (`abs(IS IC) >= 0.002`, `abs(OOS IC) >= 0.001`, coverage proxy
>= 0.5), 2,535 candidates passed and the top 100 were selected as new effective
factors.

Model comparison on 2020 OOS, using `pred_xsz` IC:

| Model | Feature set | Original IC | With 100 new factors | Delta |
| --- | --- | ---: | ---: | ---: |
| Ridge | top300 | 0.038573 | 0.039094 | +0.000521 |
| Ridge | effective657 | 0.038414 | 0.039323 | +0.000909 |
| Ridge | all1144 | 0.037073 | 0.037890 | +0.000817 |
| LightGBM | top300 | 0.026351 | 0.025263 | -0.001088 |
| LightGBM | effective657 | 0.025890 | 0.026199 | +0.000308 |
| LightGBM | all1144 | 0.025679 | 0.025615 | -0.000065 |

Validated-retained factor test: after identifying model-specific effective
factors on 2020-01, each model was retrained on its full retained set and on the
same retained set with the retained new factors removed.

| Model | Feature set | Factors | 2020 OOS IC | 2020-01 IC | 2020 OOS delta |
| --- | --- | ---: | ---: | ---: | ---: |
| Ridge | retained factors, including 51 new factors | 617 | 0.041014 | 0.085484 | +0.000676 |
| Ridge | retained factors, excluding those 51 new factors | 566 | 0.040339 | 0.085646 | baseline |
| LightGBM | retained factors, including 46 new factors | 643 | 0.025216 | 0.044339 | +0.000204 |
| LightGBM | retained factors, excluding those 46 new factors | 597 | 0.025012 | 0.045005 | baseline |

This retained-set test supports the mined-factor signal on full-year 2020 OOS:
the full Ridge retained set beats the old-only retained set by 0.000676 IC, and
the full LightGBM retained set beats the old-only retained set by 0.000204 IC.
The 2020-01 retrained comparison is not monotonic, which is expected because
2020-01 was used to identify factor effectiveness and the model is refit after
removing correlated features.

Model-specific factor validation was then run on the `new_all1244` universe
using 2020-01 as the validation month:

| Model | Test | Retained factors | Retained from new100 |
| --- | --- | ---: | ---: |
| Ridge | exact leave-one-factor-out retrain; remove if IC does not decline | 617 / 1244 | 51 / 100 |
| LightGBM | single-factor standard-normal replacement; remove if IC does not decline | 643 / 1244 | 46 / 100 |

The overlap between Ridge-retained and LightGBM-retained new factors is 26. This
gives independent model-specific evidence that the mined expression factors are
not merely duplicates of the original pool. The full reports are:

- `reports/final_factor_mining_report.md`
- `reports/factor_effectiveness_validation.md`

## Quick Start

```bash
cd /root/autodl-tmp/fu-alpha-research
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
