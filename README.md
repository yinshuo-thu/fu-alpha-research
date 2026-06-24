# FU Alpha Research

Local China futures alpha mining, factor validation, and model-feature research
project.

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
- A local library of existing futures factors, including price-volume,
  technical, time-series normalized, and cross-sectional normalized views. The
  exact catalog lives in `references/futures/factor_catalog.csv`.
- In-sample window defaults to 2018-2019.
- OOS window defaults to 2020.
- Baselines: Ridge and LightGBM.
- Factor mining: dynamic expression generation plus a multi-layer scorecard.
  Full acceptance is not based on full-sample IC or a single backtest.
- Backtest: simple timestamp-level long-short spread on prediction ranks.
- Factor acceptance now follows the multi-layer framework in
  `docs/factor_evaluation_framework.md`, covering data quality, IC, bucket,
  regime, trading simulation, incremental value, robustness, and A-E decisions.

## Validation Evidence

The project has been run end-to-end on the local futures panel with 2018-2019
as in-sample data and 2020 as OOS data.

Earlier experiments used a fast same-sign IS/OOS IC screen as a candidate
prefilter. That screen is now treated only as a cheap triage stage. New factors
must pass the scorecard workflow before they can be called effective: data
quality, Pearson and rank IC, daily/monthly stability, product-level IC, bucket
monotonicity, top-bottom spread, liquidity/volatility regimes, turnover/cost
proxies, correlation with the existing library, residualized IC, and model
incremental tests.

The current mining skill is designed as a continuous loop:

1. Generate auditable expression candidates from a diversified seed pool rather
   than only the strongest IC seeds.
2. Aggregate cheap IC parts to remove obvious failures.
3. Run `scripts/evaluate_expression_scorecard.py` on the remaining candidates.
   This computes bucket, product, regime, turnover, library-correlation, and
   residual-IC diagnostics.
4. Enforce low-correlation gates: a candidate should have max absolute
   correlation <= 0.90 against the existing library, unless its residualized IC
   still justifies keeping it as a model feature.
5. Select candidates greedily with family/operator diversity and candidate-peer
   correlation checks, then write `outputs/expression_sets/new100.csv`.
6. Validate selected factors inside Ridge and LightGBM with leave-one-factor-out
   or shuffle tests before reporting model IC lift.

This design borrows the useful v3 idea of a persistent accepted pool,
low-correlation gating, train-only discovery, OOS reporting, and full audit logs,
but adapts it to this futures factor panel and local Ridge/LightGBM workflow.

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

Second-round skill optimization: using the expanded multi-layer mining skill,
another 100 formula factors were generated from the remaining candidate pool and
tested only as incremental additions to the already validated model sets. Ridge
started from its 617 retained factors; LightGBM started from its 643 retained
factors. After model-specific validation on 2020-01, the retained new factors
were added back, models were retrained on 2018-2019, and 2020 full-year OOS IC
was recomputed.

| Model | Validation base | Test | Effective from new100 | Final factors | Base 2020 OOS IC | +100 candidate IC | +effective-new IC | Delta vs base | Delta vs +100 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 617 | exact leave-one-factor-out retrain | 42 / 100 | 659 | 0.041014 | 0.041064 | 0.041071 | +0.000056 | +0.000006 |
| LightGBM | 643 | single-factor standard-normal replacement | 40 / 100 | 683 | 0.025216 | 0.025852 | 0.026121 | +0.000905 | +0.000269 |

The second round therefore adds positive full-year 2020 OOS IC after validation:
Ridge improves by 0.000056 IC over the 617-factor base, and LightGBM improves by
0.000905 IC over the 643-factor base. The filtered LightGBM set also beats
directly adding all 100 candidates by 0.000269 IC, showing that the shuffle-based
incremental filter removed harmful or redundant candidates. Ridge and LightGBM
jointly retained 14 of the second-round new factors. The compact round-two
factor summary, including the 100 generated formulas and the model-specific
retained/removed lists, is committed at
`references/futures/round2_effective_factors_summary.json`; the local generated
expression and feature-set artifacts are:

- `outputs/expression_sets/new100_round2.csv`
- `outputs/expression_sets/new200.csv`
- `outputs/model_feature_sets/ridge617_plus_round2_retained42.txt`
- `outputs/model_feature_sets/lgbm643_plus_round2_retained40.txt`

For the first 100-factor round, the overlap between Ridge-retained and
LightGBM-retained new factors was 26. These model-specific validations give
independent evidence that the mined expression factors are not merely duplicates
of the original pool. The full reports are:

- `reports/final_factor_mining_report.md`
- `reports/factor_effectiveness_validation.md`
- `references/futures/old_effective_factors_summary.json`
- `references/futures/round2_effective_factors_summary.json`

## Quick Start

```bash
cd /root/autodl-tmp/fu-alpha-research
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml audit-data
# Existing-library IC prefilter, not final new-expression acceptance.
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml mine-factors
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml baseline --models ridge,lightgbm
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml incremental --sets 100,300,all
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml report
```

For the current scorecard-based expression mining loop:

```bash
PYTHONPATH=src python scripts/run_continuous_factor_mining.py --config configs/futures.yaml --target 100
```

Or run the detailed scorecard directly after generating candidates and IC parts:

```bash
PYTHONPATH=src python scripts/evaluate_expression_scorecard.py \
  --config configs/futures.yaml \
  --target 100 \
  --rows-per-month 3000 \
  --max-corr 0.90
```

The same CLI is exposed as `fu-alpha` after installing the package:

```bash
pip install -e .
fu-alpha --config configs/futures.yaml run-all
```

## Rebuild Factors

If the final panel is missing, the loader can materialize the existing factor
library on the fly from intermediate month partitions. To rebuild those
partitions from raw CSV files:

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
- `old_effective_factors_summary.json`
- `round2_effective_factors_summary.json`

Do not commit `/root/autodl-tmp/quant/data/raw`, `selected_month_parts`,
`data_factors_big.parquet`, or prediction parquet files.
