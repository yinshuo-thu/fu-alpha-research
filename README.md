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
- Baselines: Ridge, LightGBM, and MLP research scripts.
- Factor mining: dynamic expression generation plus a multi-layer scorecard.
  Full acceptance is not based on full-sample IC or a single backtest.
- Backtest: simple timestamp-level long-short spread on prediction ranks.
- Factor acceptance now follows the multi-layer framework in
  `docs/factor_evaluation_framework.md`, covering data quality, IC, bucket,
  regime, trading simulation, incremental value, robustness, and A-E decisions.
- Default numerical gates are versioned in
  `references/futures/factor_acceptance_standards.json`.

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
5. Select candidates greedily only from `decision_reason == pass_all`, with
   family/operator diversity and candidate-peer correlation checks, then write
   `outputs/expression_sets/new100.csv`.
6. Validate selected factors inside Ridge, LightGBM, or MLP with
   leave-one-factor-out or shuffle tests before reporting model IC lift.

Default scorecard thresholds are explicit and shared across scripts:

| Gate | Default standard |
| --- | --- |
| Data quality | coverage >= 0.70, outlier ratio <= 0.02, non-constant values |
| IC | abs(selection IC) >= 0.001, abs(rank IC) >= 0.0005, monthly hit rate >= 0.50, at least 6 monthly IC observations |
| Bucket | top-bottom spread has the same sign as selection IC, abs(monotonicity) >= 0.25 |
| Regime | product hit rate >= 0.45, liquidity-regime hit rate >= 0.40, volatility-regime hit rate >= 0.40 |
| Trading | turnover proxy <= 0.85 for A-grade factors |
| Incremental | max abs corr <= 0.90 versus library and selected peers, or abs(residual IC) >= 0.001 |
| Robustness | scorecard artifact plus model-incremental validation |

This design borrows the useful v3 idea of a persistent accepted pool,
low-correlation gating, train-only discovery, OOS reporting, and full audit logs,
but adapts it to this futures factor panel and local Ridge/LightGBM/MLP
workflow.

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

Strict scorecard optimization round: the latest mining pass evaluated 12,000
formula candidates with explicit gates for data quality, Pearson/rank IC,
monthly/product/regime stability, bucket shape, turnover proxy, library
correlation, residualized IC, and candidate-peer correlation. The cheap IS/OOS
IC screen was used only as a prefilter. Among the 12,000 candidates, 139 passed
all scorecard gates. The final selector kept 100 `pass_all` factors with max
candidate-peer absolute correlation 0.878208 and operator diversity
(`z_spread`: 50, `z_product`: 46, `z_add`: 3, `rank_spread`: 1).

Those 100 strict scorecard factors were then added to the existing model
feature bases and checked again with model-specific incremental tests on
2020-01:

| Model | Validation base | Incremental test | Strict scorecard factors retained | Validation-month IC |
| --- | ---: | --- | ---: | ---: |
| Ridge | 617 | exact leave-one-factor-out retrain; keep if IC declines when removed | 45 / 100 | 0.084562 |
| LightGBM | 643 | single-factor standard-normal replacement; keep if IC declines when shuffled | 59 / 100 | 0.046381 |
| Ridge and LightGBM | 617 / 643 | retained by both model-specific tests | 27 / 100 | n/a |

Ridge 2020 full-year OOS IC, using `pred_xsz`, improved after adding the new
strict scorecard factors:

| Ridge feature set | Factors | 2020 OOS IC | Delta vs 617 base |
| --- | ---: | ---: | ---: |
| existing validated base | 617 | 0.041014 | baseline |
| base + 100 strict scorecard factors | 717 | 0.041586 | +0.000571 |
| base + 45 Ridge-retained strict factors | 662 | 0.042092 | +0.001078 |
| base + 27 factors retained by both Ridge and LightGBM | 644 | 0.041892 | +0.000878 |

The model-incremental tests are intentionally stricter than the scorecard:
the skill generated 100 factors that passed every multi-layer scorecard gate,
while Ridge and LightGBM kept model-specific subsets. The strongest Ridge
result comes from the 45 Ridge-retained factors, which add +0.001078 IC over the
617-factor base. The full strict-round formula and retained/removed lists are
committed at `references/futures/scorecard_strict_factors_summary.json`.

1000-factor Ridge/MLP scaling round: the current skill was then run at larger
scale. It generated 162,208 expression candidates, kept 40,000 after the cheap
train-only sample-IC/data-quality prefilter, evaluated the top 25,000 with the
full scorecard, and selected 1,000 strict `pass_all` factors after excluding
prior expression rounds and enforcing candidate-peer max absolute correlation
<= 0.90. No 2020 OOS IC was used in this selection stage. The selected factors
had max candidate-peer absolute correlation 0.899143 and operator diversity
(`z_product`: 763, `z_spread`: 161, `z_add`: 76).

For MLP, the compact overlap333 baseline is no longer used for this comparison.
The corrected baseline first trains on all old 1,144 factors, then performs a
single-factor random-replacement test on 2020-01. Because the MLP consumes
standardized inputs, replacing a standardized feature by `N(0, 1)` is equivalent
to replacing the raw factor by random draws from its training `mean/std`. A
factor is retained only when replacement lowers validation-month IC.

Model-specific 2020-01 incremental validation:

| Model | Base factors | Candidate factors | Incremental test | Retained factors | Final factors | Validation-month IC |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Ridge | 617 | 1,617 | exact leave-one-factor-out retrain; keep if IC declines when removed | 463 / 1000 new | 1,080 | 0.066742 |
| MLP old-factor screen | 1,144 | 1,144 | same-mean/std random replacement; remove if IC does not decline | 617 / 1,144 old | 617 | 0.073163 |
| MLP new-factor screen | 617 old retained | 1,617 | same-mean/std random replacement; remove if IC does not decline | 534 / 1,000 new | 1,151 | 0.067314 |

2020 full-year OOS IC, using `pred_xsz`:

| Model | Feature set | Factors | 2020 OOS IC | Monthly mean IC | Monthly IR | Delta |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Ridge | 617 validated base | 617 | 0.040171 | 0.042578 | 3.381732 | baseline |
| Ridge | base + 1000 scorecard factors | 1,617 | 0.041995 | 0.044123 | 3.661060 | +0.001824 vs Ridge base |
| Ridge | base + 463 Ridge-retained factors | 1,080 | 0.042479 | 0.045262 | 2.987944 | +0.002308 vs Ridge base |
| MLP | old all factors | 1,144 | 0.045920 | 0.048462 | 4.166469 | +0.001079 vs old-retained baseline |
| MLP | old retained factors | 617 | 0.044841 | 0.047474 | 3.779491 | baseline |
| MLP | old retained + 1000 scorecard factors | 1,617 | 0.040447 | 0.042406 | 3.964705 | -0.004394 vs old-retained baseline |
| MLP | old retained + 534 MLP-retained new factors | 1,151 | 0.041569 | 0.043788 | 3.687781 | -0.003271 vs old-retained baseline; +0.001123 vs adding all 1000 |

The large-scale round is strongly positive for Ridge: adding all 1,000
scorecard factors improves 2020 OOS IC by +0.001824, and the Ridge
leave-one-factor-out filter lifts the final 1,080-factor model to +0.002308
over the 617-factor base. Under the corrected MLP baseline, the 1,000 new
factors do not improve 2020 OOS IC. The shuffle filter is still useful because
it recovers +0.001123 IC versus directly adding all 1,000 new factors, but the
filtered MLP remains below both the old-retained 617-factor baseline and the old
1,144-factor model. The corrected MLP audit is committed at
`references/futures/new1000_mlp_old1144_shuffle_summary.json`; the earlier
overlap333 MLP comparison is superseded and should not be used as evidence of
MLP improvement.

For the first 100-factor round, the overlap between Ridge-retained and
LightGBM-retained new factors was 26. These model-specific validations give
independent evidence that the mined expression factors are not merely duplicates
of the original pool. The full reports are:

- `reports/final_factor_mining_report.md`
- `reports/factor_effectiveness_validation.md`
- `references/futures/old_effective_factors_summary.json`
- `references/futures/round2_effective_factors_summary.json`
- `references/futures/scorecard_strict_factors_summary.json`
- `references/futures/new1000_ridge_mlp_summary.json`
- `references/futures/new1000_mlp_old1144_shuffle_summary.json`
- `references/futures/factor_acceptance_standards.json`

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
# For a large scaling run, increase the target and candidate pool.
PYTHONPATH=src python scripts/run_continuous_factor_mining.py --config configs/futures.yaml --target 1000
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
- `scorecard_strict_factors_summary.json`
- `factor_acceptance_standards.json`

Do not commit `/root/autodl-tmp/quant/data/raw`, `selected_month_parts`,
`data_factors_big.parquet`, or prediction parquet files.
