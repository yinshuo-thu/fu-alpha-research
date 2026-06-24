# Futures Migration Design

## Objective

Migrate the original alpha-research shell into a local China futures factor
mining project with a reproducible IS/OOS protocol:

- IS: 2018-2019.
- OOS: 2020.
- Label: 30-bar forward return.
- Metric: pooled cosine IC.
- Models: Ridge and LightGBM baselines.

## Factor Storage

The repository stores only metadata for the existing local factor library.
Large data is external. The loader supports two storage modes:

- `final_panel`: read `data_factors_big.parquet` directly when it exists.
- `month_partitions`: read intermediate `month=YYYY-MM/*.parquet` partitions
  and reconstruct cross-sectional z-score/rank factors by timestamp.

The second mode lets the research continue while a low-memory full-panel build
is still in progress.

## Evaluation

`fu_alpha_research.metrics.compute_ic` implements the official pooled IC:

```text
mean(pred * label) / sqrt(mean(pred^2) * mean(label^2))
```

Predictions are also evaluated after timestamp-level z-score and rank
transforms. The long-short backtest is intentionally simple: at each timestamp,
long the top prediction quantile and short the bottom quantile, then average
the 30-bar labels.

## Effective Factors

Fast single-factor mining computes IS and OOS IC for the existing factor
library, but this is only a prefilter. A candidate is not effective until it
passes the scorecard and model-incremental checks described in
`docs/factor_evaluation_framework.md`.

The scorecard includes:

- data quality and coverage;
- Pearson/rank IC and monthly stability;
- product, liquidity, and volatility regimes;
- bucket monotonicity and top-bottom spread;
- turnover/cost proxies;
- max correlation to the existing library and selected peers;
- residualized IC and model-incremental validation.

Generated reports under `reports/generated/` are excluded from git but can be
regenerated from the local factor panel.
