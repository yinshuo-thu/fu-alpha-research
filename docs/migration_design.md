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

The repository stores only metadata for 1,144 selected factors. Large data is
external. The loader supports two storage modes:

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

Single-factor mining computes IS and OOS IC for all selected factors. A factor
is effective when:

- `abs(IS IC)` passes the configured threshold;
- `abs(OOS IC)` passes the configured threshold;
- IS and OOS IC signs agree;
- coverage is broad enough in both windows.

The generated `reports/generated/effective_factors.csv` is excluded from git
but can be regenerated from the local factor panel.
