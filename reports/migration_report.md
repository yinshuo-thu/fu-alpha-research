# Futures Alpha Migration Report

IS window: 2018-01-01 to 2019-12-31.
OOS window: 2020-01-01 to 2020-12-31.

## Data Audit

- Partition months available: 48 (2017-01 to 2020-12).
- Symbols in partitions: 51.
- Selected factor catalog: 1144 factors (402 raw, 164 tsz, 299 csz, 279 csr).

## Factor Mining

- Effective factors: 657 / 1144.
- Thresholds: abs(IS IC) >= 0.002, abs(OOS IC) >= 0.001, same sign.
- Top names: csr_cpos, csz_macd_3_13, csz_macd_5_21, csr_bop, macd_3_13, csz_willr_4, csz_px_sma_4, tsz_cpos, csz_stoch_dev_4, csr_stoch_dev_8.

## Baseline

```text
     name   pred_col    rows  label_rows  coverage  total_ic  monthly_mean  monthly_std  monthly_ir   ic_2020
ridge_all       pred 3915511     2939535  0.750741 -0.000475      0.037783     0.020758    1.820167 -0.000475
ridge_all   pred_xsz 3915511     2939535  0.750732  0.037073      0.039677     0.011071    3.583894  0.037073
ridge_all pred_xrank 3915511     2939535  0.750741  0.036099      0.038570     0.010380    3.715756  0.036099
```

## Baseline

```text
        name   pred_col    rows  label_rows  coverage  total_ic  monthly_mean  monthly_std  monthly_ir  ic_2020
ridge_top100       pred 3915511     2939535  0.750741  0.033190      0.036115     0.014422    2.504125 0.033190
ridge_top100   pred_xsz 3915511     2939535  0.750732  0.035914      0.038071     0.007851    4.849484 0.035914
ridge_top100 pred_xrank 3915511     2939535  0.750741  0.034574      0.036649     0.007392    4.958001 0.034574
```

## Baseline

```text
        name   pred_col    rows  label_rows  coverage  total_ic  monthly_mean  monthly_std  monthly_ir  ic_2020
ridge_top300       pred 3915511     2939535  0.750741  0.036827      0.040056     0.014792    2.707969 0.036827
ridge_top300   pred_xsz 3915511     2939535  0.750732  0.038573      0.040795     0.008498    4.800481 0.038573
ridge_top300 pred_xrank 3915511     2939535  0.750741  0.036704      0.038875     0.008211    4.734649 0.036704
```
