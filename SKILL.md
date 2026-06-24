---
name: fu-alpha-research
description: "Use for local China futures alpha research: constructing new expression factors, rebuilding selected factor panels, performing multi-layer factor validation, running Ridge/LightGBM baselines, and checking 2018-2019 IS versus 2020 OOS performance."
---

# FU Alpha Research Playbook

This repository is now a local futures alpha research workflow, not a
WorldQuant BRAIN submission helper.

## Research Loop

1. Audit local data, factor availability, label horizon, trading calendar, and
   contract-roll metadata before using any factor.
2. Construct candidate factors from transparent expression templates:
   cross-sectional rank/zscore combinations, spreads, gated signals,
   residualized signals, regime-conditioned signals, and simple temporal
   transforms. Avoid black-box expressions that cannot be audited.
3. Materialize or read the 1,144 selected factor views from the final parquet
   panel or month partitions, then evaluate new expression factors using the
   same timestamp/symbol alignment.
4. Run the multi-layer factor gate. Do not keep a factor only because full-sample
   IC or one backtest looks good.
5. Train model-specific feature sets only after the factor passes data quality,
   IC, bucket, regime, trading simulation, incremental, and robustness checks.
6. Predict 2020 OOS and report pooled IC, Pearson IC, rank IC, monthly/daily IC,
   bucket monotonicity, long-short spread, turnover, cost sensitivity, and
   incremental model lift.
7. Classify each factor as A/B/C/D/E:
   A core trading alpha, B model feature, C conditional alpha, D watchlist,
   or E discard.

## Multi-Layer Factor Gate

Use `docs/factor_evaluation_framework.md` as the required evaluation standard.

### Step 1: Data Quality

- coverage by timestamp, symbol, product, and regime;
- missing/zero/constant values;
- outliers and winsorization sensitivity;
- distribution drift between IS/OOS and across months;
- potential contract-roll, session-boundary, and cross-contract leakage issues.

### Step 2: IC Tests

- pooled Pearson IC and rank IC;
- daily/monthly IC mean, volatility, hit rate, t-stat, and drawdown;
- horizon decay over multiple forward-return horizons;
- sign stability across products and time.

### Step 3: Bucket Tests

- quantile bucket returns;
- top-bottom spread and extreme-bucket contribution;
- monotonicity score and whether the signal only works in one tail;
- stability of bucket shape by month and product.

### Step 4: Regime Tests

- product family;
- liquidity regime;
- volatility regime;
- intraday session and close/open boundary;
- roll window versus non-roll window.

### Step 5: Trading Simulation

- long-short rank portfolio;
- turnover and holding-period decay;
- transaction-cost stress;
- break-even cost;
- exposure concentration and capacity proxy.

### Step 6: Incremental Tests

- correlation with existing factor library;
- residualized IC after removing known factor exposures;
- baseline model versus baseline plus candidate;
- leave-one-factor-out or permutation/shuffle checks.

### Step 7: Robustness and Audit

- walk-forward train/test splits;
- random permutation and label-shift controls;
- leakage checks;
- multiple-testing penalty and false discovery control;
- reproducible artifact paths and exact decision rationale.

### Step 8: Decision

- A: core trading alpha. Strong standalone, stable bucket/spread, robust after
  costs, and incremental to existing models.
- B: model feature. Weak standalone but improves model OOS or is useful after
  residualization.
- C: conditional alpha. Works only in explicit product/liquidity/volatility/
  session regimes.
- D: watchlist. Promising but fails one important robustness or cost gate.
- E: discard. Poor quality, unstable sign, no incremental value, or leakage risk.

## Commands

```bash
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml audit-data
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml mine-factors
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml baseline --models ridge,lightgbm
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml incremental --sets 100,300,all
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml report
```

Useful local scripts for the expanded workflow:

```bash
PYTHONPATH=src python scripts/run_continuous_factor_mining.py --config configs/futures.yaml
PYTHONPATH=src python scripts/validate_model_factor_effectiveness.py --mode ridge --config configs/futures.yaml
PYTHONPATH=src python scripts/validate_model_factor_effectiveness.py --mode lgbm --config configs/futures.yaml
PYTHONPATH=src python scripts/build_effectiveness_validation_report.py --config configs/futures.yaml
```

## Data Policy

Commit source code and small factor metadata only. Keep raw bars, factor
panels, generated predictions, and model outputs local and untracked.
