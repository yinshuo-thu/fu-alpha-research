---
name: fu-alpha-research
description: "Use for local China futures alpha research: continuously generating auditable expression factors, evaluating candidates with a multi-layer scorecard, enforcing low-correlation and robustness gates, running Ridge/LightGBM incremental validation, and checking 2018-2019 IS versus 2020 OOS performance."
---

# FU Alpha Research Playbook

This repository is a local futures alpha research workflow and reusable skill.
Do not accept factors only because they rank high by full-sample IC or a single
backtest.

## Research Loop

1. Audit local data, factor availability, label horizon, trading calendar, and
   contract-roll metadata before using any factor.
2. Construct candidate factors from transparent expression templates and a
   diversified seed pool: cross-sectional rank/zscore combinations, spreads,
   gated signals, residualized signals, regime-conditioned signals, and simple
   temporal transforms. Avoid black-box expressions that cannot be audited.
3. Read the existing local factor library from the final parquet panel or month
   partitions, then evaluate new expression factors using the same timestamp and
   symbol alignment.
4. Use fast IC aggregation only as a prefilter. Final selection must run
   `scripts/evaluate_expression_scorecard.py`, which computes data quality,
   Pearson/rank IC, monthly/product IC, bucket, regime, turnover, library
   correlation, residual IC, and candidate-peer correlation diagnostics.
5. Enforce low-correlation gates. A candidate should have max absolute
   correlation <= 0.90 against the existing factor library and selected peers,
   unless residualized IC justifies keeping it as a model feature.
6. Train model-specific feature sets only after the factor passes data quality,
   IC, bucket, regime, trading simulation, incremental, and robustness checks.
7. Predict 2020 OOS and report pooled IC, Pearson IC, rank IC, monthly/daily IC,
   bucket monotonicity, long-short spread, turnover, cost sensitivity, and
   incremental model lift.
8. Classify each factor as A/B/C/D/E:
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
- candidate-peer correlation inside the accepted pool;
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
PYTHONPATH=src python scripts/evaluate_expression_scorecard.py --config configs/futures.yaml --target 100 --max-corr 0.90
PYTHONPATH=src python scripts/validate_model_factor_effectiveness.py --mode ridge --config configs/futures.yaml
PYTHONPATH=src python scripts/validate_model_factor_effectiveness.py --mode lgbm --config configs/futures.yaml
PYTHONPATH=src python scripts/build_effectiveness_validation_report.py --config configs/futures.yaml
```

## Selection Rule

The default accepted-factor path is:

```text
candidate expressions
  -> IC prefilter
  -> multi-layer scorecard
  -> low-correlation greedy selection
  -> Ridge leave-one-factor-out / LightGBM shuffle validation
  -> README/report evidence
```

Never call a factor "effective" merely because it is in the top 100 by OOS IC.

## Data Policy

Commit source code and small factor metadata only. Keep raw bars, factor
panels, generated predictions, and model outputs local and untracked.
