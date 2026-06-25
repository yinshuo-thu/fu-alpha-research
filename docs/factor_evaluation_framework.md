# Multi-Layer Futures Factor Evaluation Framework

This document is the required evaluation standard for constructed factors in
`fu-alpha-research`. A factor should not be accepted only because full-sample IC
or one long-short backtest is positive.

The default research split remains:

- IS: 2018-2019
- OOS: 2020
- Label: 30-bar forward return unless otherwise stated
- Main model-view metric: `pred_xsz`

## Candidate Construction

New factors should be generated from auditable templates:

- cross-sectional rank/zscore transforms;
- additive and spread combinations of strong but not identical seeds;
- gated signals such as `rank(a) * I[rank(b) > 0]`;
- products only after checking outlier sensitivity;
- residualized signals against known factor families;
- regime-conditioned expressions that explicitly state their active condition;
- simple horizon variants that support decay testing.

Each candidate must store:

- factor name;
- formula;
- source dependencies;
- transform family;
- intended intuition;
- creation timestamp or generation round.

Continuous mining should maintain an accepted pool and a rejected ledger. The
accepted pool is used for low-correlation checks and family/operator diversity;
the rejected ledger prevents repeating candidates that failed data quality,
leakage, correlation, or robustness gates. OOS labels are for reporting and
final evidence, not for repeatedly tuning candidate formulas.

The current command-level implementation is:

```bash
PYTHONPATH=src python scripts/run_continuous_factor_mining.py --config configs/futures.yaml --target 100
```

The detailed evaluator can also be run directly:

```bash
PYTHONPATH=src python scripts/evaluate_expression_scorecard.py \
  --config configs/futures.yaml \
  --target 100 \
  --rows-per-month 3000 \
  --max-corr 0.90
```

This script writes:

- `reports/generated/new_factor_scorecard.csv`
- `reports/generated/new_effective_factors_scorecard.csv`
- `reports/generated/new_factor_scorecard_summary.json`
- `outputs/expression_sets/new100.csv`

Default final selection is strict: `outputs/expression_sets/new100.csv` is
written only from candidates whose `decision_reason` is `pass_all`, then filtered
by candidate-peer absolute correlation <= 0.90 and operator/family diversity.
The A/B/C/D/E labels remain useful for diagnostics and watchlists, but the
default accepted-factor artifact should not include a candidate that fails any
scorecard gate.

The default numerical standards are committed in
`references/futures/factor_acceptance_standards.json`. Override them only when a
new experiment explicitly documents why the standard changed.

## Step 1: Data Quality Check

Required checks:

- coverage by timestamp, symbol, product, month, session, roll window, liquidity
  regime, and volatility regime;
- missing, infinite, zero, constant, or near-constant values;
- extreme-value ratio and winsorization sensitivity;
- cross-sectional dispersion by timestamp;
- IS/OOS distribution drift using mean, standard deviation, quantiles, and PSI or
  KS-style distance;
- possible contract-roll contamination;
- session-boundary and look-ahead leakage risk;
- product-specific availability gaps.

Suggested gates:

- timestamp coverage >= 80% unless the factor is explicitly conditional;
- no unexplained product coverage hole;
- no month where coverage collapses without a documented market reason;
- winsorized and raw IC signs should not contradict materially;
- roll-window behavior must be separated from normal-window behavior.

Project default implementation gate:

- coverage >= 0.70;
- outlier ratio <= 0.02;
- standard deviation > 1e-8.

## Step 2: IC Test

Run the candidate against multiple return horizons, not only the target horizon.

Required metrics:

- Pearson IC;
- rank IC;
- pooled IC;
- daily IC;
- monthly IC;
- IC hit rate;
- IC t-stat;
- horizon decay curve;
- product-level IC;
- IS/OOS sign consistency.

Suggested gates:

- IS and OOS signs are consistent;
- horizon decay is economically plausible;
- monthly IC is not driven by one month;
- rank IC and Pearson IC are not in severe conflict unless the distribution
  explains why;
- OOS IC remains meaningful after excluding the best month.

Project default implementation gate:

- abs(selection IC) >= 0.001;
- abs(rank IC) >= 0.0005;
- monthly IC hit rate >= 0.50;
- at least 6 monthly IC observations;
- selection IC and sample Pearson IC have the same sign.

## Step 3: Bucket Test

The bucket test checks whether signal ordering is usable.

Required metrics:

- quantile bucket return table;
- top-bottom spread;
- monotonicity score;
- tail contribution;
- bucket turnover;
- bucket stability by month and product.

Suggested gates:

- top-bottom spread has the expected sign;
- bucket shape is at least approximately monotonic or has a documented tail-only
  interpretation;
- the edge is not entirely from one extreme bucket unless the factor is designed
  as a tail alpha;
- bucket behavior is not reversed in major product groups.

Project default implementation gate:

- top-bottom spread has the same sign as selection IC;
- abs(bucket monotonicity) >= 0.25.

## Step 4: Regime Test

Evaluate the factor under explicit regimes:

- product family;
- liquidity quantile;
- volatility quantile;
- intraday session;
- roll window versus non-roll window;
- trend/range market state if available;
- high/low cross-sectional dispersion states.

Outputs:

- regime IC table;
- regime bucket spread table;
- active regime recommendation;
- failure regimes to exclude or downweight.

Decision rule:

- broad performance can qualify as A/B;
- narrow but repeatable performance can qualify as C;
- unstable or sign-flipping regime behavior should be D/E.

Project default implementation gate:

- product IC hit rate >= 0.45;
- liquidity-regime IC hit rate >= 0.40;
- volatility-regime IC hit rate >= 0.40.

## Step 5: Trading Simulation

Run a simple but explicit simulation before calling a factor tradable.

Required metrics:

- long-short portfolio return;
- turnover;
- cost-stressed return;
- break-even cost;
- hit rate;
- drawdown;
- exposure concentration;
- product contribution;
- capacity proxy using liquidity participation.

Suggested gates:

- factor survives plausible cost assumptions for A;
- if it fails costs but improves a model, it may still be B;
- break-even cost must be reported, not inferred;
- turnover spikes around roll/session boundaries must be audited.

Project default implementation gate:

- turnover proxy <= 0.85 for A-grade standalone factors;
- factors above this can only be B/C when residualized/model-incremental
  evidence remains strong.

## Step 6: Incremental Test

Standalone quality is not enough. Check incremental value.

Required metrics:

- correlation with existing factor library;
- correlation with already selected candidates in the same mining round;
- cluster/family assignment;
- residualized IC after regressing on existing factors or family components;
- baseline model versus baseline plus candidate;
- leave-one-factor-out effect inside the selected model;
- permutation or shuffle importance for tree models.

Suggested gates:

- max absolute correlation should normally be <= 0.90 against the existing
  library and selected peers;
- high correlation requires residualized IC or model lift evidence;
- a factor with weak standalone IC can be B if it improves OOS model IC;
- a factor with high standalone IC can still be rejected if it is redundant.

Project default implementation gate:

- max absolute correlation <= 0.90 against the existing library;
- max absolute correlation <= 0.90 against selected peers in the same round;
- high-correlation escape requires abs(residual IC) >= 0.001 or model lift.

## Step 7: Robustness and Audit

Required checks:

- walk-forward splits, not just one IS/OOS split;
- random label permutation control;
- feature shuffle/permutation control;
- leakage check around label construction, session boundaries, and roll;
- multiple-testing penalty;
- sensitivity to winsorization, standardization, and universe filters;
- reproducible command log and artifact paths.
- clear separation of discovery metrics and OOS report metrics.

Suggested gates:

- random controls should collapse to near-zero IC;
- walk-forward performance should not rely on a single period;
- multiple-testing adjusted results should still be plausible;
- leakage suspicion is an automatic E until resolved.

Project default implementation gate:

- scorecard artifact must be saved;
- Ridge leave-one-factor-out and LightGBM shuffle validation are required before
  a generated factor is reported as final model-effective evidence.

## Step 8: Decision

Each factor receives one decision label.

| Grade | Meaning | Typical Requirement |
| --- | --- | --- |
| A | Core trading alpha | Strong standalone evidence, robust buckets, survives costs, incremental to models |
| B | Model feature | Improves model OOS or residualized IC, but may be weak standalone |
| C | Conditional alpha | Works in a documented product/liquidity/volatility/session regime |
| D | Watchlist | Promising but fails one important stability, cost, or audit gate |
| E | Discard | Bad quality, leakage risk, unstable sign, or no incremental value |

Project default final artifact rule:

- only `decision_reason == pass_all` candidates are eligible for the generated
  `new100` expression artifact;
- model-specific effectiveness is reported separately. Ridge leave-one-factor-
  out and LightGBM shuffle may retain smaller subsets of the strict scorecard
  factors; those subsets should be named explicitly instead of calling every
  scorecard-passed factor model-effective.

## Factor Scorecard

Recommended score fields:

```text
data_quality_score
ic_score
bucket_score
regime_score
trading_score
incremental_score
robustness_score
max_abs_corr_to_library
candidate_peer_max_abs_corr
residual_ic
audit_penalty
final_grade
decision_reason
```

The final grade is not a simple average. Leakage or severe data-quality failure
overrides every positive performance metric. Trading-cost failure prevents A but
does not necessarily prevent B. Narrow regime dependence should become C, not
A.

Fast same-sign IS/OOS IC screens are allowed only as prefilters. They should not
write the final accepted list unless followed by the scorecard and low-correlation
selection stages.

## Minimum Artifacts

For every accepted A/B/C factor, keep:

- formula and dependencies;
- coverage report;
- IC report;
- bucket report;
- regime report;
- trading simulation report;
- incremental-value report;
- robustness/audit report;
- final decision row.

Generated heavy artifacts stay local and untracked. Compact metadata and final
approved lists can be stored under `references/futures/`.
