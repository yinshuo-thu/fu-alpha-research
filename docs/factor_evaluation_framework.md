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

## Step 6: Incremental Test

Standalone quality is not enough. Check incremental value.

Required metrics:

- correlation with existing factor library;
- cluster/family assignment;
- residualized IC after regressing on existing factors or family components;
- baseline model versus baseline plus candidate;
- leave-one-factor-out effect inside the selected model;
- permutation or shuffle importance for tree models.

Suggested gates:

- high correlation requires residualized IC or model lift evidence;
- a factor with weak standalone IC can be B if it improves OOS model IC;
- a factor with high standalone IC can still be rejected if it is redundant.

## Step 7: Robustness and Audit

Required checks:

- walk-forward splits, not just one IS/OOS split;
- random label permutation control;
- feature shuffle/permutation control;
- leakage check around label construction, session boundaries, and roll;
- multiple-testing penalty;
- sensitivity to winsorization, standardization, and universe filters;
- reproducible command log and artifact paths.

Suggested gates:

- random controls should collapse to near-zero IC;
- walk-forward performance should not rely on a single period;
- multiple-testing adjusted results should still be plausible;
- leakage suspicion is an automatic E until resolved.

## Step 8: Decision

Each factor receives one decision label.

| Grade | Meaning | Typical Requirement |
| --- | --- | --- |
| A | Core trading alpha | Strong standalone evidence, robust buckets, survives costs, incremental to models |
| B | Model feature | Improves model OOS or residualized IC, but may be weak standalone |
| C | Conditional alpha | Works in a documented product/liquidity/volatility/session regime |
| D | Watchlist | Promising but fails one important stability, cost, or audit gate |
| E | Discard | Bad quality, leakage risk, unstable sign, or no incremental value |

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
audit_penalty
final_grade
decision_reason
```

The final grade is not a simple average. Leakage or severe data-quality failure
overrides every positive performance metric. Trading-cost failure prevents A but
does not necessarily prevent B. Narrow regime dependence should become C, not
A.

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
