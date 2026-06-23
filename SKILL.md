---
name: fu-alpha-research
description: "Use for local China futures alpha research: rebuilding selected factor panels, computing IC, mining effective factors, running Ridge/LightGBM baselines, and checking 2018-2019 IS versus 2020 OOS performance."
---

# FU Alpha Research Playbook

This repository is now a local futures alpha research workflow, not a
WorldQuant BRAIN submission helper.

## Research Loop

1. Audit local data and factor availability.
2. Materialize the 1,144 selected factor views from either the final parquet
   panel or month partitions.
3. Compute single-factor IC on 2018-2019 IS and 2020 OOS.
4. Keep effective factors only when IS and OOS IC have the same sign and pass
   minimum absolute IC thresholds.
5. Train a base Ridge or LightGBM model on IS rows.
6. Predict 2020 OOS and report pooled IC, monthly IC, yearly IC, and a simple
   long-short rank backtest.
7. Compare incremental Ridge feature sets such as top 100, top 300, and all
   selected factors to test whether adding factors improves OOS IC.

## Commands

```bash
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml audit-data
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml mine-factors
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml baseline --models ridge,lightgbm
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml incremental --sets 100,300,all
PYTHONPATH=src python -m fu_alpha_research.cli --config configs/futures.yaml report
```

## Data Policy

Commit source code and small factor metadata only. Keep raw bars, factor
panels, generated predictions, and model outputs local and untracked.
