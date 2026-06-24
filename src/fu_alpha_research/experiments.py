from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .factor_store import FactorStore
from .metrics import add_prediction_views, long_short_backtest, summarize_backtest, summarize_predictions
from .mining import attach_catalog, mine_effective_factors
from .modeling import fit_lightgbm, fit_ridge, predict_lightgbm


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _text_table(df: pd.DataFrame) -> str:
    return "```text\n" + df.to_string(index=False) + "\n```"


def audit_data(cfg: Config) -> dict[str, object]:
    store = FactorStore(cfg)
    audit = store.audit()
    audit["is_months"] = store.available_months(cfg.is_start, cfg.is_end)
    audit["oos_months"] = store.available_months(cfg.oos_start, cfg.oos_end)
    save_json(audit, cfg.reports_dir / "data_audit.json")
    return audit


def _predict_oos(
    store: FactorStore,
    feature_cols: list[str],
    model_name: str,
    model,
    cfg: Config,
) -> pd.DataFrame:
    parts = []
    for month, df in store.iter_months(cfg.oos_start, cfg.oos_end, columns=feature_cols):
        out = df[["symbol", "datetime", "label"]].copy()
        if model_name == "ridge":
            out["pred"] = model.predict(df).astype(np.float32)
        elif model_name == "lightgbm":
            out["pred"] = predict_lightgbm(model, df, feature_cols).astype(np.float32)
        else:
            raise ValueError(model_name)
        parts.append(out)
        print(f"  [predict][{model_name}] {month} rows={len(out)}", flush=True)
    pred = pd.concat(parts, ignore_index=True)
    pred = add_prediction_views(pred, "pred")
    return pred


def run_baseline(
    cfg: Config,
    model_names: list[str] | None = None,
    feature_cols: list[str] | None = None,
    name_suffix: str = "all",
) -> pd.DataFrame:
    store = FactorStore(cfg)
    features = feature_cols or store.selected
    models = model_names or list(cfg.baseline_models)
    summary_rows = []
    for model_name in models:
        train_rows = cfg.lgb_train_rows if model_name == "lightgbm" else cfg.ridge_train_rows
        train = store.sample_rows(cfg.is_start, cfg.is_end, train_rows, columns=features, seed=cfg.seed)
        print(
            f"[baseline] model={model_name} features={len(features)} train_rows={len(train)} "
            f"is={cfg.is_start}..{cfg.is_end} oos={cfg.oos_start}..{cfg.oos_end}",
            flush=True,
        )
        if model_name == "ridge":
            model = fit_ridge(train, features, alpha=1.0)
        elif model_name == "lightgbm":
            model = fit_lightgbm(train, features, seed=cfg.seed)
        else:
            raise ValueError(f"unknown model: {model_name}")
        pred = _predict_oos(store, features, model_name, model, cfg)
        pred_file = cfg.output_dir / f"predictions_{model_name}_{name_suffix}.parquet"
        pred.to_parquet(pred_file, index=False)
        summary = summarize_predictions(pred)
        summary.insert(0, "model", model_name)
        summary.insert(1, "feature_set", name_suffix)
        summary.insert(2, "feature_count", len(features))
        summary.insert(3, "prediction_file", str(pred_file))
        summary_rows.append(summary)
        summary_path = cfg.reports_dir / f"baseline_{model_name}_{name_suffix}.csv"
        summary.to_csv(summary_path, index=False)
        bt = long_short_backtest(pred, pred_col="pred_xrank")
        bt_summary = summarize_backtest(bt)
        save_json(bt_summary, cfg.reports_dir / f"backtest_{model_name}_{name_suffix}.json")
        bt.to_csv(cfg.output_dir / f"backtest_{model_name}_{name_suffix}.csv", index=False)
    out = pd.concat(summary_rows, ignore_index=True)
    out.to_csv(cfg.reports_dir / f"baseline_summary_{name_suffix}.csv", index=False)
    return out


def run_factor_mining(
    cfg: Config,
    min_is_ic: float = 0.002,
    min_oos_ic: float = 0.001,
) -> pd.DataFrame:
    store = FactorStore(cfg)
    scores = mine_effective_factors(
        store,
        cfg.is_start,
        cfg.is_end,
        cfg.oos_start,
        cfg.oos_end,
        min_is_ic=min_is_ic,
        min_oos_ic=min_oos_ic,
        cache_dir=cfg.reports_dir / "ic_parts",
    )
    scores = attach_catalog(scores, str(cfg.factor_catalog_path))
    out_file = cfg.reports_dir / "factor_scores.csv"
    scores.to_csv(out_file, index=False)
    effective = scores[scores["effective"]].copy()
    effective.to_csv(cfg.reports_dir / "effective_factors.csv", index=False)
    save_json(
        {
            "total_factors": int(len(scores)),
            "effective_factors": int(len(effective)),
            "min_is_ic": min_is_ic,
            "min_oos_ic": min_oos_ic,
            "top_effective": effective.head(20)["factor"].tolist(),
        },
        cfg.reports_dir / "factor_mining_summary.json",
    )
    return scores


def run_incremental_models(cfg: Config, top_sets: list[int | str]) -> pd.DataFrame:
    score_path = cfg.reports_dir / "factor_scores.csv"
    if not score_path.exists():
        run_factor_mining(cfg)
    scores = pd.read_csv(score_path)
    ranked = scores.sort_values("abs_is_ic", ascending=False)["factor"].tolist()
    rows = []
    for item in top_sets:
        if str(item).lower() == "all":
            features = FactorStore(cfg).selected
            suffix = "top_all"
        else:
            k = int(item)
            features = ranked[:k]
            suffix = f"top_{k}"
        summary = run_baseline(cfg, model_names=["ridge"], feature_cols=features, name_suffix=suffix)
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(cfg.reports_dir / "incremental_model_summary.csv", index=False)
    return out


def write_markdown_report(cfg: Config) -> Path:
    audit_path = cfg.reports_dir / "data_audit.json"
    mining_path = cfg.reports_dir / "factor_mining_summary.json"
    baseline_files = sorted(cfg.reports_dir.glob("baseline_summary_*.csv"))
    incremental_path = cfg.reports_dir / "incremental_model_summary.csv"
    lines = [
        "# Futures Alpha Migration Report",
        "",
        f"IS window: {cfg.is_start} to {cfg.is_end}.",
        f"OOS window: {cfg.oos_start} to {cfg.oos_end}.",
        "",
    ]
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        lines += [
            "## Data Audit",
            "",
            f"- Partition months available: {audit.get('month_count')} ({audit.get('months', [''])[0]} to {audit.get('months', [''])[-1]}).",
            f"- Symbols in partitions: {audit.get('symbol_count')}.",
            f"- Selected factor catalog: {audit.get('selected_factor_count')} factors "
            f"({audit.get('raw_count')} raw, {audit.get('tsz_count')} tsz, "
            f"{audit.get('csz_count')} csz, {audit.get('csr_count')} csr).",
            "",
        ]
    if mining_path.exists():
        mining = json.loads(mining_path.read_text(encoding="utf-8"))
        lines += [
            "## Existing-Library IC Prefilter",
            "",
            f"- Effective factors: {mining.get('effective_factors')} / {mining.get('total_factors')}.",
            f"- Thresholds: abs(IS IC) >= {mining.get('min_is_ic')}, "
            f"abs(OOS IC) >= {mining.get('min_oos_ic')}, same sign.",
            "- This is a fast prefilter, not the final new-expression acceptance rule.",
            f"- Top prefilter names: {', '.join(mining.get('top_effective', [])[:10])}.",
            "",
        ]
    for path in baseline_files:
        df = pd.read_csv(path)
        lines += ["## Baseline", "", _text_table(df), ""]
    if incremental_path.exists():
        df = pd.read_csv(incremental_path)
        lines += ["## Incremental Ridge Feature Sets", "", _text_table(df), ""]
    report = cfg.reports_dir / "migration_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
