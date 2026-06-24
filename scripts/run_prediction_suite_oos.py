#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from fu_alpha_research.config import load_config
from fu_alpha_research.factor_store import FactorStore


def run_job(cmd: list[str], retries: int = 1) -> None:
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("LIGHTGBM_NUM_THREADS", "4")
    env.setdefault("PREDICT_CHUNK_ROWS", "50000")
    for attempt in range(retries + 1):
        print("[predict-runner] " + " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, env=env)
        if proc.returncode == 0:
            return
        if attempt < retries:
            print(f"[predict-runner] retry returncode={proc.returncode}", flush=True)
            continue
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/futures.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    months = FactorStore(cfg).available_months(args.start or cfg.oos_start, args.end or cfg.oos_end)
    paired_sets = ["orig_top300", "new_top300"]
    split_sets = ["orig_effective657", "new_effective757", "orig_all1144", "new_all1244"]
    for month in months:
        print(f"[predict-runner] month={month}", flush=True)
        for set_name in paired_sets:
            cmd = [
                sys.executable,
                "scripts/predict_model_suite_month.py",
                "--config",
                args.config,
                "--month",
                month,
                "--models",
                "ridge,lgbm",
                "--sets",
                set_name,
            ]
            if args.force:
                cmd.append("--force")
            run_job(cmd)
        for set_name in split_sets:
            for model in ["ridge", "lgbm"]:
                cmd = [
                    sys.executable,
                    "scripts/predict_model_suite_month.py",
                    "--config",
                    args.config,
                    "--month",
                    month,
                    "--models",
                    model,
                    "--sets",
                    set_name,
                ]
                if args.force:
                    cmd.append("--force")
                run_job(cmd)


if __name__ == "__main__":
    main()
