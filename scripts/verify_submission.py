"""Verify submission.csv matches sample schema and probabilities are sane."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("verify")

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "raw" / "sample_sol_prob_death.csv"
SUB = ROOT / "submissions" / "submission_lgbm.csv"


def main() -> int:
    sample = pd.read_csv(SAMPLE)
    sub = pd.read_csv(SUB)

    issues: list[str] = []

    if list(sample.columns) != list(sub.columns):
        issues.append(f"column mismatch: sample={list(sample.columns)} sub={list(sub.columns)}")
    if len(sample) != len(sub):
        issues.append(f"row count: sample={len(sample)} sub={len(sub)}")

    probs = sub["HOSPITAL_EXPIRE_FLAG"]
    if probs.isna().any():
        issues.append(f"NaN probabilities: {int(probs.isna().sum())}")
    if (probs < 0).any() or (probs > 1).any():
        issues.append(f"out-of-range probs: min={probs.min()} max={probs.max()}")

    # ID coverage
    missing_ids = set(sample["icustay_id"]) - set(sub["icustay_id"])
    extra_ids = set(sub["icustay_id"]) - set(sample["icustay_id"])
    if missing_ids:
        issues.append(f"IDs missing from sub: {len(missing_ids)}")
    if extra_ids:
        issues.append(f"IDs in sub not in sample: {len(extra_ids)}")

    log.info("submission rows=%d, cols=%s", len(sub), list(sub.columns))
    log.info("prob stats: min=%.6f mean=%.4f max=%.6f std=%.4f",
             probs.min(), probs.mean(), probs.max(), probs.std())
    log.info("quantiles 10/50/90: %.4f / %.4f / %.4f",
             probs.quantile(0.1), probs.quantile(0.5), probs.quantile(0.9))
    log.info("sample head:\n%s", sub.head().to_string())

    if issues:
        for i in issues:
            log.error(i)
        return 1
    log.info("OK - submission is schema-valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
