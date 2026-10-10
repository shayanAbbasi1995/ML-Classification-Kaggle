"""One-off inspection script: schema, dtypes, missingness, class balance."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("inspect")

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def summarise(name: str, df: pd.DataFrame) -> None:
    log.info("---- %s ----", name)
    log.info("shape: %s", df.shape)
    log.info("columns: %s", list(df.columns))
    log.info("dtypes:\n%s", df.dtypes.to_string())
    miss = df.isna().mean().mul(100).round(2)
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        log.info("missing %%:\n%s", miss.to_string())
    else:
        log.info("missing %%: none")
    log.info("head:\n%s", df.head(3).to_string())


def main() -> None:
    train = pd.read_csv(RAW / "train_prob_death.csv")
    test = pd.read_csv(RAW / "test_prob_death.csv")
    sample = pd.read_csv(RAW / "sample_sol_prob_death.csv")
    extra = pd.read_csv(RAW / "extra_diagnoses.csv")
    meta_dx = pd.read_csv(RAW / "metadata_diagnose.csv")
    try:
        patient_meta = pd.read_excel(RAW / "patient_metadata.xlsx")
    except Exception as exc:  # noqa: BLE001
        log.warning("patient_metadata.xlsx read failed: %s", exc)
        patient_meta = None

    summarise("train", train)
    summarise("test", test)
    summarise("sample_submission", sample)
    summarise("extra_diagnoses", extra)
    summarise("metadata_diagnose", meta_dx)
    if patient_meta is not None:
        summarise("patient_metadata", patient_meta)

    log.info("==== cross-checks ====")
    only_train = set(train.columns) - set(test.columns)
    only_test = set(test.columns) - set(train.columns)
    log.info("train-only cols: %s", sorted(only_train))
    log.info("test-only cols: %s", sorted(only_test))

    # Candidate target column
    for cand in ("HOSPITAL_EXPIRE_FLAG", "target", "label", "y"):
        if cand in train.columns:
            vc = train[cand].value_counts(dropna=False).to_dict()
            log.info("target '%s' value counts: %s (pos rate %.4f)",
                     cand, vc, train[cand].mean())
            break

    # ID column
    for cand in ("icustay_id", "ICUSTAY_ID", "id", "ID"):
        if cand in test.columns:
            log.info("test ID '%s' example: %s", cand, test[cand].head(3).tolist())
            break

    log.info("sample submission columns: %s", list(sample.columns))
    log.info("sample submission head:\n%s", sample.head().to_string())
    log.info("sample submission rows=%d, test rows=%d", len(sample), len(test))


if __name__ == "__main__":
    main()
