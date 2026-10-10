"""Shared pytest fixtures.

The real Kaggle data is not committed to the repo (gitignored). Tests
instead use ``synthetic_dataset`` which mimics the real column schema
with a few hundred random rows, so the full training pipeline can run
end-to-end on every ``pytest`` invocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pod.config import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL


def _rand_dates(n: int, rng: np.random.Generator, year_lo: int, year_hi: int) -> pd.Series:
    years = rng.integers(year_lo, year_hi, size=n)
    months = rng.integers(1, 13, size=n)
    days = rng.integers(1, 28, size=n)
    return pd.to_datetime(
        {"year": years, "month": months, "day": days}
    ).dt.strftime("%Y-%m-%d %H:%M:%S")


def _make_frame(n: int, seed: int, *, labelled: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray | pd.Series] = {
        "subject_id": rng.integers(1, 200, size=n),
        "hadm_id": rng.integers(1_000, 9_999, size=n),
        "icustay_id": np.arange(1, n + 1),
        "ADMITTIME": _rand_dates(n, rng, 2001, 2012),
        "DOB": _rand_dates(n, rng, 1920, 2000),
    }
    for col in NUMERIC_COLS:
        if "HeartRate" in col:
            data[col] = rng.normal(80, 15, size=n)
        elif "SysBP" in col:
            data[col] = rng.normal(120, 20, size=n)
        elif "DiasBP" in col:
            data[col] = rng.normal(70, 10, size=n)
        elif "MeanBP" in col:
            data[col] = rng.normal(85, 12, size=n)
        elif "RespRate" in col:
            data[col] = rng.normal(18, 5, size=n)
        elif "TempC" in col:
            data[col] = rng.normal(37.0, 0.5, size=n)
        elif "SpO2" in col:
            data[col] = rng.normal(96, 2, size=n)
        elif "Glucose" in col:
            vals = rng.normal(130, 30, size=n)
            # inject a sentinel the real data contains
            if n > 10:
                vals[0] = 999999.0
            data[col] = vals
        else:
            data[col] = rng.normal(0, 1, size=n)

    cat_choices: dict[str, list[str]] = {
        "GENDER": ["M", "F"],
        "ADMISSION_TYPE": ["EMERGENCY", "URGENT", "ELECTIVE"],
        "INSURANCE": ["Medicare", "Private", "Medicaid", "Self Pay"],
        "RELIGION": ["CATHOLIC", "PROTESTANT", "JEWISH", "UNOBTAINABLE"],
        "MARITAL_STATUS": ["MARRIED", "SINGLE", "WIDOWED", "UNKNOWN (DEFAULT)"],
        "ETHNICITY": ["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", "HISPANIC/LATINO"],
        "DIAGNOSIS": ["SEPSIS", "PNEUMONIA", "CHF", "COPD", "CVA"],
        "ICD9_diagnosis": ["038", "486", "428", "496", "431"],
        "FIRST_CAREUNIT": ["MICU", "SICU", "CCU", "TSICU"],
    }
    for col in CATEGORICAL_COLS:
        data[col] = rng.choice(cat_choices[col], size=n)
    # Inject a few NaN marital statuses so basic_clean has something to do.
    data["MARITAL_STATUS"] = pd.Series(data["MARITAL_STATUS"]).astype(object)

    df = pd.DataFrame(data)
    if labelled:
        # Create a target weakly correlated with HeartRate_Mean + Age signal.
        age = pd.to_datetime(df["ADMITTIME"]).dt.year - pd.to_datetime(df["DOB"]).dt.year
        logit = -2.5 + 0.015 * (df["HeartRate_Mean"] - 80) + 0.02 * (age - 60)
        prob = 1.0 / (1.0 + np.exp(-logit))
        df[TARGET_COL] = (rng.uniform(size=n) < prob).astype(int)
    return df


@pytest.fixture
def synthetic_train() -> pd.DataFrame:
    return _make_frame(300, seed=11, labelled=True)


@pytest.fixture
def synthetic_test() -> pd.DataFrame:
    return _make_frame(80, seed=22, labelled=False)


@pytest.fixture
def synthetic_extra() -> pd.DataFrame:
    rng = np.random.default_rng(33)
    rows = []
    for sub in range(1, 200):
        for had in rng.choice(range(1000, 9999), size=rng.integers(1, 4), replace=False):
            for seq in range(1, rng.integers(2, 6)):
                rows.append((sub, int(had), seq, f"ICD{rng.integers(100, 999)}"))
    return pd.DataFrame(rows, columns=["subject_id", "hadm_id", "SEQ_NUM", "ICD9_diagnosis"])
