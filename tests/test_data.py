"""Tests for ``pod.data``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pod import data
from pod.config import GLUCOSE_SENTINEL, TARGET_COL


def test_basic_clean_removes_glucose_sentinel(synthetic_train: pd.DataFrame) -> None:
    assert (synthetic_train["Glucose_Max"] == GLUCOSE_SENTINEL).any()
    cleaned = data.basic_clean(synthetic_train, is_train=True)
    assert not (cleaned["Glucose_Max"] == GLUCOSE_SENTINEL).any()
    # And the row count shouldn't change just because of sentinel scrubbing.
    assert len(cleaned) <= len(synthetic_train)  # may drop age outliers


def test_basic_clean_drops_implausible_ages() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "icustay_id": [100, 200],
            "ADMITTIME": ["2010-01-01", "2010-01-01"],
            # Second row: 2010 - 1700 = 310 years, must be dropped.
            "DOB": ["1950-01-01", "1700-01-01"],
            "Glucose_Mean": [100.0, 110.0],
        }
    )
    cleaned = data.basic_clean(df, is_train=True)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["subject_id"] == 1


def test_basic_clean_replaces_inf() -> None:
    df = pd.DataFrame({"Glucose_Max": [100.0, np.inf, -np.inf]})
    cleaned = data.basic_clean(df, is_train=False)
    assert cleaned["Glucose_Max"].isna().sum() == 2


def test_make_holdout_split_stratified(synthetic_train: pd.DataFrame) -> None:
    cleaned = data.basic_clean(synthetic_train, is_train=True)
    X_tr, X_hold, y_tr, y_hold = data.make_holdout_split(cleaned, test_size=0.25, seed=7)
    assert len(X_tr) + len(X_hold) == len(cleaned)
    assert TARGET_COL not in X_tr.columns and TARGET_COL not in X_hold.columns
    # Positive-class rate within 10 percentage points between splits.
    assert abs(y_tr.mean() - y_hold.mean()) < 0.10


def test_dataframe_hash_is_deterministic(synthetic_train: pd.DataFrame) -> None:
    h1 = data.dataframe_hash(synthetic_train)
    h2 = data.dataframe_hash(synthetic_train.copy())
    assert h1 == h2
    perturbed = synthetic_train.copy()
    perturbed.iloc[0, 0] = 999999
    assert data.dataframe_hash(perturbed) != h1


def test_load_raw_requires_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_raw(tmp_path)
