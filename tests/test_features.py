"""Tests for ``pod.features``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pod import features


def test_add_diff_features_computes_deltas() -> None:
    df = pd.DataFrame(
        {
            "HeartRate_Max": [100.0], "HeartRate_Min": [60.0],
            "SysBP_Max": [140.0], "SysBP_Min": [100.0],
            "DiasBP_Max": [90.0], "DiasBP_Min": [60.0],
            "SysBP_Mean": [120.0], "DiasBP_Mean": [80.0],
            "MeanBP_Max": [100.0], "MeanBP_Min": [70.0],
            "RespRate_Max": [20.0], "RespRate_Min": [12.0],
            "TempC_Max": [37.5], "TempC_Min": [36.5],
            "SpO2_Max": [100.0], "SpO2_Min": [92.0],
            "Glucose_Max": [180.0], "Glucose_Min": [90.0],
        }
    )
    out = features.add_diff_features(df)
    assert out["heartRate_fluc"].iloc[0] == 40.0
    assert out["BP_fluc"].iloc[0] == 40.0
    # Renal resistive index = 80/120.
    assert out["renal_resistive_index"].iloc[0] == 80.0 / 120.0


def test_add_age_year_difference() -> None:
    df = pd.DataFrame(
        {"ADMITTIME": ["2010-05-01 00:00:00", "2011-07-14 08:00:00"],
         "DOB": ["1950-01-01 00:00:00", "1975-01-01 00:00:00"]}
    )
    out = features.add_age(df)
    assert list(out["Age"].astype(int)) == [60, 36]


def test_add_previous_stay_count_orders_by_admittime() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1, 1, 1, 2],
            "icustay_id": [10, 11, 12, 20],
            "ADMITTIME": ["2010-01-01", "2008-01-01", "2012-01-01", "2010-01-01"],
        }
    )
    out = features.add_previous_stay_count(df)
    # Order by admittime for subject 1: 11 (2008), 10 (2010), 12 (2012) -> 0,1,2.
    counts = dict(zip(out["icustay_id"], out["num_prev_stay"], strict=True))
    assert counts == {10: 1, 11: 0, 12: 2, 20: 0}


def test_regime_features_handle_nans() -> None:
    df = pd.DataFrame(
        {
            "SpO2_Mean": [97.0, 90.0, 80.0, np.nan],
            "TempC_Mean": [37.0, 42.0, 44.5, np.nan],
            "SysBP_Mean": [120.0, 100.0, 130.0, np.nan],
            "DiasBP_Mean": [70.0, 70.0, 70.0, np.nan],
            "HeartRate_Mean": [70.0, 200.0, 50.0, np.nan],
            "RespRate_Mean": [16.0, 40.0, 10.0, np.nan],
            "Glucose_Mean": [100.0, 150.0, 250.0, np.nan],
            "Age": [45.0, 30.0, 70.0, 50.0],
        }
    )
    out = features.add_regime_features(df)
    assert out["Spo2_situation"].iloc[0] == "normal"
    assert out["Spo2_situation"].iloc[1] == "considerable"
    assert out["Spo2_situation"].iloc[2] == "critical"
    assert out["Spo2_situation"].iloc[3] == "unknown"
    assert out["Diabetes_situation"].iloc[0] == "normal"
    assert out["Diabetes_situation"].iloc[1] == "prediabetes"
    assert out["Diabetes_situation"].iloc[2] == "diabetes"
    assert out["TempC_situation"].iloc[2] == "deadly"


def test_add_extra_diagnoses_count(synthetic_train, synthetic_extra) -> None:
    out = features.add_extra_diagnoses_count(synthetic_train, synthetic_extra)
    assert "n_extra_diagnoses" in out.columns
    assert out["n_extra_diagnoses"].min() >= 0
    # When extra is None we still get a column of zeros.
    out2 = features.add_extra_diagnoses_count(synthetic_train, None)
    assert (out2["n_extra_diagnoses"] == 0).all()


def test_build_features_produces_all_engineered_cols(synthetic_train, synthetic_extra) -> None:
    out = features.build_features(synthetic_train, synthetic_extra)
    for col in (
        *features.DIFF_FEATURES,
        *features.REGIME_FEATURES,
        *features.EXTRA_FEATURES,
    ):
        assert col in out.columns, f"missing engineered column: {col}"
    assert "ADMITTIME" not in out.columns
    assert "DOB" not in out.columns
