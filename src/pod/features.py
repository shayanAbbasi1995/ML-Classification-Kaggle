"""Feature engineering.

Everything here is a *stateless* row-wise transform of a dataframe. The
functions that could leak target information (imputation, scaling,
target encoding) live inside the sklearn ``Pipeline`` in
``pod.models``, where they get re-fit each CV fold.

Original-project feature ideas we preserve:

- previous-ICU-stay count per subject
- age in years from ``ADMITTIME`` - ``DOB``
- min/max fluctuation deltas (``HeartRate_Max - HeartRate_Min`` etc.)
- "renal resistive index" (``DiasBP_Mean / SysBP_Mean``)
- medically-informed categorical regimes for SpO2, TempC, pulse
  pressure, heart rate, respiratory rate, blood glucose
- a count of extra diagnoses per admission from ``extra_diagnoses.csv``
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns the feature functions produce. Exported so ``pod.models`` can
# tell the preprocessor what's numeric vs. categorical.
DIFF_FEATURES: tuple[str, ...] = (
    "heartRate_fluc",
    "SysBP_fluc",
    "DiasBP_fluc",
    "BP_fluc",
    "MeanBP_fluc",
    "renal_resistive_index",
    "RespRate_fluc",
    "TempC_fluc",
    "SpO2_fluc",
    "Glucose_fluc",
)

REGIME_FEATURES: tuple[str, ...] = (
    "Spo2_situation",
    "TempC_situation",
    "Pulse_pressure",
    "Heart_rate_situation",
    "Resp_rate_situation",
    "Diabetes_situation",
)

EXTRA_FEATURES: tuple[str, ...] = ("num_prev_stay", "n_extra_diagnoses", "Age")


# ---------------------------------------------------------------------
# Diff / ratio features
# ---------------------------------------------------------------------

def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max fluctuation deltas and the renal-resistive index."""
    df = df.copy()
    pairs = [
        ("heartRate_fluc", "HeartRate_Max", "HeartRate_Min"),
        ("SysBP_fluc", "SysBP_Max", "SysBP_Min"),
        ("DiasBP_fluc", "DiasBP_Max", "DiasBP_Min"),
        ("BP_fluc", "SysBP_Mean", "DiasBP_Mean"),
        ("MeanBP_fluc", "MeanBP_Max", "MeanBP_Min"),
        ("RespRate_fluc", "RespRate_Max", "RespRate_Min"),
        ("TempC_fluc", "TempC_Max", "TempC_Min"),
        ("SpO2_fluc", "SpO2_Max", "SpO2_Min"),
        ("Glucose_fluc", "Glucose_Max", "Glucose_Min"),
    ]
    for new, a, b in pairs:
        if a in df.columns and b in df.columns:
            df[new] = df[a] - df[b]
        else:
            df[new] = np.nan

    if {"DiasBP_Mean", "SysBP_Mean"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["renal_resistive_index"] = df["DiasBP_Mean"] / df["SysBP_Mean"]
        df["renal_resistive_index"] = df["renal_resistive_index"].replace(
            [np.inf, -np.inf], np.nan
        )
    else:
        df["renal_resistive_index"] = np.nan
    return df


# ---------------------------------------------------------------------
# Age + previous-stay
# ---------------------------------------------------------------------

def add_age(df: pd.DataFrame) -> pd.DataFrame:
    """Age in years, derived from ``ADMITTIME`` and ``DOB``.

    The original notebook sliced the first 4 characters of each string
    which is fine for the Kaggle CSVs but breaks if the datetime
    parser changes. We parse to ``datetime`` and take the year
    difference; missing values stay NaN rather than crashing.
    """
    df = df.copy()
    if {"ADMITTIME", "DOB"}.issubset(df.columns):
        admit_y = pd.to_datetime(df["ADMITTIME"], errors="coerce").dt.year
        dob_y = pd.to_datetime(df["DOB"], errors="coerce").dt.year
        df["Age"] = (admit_y - dob_y).astype("float64")
    else:
        df["Age"] = np.nan
    return df


def add_previous_stay_count(df: pd.DataFrame) -> pd.DataFrame:
    """Zero-indexed count of prior ICU stays per ``(subject_id, icustay_id)``.

    This is a vectorised version of the original per-group Python
    loop. For each ``subject_id`` we sort by ``ADMITTIME`` and
    cumulatively count. The *first* stay gets 0, the second 1, etc.
    """
    df = df.copy()
    required = {"subject_id", "icustay_id", "ADMITTIME"}
    if not required.issubset(df.columns):
        logger.warning("Columns %s not all present; filling num_prev_stay with 0",
                       required - set(df.columns))
        df["num_prev_stay"] = 0
        return df

    admit = pd.to_datetime(df["ADMITTIME"], errors="coerce")
    order = (
        pd.DataFrame({"subject_id": df["subject_id"], "admit": admit})
        .groupby("subject_id")["admit"]
        .rank(method="first")
        .fillna(1)
        .astype(int)
        - 1
    )
    df["num_prev_stay"] = order.to_numpy()
    return df


# ---------------------------------------------------------------------
# Medical-regime categoricals
# ---------------------------------------------------------------------

def _spo2(x: float) -> str:
    if pd.isna(x):
        return "unknown"
    if x >= 95:
        return "normal"
    if x > 85:
        return "considerable"
    return "critical"


def _tempc(x: float) -> str:
    if pd.isna(x):
        return "unknown"
    if x >= 44:
        return "deadly"
    if x > 41:
        return "critical"
    if x > 37.8:
        return "fever"
    if x < 36.1:
        return "hypothermy"
    return "normal"


def _pulse_pressure(sys_mean: float, dias_mean: float) -> str:
    if pd.isna(sys_mean) or pd.isna(dias_mean):
        return "unknown"
    pp = sys_mean - dias_mean
    if 40 <= pp <= 60:
        return "normal"
    if pp < 40:
        return "heart_failure"
    return "hypertension"


def _heart_rate(hr: float, age: float) -> str:
    if pd.isna(hr) or pd.isna(age):
        return "unknown"
    # Pediatric age-dependent normal bands + adult band.
    ranges: list[tuple[float, float, float, float]] = [
        (0, 1, 90, 170),
        (1, 3, 80, 150),
        (3, 6, 70, 120),
        (6, 12, 60, 110),
        (12, np.inf, 60, 100),
    ]
    for lo_age, hi_age, lo_hr, hi_hr in ranges:
        if lo_age <= age < hi_age:
            return "normal" if lo_hr <= hr <= hi_hr else "bad"
    return "unknown"


def _resp_rate(rr: float, age: float) -> str:
    if pd.isna(rr) or pd.isna(age):
        return "unknown"
    ranges: list[tuple[float, float, float, float]] = [
        (0, 1, 22, 70),
        (1, 3, 22, 30),
        (3, 6, 20, 24),
        (6, 12, 16, 22),
        (12, np.inf, 12, 20),
    ]
    for lo_age, hi_age, lo_rr, hi_rr in ranges:
        if lo_age <= age < hi_age:
            return "normal" if lo_rr <= rr <= hi_rr else "bad"
    return "unknown"


def _diabetes(glu: float) -> str:
    if pd.isna(glu):
        return "unknown"
    if glu < 140:
        return "normal"
    if glu < 200:
        return "prediabetes"
    return "diabetes"


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add six medically-informed categorical columns."""
    df = df.copy()
    df["Spo2_situation"] = df.get("SpO2_Mean", pd.Series(np.nan, index=df.index)).map(_spo2)
    df["TempC_situation"] = df.get("TempC_Mean", pd.Series(np.nan, index=df.index)).map(_tempc)
    if {"SysBP_Mean", "DiasBP_Mean"}.issubset(df.columns):
        df["Pulse_pressure"] = [
            _pulse_pressure(s, d) for s, d in zip(df["SysBP_Mean"], df["DiasBP_Mean"], strict=True)
        ]
    else:
        df["Pulse_pressure"] = "unknown"
    age = df["Age"] if "Age" in df.columns else pd.Series(np.nan, index=df.index)
    df["Heart_rate_situation"] = [
        _heart_rate(hr, a) for hr, a in zip(df.get("HeartRate_Mean", age * np.nan), age, strict=True)
    ]
    df["Resp_rate_situation"] = [
        _resp_rate(rr, a) for rr, a in zip(df.get("RespRate_Mean", age * np.nan), age, strict=True)
    ]
    df["Diabetes_situation"] = df.get("Glucose_Mean", pd.Series(np.nan, index=df.index)).map(_diabetes)
    return df


# ---------------------------------------------------------------------
# Extra diagnoses count
# ---------------------------------------------------------------------

def add_extra_diagnoses_count(df: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    """Add ``n_extra_diagnoses`` from ``extra_diagnoses.csv`` (0 if missing)."""
    df = df.copy()
    if extra is None or not {"subject_id", "hadm_id"}.issubset(df.columns):
        df["n_extra_diagnoses"] = 0
        return df
    counts = (
        extra.groupby(["subject_id", "hadm_id"]).size().rename("n_extra_diagnoses").reset_index()
    )
    df = df.merge(counts, on=["subject_id", "hadm_id"], how="left")
    df["n_extra_diagnoses"] = df["n_extra_diagnoses"].fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------
# One-call public API
# ---------------------------------------------------------------------

def build_features(df: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    """Apply every feature step. Order matters (Age must precede regimes)."""
    logger.info("Building features on dataframe of shape %s", df.shape)
    df = add_previous_stay_count(df)
    df = add_age(df)
    df = add_diff_features(df)
    df = add_regime_features(df)
    df = add_extra_diagnoses_count(df, extra)

    # Columns that were only inputs to feature engineering can be dropped
    # before modelling. ID columns are kept so the caller can split
    # them off later.
    drop_cols = [c for c in ("ADMITTIME", "DOB") if c in df.columns]
    if drop_cols:
        logger.debug("Dropping input-only columns: %s", drop_cols)
        df = df.drop(columns=drop_cols)
    logger.info("Feature-engineered shape: %s", df.shape)
    return df
