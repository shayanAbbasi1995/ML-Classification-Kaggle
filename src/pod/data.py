"""Loading raw Kaggle CSVs and producing train / holdout splits.

The original project kept all preprocessing in a flat notebook and merged
train + test together before fitting a KNN imputer and target encoder,
which leaks test information at training time. The new pipeline loads
raw CSVs verbatim, scrubs obvious sentinels, and keeps all learned
transforms inside the sklearn ``Pipeline`` (see ``pod.models``).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pod.config import (
    EXTRA_DIAGNOSES_CSV,
    GLUCOSE_SENTINEL,
    ID_COL,
    MAX_PLAUSIBLE_AGE,
    RAW_DATA_DIR,
    SEED,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawFrames:
    """Container for the three CSVs the Kaggle competition ships."""

    train: pd.DataFrame
    test: pd.DataFrame
    extra_diagnoses: pd.DataFrame | None


def load_raw(raw_dir: Path = RAW_DATA_DIR) -> RawFrames:
    """Load ``train``, ``test``, and (if present) ``extra_diagnoses``.

    Parameters
    ----------
    raw_dir:
        Directory containing the Kaggle download. The function does
        *not* unzip anything — ``scripts/download_data.py`` is
        responsible for that.
    """
    train_path = raw_dir / TRAIN_CSV
    test_path = raw_dir / TEST_CSV
    extra_path = raw_dir / EXTRA_DIAGNOSES_CSV

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected {TRAIN_CSV} and {TEST_CSV} in {raw_dir}. Run "
            "`uv run python scripts/download_data.py` first."
        )

    logger.info("Loading train from %s", train_path)
    train = pd.read_csv(train_path)
    logger.info("Loading test from %s", test_path)
    test = pd.read_csv(test_path)

    extra: pd.DataFrame | None = None
    if extra_path.exists():
        logger.info("Loading extra diagnoses from %s", extra_path)
        extra = pd.read_csv(extra_path)
        # The original notebook renamed anonymous columns; replicate so
        # downstream feature code can rely on these names.
        if list(extra.columns)[:4] != ["subject_id", "hadm_id", "SEQ_NUM", "ICD9_diagnosis"]:
            extra.columns = ["subject_id", "hadm_id", "SEQ_NUM", "ICD9_diagnosis", *extra.columns[4:]]
    else:
        logger.warning("extra_diagnoses CSV not found at %s; co-diagnosis features will be skipped",
                       extra_path)

    logger.info("Shapes - train=%s test=%s extra=%s",
                train.shape, test.shape, (extra.shape if extra is not None else "n/a"))
    return RawFrames(train=train, test=test, extra_diagnoses=extra)


def basic_clean(df: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
    """Scrub obvious sentinels and nonsense values **row-wise only**.

    This function must never learn any statistics from ``df`` — those
    belong inside the sklearn Pipeline so they can be re-fit per CV
    fold. Here we only do things that are safe in isolation:

    * Replace ``Glucose_* == 999999`` with NaN (documented sentinel).
    * Replace +/- inf with NaN.
    * Drop training rows whose derived Age is implausibly large.
    """
    df = df.copy()
    glucose_cols = [c for c in ("Glucose_Min", "Glucose_Max", "Glucose_Mean") if c in df.columns]
    for c in glucose_cols:
        sentinel_mask = df[c] == GLUCOSE_SENTINEL
        n_sent = int(sentinel_mask.sum())
        if n_sent:
            logger.info("Scrubbing %d sentinel (%s) values in %s", n_sent, GLUCOSE_SENTINEL, c)
            df.loc[sentinel_mask, c] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)

    if is_train and {"ADMITTIME", "DOB"}.issubset(df.columns):
        age = _age_from_dates(df["ADMITTIME"], df["DOB"])
        before = len(df)
        df = df.loc[age < MAX_PLAUSIBLE_AGE].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d training rows with Age >= %s (anonymised very-old patients)",
                        dropped, MAX_PLAUSIBLE_AGE)
    return df


def _age_from_dates(admit: pd.Series, dob: pd.Series) -> pd.Series:
    """Year(ADMITTIME) - Year(DOB). Matches the original notebook."""
    admit_year = pd.to_datetime(admit, errors="coerce").dt.year
    dob_year = pd.to_datetime(dob, errors="coerce").dt.year
    return (admit_year - dob_year).astype("float64")


def make_holdout_split(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train / holdout split on the labelled data."""
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not in dataframe")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    X_tr, X_hold, y_tr, y_hold = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    logger.info("Holdout split: train=%d (%.3f pos) holdout=%d (%.3f pos)",
                len(X_tr), y_tr.mean(), len(X_hold), y_hold.mean())
    return X_tr, X_hold, y_tr, y_hold


def dataframe_hash(df: pd.DataFrame) -> str:
    """SHA-256 of the dataframe's byte representation.

    Logged to MLflow so a run can be tied back to the exact data it
    saw.
    """
    hasher = hashlib.sha256()
    hasher.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    hasher.update(",".join(df.columns.astype(str)).encode("utf-8"))
    return hasher.hexdigest()


def submission_ids(test_df: pd.DataFrame, *, id_col: str = ID_COL) -> pd.Series:
    """Return the ID column needed for the Kaggle submission CSV."""
    if id_col not in test_df.columns:
        raise KeyError(f"ID column {id_col!r} not present in test dataframe")
    return test_df[id_col].copy()
