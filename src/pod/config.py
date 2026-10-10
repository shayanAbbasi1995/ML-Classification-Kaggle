"""Project-wide paths, constants, and seed.

Everything that another module might want to reference by name (column
lists, filenames, the random seed, path to the MLflow store) lives here
so the rest of the package has no magic strings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

#: Root of the git repository (the folder that contains ``pyproject.toml``).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SUBMISSIONS_DIR: Path = PROJECT_ROOT / "submissions"

#: SQLite-backed MLflow tracking store, kept under the repo root so a
#: fresh clone can start logging immediately.
MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlruns.db').as_posix()}",
)
MLFLOW_EXPERIMENT_NAME: str = os.environ.get("POD_MLFLOW_EXPERIMENT", "probability-of-death")

# Expected raw-data file names. The Kaggle download contains these.
TRAIN_CSV: str = "train_prob_death.csv"
TEST_CSV: str = "test_prob_death.csv"
EXTRA_DIAGNOSES_CSV: str = "extra_diagnoses.csv"

# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------

TARGET_COL: str = "HOSPITAL_EXPIRE_FLAG"
ID_COL: str = "icustay_id"

#: Numeric physiological-measurement columns present in train/test.
NUMERIC_COLS: tuple[str, ...] = (
    "HeartRate_Min", "HeartRate_Max", "HeartRate_Mean",
    "SysBP_Min", "SysBP_Max", "SysBP_Mean",
    "DiasBP_Min", "DiasBP_Max", "DiasBP_Mean",
    "MeanBP_Min", "MeanBP_Max", "MeanBP_Mean",
    "RespRate_Min", "RespRate_Max", "RespRate_Mean",
    "TempC_Min", "TempC_Max", "TempC_Mean",
    "SpO2_Min", "SpO2_Max", "SpO2_Mean",
    "Glucose_Min", "Glucose_Max", "Glucose_Mean",
)

#: Categorical demographic / admission columns.
CATEGORICAL_COLS: tuple[str, ...] = (
    "GENDER",
    "ADMISSION_TYPE",
    "INSURANCE",
    "RELIGION",
    "MARITAL_STATUS",
    "ETHNICITY",
    "DIAGNOSIS",
    "ICD9_diagnosis",
    "FIRST_CAREUNIT",
)

# ---------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------

SEED: int = 20260101  # yyyymmdd of a cold-but-sunny Thursday
CV_FOLDS: int = 5
#: Glucose placeholder in raw data; the original notebook noted ``999999``
#: appears in the Glucose columns and must be scrubbed to NaN.
GLUCOSE_SENTINEL: float = 999999.0
#: Patients older than this are clearly data artefacts (MIMIC anonymises
#: 300+ year-old ages for very-old patients).
MAX_PLAUSIBLE_AGE: float = 122.0


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger once.

    Call from script entrypoints. Library modules should only ever call
    ``logging.getLogger(__name__)``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
