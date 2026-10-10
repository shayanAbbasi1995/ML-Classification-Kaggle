"""Fit a chosen model on all training data and write ``submission.csv``.

Separate from ``pod.train`` so the submission step can be re-run
without redoing CV. Reuses the same feature pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from pod.config import (
    ID_COL,
    SEED,
    SUBMISSIONS_DIR,
    TARGET_COL,
    configure_logging,
)
from pod.data import basic_clean, load_raw, submission_ids
from pod.features import build_features
from pod.models import MODEL_BUILDERS
from pod.train import _feature_columns

logger = logging.getLogger(__name__)


def fit_and_predict(model_name: str, *, seed: int = SEED) -> pd.DataFrame:
    """Train on all labelled data and return predictions for the Kaggle test set.

    Returns a dataframe with two columns, ``icustay_id`` and
    ``HOSPITAL_EXPIRE_FLAG`` (predicted probability of the positive
    class), matching the Kaggle submission schema.
    """
    if model_name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model {model_name!r}; known={list(MODEL_BUILDERS)}")

    frames = load_raw()
    train = basic_clean(frames.train, is_train=True)
    test = basic_clean(frames.test, is_train=False)

    train_feat = build_features(train, frames.extra_diagnoses)
    test_feat = build_features(test, frames.extra_diagnoses)

    y = train_feat[TARGET_COL].astype(int)
    X_train = train_feat.drop(columns=[TARGET_COL])
    numeric, categorical = _feature_columns()
    builder = MODEL_BUILDERS[model_name]
    pipeline: Pipeline = builder(numeric, categorical, seed=seed)

    logger.info("Fitting %s on %d rows", model_name, len(X_train))
    pipeline.fit(X_train, y)
    logger.info("Predicting on %d test rows", len(test_feat))
    probs = pipeline.predict_proba(test_feat)[:, 1]

    ids = submission_ids(test_feat)
    submission = pd.DataFrame({ID_COL: ids.values, TARGET_COL: probs})
    return submission


def write_submission(submission: pd.DataFrame, path: Path) -> Path:
    """Write submission CSV and return the absolute path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    logger.info("Wrote submission with %d rows to %s", len(submission), path)
    return path.resolve()


def main(model_name: str = "lgbm", out_path: Path | None = None) -> Path:
    configure_logging()
    submission = fit_and_predict(model_name)
    out = out_path or (SUBMISSIONS_DIR / f"submission_{model_name}.csv")
    return write_submission(submission, out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a Kaggle submission CSV.")
    parser.add_argument("--model", default="lgbm", choices=list(MODEL_BUILDERS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.model, args.out)
