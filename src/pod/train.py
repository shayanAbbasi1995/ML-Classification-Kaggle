"""CV training entrypoint.

Runs stratified K-fold cross-validation for each registered model and
logs hyperparameters, metrics, dataset hash, git commit, and seed to a
local MLflow store. Also fits a final model on all training data and
logs it as an artifact so ``pod.predict`` can load it later.

Usage::

    uv run python -m pod.train
    uv run python -m pod.train --model lgbm
    uv run python -m pod.train --no-mlflow  # dry run, prints only
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from pod.config import (
    CATEGORICAL_COLS,
    CV_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    NUMERIC_COLS,
    SEED,
    TARGET_COL,
    configure_logging,
)
from pod.data import basic_clean, dataframe_hash, load_raw
from pod.evaluate import ClassificationMetrics, compute_metrics, summarise_cv
from pod.features import DIFF_FEATURES, EXTRA_FEATURES, REGIME_FEATURES, build_features
from pod.models import MODEL_BUILDERS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    cv_summary: dict[str, float]
    per_fold: list[ClassificationMetrics]
    fitted: Pipeline


def _git_sha() -> str:
    """Best-effort git commit SHA; returns 'unknown' outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read git SHA: %s", exc)
        return "unknown"


def _feature_columns() -> tuple[list[str], list[str]]:
    """Lists of numeric and categorical columns fed to the preprocessor."""
    numeric = list(NUMERIC_COLS) + list(DIFF_FEATURES) + list(EXTRA_FEATURES)
    categorical = list(CATEGORICAL_COLS) + list(REGIME_FEATURES)
    return numeric, categorical


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = CV_FOLDS,
    seed: int = SEED,
) -> list[ClassificationMetrics]:
    """Stratified K-fold CV returning per-fold metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_fold: list[ClassificationMetrics] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        logger.info("Fold %d/%d - train=%d val=%d", fold, n_splits, len(train_idx), len(val_idx))
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        m = compute_metrics(y_val.to_numpy(), y_prob)
        logger.info("Fold %d metrics: log_loss=%.4f roc_auc=%.4f pr_auc=%.4f",
                    fold, m.log_loss, m.roc_auc, m.pr_auc)
        per_fold.append(m)
    return per_fold


def train_one_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = SEED,
    n_splits: int = CV_FOLDS,
    use_mlflow: bool = True,
    data_hash: str | None = None,
) -> TrainingResult:
    """CV one named model, log to MLflow, refit on all data, return result."""
    if model_name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model {model_name!r}; known={list(MODEL_BUILDERS)}")

    numeric, categorical = _feature_columns()
    builder = MODEL_BUILDERS[model_name]
    pipeline = builder(numeric, categorical, seed=seed)

    logger.info("Running %d-fold CV for %s", n_splits, model_name)
    per_fold = cross_validate_model(pipeline, X, y, n_splits=n_splits, seed=seed)
    summary = summarise_cv(per_fold)
    logger.info("CV summary for %s: %s", model_name, summary)

    # Refit on all training data.
    logger.info("Refitting %s on all %d rows", model_name, len(X))
    final_pipeline = builder(numeric, categorical, seed=seed)
    final_pipeline.fit(X, y)

    if use_mlflow:
        _log_to_mlflow(
            model_name=model_name,
            pipeline=final_pipeline,
            summary=summary,
            per_fold=per_fold,
            seed=seed,
            n_splits=n_splits,
            n_rows=len(X),
            data_hash=data_hash,
        )

    return TrainingResult(
        model_name=model_name,
        cv_summary=summary,
        per_fold=per_fold,
        fitted=final_pipeline,
    )


def _log_to_mlflow(
    *,
    model_name: str,
    pipeline: Pipeline,
    summary: dict[str, float],
    per_fold: list[ClassificationMetrics],
    seed: int,
    n_splits: int,
    n_rows: int,
    data_hash: str | None,
) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(
            {
                "model": model_name,
                "seed": seed,
                "cv_folds": n_splits,
                "n_rows": n_rows,
                "git_sha": _git_sha(),
                "data_hash": data_hash or "unknown",
            }
        )
        # Flatten estimator-specific params.
        try:
            clf_params = pipeline.named_steps["clf"].get_params()
            mlflow.log_params({f"clf.{k}": v for k, v in clf_params.items()
                               if isinstance(v, (str, int, float, bool, type(None)))})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not log clf params: %s", exc)

        mlflow.log_metrics(summary)
        for i, m in enumerate(per_fold, start=1):
            mlflow.log_metrics({f"fold{i}_{k}": v for k, v in m.as_dict().items()})

        try:
            # ``skops`` (MLflow 3's default) can't serialise ``category_encoders``
            # transformers; fall back to cloudpickle which handles them.
            mlflow.sklearn.log_model(
                pipeline,
                name=f"model-{model_name}",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow model log failed: %s", exc)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train POD classifiers with CV + MLflow.")
    p.add_argument(
        "--model",
        choices=[*MODEL_BUILDERS.keys(), "all"],
        default="all",
        help="Which model(s) to train.",
    )
    p.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging (debug).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, TrainingResult]:
    args = _parse_args(argv)
    configure_logging()
    rng = np.random.default_rng(args.seed)
    _ = rng.integers(0, 2**32 - 1)  # just to make the seed get used

    frames = load_raw()
    train_df = basic_clean(frames.train, is_train=True)
    train_feat = build_features(train_df, frames.extra_diagnoses)

    y = train_feat[TARGET_COL].astype(int)
    X = train_feat.drop(columns=[TARGET_COL])
    data_hash = dataframe_hash(train_feat)
    logger.info("Training data hash=%s", data_hash[:16])

    models = list(MODEL_BUILDERS) if args.model == "all" else [args.model]
    results: dict[str, TrainingResult] = {}
    for name in models:
        results[name] = train_one_model(
            name, X, y,
            seed=args.seed,
            n_splits=args.cv_folds,
            use_mlflow=not args.no_mlflow,
            data_hash=data_hash,
        )

    # Print a comparison summary.
    logger.info("=" * 72)
    logger.info("CV comparison:")
    for name, r in results.items():
        s = r.cv_summary
        logger.info("  %-6s | log_loss %.4f +/- %.4f | roc_auc %.4f +/- %.4f | pr_auc %.4f +/- %.4f",
                    name, s["log_loss_mean"], s["log_loss_std"],
                    s["roc_auc_mean"], s["roc_auc_std"],
                    s["pr_auc_mean"], s["pr_auc_std"])
    return results


if __name__ == "__main__":
    main()
