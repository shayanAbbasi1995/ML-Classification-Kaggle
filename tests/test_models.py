"""Tests for ``pod.models``, ``pod.train``, ``pod.evaluate``, ``pod.predict``.

Covers the two headline integration guarantees:

1. Both model pipelines ``.fit`` and ``.predict_proba`` on the engineered
   feature frame without errors.
2. ``train.train_one_model`` runs 2 folds end-to-end on synthetic data
   and ``predict.fit_and_predict`` produces a submission dataframe
   with the right schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pod import features, models, train
from pod.config import ID_COL, TARGET_COL
from pod.data import basic_clean
from pod.evaluate import compute_metrics, summarise_cv
from pod.predict import fit_and_predict, write_submission


def _xy_from_synthetic(synthetic_train, synthetic_extra):
    df = basic_clean(synthetic_train, is_train=True)
    df = features.build_features(df, synthetic_extra)
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def test_logreg_pipeline_fits_and_predicts(synthetic_train, synthetic_extra) -> None:
    X, y = _xy_from_synthetic(synthetic_train, synthetic_extra)
    numeric, categorical = train._feature_columns()
    pipe = models.build_logreg_pipeline(numeric, categorical)
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)[:, 1]
    assert probs.shape == (len(X),)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_lgbm_pipeline_fits_and_predicts(synthetic_train, synthetic_extra) -> None:
    X, y = _xy_from_synthetic(synthetic_train, synthetic_extra)
    numeric, categorical = train._feature_columns()
    pipe = models.build_lgbm_pipeline(numeric, categorical, n_estimators=20)
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)[:, 1]
    assert probs.shape == (len(X),)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_compute_metrics_bounds() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3])
    m = compute_metrics(y_true, y_prob)
    assert 0.0 <= m.roc_auc <= 1.0
    assert 0.0 <= m.pr_auc <= 1.0
    assert m.log_loss > 0.0


def test_summarise_cv_shape() -> None:
    fakes = [compute_metrics(np.array([0, 1, 1, 0]), np.array([0.2, 0.8, 0.7, 0.3])) for _ in range(3)]
    s = summarise_cv(fakes)
    assert {"log_loss_mean", "log_loss_std", "roc_auc_mean", "roc_auc_std",
            "pr_auc_mean", "pr_auc_std"} == set(s)


@pytest.mark.parametrize("model_name", ["logreg", "lgbm"])
def test_train_one_model_runs_cv(model_name, synthetic_train, synthetic_extra) -> None:
    X, y = _xy_from_synthetic(synthetic_train, synthetic_extra)
    result = train.train_one_model(
        model_name, X, y, n_splits=2, seed=1, use_mlflow=False, data_hash="test",
    )
    assert result.model_name == model_name
    assert len(result.per_fold) == 2
    s = result.cv_summary
    assert s["log_loss_mean"] > 0.0
    # Final refit yielded a fitted pipeline we can predict with.
    probs = result.fitted.predict_proba(X)[:, 1]
    assert probs.shape == (len(X),)


def test_fit_and_predict_schema(monkeypatch, tmp_path, synthetic_train, synthetic_test, synthetic_extra) -> None:
    """fit_and_predict should produce the Kaggle-expected 2-column dataframe."""
    from pod import data as data_mod
    from pod import predict as predict_mod

    def _fake_load_raw(*a, **kw):
        return data_mod.RawFrames(
            train=synthetic_train, test=synthetic_test, extra_diagnoses=synthetic_extra,
        )

    monkeypatch.setattr(predict_mod, "load_raw", _fake_load_raw)
    out = fit_and_predict("logreg", seed=1)
    assert list(out.columns) == [ID_COL, TARGET_COL]
    assert len(out) == len(synthetic_test)
    assert out[TARGET_COL].between(0, 1).all()

    # And writing it out works.
    sub_path = tmp_path / "submission.csv"
    returned = write_submission(out, sub_path)
    assert returned.exists()
    reread = pd.read_csv(returned)
    assert list(reread.columns) == [ID_COL, TARGET_COL]
    assert len(reread) == len(out)


def test_write_submission_creates_parent_dir(tmp_path: Path) -> None:
    df = pd.DataFrame({ID_COL: [1, 2], TARGET_COL: [0.1, 0.9]})
    target = tmp_path / "nested" / "deeper" / "sub.csv"
    out = write_submission(df, target)
    assert out.exists()
