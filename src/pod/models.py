"""Model factories.

Each factory returns a sklearn ``Pipeline`` that bundles preprocessing
(imputation + encoding + scaling) with the estimator. All *stateful*
preprocessing — including target encoding, which the original project
fit on the full training set and so leaked target info — lives *inside*
the pipeline so ``cross_val_score`` refits it each fold.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from category_encoders import TargetEncoder
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pod.config import SEED

logger = logging.getLogger(__name__)


def _numeric_branch() -> Pipeline:
    """Impute medians, then standard-scale."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _categorical_branch() -> Pipeline:
    """Fill a sentinel string, then target-encode (per fold)."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("target_enc", TargetEncoder(smoothing=10.0)),
        ]
    )


def build_preprocessor(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ColumnTransformer:
    """Shared preprocessing for both model families.

    LightGBM technically doesn't need scaling, but keeping one
    ``ColumnTransformer`` makes the two pipelines directly comparable
    and the cost is negligible.
    """
    return ColumnTransformer(
        transformers=[
            ("num", _numeric_branch(), list(numeric_cols)),
            ("cat", _categorical_branch(), list(categorical_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_logreg_pipeline(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    C: float = 1.0,
    seed: int = SEED,
) -> Pipeline:
    """Baseline: L2-regularised logistic regression."""
    pre = build_preprocessor(numeric_cols, categorical_cols)
    # sklearn >= 1.8 deprecates ``penalty`` / ``n_jobs`` on LogisticRegression.
    # Default solver=lbfgs with no ``penalty`` kwarg == L2 regularisation.
    clf = LogisticRegression(
        C=C,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=5_000,
        random_state=seed,
    )
    logger.debug("Built logreg pipeline (C=%s, seed=%s)", C, seed)
    return Pipeline(steps=[("preprocess", pre), ("clf", clf)])


def build_lgbm_pipeline(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    min_child_samples: int = 20,
    reg_lambda: float = 1.0,
    seed: int = SEED,
) -> Pipeline:
    """LightGBM gradient boosting with sensible defaults."""
    pre = build_preprocessor(numeric_cols, categorical_cols)
    clf = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        reg_lambda=reg_lambda,
        class_weight="balanced",
        objective="binary",
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    logger.debug(
        "Built LGBM pipeline (n=%d lr=%s leaves=%d seed=%s)",
        n_estimators, learning_rate, num_leaves, seed,
    )
    return Pipeline(steps=[("preprocess", pre), ("clf", clf)])


# Registry so ``train.py`` / ``predict.py`` can refer to models by name.
MODEL_BUILDERS = {
    "logreg": build_logreg_pipeline,
    "lgbm": build_lgbm_pipeline,
}
