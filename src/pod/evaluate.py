"""Metric helpers.

Kaggle's "probability-of-death-21" competition uses **binary log-loss**
on predicted probabilities. We also track ROC-AUC and PR-AUC because
they're useful for diagnosing class-imbalance behaviour independent of
calibration.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metrics computed on predicted probabilities for the positive class."""

    log_loss: float
    roc_auc: float
    pr_auc: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        d = asdict(self)
        return {f"{prefix}{k}": v for k, v in d.items()} if prefix else d


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> ClassificationMetrics:
    """Compute log-loss, ROC-AUC, PR-AUC on positive-class probabilities.

    Probabilities are clipped to ``(1e-15, 1 - 1e-15)`` before log-loss
    to avoid ``inf`` on perfectly confident wrong answers.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)

    ll = float(log_loss(y_true, y_prob_clipped, labels=[0, 1]))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        logger.warning("ROC-AUC undefined (likely single-class fold); returning NaN")
        auc = float("nan")
    try:
        pr = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr = float("nan")

    return ClassificationMetrics(log_loss=ll, roc_auc=auc, pr_auc=pr)


def summarise_cv(per_fold: list[ClassificationMetrics]) -> dict[str, float]:
    """Mean +/- std of each metric across CV folds."""
    arr = {
        "log_loss": np.array([m.log_loss for m in per_fold]),
        "roc_auc": np.array([m.roc_auc for m in per_fold]),
        "pr_auc": np.array([m.pr_auc for m in per_fold]),
    }
    out: dict[str, float] = {}
    for k, v in arr.items():
        out[f"{k}_mean"] = float(np.nanmean(v))
        out[f"{k}_std"] = float(np.nanstd(v, ddof=0))
    return out
