"""pod - probability of death (MIMIC-III) classification.

A 2026 rewrite of a 5-year-old Kaggle project. Package layout:

- ``pod.config``   - paths, constants, random seed.
- ``pod.data``     - loading raw CSVs and producing splits.
- ``pod.features`` - feature engineering: previous-ICU-stay count,
  age-from-DOB, physiological regime categoricals, min-max difference
  features, extra-diagnoses aggregation.
- ``pod.models``   - sklearn ``Pipeline`` factories for the baseline
  logistic regression and a LightGBM gradient-boosted model.
- ``pod.train``    - CV training entrypoint with MLflow logging.
- ``pod.evaluate`` - metric helpers.
- ``pod.predict``  - fit on all training data and produce a Kaggle
  ``submission.csv``.
"""

from __future__ import annotations

__version__ = "0.2.0"
