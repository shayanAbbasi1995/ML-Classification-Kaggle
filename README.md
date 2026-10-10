# bgse-ml-classification: probability of death (MIMIC-III)

A 2026 rewrite of a Kaggle group project from the Barcelona Graduate School
of Economics ML class (2021). The task is the [`probability-of-death-21`][kaggle]
Kaggle competition: given one ICU admission's demographics, physiological
summary statistics, and ICD-9 diagnoses, predict the probability that the
patient dies in hospital. The data is derived from [MIMIC-III][mimic].

[kaggle]: https://www.kaggle.com/c/probability-of-death-21
[mimic]: https://mimic.physionet.org/

The scoring metric is binary log-loss on predicted probabilities.

## What this repository is (and isn't)

The original project lived in a single 148-cell notebook with hardcoded
Google-Drive paths, no train/validation split outside hand-rolled splits
inside each model section, a KNN imputer and a target encoder fit on
`concat([train, test])` (test-set leakage), and a final submission chosen
by eyeballing training-set accuracy. The notebook is preserved in
git history (commit `2687e77`) if you want to see the archaeology.

This rewrite keeps the useful ideas from that notebook - the medically-
informed categorical regimes (SpO2, TempC, pulse-pressure, heart-rate-by-
age, resp-rate-by-age, blood-glucose), the max-min fluctuation deltas,
the renal resistive index, the per-subject "previous ICU stay" count,
and the extra-diagnoses count - and discards everything else. It now
has a proper package layout, a sklearn `Pipeline` that keeps all
stateful preprocessing inside the CV loop (no more leakage), MLflow
tracking, type hints, structured logging, pinned dependencies, and a
test suite that runs end-to-end on every commit.

It is intentionally modest. The point was to have something that would
not be embarrassing to re-open five years from now, not to climb the
public leaderboard.

## Project layout

```
bgse-ml-classification/
|-- pyproject.toml          # uv project, pinned versions
|-- .python-version         # pinned to 3.12
|-- README.md               # this file
|-- src/pod/                # library ("probability of death")
|   |-- config.py           # paths, constants, seed, logging
|   |-- data.py             # raw CSV loading, cleaning, splits, hashing
|   |-- features.py         # stateless row-wise feature engineering
|   |-- models.py           # logreg + LightGBM Pipeline factories
|   |-- train.py            # CV + MLflow entrypoint
|   |-- evaluate.py         # log-loss / ROC-AUC / PR-AUC helpers
|   `-- predict.py          # produce submission.csv
|-- scripts/
|   |-- download_data.py    # Kaggle CLI wrapper
|   `-- make_submission.py
|-- tests/                  # pytest, runs on synthetic data
`-- data/raw/               # gitignored; populated by download_data.py
```

## Install and run

Everything is driven by [`uv`](https://docs.astral.sh/uv/).

```powershell
uv sync                                                   # install pinned deps
uv run python scripts/download_data.py                    # pull Kaggle data
uv run python -m pod.train                                # CV + MLflow logging
uv run python scripts/make_submission.py --model lgbm    # write submission.csv
```

The MLflow UI is local-SQLite backed:

```powershell
uv run mlflow ui --backend-store-uri "sqlite:///mlruns.db"
```

Then open <http://localhost:5000>.

### Prerequisites

Downloading the data requires a Kaggle API token at
`%USERPROFILE%\.kaggle\kaggle.json` (or `KAGGLE_USERNAME` and `KAGGLE_KEY`
environment variables). You must also click **Accept** on the competition's
[rules page][kaggle] before the first download, or the Kaggle CLI returns
403.

## Results

5-fold stratified CV on the 13,840-row training set, seed `20260101`,
`class_weight="balanced"` on both models, all preprocessing (median
impute + standard scale on numerics; constant-impute + target-encode
on categoricals) refit per fold inside the sklearn Pipeline. Positive
class rate in training is ~11.0%.

| Model               | log-loss (mean +/- std) | ROC-AUC (mean +/- std) | PR-AUC (mean +/- std) |
|---------------------|-------------------------|------------------------|-----------------------|
| Logistic regression | 0.4257 +/- 0.0048       | 0.8534 +/- 0.0062      | 0.4542 +/- 0.0220     |
| LightGBM            | 0.3296 +/- 0.0110       | 0.8546 +/- 0.0064      | 0.5244 +/- 0.0126     |

LightGBM wins decisively on log-loss (the Kaggle metric) and PR-AUC;
ROC-AUC is essentially a tie between the two. These are CV scores on
the training set only - the public-leaderboard figure will be whatever
Kaggle reports once `submissions/submission_lgbm.csv` is uploaded.

The original notebook reported only training-set accuracy, which is not
comparable - and is misleading on an imbalanced classification problem
regardless.

## Known limitations and next steps

- No hyperparameter search. LightGBM uses sensible defaults; a modest
  `optuna` sweep over `num_leaves`, `learning_rate`, and `n_estimators`
  would almost certainly help, but tuning is deferred until there's real
  CV evidence that it's the bottleneck.
- SMOTE and undersampling were tried in the original notebook. Modern
  gradient boosters with `class_weight="balanced"` typically match or
  beat SMOTE on tabular data, so neither is included here. Worth
  revisiting if the LightGBM baseline underperforms.
- Only the *count* of extra diagnoses from `extra_diagnoses.csv` is used.
  The original project also plucked the 2nd and 3rd ICD-9 code per
  admission by index into a `groupby`-list, which is both fragile and
  low signal; a proper multi-hot or embedded encoding of ICD-9 chapters
  would be a better upgrade.
- Calibration isn't checked. Since the metric is log-loss, a sigmoid or
  isotonic calibration layer wrapped around LightGBM is worth trying.

## Development

```powershell
uv run pytest                           # 20 tests
uv run ruff check src tests scripts     # linter
```

## License

MIT. See `LICENSE`.
