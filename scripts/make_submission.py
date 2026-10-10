"""Convenience wrapper: produce ``submissions/submission_<model>.csv``.

Usage::

    uv run python scripts/make_submission.py
    uv run python scripts/make_submission.py --model logreg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pod.models import MODEL_BUILDERS  # noqa: E402
from pod.predict import main as predict_main  # noqa: E402


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Kaggle submission CSV.")
    parser.add_argument("--model", default="lgbm", choices=list(MODEL_BUILDERS))
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse()
    path = predict_main(args.model, args.out)
    print(f"Submission written to: {path}")
