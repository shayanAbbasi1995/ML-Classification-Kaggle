"""Download the Kaggle competition data into ``data/raw/``.

Requires ``~/.kaggle/kaggle.json`` to be present, or the environment
variables ``KAGGLE_USERNAME`` + ``KAGGLE_KEY`` to be set.

We shell out to the Kaggle CLI (via ``uvx``) rather than taking a
direct dependency, because the kaggle package pulls in a lot of
transitive deps that are irrelevant to training.

Usage::

    uv run python scripts/download_data.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Package import for config lookup only; keeps script and lib in sync.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pod.config import RAW_DATA_DIR, configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

COMPETITION = "probability-of-death-21"


def _have_credentials() -> bool:
    home = Path(os.environ.get("USERPROFILE") or os.environ.get("HOME") or "")
    kaggle_json = home / ".kaggle" / "kaggle.json"
    return kaggle_json.exists() or {"KAGGLE_USERNAME", "KAGGLE_KEY"}.issubset(os.environ)


def download(dest: Path = RAW_DATA_DIR) -> None:
    """Download and unzip the Kaggle competition archive."""
    if not _have_credentials():
        raise RuntimeError(
            "Kaggle credentials missing. Put kaggle.json in %USERPROFILE%\\.kaggle\\ "
            "or set KAGGLE_USERNAME / KAGGLE_KEY."
        )

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading competition %s to %s", COMPETITION, dest)

    cmd = [
        "uvx", "--quiet", "kaggle", "competitions", "download",
        "-c", COMPETITION, "-p", str(dest),
    ]
    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Kaggle CLI failed (exit {exc.returncode}). "
            "If this is a '403: You must accept this competition's rules before you'll "
            "be able to download files' error, visit "
            "https://www.kaggle.com/c/probability-of-death-21/rules and click Accept."
        ) from exc

    zips = list(dest.glob("*.zip"))
    if not zips:
        logger.warning("No .zip archive produced; nothing to extract.")
        return
    for z in zips:
        logger.info("Extracting %s", z.name)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()

    logger.info("Files in %s: %s", dest, sorted(p.name for p in dest.iterdir()))


if __name__ == "__main__":
    configure_logging()
    download()
