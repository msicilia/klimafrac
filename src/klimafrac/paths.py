"""Canonical file locations, so every module works from any working directory.

`KLIMAFRAC_DATA_DIR` and `KLIMAFRAC_FIGURE_DIR` override the defaults, which
lets a test run write somewhere other than the real dataset.
"""

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("KLIMAFRAC_DATA_DIR", CODE_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
FIGURE_DIR = Path(os.environ.get("KLIMAFRAC_FIGURE_DIR", CODE_ROOT / "figures"))


def data(name: str) -> str:
    """Path to a derived data file, as a string for pandas."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR / name)


def raw(*parts: str) -> Path:
    """Path under data/raw/, creating parent directories."""
    path = RAW_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure(name: str) -> str:
    """Path to a figure output, as a string for matplotlib."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return str(FIGURE_DIR / name)
