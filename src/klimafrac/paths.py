"""Canonical file locations, so every module works from any working directory.

`KLIMAFRAC_DATA_DIR` and `KLIMAFRAC_FIGURE_DIR` override the defaults, which
lets a test run write somewhere other than the real dataset.

`KLIMAFRAC_PRICE` chooses what a swap's price is, for every analysis at once:
  pool       the pool's reserve ratio just after the swap (fetch/pool_price.py),
             which carries neither the fee nor the rounding of small amounts;
             the default, and the paper's
  execution  quote amount over base amount, what the trader paid; the
             submitted version of the paper used it

`KLIMAFRAC_QMAX` sets the largest moment of the positive-side width, q running
from 0.5 to it in steps of 0.5. The default, 3, is the paper's; the submitted
version used 5.

The paper's configuration writes to data/results and figures/. Any other
combination gets its own folders, named by both switches, so that results of
different configurations never mix: data/results_execution_q5 and
figures_execution_q5 hold the submitted version's, data/results_pool_q5 and
data/results_execution_q3 the two intermediate ones.
"""

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]

PRICE = os.environ.get("KLIMAFRAC_PRICE", "pool")
if PRICE not in ("execution", "pool"):
    raise SystemExit(f"KLIMAFRAC_PRICE must be 'execution' or 'pool', not {PRICE!r}")

Q_MAX = float(os.environ.get("KLIMAFRAC_QMAX", "3"))
if Q_MAX < 1 or (2 * Q_MAX) % 1:
    raise SystemExit(f"KLIMAFRAC_QMAX must be a multiple of 0.5, at least 1, not {Q_MAX:g}")

CONFIG_SUFFIX = "" if (PRICE, Q_MAX) == ("pool", 3.0) else f"_{PRICE}_q{Q_MAX:g}"

DATA_DIR = Path(os.environ.get("KLIMAFRAC_DATA_DIR", CODE_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = DATA_DIR / f"results{CONFIG_SUFFIX}"
FIGURE_DIR = Path(os.environ.get("KLIMAFRAC_FIGURE_DIR", CODE_ROOT / f"figures{CONFIG_SUFFIX}"))


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
