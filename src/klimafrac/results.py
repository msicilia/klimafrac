"""
Where analyses put what they produce.

Tables go to data/results/<name>.csv. Every number the paper cites goes to
data/results/<module>.json under a stable key (data/results_pool under
KLIMAFRAC_PRICE=pool, see paths.py), so verify_paper.py can compare
the manuscript with the pipeline's output rather than with hand-copied values.
"""

import json
import math

import pandas as pd

from klimafrac.paths import RESULTS_DIR  # noqa: F401  (re-exported)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def table(name: str, df: pd.DataFrame) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def read_table(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"{name}.csv", **kwargs)


def numbers(module: str, values: dict) -> str:
    """Write (replacing) the cited numbers of one module."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{module}.json"
    path.write_text(json.dumps(_clean(values), indent=2, sort_keys=True))
    return str(path)


def read_numbers(module: str) -> dict:
    return json.loads((RESULTS_DIR / f"{module}.json").read_text())
