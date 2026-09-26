"""
Where analyses put what they produce.

Tables go to data/results/<name>.csv. The numbers the paper cites go to
data/results/<module>.json under stable keys.
"""

import json
import math

import pandas as pd

from klimafrac.paths import DATA_DIR

RESULTS_DIR = DATA_DIR / "results"


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


def numbers(module: str, values: dict) -> str:
    """Write (replacing) the cited numbers of one module."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{module}.json"
    path.write_text(json.dumps(_clean(values), indent=2, sort_keys=True))
    return str(path)


def read_numbers(module: str) -> dict:
    return json.loads((RESULTS_DIR / f"{module}.json").read_text())
