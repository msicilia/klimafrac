"""
Reconstructed trades: loading, cleaning, the estimation sample, returns and
non-overlapping windows. Every analysis takes its trades from here.

Cleaning (paper Section 3.2), applied to each pool's full data window:
  1. drop swaps with a dollar value below MIN_USD or a base amount below
     MIN_BASE; dollar value is the quote amount for USDC-quoted pools and the
     quote amount times the KLIMA price for the two KLIMA-quoted pools
  2. drop swaps whose log price deviates from the centred rolling median of
     ROLL_WINDOW surviving swaps by more than a factor of MAX_DEVIATION

The estimation sample then keeps trades before SAMPLE_END. On 27 May 2026 at
18:14 UTC a single transaction sold a large block of pool tokens into the
remaining liquidity; the sample ends at the start of that day so that no
result depends on it.

Returns are log price changes between consecutive trades in execution order.
The first trade has no return and is dropped.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from klimafrac.mfdfa import positive_spectrum
from klimafrac.paths import data

MIN_USD = 0.10
MIN_BASE = 1e-4
ROLL_WINDOW = 251
ROLL_MIN_PERIODS = 25
MAX_DEVIATION = 3.0
SAMPLE_END = pd.Timestamp("2026-05-27", tz="UTC")   # exclusive
WINDOW = 4000
S_MAX_DIV = 10

KLIMA_QUOTED = {"polygon_UBO", "polygon_NBO"}


def load(key: str) -> pd.DataFrame:
    """Every reconstructed swap of one pool, in execution order."""
    df = pd.read_csv(data(f"swaps_{key}.csv.gz"),
                     dtype={"tx_hash": str, "base_raw": str, "quote_raw": str,
                            "side": str, "source": str})
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    return df.sort_values(["block", "log_index"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _klima_usd() -> pd.DataFrame:
    return clean(load("polygon_KLIMA"), "polygon_KLIMA")[["timestamp", "price"]]


def usd_value(df: pd.DataFrame, key: str) -> pd.Series:
    """Dollar value of each swap's quote leg."""
    if key not in KLIMA_QUOTED:
        return df.quote_amt.abs()
    matched = pd.merge_asof(df[["timestamp"]].reset_index(), _klima_usd(),
                            on="timestamp", direction="nearest")
    return pd.Series(df.quote_amt.abs().to_numpy() * matched.sort_values("index").price.to_numpy(),
                     index=df.index)


def clean(df: pd.DataFrame, key: str, min_usd: float = MIN_USD, min_base: float = MIN_BASE,
          max_deviation: float | None = MAX_DEVIATION) -> pd.DataFrame:
    """Size filters, then the rolling-median filter on log price (None disables it)."""
    keep = df.price.notna() & (df.price > 0) & (df.base_amt.abs() >= min_base)
    keep &= usd_value(df, key) >= min_usd
    d = df[keep].reset_index(drop=True)
    if max_deviation is None:
        return d
    lp = np.log(d.price)
    median = lp.rolling(ROLL_WINDOW, center=True, min_periods=ROLL_MIN_PERIODS).median()
    return d[(lp - median).abs() < np.log(max_deviation)].reset_index(drop=True)


def sample(df: pd.DataFrame, end: pd.Timestamp = SAMPLE_END) -> pd.DataFrame:
    return df[df.timestamp < end].reset_index(drop=True)


def trades(key: str, full_window: bool = False) -> pd.DataFrame:
    """Cleaned trades of one pool, restricted to the estimation sample unless asked."""
    d = clean(load(key), key)
    return d if full_window else sample(d)


def returns(df: pd.DataFrame) -> pd.DataFrame:
    """Trade-to-trade log returns, each stamped with the trade that ends it."""
    r = np.log(df.price.to_numpy())
    out = df.iloc[1:].copy()
    out["ret"] = np.diff(r)
    return out.reset_index(drop=True)


def calendar(df: pd.DataFrame, freq: str = "1h",
             end: pd.Timestamp | None = SAMPLE_END) -> pd.DataFrame:
    """
    Fixed-interval grid from the first trade's interval to the last interval
    before `end` (or to the last trade if `end` is None). Each interval holds
    the last traded price, carried forward through intervals with no trade;
    `no_trade` marks those intervals from the trades themselves, and `ret` is
    the log price change into the interval (exactly zero when no trade).
    """
    s = df.set_index("timestamp")
    last = s.price.resample(freq).last()
    count = s.price.resample(freq).count()
    if end is not None:
        grid = pd.date_range(last.index[0], end - pd.Timedelta(freq), freq=freq)
        last, count = last.reindex(grid), count.reindex(grid, fill_value=0)
    out = pd.DataFrame({"price": last.ffill(), "trades": count.astype(int)})
    out["no_trade"] = out.trades == 0
    out["ret"] = np.log(out.price).diff()
    out.index.name = "interval"
    return out.iloc[1:]


def windows(values: np.ndarray, window: int = WINDOW, s_max_div: int = S_MAX_DIV,
            stamps: pd.Series | None = None) -> pd.DataFrame:
    """Positive-q estimates on consecutive non-overlapping windows."""
    values = np.asarray(values, dtype=float)
    rows = []
    for k, start in enumerate(range(0, len(values) - window + 1, window)):
        res = positive_spectrum(values[start:start + window], s_max_div)
        if res is None:
            continue
        row = {"window": k, "start": start, **res}
        if stamps is not None:
            row["t_start"] = stamps.iloc[start]
            row["t_end"] = stamps.iloc[start + window - 1]
        rows.append(row)
    return pd.DataFrame(rows)
