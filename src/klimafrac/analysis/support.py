"""
Common support in staleness across the carbon tiers, in calendar time (paper
Section 4.1: how many 4,000-hour windows have under a quarter of hours empty).

For BCT, NCT and MCO2 the cleaned trades of the estimation sample are put on
an hourly grid (trades.calendar): each hour carries the last traded price
and a no_trade flag set when the hour contains no trade. The hourly log
returns are cut into consecutive non-overlapping windows of 4,000 hours, and
in each window we record
  no_trade    fraction of hours flagged no_trade, read from the trades
              themselves rather than inferred from zero returns, which
              would also count hours whose trades left the price unchanged
  width_pos   positive-q spectrum width (trades.windows)

Windows do not overlap, so each hour enters exactly one window, and have the
length used by the window-based tests.

Per token we report the minimum, median and maximum no-trade fraction, the
number of windows below 25% no-trade, the Pearson correlation between width
and no-trade fraction, and the same correlation with the three tokens pooled.

Outputs:
  data/results/support_windows.csv   one row per window
  data/results/support_summary.csv   one row per token
  data/results/support.json          cited numbers
--quick restricts to the tokens named with --tokens and suffixes outputs
with _quick.

Usage:
    uv run python -m klimafrac.analysis.support
    uv run python -m klimafrac.analysis.support --quick --tokens NCT MCO2
"""

import argparse

import numpy as np
import pandas as pd

from klimafrac import trades
from klimafrac.analysis.tiers import TOKENS, require
from klimafrac.results import numbers, table

FREQ = "1h"
SUPPORT_THRESHOLD = 0.25


def window_table(token: str) -> pd.DataFrame:
    """Non-overlapping 4,000-hour windows of one token with their no-trade fraction."""
    grid = trades.calendar(trades.trades(TOKENS[token]), freq=FREQ)
    if grid.ret.isna().any():
        raise ValueError(f"{token}: hourly grid has undefined returns")
    stamps = pd.Series(grid.index)
    w = trades.windows(grid.ret.to_numpy(), stamps=stamps)
    flag = grid.no_trade.to_numpy()
    w.insert(0, "token", token)
    w["no_trade"] = [flag[s:s + trades.WINDOW].mean() for s in w.start]
    w.attrs["hours"] = len(grid)
    w.attrs["no_trade_all"] = float(flag.mean())
    return w


def summarise(win: pd.DataFrame, out: dict) -> pd.DataFrame:
    rows = []
    for token, g in win.groupby("token", sort=False):
        corr = g.width_pos.corr(g.no_trade) if len(g) > 2 else np.nan
        rows.append({"token": token, "windows": len(g),
                     "no_trade_min": g.no_trade.min(),
                     "no_trade_median": g.no_trade.median(),
                     "no_trade_max": g.no_trade.max(),
                     "below_threshold": int((g.no_trade < SUPPORT_THRESHOLD).sum()),
                     "corr_width_no_trade": corr,
                     "width_median": g.width_pos.median()})
        pre = f"support.{token}."
        for k, v in rows[-1].items():
            if k != "token":
                out[pre + k] = v
    out["support.pooled.windows"] = len(win)
    out["support.pooled.corr_width_no_trade"] = win.width_pos.corr(win.no_trade)
    out["support.threshold"] = SUPPORT_THRESHOLD
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokens", nargs="+", default=list(TOKENS), choices=list(TOKENS))
    ap.add_argument("--quick", action="store_true", help="outputs suffixed _quick")
    args = ap.parse_args()

    tokens = args.tokens if args.quick else list(TOKENS)
    require(*tokens)
    out = {"support.window": trades.WINDOW, "support.freq": FREQ}
    parts = []
    for token in tokens:
        w = window_table(token)
        out[f"support.{token}.hours"] = w.attrs["hours"]
        out[f"support.{token}.no_trade_all_hours"] = w.attrs["no_trade_all"]
        parts.append(w)
        print(f"{token:5s} {w.attrs['hours']:6,} hours  {len(w)} windows  "
              f"no-trade {100 * w.attrs['no_trade_all']:.1f}% of hours")
    win = pd.concat(parts, ignore_index=True)
    summary = summarise(win, out)
    print(summary.round(3).to_string(index=False))
    print(f"pooled correlation {out['support.pooled.corr_width_no_trade']:+.3f}")

    suffix = "_quick" if args.quick else ""
    table("support_windows" + suffix, win)
    table("support_summary" + suffix, summary)
    print("wrote", numbers("support" + suffix, out))


if __name__ == "__main__":
    main()
