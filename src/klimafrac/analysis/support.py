"""
Common support in staleness across the carbon tiers, in calendar time: the
paper's paragraph on why the ladder is compared in transaction time. Figure
fig4_support is drawn but not in the paper.

For BCT, NCT and MCO2 the cleaned trades of the estimation sample are put on
an hourly grid (trades.calendar): each hour carries the last traded price
and a no_trade flag set when the hour contains no trade. The hourly log
returns are cut into consecutive non-overlapping windows of 4,000 hours, and
in each window we record
  no_trade    fraction of hours flagged no_trade, read from the trades
              themselves rather than inferred from zero returns, which
              would also count hours whose trades left the price unchanged
  width_pos   positive-q spectrum width (trades.windows)

Windows do not overlap, as in every test in the paper, so each hour enters
exactly one window. The diagnostic uses the window length of the inference
it qualifies, which keeps it about the windows the comparisons would use;
the cost is a small number of windows per token, which the counts report.

Per token we report the minimum, median and maximum no-trade fraction, the
number of windows below 25% no-trade, the Pearson correlation between width
and no-trade fraction, and the same correlation with the three tokens pooled.

Outputs:
  data/results/support_windows.csv   one row per window
  data/results/support_summary.csv   one row per token
  data/results/support.json          cited numbers
  figures/fig4_support.png           width against no-trade fraction
--quick restricts to the tokens named with --tokens and suffixes outputs
with _quick; the figure is written only by a full run.

Usage:
    uv run python -m klimafrac.analysis.support
    uv run python -m klimafrac.analysis.support --figure-only
    uv run python -m klimafrac.analysis.support --quick --tokens NCT MCO2
"""

import argparse

import numpy as np
import pandas as pd

from klimafrac import trades
from klimafrac.analysis.tiers import TOKENS, require
from klimafrac.paths import figure
from klimafrac.plotstyle import MUTED, SERIES, SURFACE, plt, save, style
from klimafrac.results import numbers, read_table, table

FREQ = "1h"
SUPPORT_THRESHOLD = 0.25
COLORS = {t: SERIES[t] for t in TOKENS}


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


def plot(win: pd.DataFrame, path: str) -> None:
    """Width against no-trade fraction, one point per window, by token."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.axvspan(0, 100 * SUPPORT_THRESHOLD, color=MUTED, alpha=0.08, lw=0, zorder=0)
    for token in COLORS:
        g = win[win.token == token]
        if g.empty:
            continue
        corr = g.width_pos.corr(g.no_trade) if len(g) > 2 else np.nan
        ax.scatter(100 * g.no_trade, g.width_pos, s=46, color=COLORS[token],
                   edgecolor=SURFACE, linewidth=1.2, alpha=0.9, zorder=3,
                   label=f"{token}  ({len(g)} windows, r = {corr:+.2f})")
    ax.set_xlim(0, 100)
    ax.set_xlabel("hours with no trade in the window (%)")
    style(ax, ylabel=r"positive-$q$ width $\Delta\alpha^{+}$")
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokens", nargs="+", default=list(TOKENS), choices=list(TOKENS))
    ap.add_argument("--quick", action="store_true", help="outputs suffixed _quick, no figure")
    ap.add_argument("--figure-only", action="store_true",
                    help="redraw the figure from data/results/support_windows.csv")
    args = ap.parse_args()

    if args.figure_only:
        win = read_table("support_windows")
        plot(win, figure("fig4_support.png"))
        print("wrote", figure("fig4_support.png"))
        return

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
    if not args.quick:
        plot(win, figure("fig4_support.png"))
        print("wrote", figure("fig4_support.png"))


if __name__ == "__main__":
    main()
