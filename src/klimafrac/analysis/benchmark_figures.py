"""
The two figures of the benchmark section, drawn from the tables that
analysis/thin_benchmark.py and analysis/size.py write.

  fig6_benchmark.png   positive-side width of every non-overlapping window of
                       4000 hourly returns, one row per token, grouped as
                       carbon, thin conventional, active conventional and deep
                       conventional; token medians as bars, group medians as
                       dashed lines.
  fig9_size.png        each conventional token's median width against the
                       share of hours with no trade and against its asset's
                       market-capitalisation rank; carbon tokens shown but,
                       as in the test, not fitted.

Usage:
    uv run python -m klimafrac.analysis.benchmark_figures
"""

import numpy as np
import pandas as pd

from klimafrac.paths import figure
from klimafrac.plotstyle import (ACCENT, INK, INK2, MUTED, PAGE_WIDTH, PAPER, PAPER_DPI,
                                 SURFACE, plt, save, style)
from klimafrac.results import read_table

GROUPS = [("carbon", "carbon-linked tokens", ACCENT[0]),
          ("thin", "conventional, as thin as the carbon pools", ACCENT[1]),
          ("active", "conventional, busier than any carbon pool", ACCENT[2]),
          ("deep", "conventional, the deepest pools", INK2)]
COLOUR = {g: c for g, _, c in GROUPS}
SEED = 20260915


def plot_benchmark(win: pd.DataFrame, tokens: pd.DataFrame, path: str) -> None:
    with plt.rc_context(PAPER):
        _plot_benchmark(win, tokens, path)


def _plot_benchmark(win: pd.DataFrame, tokens: pd.DataFrame, path: str) -> None:
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 7.2))
    y, ticks, labels = 0.0, [], []
    for group, title, colour in GROUPS:
        members = tokens[tokens.group == group].sort_values("width_median", ascending=False)
        top = y
        for r in members.itertuples():
            v = win[win.token == r.token].width_pos.to_numpy()
            ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=7, color=colour, alpha=0.75,
                       edgecolor=SURFACE, linewidth=0.3, zorder=3)
            ax.plot([r.width_median, r.width_median], [y - 0.38, y + 0.38], color=INK, lw=1.1,
                    zorder=4)
            ticks.append(y)
            labels.append(r.token.replace("_S", " (Sushiswap)").replace("_Q", " (Quickswap)"))
            y += 1
        med = float(np.median(members.width_median))
        ax.plot([med, med], [top - 0.5, y - 0.5], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=2)
        ax.text(0.995, top - 0.7, f"{title}: median of token medians {med:.3f}, "
                f"{len(members)} tokens", transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", color=INK2)
        y += 1.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_ylim(-1.6, y - 1.9)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta\alpha^{+}$ per non-overlapping window of 4000 hourly returns")
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, path, dpi=PAPER_DPI)


def plot_size(t: pd.DataFrame, path: str) -> None:
    with plt.rc_context(PAPER):
        _plot_size(t, path)


def _plot_size(t: pd.DataFrame, path: str) -> None:
    conv = t[t.group != "carbon"]
    carbon = t[t.group == "carbon"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 2.85))
    for ax, x, xlabel in ((ax1, "no_trade", "share of hours with no trade"),
                          (ax2, "snapshot_rank_filled", "market-capitalisation rank\n"
                                                        "(unlisted assets at the right edge)")):
        for group, _, colour in GROUPS[1:]:
            g = conv[conv.group == group]
            ax.scatter(g[x], g.width_median, s=12, color=colour, alpha=0.85, edgecolor=SURFACE,
                       linewidth=0.4, zorder=3, label=dict((a, b) for a, b, _ in GROUPS)[group])
        ax.scatter(carbon[x], carbon.width_median, s=18, facecolor="none",
                   edgecolor=COLOUR["carbon"], linewidth=1.0, zorder=4,
                   label="carbon-linked tokens (not in the test)")
        for r in carbon.itertuples():
            ax.annotate(r.token, (getattr(r, x), r.width_median), xytext=(3, 2),
                        textcoords="offset points", fontsize=6.5, color=COLOUR["carbon"])
        style(ax, ylabel=r"median $\Delta\alpha^{+}$ of the token's windows")
        ax.set_xlabel(xlabel)
    ax2.set_xscale("log")
    for r in conv[conv.token.isin(("TEL", "LUNA", "UNI", "WPOL", "QUICK"))].itertuples():
        ax2.annotate(r.token, (r.snapshot_rank_filled, r.width_median), xytext=(3, 2),
                     textcoords="offset points", fontsize=6.5, color=MUTED)
    handles, names = ax1.get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.5, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    save(fig, path, dpi=PAPER_DPI)


def main() -> None:
    win = read_table("thin_benchmark_windows")
    tokens = read_table("thin_benchmark_tokens")
    size = read_table("size_tokens")
    plot_benchmark(win, tokens, figure("fig6_benchmark.png"))
    plot_size(size, figure("fig9_size.png"))
    print("wrote", figure("fig6_benchmark.png"), "and", figure("fig9_size.png"))


if __name__ == "__main__":
    main()
