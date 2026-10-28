"""
Diagnostics of the log-log fits behind every width.

For every window the paper analyses, the hourly windows of the groups of
analysis/thin_benchmark.py and the transaction-time windows of the ladder,
each q is refitted and the fit recorded: r2, the largest residual, and the
slopes over the lower and the upper half of the scales, whose difference is
a crossover. Fits are made for q from 0.5 to 5 (DIAG_QS) whatever the range
in force; the width uses mfdfa.QS_POS.

Outputs (data/results/): scaling_windows.csv, scaling_summary.csv,
scaling.json; figures/fig8_scaling.png, the window of median width in each
group with its fitted lines.

Usage:
    uv run python -m klimafrac.analysis.scaling
"""

import numpy as np
import pandas as pd

from klimafrac import stats, trades
from klimafrac.analysis.thin_benchmark import GROUPS
from klimafrac.analysis.tiers import TOKENS as LADDER
from klimafrac.mfdfa import QS_POS, fluctuation_functions
from klimafrac.paths import PRICE, figure
from klimafrac.plotstyle import ACCENT, INK2, PAGE_WIDTH, PAPER, PAPER_DPI, plt, save, style
from klimafrac.results import numbers, table

DIAG_QS = np.arange(0.5, 5.5, 0.5)                    # the q values fitted
IN_USE = DIAG_QS <= QS_POS[-1] + 1e-9                  # those the width uses
PLOTTED = (0.5, 2.0, float(QS_POS[-1]))                # the q values drawn
SHOWN = tuple(sorted(set(PLOTTED) | {5.0}))            # those summarised in the json
POOR = 0.95                                            # an r2 below this is called a poor fit


def diagnose(x: np.ndarray) -> tuple[list[dict], dict] | None:
    """Fit quality for every q of one window, and what is needed to draw it."""
    s, f = fluctuation_functions(x, s_max=len(x) // trades.S_MAX_DIV, qs=DIAG_QS)
    if s is None:
        return None
    logs, logf = np.log(s), np.log(f)
    half = len(s) // 2
    rows, h = [], []
    for j, q in enumerate(DIAG_QS):
        slope, intercept = np.polyfit(logs, logf[:, j], 1)
        resid = logf[:, j] - (slope * logs + intercept)
        rows.append({"q": float(q), "h": slope,
                     "r2": 1 - resid.var() / logf[:, j].var(),
                     "max_resid": float(np.abs(resid).max()),
                     "h_low": np.polyfit(logs[:half], logf[:half, j], 1)[0],
                     "h_high": np.polyfit(logs[half:], logf[half:, j], 1)[0],
                     "scales": len(s)})
        h.append(slope)
    alpha = np.gradient(QS_POS * np.array(h)[IN_USE] - 1.0, QS_POS)
    return rows, {"s": s, "f": f, "width": float(alpha.max() - alpha.min())}


def series() -> list[tuple[str, str, str, np.ndarray]]:
    """(clock, group, token, returns) for every series the paper windows."""
    out = []
    for group, tokens in GROUPS.items():
        for token, key in tokens.items():
            grid = trades.calendar(trades.trades(key), freq="1h")
            out.append(("hourly", group, token, grid.ret.to_numpy()))
    for token, key in LADDER.items():
        r = trades.returns(trades.trades(key))
        out.append(("transaction", token, token, r.ret.to_numpy()))
    return out


def plot(shown: dict, path: str) -> None:
    with plt.rc_context(PAPER):
        _plot(shown, path)


def _plot(shown: dict, path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_WIDTH, 3.55), sharex="row")
    for ax, (title, d) in zip(axes.ravel(), shown.items()):
        logs = np.log10(d["s"])
        for q, colour in zip(PLOTTED, ACCENT):
            j = int(np.flatnonzero(np.isclose(DIAG_QS, q))[0])
            y = np.log10(d["f"][:, j])
            slope, intercept = np.polyfit(logs, y, 1)
            ax.plot(logs, slope * logs + intercept, color=colour, lw=0.8, zorder=2)
            ax.scatter(logs, y, s=5, color=colour, zorder=3, label=f"$q={q:g}$")
        style(ax)
        ax.set_title(title, loc="left", fontsize=7, fontweight="normal", color=INK2,
                     linespacing=1.3)
        ax.text(0.97, 0.05, rf"$\Delta\alpha^{{+}}={d['width']:.3f}$", transform=ax.transAxes,
                ha="right", fontsize=6.5, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10} F_q(s)$")
    for ax in axes[1]:
        ax.set_xlabel(r"$\log_{10} s$")
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left", handletextpad=0.1,
                      borderaxespad=0.2)
    fig.tight_layout()
    save(fig, path, dpi=PAPER_DPI)


def main() -> None:
    rows, drawings = [], {}
    for clock, group, token, x in series():
        for k, start in enumerate(range(0, len(x) - trades.WINDOW + 1, trades.WINDOW)):
            res = diagnose(x[start:start + trades.WINDOW])
            if res is None:
                continue
            fits, drawing = res
            for r in fits:
                rows.append({"clock": clock, "group": group, "token": token, "window": k,
                             "width": drawing["width"], **r})
            drawings[(clock, group, token, k)] = drawing
    win = pd.DataFrame(rows)
    win["crossover"] = win.h_low - win.h_high
    win["poor"] = win.r2 < POOR

    summary = (win.groupby(["clock", "group", "q"], sort=False)
               .agg(windows=("window", "size"), r2_median=("r2", "median"),
                    r2_p10=("r2", lambda v: v.quantile(0.10)), r2_min=("r2", "min"),
                    poor_share=("poor", "mean"), max_resid_median=("max_resid", "median"),
                    h_median=("h", "median"), h_low_median=("h_low", "median"),
                    h_high_median=("h_high", "median"),
                    crossover_median=("crossover", "median"),
                    crossover_abs_median=("crossover", lambda v: v.abs().median()))
               .reset_index())

    # the window whose width is its group's median, one per panel
    shown = {}
    one = win[win.q == SHOWN[0]]
    # the caption names the clocks, so the titles name only the group and
    # the window shown; the ladder tokens are their own group
    for (clock, group), label in ((("hourly", "carbon"), "carbon"),
                                  (("hourly", "thin"), "thin conventional"),
                                  (("hourly", "deep"), "deep conventional"),
                                  (("transaction", "BCT"), "BCT"),
                                  (("transaction", "NCT"), "NCT"),
                                  (("transaction", "MCO2"), "MCO2")):
        g = one[(one.clock == clock) & (one.group == group)]
        pick = g.iloc[(g.width - g.width.median()).abs().argsort().iloc[0]]
        name = pick.token.replace("_S", " (Sushiswap)").replace("_Q", " (Quickswap)")
        detail = (f"window {pick.window}" if clock == "transaction"
                  else f"{name}, window {pick.window}")
        shown[f"{label}\n{detail}"] = drawings[(clock, group, pick.token, pick.window)]

    out = {"scaling.price": PRICE, "scaling.poor_below_r2": POOR}
    # Does a poor fit at the top of the customary range go with a wide window?
    top = win[(win.clock == "hourly") & np.isclose(win.q, 5.0)]
    rho = top[["width", "r2"]].corr(method="spearman").iloc[0, 1]
    out["scaling.hourly.width_vs_r2_at_q5.spearman"] = float(rho)
    out["scaling.hourly.width_vs_r2_at_q5.windows"] = len(top)
    for r in summary[summary.q.isin(SHOWN)].to_dict("records"):
        pre = f"scaling.{r['clock']}.{r['group']}.q{r['q']:g}."
        for k in ("windows", "r2_median", "r2_p10", "r2_min", "poor_share", "h_low_median",
                  "h_high_median", "crossover_median", "crossover_abs_median"):
            out[pre + k] = r[k]
    # The ladder contrasts again, on the windows whose every fit, over the q
    # grid in force, is good.
    worst = win[win.q <= QS_POS[-1] + 1e-9].groupby(["clock", "group", "token", "window"],
                                                    sort=False).agg(
        width=("width", "first"), r2_min=("r2", "min")).reset_index()
    ladder = worst[worst.clock == "transaction"]
    rows = []
    for other in ("NCT", "MCO2"):
        for label, keep in (("all windows", ladder), ("every fit good", ladder[ladder.r2_min >= POOR])):
            a, b = keep[keep.token == "BCT"].width, keep[keep.token == other].width
            test = stats.permutation_test(a.to_numpy(), b.to_numpy()) if len(a) and len(b) else {}
            rows.append({"comparison": f"{other} vs BCT", "windows": label, "n_base": len(a),
                         "n_other": len(b), "median_base": a.median(), "median_other": b.median(),
                         "gap": test.get("gap", np.nan), "p": test.get("p", np.nan)})
            pre = f"scaling.ladder.{other}_vs_BCT.{label.replace(' ', '_')}."
            out[pre + "gap"], out[pre + "p"] = rows[-1]["gap"], rows[-1]["p"]
            out[pre + "windows"] = [len(a), len(b)]
    table("scaling_ladder", pd.DataFrame(rows))
    table("scaling_windows", win)
    table("scaling_summary", summary)
    print("wrote", numbers("scaling", out))
    plot(shown, figure("fig8_scaling.png"))
    print("wrote", figure("fig8_scaling.png"))
    show = summary[summary.q.isin(SHOWN)][["clock", "group", "q", "windows", "r2_median", "r2_p10",
                                           "r2_min", "poor_share", "h_low_median",
                                           "h_high_median", "crossover_median"]]
    print(show.round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
