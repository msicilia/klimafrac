"""
Transaction-time comparison of the carbon pool tiers: NCT against BCT and
MCO2 against BCT (paper Section 4.2.1, "Within the quality ladder", with its
Robustness paragraph and Table tab:robust).

Main comparison. Each token's cleaned trades in the estimation sample
(trades.trades) give trade-to-trade log returns (trades.returns, first trade
dropped). These are cut into consecutive non-overlapping windows of 4,000
returns and the positive-q width is estimated in each (trades.windows). A
comparison is the difference of window medians, other minus BCT, tested with
stats.permutation_test. BCT on Sushiswap is the base; NCT trades on the same
Sushiswap venue, MCO2 on Quickswap, so only NCT vs BCT holds venue fixed.

What a window covers. A window's trades are the 4,000 trades that end its
returns, and its calendar span runs from the first of those to the last.
For each window we record
  span_days        calendar span
  usd_total        dollar value traded (sum of trades.usd_value)
  usd_median       median trade size in dollars
  intensity        trades per day, 4,000 / span_days
and summarise them per token: median span and dollar value, how many
windows begin in 2021, and the overlap in trade intensity, i.e. how many
windows of each group lie inside the intensity range the two groups share.

Calendar matching. A window is kept if its own [t_start, t_end] intersects
at least one window of the other group, and the test is rerun on the kept
windows. Matching against individual windows rather than against the other
group's overall range matters: NCT's windows run from 2022 to 2026, so
nearly every BCT window lies inside NCT's overall range while few of them
coincide with an NCT window.

Conditioning. Window width is regressed on NCT and MCO2 dummies (BCT the
omitted category), log trade intensity (natural log of trades per day) and
log median trade size (natural log of dollars), windows of the three tokens
pooled, with HC3 standard errors. Non-overlapping windows need no
autocorrelation correction; HC3 guards against the unequal group sizes.

Robustness (Table tab:robust). Each specification reruns both comparisons:
  estimation sample   the specification above
  full data window    cleaned trades without the SAMPLE_END restriction
  isolated trades     only swaps that are alone in their block, with returns
                      recomputed on the price path of the isolated trades.
                      Block membership is read from every swap of the pool
                      before cleaning, because whether a block held another
                      swap is a fact about the chain, not about the filter.
                      The share of cleaned trades removed is reported.
  fill first return   the undefined first return set to zero instead of
                      dropped, which shifts every window boundary by one
  cleaning grid       MIN_USD in {0, 0.10, 1, 10}, MIN_BASE in {0, 1e-4},
                      MAX_DEVIATION in {2, 3, 5, off}; 32 combinations, each
                      applied to the full data window before the estimation
                      sample is taken, exactly as trades.clean is. The range
                      of BCT's median window width across the grid is the
                      instability of the level that the paper reports.
Every specification, the main one included, is cleaned by trades.clean.

Outputs (via klimafrac.results):
  data/results/tiers_windows.csv       every window of every specification
  data/results/tiers_comparisons.csv   gap, p and window counts per test
  data/results/tiers_support.csv       per-token window coverage summary
  data/results/tiers_regression.csv    conditioning regression
  data/results/tiers_cleaning.csv      the cleaning grid
  data/results/tiers.json              every number the paper cites
--quick uses 999 permutation draws and three grid points and writes the
same files with a _quick suffix, so a test run never replaces paper outputs.

Usage:
    uv run python -m klimafrac.analysis.tiers
    uv run python -m klimafrac.analysis.tiers --draws 20000 --no-grid
    uv run python -m klimafrac.analysis.tiers --quick
"""

import argparse
import os
from itertools import product

import numpy as np
import pandas as pd
import statsmodels.api as sm

from klimafrac import stats, trades
from klimafrac.paths import data
from klimafrac.results import numbers, table

TOKENS = {"BCT": "polygon_BCT", "NCT": "polygon_NCT", "MCO2": "polygon_MCO2"}
BASE = "BCT"
OTHERS = ("NCT", "MCO2")

GRID_MIN_USD = (0.0, 0.10, 1.0, 10.0)
GRID_MIN_BASE = (0.0, 1e-4)
GRID_MAX_DEVIATION = (2.0, 3.0, 5.0, None)       # None: no rolling-median filter
DEFAULT_THRESHOLDS = (trades.MIN_USD, trades.MIN_BASE, trades.MAX_DEVIATION)
QUICK_GRID = [DEFAULT_THRESHOLDS, (1.0, 1e-4, 3.0), (0.10, 1e-4, None)]


def require(*tokens: str) -> None:
    """Stop with a clear message if a pool's swap file has not been built."""
    missing = [TOKENS.get(t, t) for t in tokens
               if not os.path.exists(data(f"swaps_{TOKENS.get(t, t)}.csv.gz"))]
    if missing:
        files = ", ".join(f"swaps_{k}.csv.gz" for k in missing)
        raise SystemExit(f"missing swap file(s) in {data('')}: {files}. "
                         "Build them with klimafrac.fetch.swaps first.")


clean = trades.clean


def isolated(cleaned: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaned trades whose block contains no other swap of the same pool."""
    per_block = raw.groupby("block").size()
    return cleaned[cleaned.block.map(per_block).eq(1)].reset_index(drop=True)


def window_table(df: pd.DataFrame, token: str, fill_first: bool = False) -> pd.DataFrame:
    """Non-overlapping 4,000-return windows of one token, with their coverage."""
    if fill_first:
        r = df.copy()
        r["ret"] = np.concatenate([[0.0], np.diff(np.log(df.price.to_numpy()))])
    else:
        r = trades.returns(df)
    w = trades.windows(r.ret.to_numpy(), stamps=r.timestamp)
    if w.empty:
        return w
    usd = trades.usd_value(r, TOKENS[token]).to_numpy()
    size = trades.WINDOW
    w.insert(0, "token", token)
    w["span_days"] = [max((e - s).total_seconds(), 1.0) / 86400
                      for s, e in zip(w.t_start, w.t_end)]
    w["usd_total"] = [usd[s:s + size].sum() for s in w.start]
    w["usd_median"] = [np.median(usd[s:s + size]) for s in w.start]
    w["intensity"] = size / w.span_days
    return w


def compare(win: pd.DataFrame, draws: int, spec: str) -> list[dict]:
    """Both comparisons against BCT on one set of windows."""
    base = win.loc[win.token == BASE, "width_pos"].to_numpy()
    rows = []
    for other in OTHERS:
        alt = win.loc[win.token == other, "width_pos"].to_numpy()
        row = {"spec": spec, "comparison": f"{other} vs {BASE}",
               "n_base": len(base), "n_other": len(alt),
               "median_base": np.median(base) if len(base) else np.nan,
               "median_other": np.median(alt) if len(alt) else np.nan}
        if len(base) and len(alt):
            res = stats.permutation_test(base, alt, draws=draws)
            row.update(gap=res["gap"], p=res["p"], exact=res["exact"],
                       mc_se=res["mc_se"])
        rows.append(row)
    return rows


def overlapping(win: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """Mask of `win` rows whose calendar span intersects any row of `other`."""
    s, e = win.t_start.to_numpy()[:, None], win.t_end.to_numpy()[:, None]
    return ((s <= other.t_end.to_numpy()[None, :])
            & (e >= other.t_start.to_numpy()[None, :])).any(axis=1)


def support(win: pd.DataFrame, draws: int, out: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Coverage summaries, intensity overlap and the calendar-matched tests."""
    rows = []
    for token, g in win.groupby("token", sort=False):
        rows.append({"token": token, "windows": len(g),
                     "span_days_median": g.span_days.median(),
                     "usd_total_median": g.usd_total.median(),
                     "usd_median_trade_median": g.usd_median.median(),
                     "intensity_min": g.intensity.min(),
                     "intensity_median": g.intensity.median(),
                     "intensity_max": g.intensity.max(),
                     "windows_start_2021": int((g.t_start.dt.year == 2021).sum()),
                     "width_median": g.width_pos.median()})
        pre = f"tiers.{token}."
        out[pre + "windows"] = len(g)
        out[pre + "span_days_median"] = rows[-1]["span_days_median"]
        out[pre + "usd_per_window_median"] = rows[-1]["usd_total_median"]
        out[pre + "windows_start_2021"] = rows[-1]["windows_start_2021"]
        out[pre + "trades_per_day_median"] = rows[-1]["intensity_median"]
        out[pre + "width_median"] = rows[-1]["width_median"]

    base = win[win.token == BASE]
    matched = []
    for other in OTHERS:
        alt = win[win.token == other]
        pre = f"tiers.{other}_vs_{BASE}."
        lo = max(base.intensity.min(), alt.intensity.min())
        hi = min(base.intensity.max(), alt.intensity.max())
        out[pre + "intensity_shared_range"] = [lo, hi] if lo <= hi else None
        out[pre + "intensity_shared_base_windows"] = int(base.intensity.between(lo, hi).sum())
        out[pre + "intensity_shared_other_windows"] = int(alt.intensity.between(lo, hi).sum())

        b, a = base[overlapping(base, alt)], alt[overlapping(alt, base)]
        spec = "calendar-matched"
        row = {"spec": spec, "comparison": f"{other} vs {BASE}",
               "n_base": len(b), "n_other": len(a),
               "median_base": b.width_pos.median(), "median_other": a.width_pos.median()}
        if len(b) and len(a):
            res = stats.permutation_test(b.width_pos, a.width_pos, draws=draws)
            row.update(gap=res["gap"], p=res["p"], exact=res["exact"], mc_se=res["mc_se"])
        matched.append(row)
        out[pre + "matched.n_base"] = len(b)
        out[pre + "matched.n_other"] = len(a)
        out[pre + "matched.gap"] = row.get("gap")
        out[pre + "matched.p"] = row.get("p")
    return pd.DataFrame(rows), matched


def regression(win: pd.DataFrame, out: dict) -> pd.DataFrame:
    """Window width on asset dummies, log intensity and log median trade size, HC3."""
    X = pd.DataFrame({"const": 1.0,
                      **{t: (win.token == t).astype(float) for t in OTHERS},
                      "log_intensity": np.log(win.intensity),
                      "log_median_usd": np.log(win.usd_median)})
    keep = np.isfinite(X).all(axis=1).to_numpy()
    fit = sm.OLS(win.width_pos.to_numpy()[keep], X[keep]).fit(cov_type="HC3")
    res = pd.DataFrame({"term": fit.params.index, "coef": fit.params.to_numpy(),
                        "se_hc3": fit.bse.to_numpy(), "p": fit.pvalues.to_numpy()})
    res["n"] = int(fit.nobs)
    res["r2"] = fit.rsquared
    for _, r in res.iterrows():
        out[f"tiers.regression.{r.term}.coef"] = r.coef
        out[f"tiers.regression.{r.term}.p"] = r.p
    out["tiers.regression.n"] = int(fit.nobs)
    return res


def record(rows: list[dict], out: dict, key: str) -> None:
    for r in rows:
        pre = f"{key}.{r['comparison'].replace(' vs ', '_vs_')}."
        for field in ("gap", "p", "n_base", "n_other"):
            out[pre + field] = r.get(field)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draws", type=int, default=stats.DRAWS,
                    help="permutation draws when enumeration is not possible")
    ap.add_argument("--no-grid", action="store_true", help="skip the cleaning grid")
    ap.add_argument("--quick", action="store_true",
                    help="999 draws, three grid points, outputs suffixed _quick")
    args = ap.parse_args()
    draws = 999 if args.quick else args.draws
    suffix = "_quick" if args.quick else ""
    require(*TOKENS)

    out = {"tiers.window": trades.WINDOW, "tiers.draws": draws}
    raw, est, full = {}, {}, {}
    for t, key in TOKENS.items():
        raw[t] = trades.load(key)
        full[t] = trades.clean(raw[t], key)
        est[t] = trades.sample(full[t])
        r = trades.returns(est[t]).ret
        out[f"tiers.{t}.trades"] = len(est[t])
        out[f"tiers.{t}.zero_return_share"] = float((r == 0).mean())
        print(f"{t:5s} {len(raw[t]):8,} swaps  {len(full[t]):8,} cleaned  "
              f"{len(est[t]):8,} in estimation sample")

    comparisons, all_windows = [], []

    def run(spec, frames, fill_first=False):
        win = pd.concat([window_table(frames[t], t, fill_first) for t in TOKENS],
                        ignore_index=True)
        rows = compare(win, draws, spec)
        comparisons.extend(rows)
        all_windows.append(win.assign(spec=spec))
        for r in rows:
            print(f"  {spec:20s} {r['comparison']:12s} gap {r.get('gap', np.nan):+.3f}  "
                  f"p={r.get('p', np.nan):.3f}  {r['n_other']} vs {r['n_base']} windows")
        return win, rows

    win, rows = run("estimation sample", est)
    record(rows, out, "tiers.main")

    summary, matched = support(win, draws, out)
    comparisons.extend(matched)
    for r in matched:
        print(f"  {'calendar-matched':20s} {r['comparison']:12s} gap {r.get('gap', np.nan):+.3f}  "
              f"p={r.get('p', np.nan):.3f}  {r['n_other']} vs {r['n_base']} windows")
    reg = regression(win, out)
    print(reg.round(4).to_string(index=False))

    _, rows = run("full data window", full)
    record(rows, out, "tiers.robust.full_window")

    iso = {t: isolated(est[t], raw[t]) for t in TOKENS}
    for t in TOKENS:
        out[f"tiers.robust.isolated.removed_share.{t}"] = 1 - len(iso[t]) / len(est[t])
    _, rows = run("isolated trades only", iso)
    record(rows, out, "tiers.robust.isolated")

    _, rows = run("fill first return", est, fill_first=True)
    record(rows, out, "tiers.robust.fill_first")

    specs = [r for r in comparisons if r["spec"] != "calendar-matched"]
    for other in OTHERS:
        gaps = [r["gap"] for r in specs if r["comparison"].startswith(other) and "gap" in r]
        out[f"tiers.robust.{other}_vs_{BASE}.gap_range"] = [min(gaps), max(gaps)]
    out["tiers.robust.tests"] = sum("p" in r for r in specs)
    out["tiers.robust.significant_5pct"] = sum(r.get("p", 1) < 0.05 for r in specs)

    if not args.no_grid:
        grid = QUICK_GRID if args.quick else list(product(GRID_MIN_USD, GRID_MIN_BASE,
                                                          GRID_MAX_DEVIATION))
        rows_grid = []
        for min_usd, min_base, max_dev in grid:
            frames = {t: trades.sample(clean(raw[t], TOKENS[t], min_usd, min_base, max_dev))
                      for t in TOKENS}
            win_g = pd.concat([window_table(frames[t], t) for t in TOKENS], ignore_index=True)
            label = f"min_usd={min_usd:g} min_base={min_base:g} max_dev={max_dev or 'off'}"
            for r in compare(win_g, draws, label):
                r.update(min_usd=min_usd, min_base=min_base,
                         max_deviation=max_dev if max_dev is not None else np.nan,
                         default=(min_usd, min_base, max_dev) == DEFAULT_THRESHOLDS,
                         **{f"trades_{t}": len(frames[t]) for t in TOKENS})
                rows_grid.append(r)
                print(f"  {label:42s} {r['comparison']:12s} gap {r.get('gap', np.nan):+.3f}  "
                      f"p={r.get('p', np.nan):.3f}  BCT median {r['median_base']:.3f}")
        grid_df = pd.DataFrame(rows_grid)
        table("tiers_cleaning" + suffix, grid_df)
        out["tiers.cleaning.combinations"] = len(grid)
        out["tiers.cleaning.BCT_width_median_range"] = [grid_df.median_base.min(),
                                                        grid_df.median_base.max()]
        for other in OTHERS:
            g = grid_df[grid_df.comparison == f"{other} vs {BASE}"]
            out[f"tiers.cleaning.{other}_vs_{BASE}.gap_range"] = [g.gap.min(), g.gap.max()]
            out[f"tiers.cleaning.{other}_vs_{BASE}.p_min"] = g.p.min()
            out[f"tiers.cleaning.{other}_vs_{BASE}.significant_5pct"] = int((g.p < 0.05).sum())
            off = g[(g.min_usd == trades.MIN_USD) & (g.min_base == trades.MIN_BASE)
                    & (g.max_deviation.isna())]
            if len(off) == 1:
                out[f"tiers.cleaning.{other}_vs_{BASE}.gap_no_deviation_filter"] = \
                    float(off.gap.iloc[0])

        # What the deviation filter actually removes, inside the estimation sample:
        # the count, and how many of those fall after the pool's first trading day.
        raw_base = trades.load(TOKENS[BASE])
        sized = trades.clean(raw_base, TOKENS[BASE], max_deviation=None)
        lp = np.log(sized.price)
        med = lp.rolling(trades.ROLL_WINDOW, center=True,
                         min_periods=trades.ROLL_MIN_PERIODS).median()
        dropped = trades.sample(sized[(lp - med).abs() >= np.log(trades.MAX_DEVIATION)])
        first_day = trades.sample(sized).timestamp.min().date()
        out[f"tiers.cleaning.deviation_removed_{BASE}"] = len(dropped)
        out[f"tiers.cleaning.deviation_removed_{BASE}_after_first_day"] = int(
            (dropped.timestamp.dt.date != first_day).sum())

    table("tiers_windows" + suffix, pd.concat(all_windows, ignore_index=True))
    table("tiers_comparisons" + suffix, pd.DataFrame(comparisons))
    table("tiers_support" + suffix, summary)
    table("tiers_regression" + suffix, reg)
    print("wrote", numbers("tiers" + suffix, out))


if __name__ == "__main__":
    main()
