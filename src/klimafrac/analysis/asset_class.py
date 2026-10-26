"""
Carbon tokens against conventional crypto: the asset-class gap in the
positive-q spectrum, what data construction contributes to it, how much the
design can resolve, and the price paths behind it: the paper's Table
tab:contrasts and the first part of its section between asset classes.
Figures fig5_prices and fig6_comparative are drawn but not in the paper.

Series. Nine hourly series in three groups:
  carbon        BCT, NCT, MCO2, KLIMA, reconstructed from their Polygon
                pools (polygon_<SYMBOL>)
  reconstructed WETH on Sushiswap (WETH_S), WETH and WBTC on Quickswap
                (WETH_Q, WBTC_Q), reconstructed by the same code path from
                the same two factories
  aggregated    BTC and ETH from the DefiLlama snapshot (fetch/benchmark.py)

Clock. Every series is on the hourly calendar grid and every window is
4,000 consecutive hourly returns (trades.WINDOW), non-overlapping, with the
positive-q width from trades.windows (s_max = 400, q in [0.5, 5]). The
estimation sample is the same for all nine: observations strictly before
trades.SAMPLE_END (2026-05-27 00:00 UTC). A reconstructed grid is
trades.calendar on trades.trades: from the hour of the pool's first cleaned
trade to the last hour before SAMPLE_END, last price carried forward, with
no_trade set for hours without a trade. The aggregated grid runs from the
first vendor instant (2021-10-08 00:00) to the same last hour; the vendor
point for instant t labels hour t. That point is a price near t rather than
the close of [t, t+1h), a one-hour labelling offset that cannot affect a
single series' width.

Vendor missing bars. The vendor returns nothing for some instants (listed in
data/vendor_manifest.json). They are forward-filled on the complete grid, so
every window is 4,000 consecutive hours as on the reconstructed side, and
each window records how many filled bars its prices touch. Differencing only
the priced bars instead would silently stitch 4,000 "hourly" returns across
holes of up to 11 hours. Every contrast is repeated with the aggregated
windows that touch a filled bar dropped, and the number dropped is reported.

Staleness restriction. A reconstructed window is kept under a restriction
when its no-trade fraction -- the share of its hours flagged no_trade from
the trades themselves -- is at most 0.30 (the tolerance derived in Section
4.1.2) or 0.05. The fraction of zero returns is not used: an hour whose
trades leave the price unchanged is not an hour without trading. An
aggregated series has no trade record, so no no-trade fraction exists for it
and it is not restricted: every aggregated window enters every restricted
row, and in contrasts B and C the restriction acts on the reconstructed side
only. Its filled-bar fraction is a vendor coverage property and is not
substituted for staleness.

Contrasts. Gap = median window width of the second side minus the first,
tested with stats.permutation_test (two-sided, exact when enumerable):
  A  carbon       - reconstructed   asset class only (the estimate)
  B  aggregated   - reconstructed   construction only, same exposures
  C  carbon       - aggregated      both (the conventional comparison)
With the same window subsets, A - B = C holds exactly for medians of the same
windows, whatever the data; it is written as an identity check, not as
evidence that the two factors decompose. The window-level test treats
windows as exchangeable although windows of one token are contiguous and
tokens are linked (KLIMA is backed by BCT); the token-level test below is
the inference that respects the design.

Composition. Per contrast and restriction, the windows each token
contributes and the token supplying most of each side.

Token level. For every contrast and restriction, an exact test over all
assignments of whole tokens, windows attached, to the two sides, keeping the
side sizes (A: C(7,4) = 35). The statistic is the same median-of-windows gap.
Reported: the observed rank among assignments, the upper-tail p (share of
assignments with gap >= observed), the two-sided p (|gap| >= |observed|), and
the minimum attainable value of each. With unequal sides an assignment's
complement is not an assignment, so the null distribution is not symmetric
and the minimum two-sided p is 1/35, not 2/35; the attainable minimum is
computed from the realised gaps, so ties raise it. Also: the gap with each
token left out, per-token medians, and whether every carbon token's median
exceeds every reconstructed conventional token's.

Size and power. For each contrast and restriction at its empirical window
counts. Each window is an independent multifractal random walk of 4,000
increments (synthetic.mrw, integral scale synthetic.INTEGRAL_SCALE, the same
for every window and group). A calibration curve maps lambda2 to the median
simulated window width; each side's lambda2 is set so its calibrated width
equals that side's empirical median width. Points:
  size       both sides at the first side's lambda2
  observed   each side at its own calibrated lambda2
  ladder     second side at the lambda2 whose calibrated width exceeds the
             first side's by each gap in GAP_LADDER
Targets beyond the calibration curve are clipped and flagged; the calibrated
gap is reported next to the target. Two variants share the same draws:
  clean      the MRW increments as generated
  no_trade   each reconstructed window (carbon or conventional) is replaced
             by carry-forward sampling under the real no_trade flags of the
             empirical window it stands for: an untraded hour returns zero
             and the next traded hour returns everything accumulated since.
             Aggregated windows have no flags and stay clean.
Under no_trade the size point measures whether the staleness difference
alone, at an equal process, produces a rejection. Rejection is p < 0.05 from
stats.permutation_test with POWER_DRAWS draws; the rejection rate's Monte
Carlo standard error is reported. The first side's windows are shared by all
ladder points of a replication (common random numbers). Every stream is
seeded from np.random.SeedSequence([SEED, cell, rep, side, point]), and each
permutation test from a distinct integer, so no two draws share a seed.
Independent windows are exactly what the window-level permutation test
assumes; the simulation therefore checks that test on its own terms and does
not address dependence between a token's windows.

Staleness bound. From data/results/staleness.json (analysis/staleness.py) at
its base cell (N = 20,000, lambda2 = 0.10): the largest deviation of the
median positive-q width from its value at p = 0, for i.i.d. inactivity and
for clustered (Markov, mean quiet runs 5 and 20), in width units, and its
ratio to the observed gap A. The range over the p grid is also taken from
staleness_curves.csv, at the base cell and at N = 5,000, the length nearest
the 4,000-hour window. If staleness.json is absent nothing is computed.

Prices (Figure fig5_prices). Daily close of day D (UTC): the last cleaned
trade price at or before the end of D for carbon tokens, the last priced
hourly vendor bar at or before D 23:00 for BTC and ETH; both carried forward
over days without an observation. Conventions:
  base      close of 18 October 2021
  peak      maximum daily close over November 2021
  trough    minimum daily close over calendar 2022
  end       last observed price strictly before SAMPLE_END
  drawdown  1 - trough / peak, and 1 - trough / base
  end value end / base - 1
A series without an observation by the end of the base day (or in November
2021) has no base (or peak) value; those tokens are listed, and their end
relative to their own first daily close is given separately. The figure
indexes series with a base to 100 at the base close and the others to 100 at
their first daily close, drawn dashed and labelled as such.

Figure fig6_comparative. Every unrestricted window of the three groups, gaps
filled, one row per token, with token and group medians, drawn from
asset_class_windows.

Outputs (via klimafrac.results):
  data/results/asset_class_windows.csv    one row per window of every series
  data/results/asset_class_contrasts.csv  A, B, C and the identity row, per
                                          restriction and vendor-gap handling,
                                          with composition
  data/results/asset_class_tokens.csv     per-token summary, leave-one-out
                                          gaps and exact token-level tests
  data/results/asset_class_power.csv      size and power per contrast,
                                          restriction, variant and point
  data/results/asset_class_prices.csv     price conventions per series
  data/results/asset_class.json           every number the paper cites
  figures/fig5_prices.png, figures/fig6_comparative.png
--quick uses 999 permutation draws, 20 replications and a short ladder,
suffixes every output _quick and writes no figures. The full run needs every
swap file; it stops naming the missing ones otherwise.

Usage:
    uv run python -m klimafrac.analysis.asset_class
    uv run python -m klimafrac.analysis.asset_class --reps 500 --draws 20000 --jobs 12
    uv run python -m klimafrac.analysis.asset_class --quick
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd

from klimafrac import stats, trades
from klimafrac.mfdfa import positive_spectrum
from klimafrac.paths import data, figure
from klimafrac.plotstyle import INK, INK2, MUTED, SERIES, SURFACE, plt, save, style
from klimafrac.results import RESULTS_DIR, numbers, read_numbers, table
from klimafrac.synthetic import INTEGRAL_SCALE, mrw

CARBON = {"BCT": "polygon_BCT", "NCT": "polygon_NCT", "MCO2": "polygon_MCO2",
          "KLIMA": "polygon_KLIMA"}
RECONSTRUCTED = {"WETH_S": "polygon_WETH_S", "WETH_Q": "polygon_WETH_Q",
                 "WBTC_Q": "polygon_WBTC_Q"}
AGGREGATED = ("BTC", "ETH")
GROUP = {**{t: "carbon" for t in CARBON}, **{t: "reconstructed" for t in RECONSTRUCTED},
         **{t: "aggregated" for t in AGGREGATED}}
VENDOR_FILE = "benchmark_hourly.csv"

RESTRICTIONS = {"all": None, "no_trade_le_0.30": 0.30, "no_trade_le_0.05": 0.05}
VENDOR_SPECS = ("gaps_filled", "gap_windows_dropped")
CONTRASTS = {
    "A": ("reconstructed", "carbon", "asset class only: carbon - reconstructed conventional"),
    "B": ("reconstructed", "aggregated", "construction only: aggregated - reconstructed, same exposures"),
    "C": ("aggregated", "carbon", "asset class and construction: carbon - aggregated conventional"),
}

DRAWS = stats.DRAWS
SEED = stats.SEED
ALPHA = 0.05
REPS = 500
POWER_DRAWS = 2000
GAP_LADDER = (0.05, 0.10, 0.15, 0.20, 0.25)
CAL_LAMBDA2 = (0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25,
               0.30, 0.40, 0.50, 0.60)
CAL_WINDOWS = 400

BASE_DAY = pd.Timestamp("2021-10-18", tz="UTC")
PEAK_MONTH = (pd.Timestamp("2021-11-01", tz="UTC"), pd.Timestamp("2021-12-01", tz="UTC"))
TROUGH_YEAR = (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"))
EPS = 1e-12


# --- data -----------------------------------------------------------------

def require() -> None:
    """Stop naming every input that has not been built."""
    missing = [f"swaps_{k}.csv.gz" for k in {**CARBON, **RECONSTRUCTED}.values()
               if not os.path.exists(data(f"swaps_{k}.csv.gz"))]
    if not os.path.exists(data(VENDOR_FILE)):
        missing.append(VENDOR_FILE)
    if missing:
        raise SystemExit(f"missing input(s) in {data('')}: {', '.join(missing)}. "
                         "Build swap files with klimafrac.fetch.swaps and the vendor "
                         "series with klimafrac.fetch.benchmark first.")


def vendor_prices() -> pd.DataFrame:
    """Hourly BTC and ETH on the complete in-sample grid, missing bars still NaN."""
    v = pd.read_csv(data(VENDOR_FILE), parse_dates=["date"]).set_index("date").sort_index()
    v.index = v.index.tz_localize("UTC") if v.index.tz is None else v.index.tz_convert("UTC")
    v = v[v.index < trades.SAMPLE_END]
    grid = pd.date_range(v.index[0], trades.SAMPLE_END - pd.Timedelta("1h"), freq="1h")
    if not v.index.isin(grid).all():
        raise ValueError(f"{VENDOR_FILE}: instants off the hourly grid")
    return v.reindex(grid)


def reconstructed_series(token: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleaned in-sample trades and their hourly grid."""
    d = trades.trades({**CARBON, **RECONSTRUCTED}[token])
    grid = trades.calendar(d, freq="1h")
    if grid.ret.isna().any():
        raise ValueError(f"{token}: hourly grid has undefined returns")
    return d, grid


def reconstructed_windows(token: str, grid: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Windows with their no-trade fraction, each window's flags, and grid facts."""
    w = trades.windows(grid.ret.to_numpy(), stamps=pd.Series(grid.index))
    flag = grid.no_trade.to_numpy()
    w["no_trade"] = [flag[s:s + trades.WINDOW].mean() for s in w.start]
    w["missing_bars"] = 0
    patterns = {int(k): flag[s:s + trades.WINDOW].copy() for k, s in zip(w.window, w.start)}
    info = {"hours": len(grid), "first_hour": grid.index[0], "no_trade_all_hours": flag.mean(),
            "windows_possible": len(grid) // trades.WINDOW, "missing_bars": 0}
    return _label(w, token), patterns, info


def vendor_windows(symbol: str, prices: pd.Series) -> tuple[pd.DataFrame, dict]:
    missing = prices.isna().to_numpy()
    if missing[0]:
        raise ValueError(f"{symbol}: first vendor bar is missing, nothing to carry forward")
    filled = prices.ffill().to_numpy()
    ret = np.diff(np.log(filled))
    w = trades.windows(ret, stamps=pd.Series(prices.index[1:]))
    w["no_trade"] = np.nan
    # A window's returns use prices start..start+WINDOW, so a filled bar at
    # either end still distorts one of its returns.
    w["missing_bars"] = [int(missing[s:s + trades.WINDOW + 1].sum()) for s in w.start]
    info = {"hours": len(ret), "first_hour": prices.index[1], "no_trade_all_hours": np.nan,
            "windows_possible": len(ret) // trades.WINDOW, "missing_bars": int(missing.sum())}
    return _label(w, symbol), info


def _label(w: pd.DataFrame, token: str) -> pd.DataFrame:
    w.insert(0, "token", token)
    w.insert(1, "group", GROUP[token])
    w.insert(2, "source", "aggregated" if token in AGGREGATED else "reconstructed")
    w["window_key"] = [f"{token}:{k}" for k in w.window]
    return w


def load_all() -> tuple[pd.DataFrame, dict, dict, dict]:
    parts, patterns, info, daily = [], {}, {}, {}
    for token in list(CARBON) + list(RECONSTRUCTED):
        d, grid = reconstructed_series(token)
        w, pat, inf = reconstructed_windows(token, grid)
        parts.append(w)
        info[token] = inf
        patterns.update({f"{token}:{k}": v for k, v in pat.items()})
        daily[token] = (d.set_index("timestamp").price, d.timestamp.iloc[0])
        print(f"  {token:7s} {inf['hours']:6,} hours from {inf['first_hour']:%Y-%m-%d %H:%M}  "
              f"{len(w)} windows  no-trade {100 * inf['no_trade_all_hours']:.1f}% of hours")
    vend = vendor_prices()
    for symbol in AGGREGATED:
        w, inf = vendor_windows(symbol, vend[symbol])
        parts.append(w)
        info[symbol] = inf
        priced = vend[symbol].dropna()
        daily[symbol] = (priced, priced.index[0])
        print(f"  {symbol:7s} {inf['hours']:6,} hours from {inf['first_hour']:%Y-%m-%d %H:%M}  "
              f"{len(w)} windows  {inf['missing_bars']} missing bars, filled; "
              f"{int((w.missing_bars > 0).sum())} windows touch one")
    win = pd.concat(parts, ignore_index=True)
    for name, thr in RESTRICTIONS.items():
        win[f"in_{name}"] = True if thr is None else (win.no_trade.isna() | (win.no_trade <= thr))
    return win, patterns, info, daily


# --- contrasts ------------------------------------------------------------

def subset(win: pd.DataFrame, restriction: str, spec: str) -> pd.DataFrame:
    s = win[win[f"in_{restriction}"]]
    if spec == "gap_windows_dropped":
        s = s[s.missing_bars == 0]
    return s


def composition(side: pd.DataFrame) -> dict:
    counts = side.token.value_counts()
    if counts.empty:
        return {"composition": "", "dominant": "", "dominant_share": np.nan}
    return {"composition": ", ".join(f"{t} {n}" for t, n in counts.items()),
            "dominant": counts.index[0], "dominant_share": counts.iloc[0] / counts.sum()}


def contrasts(win: pd.DataFrame, draws: int, out: dict) -> pd.DataFrame:
    rows = []
    for spec in VENDOR_SPECS:
        for restriction in RESTRICTIONS:
            s = subset(win, restriction, spec)
            gaps = {}
            for name, (first, second, label) in CONTRASTS.items():
                a, b = s[s.group == first], s[s.group == second]
                row = {"contrast": name, "label": label, "restriction": restriction,
                       "vendor_bars": spec, "side_a": first, "side_b": second,
                       "n_a": len(a), "n_b": len(b),
                       "median_a": a.width_pos.median(), "median_b": b.width_pos.median()}
                if len(a) and len(b):
                    res = stats.permutation_test(a.width_pos, b.width_pos, draws=draws)
                    row.update(gap=res["gap"], p=res["p"], exact=res["exact"],
                               permutations=res["draws"], mc_se=res["mc_se"])
                    gaps[name] = res["gap"]
                for side, part in (("a", a), ("b", b)):
                    row.update({f"{k}_{side}": v for k, v in composition(part).items()})
                rows.append(row)
                print(f"  {spec:19s} {restriction:17s} {name}  {len(a):3d} vs {len(b):<3d} "
                      f"gap {row.get('gap', np.nan):+.3f}  p={row.get('p', np.nan):.4f}  "
                      f"[{row['composition_b']}]")
                pre = f"asset_class.contrast.{name}.{restriction}.{spec}"
                for k in ("n_a", "n_b", "median_a", "median_b", "gap", "p", "exact", "mc_se",
                          "composition_a", "composition_b", "dominant_a", "dominant_share_a",
                          "dominant_b", "dominant_share_b"):
                    out[f"{pre}.{k}"] = row.get(k)
            if len(gaps) == 3:
                resid = gaps["A"] - gaps["B"] - gaps["C"]
                rows.append({"contrast": "identity", "restriction": restriction, "vendor_bars": spec,
                             "label": "A - B - C; zero by construction for medians of the same "
                                      "windows, an arithmetic identity and not a finding",
                             "gap": resid})
                out[f"asset_class.identity.{restriction}.{spec}.a_minus_b"] = gaps["A"] - gaps["B"]
                out[f"asset_class.identity.{restriction}.{spec}.c"] = gaps["C"]
                out[f"asset_class.identity.{restriction}.{spec}.residual"] = resid
                print(f"  {'':19s} {'':17s} identity A - B - C = {resid:+.1e} "
                      "(holds by construction; not evidence)")
    for group in ("carbon", "reconstructed", "aggregated"):
        for restriction in RESTRICTIONS:
            g = subset(win, restriction, "gaps_filled")
            g = g[g.group == group]
            out[f"asset_class.group.{group}.{restriction}.windows"] = len(g)
            out[f"asset_class.group.{group}.{restriction}.median_width_pos"] = g.width_pos.median()
    return pd.DataFrame(rows)


# --- token level ----------------------------------------------------------

def median_gap(values: dict, first: list, second: list) -> float:
    return float(np.median(np.concatenate([values[t] for t in second]))
                 - np.median(np.concatenate([values[t] for t in first])))


def token_test(values: dict, first: list, second: list) -> dict:
    """Exact test over every assignment of whole tokens keeping both side sizes."""
    pool = first + second
    observed = median_gap(values, first, second)
    gaps = []
    for chosen in combinations(pool, len(second)):
        rest = [t for t in pool if t not in chosen]
        gaps.append(median_gap(values, rest, list(chosen)))
    gaps = np.array(gaps)
    upper = np.array([np.mean(gaps >= g - EPS) for g in gaps])
    two = np.array([np.mean(np.abs(gaps) >= abs(g) - EPS) for g in gaps])
    return {"tokens_a": len(first), "tokens_b": len(second), "assignments": len(gaps),
            "gap": observed,
            "rank_upper": int((gaps > observed + EPS).sum()) + 1,
            "rank_abs": int((np.abs(gaps) > abs(observed) + EPS).sum()) + 1,
            "p_upper": float(np.mean(gaps >= observed - EPS)),
            "p_two_sided": float(np.mean(np.abs(gaps) >= abs(observed) - EPS)),
            "min_attainable_p_upper": float(upper.min()),
            "min_attainable_p_two_sided": float(two.min()),
            "assignments_abs_gap_ge_observed": int((np.abs(gaps) >= abs(observed) - EPS).sum()),
            "max_abs_gap": float(np.abs(gaps).max())}


def tokens(win: pd.DataFrame, info: dict, out: dict) -> pd.DataFrame:
    rows = []
    for token, g in win.groupby("token", sort=False):
        row = {"table": "token", "token": token, "group": GROUP[token],
               "hours": info[token]["hours"], "first_hour": info[token]["first_hour"],
               "windows_possible": info[token]["windows_possible"], "windows": len(g),
               "no_trade_all_hours": info[token]["no_trade_all_hours"],
               "no_trade_min": g.no_trade.min(), "no_trade_median": g.no_trade.median(),
               "no_trade_max": g.no_trade.max(),
               "missing_bars": info[token]["missing_bars"],
               "windows_touching_missing_bars": int((g.missing_bars > 0).sum()),
               "median_width_pos": g.width_pos.median()}
        for restriction in RESTRICTIONS:
            row[f"windows_{restriction}"] = int(g[f"in_{restriction}"].sum())
        rows.append(row)
        for k, v in row.items():
            if k not in ("table", "token"):
                out[f"asset_class.token.{token}.{k}"] = v

    med = win.groupby("token").width_pos.median()
    carbon_min, recon_max = med[list(CARBON)].idxmin(), med[list(RECONSTRUCTED)].idxmax()
    out["asset_class.separation.complete"] = bool(med[carbon_min] > med[recon_max])
    out["asset_class.separation.lowest_carbon_token"] = carbon_min
    out["asset_class.separation.lowest_carbon_median"] = med[carbon_min]
    out["asset_class.separation.highest_reconstructed_token"] = recon_max
    out["asset_class.separation.highest_reconstructed_median"] = med[recon_max]
    out["asset_class.separation.highest_carbon_token"] = med[list(CARBON)].idxmax()
    out["asset_class.separation.highest_carbon_median"] = med[list(CARBON)].max()
    out["asset_class.separation.lowest_reconstructed_token"] = med[list(RECONSTRUCTED)].idxmin()
    out["asset_class.separation.lowest_reconstructed_median"] = med[list(RECONSTRUCTED)].min()

    for name, (first_group, second_group, _) in CONTRASTS.items():
        for restriction in RESTRICTIONS:
            s = subset(win, restriction, "gaps_filled")
            values = {t: g.width_pos.to_numpy() for t, g in s.groupby("token")}
            first = [t for t in GROUP if GROUP[t] == first_group and t in values]
            second = [t for t in GROUP if GROUP[t] == second_group and t in values]
            if not first or not second:
                continue
            pre = f"asset_class.token_test.{name}.{restriction}"
            res = token_test(values, first, second)
            rows.append({"table": "token_test", "contrast": name, "restriction": restriction,
                         "tokens": ", ".join(first + second), **res})
            for k, v in res.items():
                out[f"{pre}.{k}"] = v
            loo = []
            for token in first + second:
                f, sec = [t for t in first if t != token], [t for t in second if t != token]
                if not f or not sec:
                    continue
                gap = median_gap(values, f, sec)
                loo.append(gap)
                rows.append({"table": "leave_one_out", "contrast": name, "restriction": restriction,
                             "token": token, "gap": gap})
                out[f"asset_class.leave_one_out.{name}.{restriction}.{token}"] = gap
            if loo:
                out[f"asset_class.leave_one_out.{name}.{restriction}.min"] = min(loo)
                out[f"asset_class.leave_one_out.{name}.{restriction}.max"] = max(loo)
            if restriction == "all" or name == "A":
                print(f"  {name} {restriction:17s} tokens {len(first)} vs {len(second)}: "
                      f"{res['assignments']} assignments, gap {res['gap']:+.3f}, "
                      f"rank {res['rank_upper']} (|gap| rank {res['rank_abs']}), "
                      f"p upper {res['p_upper']:.4f} (min {res['min_attainable_p_upper']:.4f}), "
                      f"two-sided {res['p_two_sided']:.4f} (min {res['min_attainable_p_two_sided']:.4f})"
                      + (f"; leave-one-out {min(loo):+.3f} to {max(loo):+.3f}" if loo else ""))
    return pd.DataFrame(rows)


# --- size and power -------------------------------------------------------

def carry_forward(x: np.ndarray, no_trade: np.ndarray) -> np.ndarray:
    """Returns observed when only traded hours reveal the accumulated price."""
    level = np.cumsum(x)
    last = np.maximum.accumulate(np.where(no_trade, -1, np.arange(len(x))))
    seen = np.where(last >= 0, level[np.maximum(last, 0)], 0.0)
    return np.diff(seen, prepend=0.0)


def _width(x: np.ndarray) -> float:
    res = positive_spectrum(x, trades.S_MAX_DIV)
    return res["width_pos"] if res else np.nan


def _simulate_side(n: int, lambda2: float, patterns: list, rng) -> tuple[np.ndarray, np.ndarray]:
    clean, stale = np.empty(n), np.empty(n)
    for i in range(n):
        x = mrw(trades.WINDOW, lambda2, rng)
        clean[i] = _width(x)
        stale[i] = clean[i] if patterns[i] is None else _width(carry_forward(x, patterns[i]))
    return clean, stale


def _calibration_point(task):
    index, lambda2, n_windows = task
    rng = np.random.default_rng([SEED, 9999, index])
    return float(np.nanmedian([_width(mrw(trades.WINDOW, lambda2, rng)) for _ in range(n_windows)]))


def _perm_seed(cell: int, point: int, variant: int, rep: int) -> int:
    return SEED + 1 + ((cell * 64 + point) * 4 + variant) * 1_000_000 + rep


def _power_chunk(task):
    cell, reps, lam_a, lam_bs, pat_a, pat_b, draws = task
    rows = []
    for rep in reps:
        a = _simulate_side(len(pat_a), lam_a, pat_a, np.random.default_rng([SEED, cell, rep, 0]))
        for point, lam_b in enumerate(lam_bs):
            b = _simulate_side(len(pat_b), lam_b, pat_b,
                               np.random.default_rng([SEED, cell, rep, 1, point]))
            for variant, (wa, wb) in enumerate(zip(a, b)):
                wa, wb = wa[np.isfinite(wa)], wb[np.isfinite(wb)]
                if not len(wa) or not len(wb):
                    continue
                res = stats.permutation_test(wa, wb, draws=draws,
                                             seed=_perm_seed(cell, point, variant, rep))
                rows.append((point, variant, res["gap"], res["p"],
                             float(np.median(wa)), float(np.median(wb))))
    return rows


def calibrate(n_windows: int, pool) -> tuple[np.ndarray, np.ndarray]:
    tasks = [(i, lam, n_windows) for i, lam in enumerate(CAL_LAMBDA2)]
    widths = list(pool.map(_calibration_point, tasks))
    return np.array(CAL_LAMBDA2), np.maximum.accumulate(np.array(widths))


def invert(lams: np.ndarray, widths: np.ndarray, target: float) -> tuple[float, bool]:
    """lambda2 whose calibrated median width is `target`, and whether it was clipped."""
    if target <= widths[0]:
        return float(lams[0]), bool(target < widths[0])
    if target >= widths[-1]:
        return float(lams[-1]), bool(target > widths[-1])
    rising = widths + np.arange(len(widths)) * 1e-12
    return float(np.interp(target, rising, lams)), False


def power(win: pd.DataFrame, patterns: dict, reps: int, draws: int, ladder, cal_windows: int,
          jobs: int, out: dict) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        lams, widths = calibrate(cal_windows, pool)
        out["asset_class.power.calibration.lambda2"] = list(lams)
        out["asset_class.power.calibration.median_width_pos"] = list(widths)
        print("  calibration lambda2 -> median width: "
              + ", ".join(f"{l:g}:{w:.3f}" for l, w in zip(lams, widths)))
        cells = []
        for name, (first, second, _) in CONTRASTS.items():
            for restriction in RESTRICTIONS:
                s = subset(win, restriction, "gaps_filled")
                a, b = s[s.group == first], s[s.group == second]
                if len(a) == 0 or len(b) == 0:
                    continue
                med_a, med_b = a.width_pos.median(), b.width_pos.median()
                lam_a, clip_a = invert(lams, widths, med_a)
                base = float(np.interp(lam_a, lams, widths))
                points = [("size", 0.0, lam_a, False)]
                lam_obs, clip_obs = invert(lams, widths, med_b)
                points.append(("observed", med_b - med_a, lam_obs, clip_obs))
                points += [("ladder", g, *invert(lams, widths, base + g)) for g in ladder]
                pat_a = [patterns.get(k) for k in a.window_key]
                pat_b = [patterns.get(k) for k in b.window_key]
                cells.append((name, restriction, len(a), len(b), med_a, med_b, lam_a, clip_a,
                              base, points, pat_a, pat_b))

        chunks = max(1, jobs * 2)
        futures = []
        for cell, (_, _, _, _, _, _, lam_a, _, _, points, pat_a, pat_b) in enumerate(cells):
            edges = np.linspace(0, reps, min(chunks, reps) + 1).astype(int)
            lam_bs = [p[2] for p in points]
            futures.append([pool.submit(_power_chunk, (cell, range(lo, hi), lam_a, lam_bs,
                                                       pat_a, pat_b, draws))
                            for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo])

        for cell, spec in enumerate(cells):
            name, restriction, n_a, n_b, med_a, med_b, lam_a, clip_a, base, points, _, _ = spec
            sims = [r for f in futures[cell] for r in f.result()]
            for point, (kind, target, lam_b, clip_b) in enumerate(points):
                for variant, vname in enumerate(("clean", "no_trade")):
                    got = [r for r in sims if r[0] == point and r[1] == variant]
                    p = np.array([r[3] for r in got])
                    rate = float(np.mean(p < ALPHA)) if len(p) else np.nan
                    row = {"contrast": name, "restriction": restriction, "variant": vname,
                           "point": kind, "target_gap": target, "n_a": n_a, "n_b": n_b,
                           "empirical_median_a": med_a, "empirical_median_b": med_b,
                           "lambda2_a": lam_a, "lambda2_b": lam_b,
                           "clipped": bool(clip_a or clip_b),
                           "calibrated_gap": float(np.interp(lam_b, lams, widths)) - base,
                           "reps": reps, "reps_used": len(got),
                           "mean_simulated_gap": np.mean([r[2] for r in got]) if got else np.nan,
                           "median_simulated_width_a": np.median([r[4] for r in got]) if got else np.nan,
                           "median_simulated_width_b": np.median([r[5] for r in got]) if got else np.nan,
                           "reject_rate": rate,
                           "mc_se": float(np.sqrt(rate * (1 - rate) / len(p))) if len(p) else np.nan}
                    rows.append(row)
                    tag = kind if kind != "ladder" else f"gap_{target:.2f}"
                    pre = f"asset_class.power.{name}.{restriction}.{vname}.{tag}"
                    for k in ("reject_rate", "mc_se", "target_gap", "calibrated_gap",
                              "mean_simulated_gap", "lambda2_a", "lambda2_b", "clipped", "reps_used"):
                        out[f"{pre}.{k}"] = row[k]
            sub = [r for r in rows if r["contrast"] == name and r["restriction"] == restriction]
            print(f"  {name} {restriction:17s} {n_a:3d} vs {n_b:<3d}  (clean / no_trade)")
            for kind, target in dict.fromkeys((r["point"], r["target_gap"]) for r in sub):
                pair = [r for r in sub if r["point"] == kind and r["target_gap"] == target]
                print(f"      {kind:8s} gap {target:+.3f}  "
                      + " / ".join(f"{r['reject_rate']:.1%}" for r in pair)
                      + ("  clipped" if pair[0]["clipped"] else ""), flush=True)
    return pd.DataFrame(rows)


# --- staleness bound ------------------------------------------------------

def staleness_bound(gap: float, out: dict) -> None:
    path = RESULTS_DIR / "staleness.json"
    if not path.exists():
        out["asset_class.staleness_bound.available"] = False
        print("  data/results/staleness.json absent: staleness bound not computed")
        return
    s = read_numbers("staleness")
    out["asset_class.staleness_bound.available"] = True
    base_width = s["bias_table_p000_width_pos"]
    pre = "asset_class.staleness_bound"
    out[f"{pre}.observed_gap_A_all"] = gap
    out[f"{pre}.base_n"] = s["design_n_base"]
    out[f"{pre}.base_lambda2"] = s["design_lambda2_base"]
    out[f"{pre}.base_width_pos_p0"] = base_width
    for clust in ("iid", "markov5", "markov20"):
        dev = s[f"base_{clust}_width_pos_inflation_max_abs_dev"] * base_width
        out[f"{pre}.{clust}.base_max_abs_dev_width_pos"] = dev
        out[f"{pre}.{clust}.base_max_abs_dev_over_gap"] = dev / gap
    out[f"{pre}.iid.table_width_pos_min"] = s["bias_table_width_pos_min"]
    out[f"{pre}.iid.table_width_pos_max"] = s["bias_table_width_pos_max"]
    out[f"{pre}.iid.table_range_over_gap"] = (s["bias_table_width_pos_max"]
                                              - s["bias_table_width_pos_min"]) / gap
    curves = RESULTS_DIR / "staleness_curves.csv"
    if curves.exists():
        c = pd.read_csv(curves)
        c = c[np.isclose(c.lambda2, s["design_lambda2_base"])]
        for n in sorted({s["design_n_base"], 5000} & set(c.n)):
            for clust, g in c[c.n == n].groupby("clustering"):
                rng_ = g.width_pos.max() - g.width_pos.min()
                out[f"{pre}.{clust}.N{n}.width_pos_min"] = g.width_pos.min()
                out[f"{pre}.{clust}.N{n}.width_pos_max"] = g.width_pos.max()
                out[f"{pre}.{clust}.N{n}.p_max"] = g.p.max()
                out[f"{pre}.{clust}.N{n}.range_over_gap"] = rng_ / gap
    for k in sorted(k for k in out if k.startswith(pre) and k.endswith("over_gap")):
        print(f"  {k.removeprefix(pre + '.')}: {out[k]:.2f}")


# --- prices ---------------------------------------------------------------

def daily_close(prices: pd.Series) -> pd.Series:
    close = prices.resample("1D").last().ffill()
    return close[close.index < trades.SAMPLE_END]


def prices_table(daily: dict, out: dict) -> tuple[pd.DataFrame, dict]:
    rows, closes = [], {}
    for token in list(CARBON) + list(AGGREGATED):
        raw, first_obs = daily[token]
        raw = raw[raw.index < trades.SAMPLE_END]
        close = daily_close(raw)
        closes[token] = close
        has_base = first_obs < BASE_DAY + pd.Timedelta("1D")
        base = close.get(BASE_DAY, np.nan) if has_base else np.nan
        nov = close[(close.index >= PEAK_MONTH[0]) & (close.index < PEAK_MONTH[1])]
        has_peak = first_obs < PEAK_MONTH[0]
        y22 = close[(close.index >= TROUGH_YEAR[0]) & (close.index < TROUGH_YEAR[1])]
        end_time, end = raw.index[-1], float(raw.iloc[-1])
        first_close = float(close.iloc[0])
        row = {"token": token, "source": "aggregated" if token in AGGREGATED else "reconstructed",
               "first_observation": first_obs, "has_base": bool(has_base),
               "base_close": base, "has_nov2021_peak": bool(has_peak),
               "nov2021_max_close": nov.max() if has_peak else np.nan,
               "nov2021_max_date": nov.idxmax() if has_peak else None,
               "trough_2022_close": y22.min(), "trough_2022_date": y22.idxmin(),
               "drawdown_from_nov2021_max_pct": 100 * (1 - y22.min() / nov.max()) if has_peak else np.nan,
               "drawdown_from_base_pct": 100 * (1 - y22.min() / base) if has_base else np.nan,
               "end_time": end_time, "end_price": end,
               "end_vs_base_pct": 100 * (end / base - 1) if has_base else np.nan,
               "first_close_date": close.index[0], "first_close": first_close,
               "end_vs_first_close_pct": 100 * (end / first_close - 1)}
        rows.append(row)
        for k, v in row.items():
            if k != "token":
                out[f"asset_class.prices.{token}.{k}"] = v
    df = pd.DataFrame(rows)
    out["asset_class.prices.lacking_base"] = df.token[~df.has_base].tolist()
    out["asset_class.prices.lacking_nov2021_peak"] = df.token[~df.has_nov2021_peak].tolist()
    print(df[["token", "has_base", "drawdown_from_nov2021_max_pct", "drawdown_from_base_pct",
              "trough_2022_date", "end_time", "end_vs_base_pct", "end_vs_first_close_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    return df, closes


# --- figures --------------------------------------------------------------

def _colour(token: str) -> str:
    return {"BTC": INK, "ETH": MUTED}.get(token, SERIES.get(token, INK2))


def plot_prices(df: pd.DataFrame, closes: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ends = []
    for _, r in df.iterrows():
        c = closes[r.token]
        if r.has_base:
            idx, label, ls = 100 * c[c.index >= BASE_DAY] / r.base_close, r.token, "-"
        else:
            idx = 100 * c / r.first_close
            label, ls = f"{r.token} (100 at {r.first_close_date:%d %b %Y})", "--"
        conv = r.token in AGGREGATED
        ax.plot(idx.index, idx.values, ls, lw=2.0 if conv else 1.4, color=_colour(r.token),
                alpha=1.0 if conv else 0.85, zorder=3 if conv else 2)
        ends.append((idx.index[-1], idx.values[-1], label, _colour(r.token)))
    ax.set_yscale("log")
    ax.axhline(100, color=MUTED, lw=1, ls=":")
    style(ax, ylabel=f"daily close, 100 at {BASE_DAY:%d %b %Y} (log)")
    lo = min(closes[t].index[0] for t in closes)
    ax.set_xlim(max(lo, BASE_DAY - pd.Timedelta("10D")), trades.SAMPLE_END + pd.Timedelta("330D"))
    fig.tight_layout()
    ends.sort(key=lambda e: e[1])
    lo_y, hi_y = (np.log10(v) for v in ax.get_ylim())
    pos = [(np.log10(y) - lo_y) / (hi_y - lo_y) for _, y, _, _ in ends]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + 0.055)
    for (x, _, text, colour), p in zip(ends, pos):
        ax.annotate(text, xy=(x, p), xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", color=colour, fontsize=8.5,
                    fontweight="bold", va="center")
    save(fig, path)


def plot_comparative(win: pd.DataFrame, path: str) -> None:
    w = subset(win, "all", "gaps_filled")
    groups = [("carbon", "carbon tokens, own pools", SERIES["BCT"],
               [("BCT", "BCT"), ("NCT", "NCT"), ("MCO2", "MCO2"), ("KLIMA", "KLIMA")]),
              ("reconstructed", "conventional, own pools", SERIES["MCO2"],
               [("WETH_S", "WETH (Sushiswap)"), ("WETH_Q", "WETH (Quickswap)"),
                ("WBTC_Q", "WBTC (Quickswap)")]),
              ("aggregated", "conventional, aggregated", MUTED,
               [("BTC", "Bitcoin"), ("ETH", "Ether")])]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    rng = np.random.default_rng(SEED)
    y, ticks, labels = 0.0, [], []
    for group, title, colour, tokens in groups:
        top = y
        for token, name in tokens:
            v = w[w.token == token].width_pos.to_numpy()
            ax.scatter(v, y + rng.uniform(-0.13, 0.13, len(v)), s=36, color=colour, alpha=0.8,
                       edgecolor=SURFACE, linewidth=1.0, zorder=3)
            med = float(np.median(v))
            ax.plot([med, med], [y - 0.30, y + 0.30], color=INK, lw=2.2, zorder=4)
            ticks.append(y)
            labels.append(f"{name}  $n={len(v)}$")
            y += 1
        v = w[w.group == group].width_pos.to_numpy()
        med = float(np.median(v))
        ax.plot([med, med], [top - 0.45, y - 0.55], color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=2)
        ax.text(0.995, top - 0.52, f"{title}: group median {med:.3f}, $n={len(v)}$",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9,
                color=INK2)
        y += 0.9
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-1.0, y - 1.3)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta\alpha^{+}$ per non-overlapping window of 4,000 hourly returns")
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, path)


# --- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--reps", type=int, default=None, help=f"power replications (default {REPS})")
    ap.add_argument("--draws", type=int, default=None,
                    help=f"permutation draws for the empirical tests (default {DRAWS})")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--quick", action="store_true",
                    help="999 draws, 20 replications, short ladder; outputs suffixed _quick, no figures")
    args = ap.parse_args()
    started = time.time()
    reps = args.reps or (20 if args.quick else REPS)
    draws = args.draws or (999 if args.quick else DRAWS)
    power_draws = 499 if args.quick else POWER_DRAWS
    ladder = (0.10, 0.20) if args.quick else GAP_LADDER
    cal_windows = 60 if args.quick else CAL_WINDOWS
    suffix = "_quick" if args.quick else ""

    require()
    out = {"asset_class.design.window": trades.WINDOW, "asset_class.design.freq": "1h",
           "asset_class.design.s_max_div": trades.S_MAX_DIV,
           "asset_class.design.sample_end_exclusive": trades.SAMPLE_END,
           "asset_class.design.restrictions": {k: v for k, v in RESTRICTIONS.items()},
           "asset_class.design.draws": draws, "asset_class.design.power_reps": reps,
           "asset_class.design.power_draws": power_draws,
           "asset_class.design.gap_ladder": list(ladder),
           "asset_class.design.integral_scale": INTEGRAL_SCALE,
           "asset_class.design.calibration_windows": cal_windows,
           "asset_class.design.seed": SEED}

    print("Hourly series")
    win, patterns, info, daily = load_all()
    print("\nContrasts (window-level permutation, windows treated as exchangeable)")
    con = contrasts(win, draws, out)
    print("\nTokens")
    tok = tokens(win, info, out)
    summary = tok[tok.table == "token"]
    print(summary[["token", "windows", "windows_no_trade_le_0.30", "windows_no_trade_le_0.05",
                   "no_trade_median", "median_width_pos", "windows_touching_missing_bars"]]
          .round(3).to_string(index=False))
    print("\nStaleness bound (ratio to the observed gap A, all windows)")
    staleness_bound(out["asset_class.contrast.A.all.gaps_filled.gap"], out)
    print("\nPrices")
    prices, closes = prices_table(daily, out)
    print(f"\nSize and power ({reps} replications, {power_draws} permutation draws, "
          f"{args.jobs} workers)")
    pw = power(win, patterns, reps, power_draws, ladder, cal_windows, args.jobs, out)

    out["asset_class.runtime_seconds"] = time.time() - started
    table("asset_class_windows" + suffix, win)
    table("asset_class_contrasts" + suffix, con)
    table("asset_class_tokens" + suffix, tok)
    table("asset_class_power" + suffix, pw)
    table("asset_class_prices" + suffix, prices)
    print("\nwrote", numbers("asset_class" + suffix, out))
    if not args.quick:
        plot_prices(prices, closes, figure("fig5_prices.png"))
        plot_comparative(win, figure("fig6_comparative.png"))
        print("wrote", figure("fig5_prices.png"), "and", figure("fig6_comparative.png"))
    print(f"runtime {out['asset_class.runtime_seconds']:.0f} s")


if __name__ == "__main__":
    main()
