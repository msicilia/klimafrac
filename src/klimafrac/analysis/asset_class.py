"""
Carbon tokens against conventional crypto: the asset-class gap in the
positive-q spectrum, what data construction contributes to it and how much
the design can resolve (paper Section 4.3, Table 4 and Figure 1).

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
the priced bars instead would stitch 4,000 "hourly" returns across holes.
Every contrast is repeated with the aggregated
windows that touch a filled bar dropped, and the number dropped is reported.

Staleness restriction. A reconstructed window is kept under a restriction
when its no-trade fraction -- the share of its hours flagged no_trade from
the trades themselves -- is at most 0.30 (the tolerance derived in Section
3.4) or 0.05. The fraction of zero returns is not used: an hour whose
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

Figure 1 of the paper (fig6_comparative.png). Every unrestricted window of the three groups, gaps
filled, with group medians, drawn from asset_class_windows.

Outputs (via klimafrac.results):
  data/results/asset_class_windows.csv    one row per window of every series
  data/results/asset_class_contrasts.csv  A, B, C and the identity row, per
                                          restriction and vendor-gap handling,
                                          with composition
  data/results/asset_class_tokens.csv     per-token summary, leave-one-out
                                          gaps and exact token-level tests
  data/results/asset_class_power.csv      size and power per contrast,
                                          restriction, variant and point
  data/results/asset_class.json           cited numbers
  figures/fig6_comparative.png
--quick uses 999 permutation draws, 20 replications and a short ladder,
suffixes every output _quick and writes no figure. The full run needs every
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
from klimafrac.plotstyle import INK, MUTED, SERIES, SURFACE, plt, save, style
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


def reconstructed_series(token: str) -> pd.DataFrame:
    """Hourly grid of the cleaned in-sample trades."""
    grid = trades.calendar(trades.trades({**CARBON, **RECONSTRUCTED}[token]), freq="1h")
    if grid.ret.isna().any():
        raise ValueError(f"{token}: hourly grid has undefined returns")
    return grid


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


def load_all() -> tuple[pd.DataFrame, dict, dict]:
    parts, patterns, info = [], {}, {}
    for token in list(CARBON) + list(RECONSTRUCTED):
        grid = reconstructed_series(token)
        w, pat, inf = reconstructed_windows(token, grid)
        parts.append(w)
        info[token] = inf
        patterns.update({f"{token}:{k}": v for k, v in pat.items()})
        print(f"  {token:7s} {inf['hours']:6,} hours from {inf['first_hour']:%Y-%m-%d %H:%M}  "
              f"{len(w)} windows  no-trade {100 * inf['no_trade_all_hours']:.1f}% of hours")
    vend = vendor_prices()
    for symbol in AGGREGATED:
        w, inf = vendor_windows(symbol, vend[symbol])
        parts.append(w)
        info[symbol] = inf
        print(f"  {symbol:7s} {inf['hours']:6,} hours from {inf['first_hour']:%Y-%m-%d %H:%M}  "
              f"{len(w)} windows  {inf['missing_bars']} missing bars, filled; "
              f"{int((w.missing_bars > 0).sum())} windows touch one")
    win = pd.concat(parts, ignore_index=True)
    for name, thr in RESTRICTIONS.items():
        win[f"in_{name}"] = True if thr is None else (win.no_trade.isna() | (win.no_trade <= thr))
    return win, patterns, info


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


# --- figure ---------------------------------------------------------------

def plot_comparative(win: pd.DataFrame, path: str) -> None:
    w = subset(win, "all", "gaps_filled")
    groups = [("carbon", "carbon tokens\nown pools", SERIES["BCT"]),
              ("reconstructed", "conventional\nown pools", SERIES["MCO2"]),
              ("aggregated", "conventional\naggregated", MUTED)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    rng = np.random.default_rng(SEED)
    labels = []
    for i, (group, label, colour) in enumerate(groups):
        v = w[w.group == group].width_pos.to_numpy()
        ax.scatter(v, i + rng.uniform(-0.16, 0.16, len(v)), s=42, color=colour, alpha=0.75,
                   edgecolor=SURFACE, linewidth=1.0, zorder=3)
        med = float(np.median(v))
        ax.plot([med, med], [i - 0.30, i + 0.30], color=INK, lw=2.4, zorder=4)
        ax.text(med, i + 0.42, f"{med:.3f}", ha="center", fontsize=9, color=INK, fontweight="bold")
        labels.append(f"{label}\n$n={len(v)}$")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-0.6, len(groups) - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta\alpha^{+}$ per non-overlapping window of 4,000 hourly returns")
    style(ax)
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
    win, patterns, info = load_all()
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
    print(f"\nSize and power ({reps} replications, {power_draws} permutation draws, "
          f"{args.jobs} workers)")
    pw = power(win, patterns, reps, power_draws, ladder, cal_windows, args.jobs, out)

    out["asset_class.runtime_seconds"] = time.time() - started
    table("asset_class_windows" + suffix, win)
    table("asset_class_contrasts" + suffix, con)
    table("asset_class_tokens" + suffix, tok)
    table("asset_class_power" + suffix, pw)
    print("\nwrote", numbers("asset_class" + suffix, out))
    if not args.quick:
        plot_comparative(win, figure("fig6_comparative.png"))
        print("wrote", figure("fig6_comparative.png"))
    print(f"runtime {out['asset_class.runtime_seconds']:.0f} s")


if __name__ == "__main__":
    main()
