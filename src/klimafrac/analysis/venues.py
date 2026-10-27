"""
One asset on two venues: BCT on Sushiswap (Polygon) against Aerodrome
(Base) over their overlap, and WETH on Sushiswap against Quickswap. Not
used in the paper.

Reports the round-trip cost from price reversals between consecutive trades
and the fee it implies; the agreement of daily prices across the two BCT
venues; venue and asset gaps in the positive-side width on windows of 2,000
(BCT) and 4,000 (WETH) returns, raw and prewhitened by a per-window AR(1);
the WETH gap by year and on the hourly clock; and the size and power of the
permutation test by smoothed bootstrap within groups and by simulation on
multifractal random walks.

Outputs (data/results/): venues_roundtrip.csv, venues_bct_windows.csv,
venues_weth_windows.csv, venues_weth_year.csv, venues_comparisons.csv,
venues_weth_activity.csv, venues_power.csv, venues.json. --quick and
--sections suffix the outputs _quick and _partial.

Usage:
    uv run python -m klimafrac.analysis.venues [--sections roundtrip bct] [--quick]
"""

import argparse
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from klimafrac import stats, trades
from klimafrac.mfdfa import positive_spectrum
from klimafrac.paths import data
from klimafrac.results import numbers, table
from klimafrac.synthetic import mrw

# MCO2 takes part in the round trip but in none of the venue comparisons: the
# case for holding the venue fixed across the quality ladder rests on all three
# tiers trading at the same effective cost, so that cost has to be measured for
# MCO2 as well.
POOLS = {"BCT_polygon": "polygon_BCT", "BCT_base": "base_BCT", "NCT": "polygon_NCT",
         "MCO2": "polygon_MCO2",
         "WETH_sushiswap": "polygon_WETH_S", "WETH_quickswap": "polygon_WETH_Q"}
VENUE = {"polygon_BCT": "Sushiswap", "polygon_NCT": "Sushiswap", "base_BCT": "Aerodrome",
         "polygon_MCO2": "Quickswap",
         "polygon_WETH_S": "Sushiswap", "polygon_WETH_Q": "Quickswap"}
DECLARED_FEE_PCT = {"polygon_BCT": 0.30, "polygon_NCT": 0.30, "polygon_MCO2": 0.30,
                    "polygon_WETH_S": 0.30, "polygon_WETH_Q": 0.30}
SECTIONS = ("roundtrip", "bct", "weth", "power")
NEEDS = {"roundtrip": tuple(POOLS),
         "bct": ("BCT_polygon", "BCT_base", "NCT"),
         "weth": ("WETH_sushiswap", "WETH_quickswap"),
         "power": ("BCT_polygon", "BCT_base", "NCT",
                   "WETH_sushiswap", "WETH_quickswap")}

BCT_WINDOW = 2000
WETH_WINDOW = trades.WINDOW
HOURLY_WINDOW = 4000

SIZE_FLOORS_USD = (trades.MIN_USD, 10.0, 100.0, 1_000.0, 10_000.0)
SIZE_CEILINGS_USD = (10.0, 100.0, 1_000.0)

POWER_REPS = 500
POWER_DRAWS = 2000
MRW_REPS = 200
CALIBRATION_WINDOWS = 200
ALPHA = 0.05
POWER_TARGET = 0.80
FIXED_SHIFTS = (0.0, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.30)
GAP_MULTIPLES = (0.5, 1.0, 2.0)
MRW_LAMBDA2 = (0.0, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)


# --- data -----------------------------------------------------------------

def require(names) -> None:
    """Stop with a clear message if a pool's swap file has not been built."""
    missing = [POOLS[n] for n in names if not os.path.exists(data(f"swaps_{POOLS[n]}.csv.gz"))]
    if missing:
        files = ", ".join(f"swaps_{k}.csv.gz" for k in sorted(set(missing)))
        raise SystemExit(f"missing swap file(s) in {data('')}: {files}. "
                         "Build them with klimafrac.fetch.swaps first.")


def between(df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    return df[(df.timestamp >= lo) & (df.timestamp <= hi)].reset_index(drop=True)


# --- round-trip cost ------------------------------------------------------

def trade_pairs(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Consecutive trades: return, reversal flag and the two trades' dollar sizes."""
    usd = trades.usd_value(df, key).to_numpy()
    side = df.side.to_numpy()
    return pd.DataFrame({"timestamp": df.timestamp.to_numpy()[1:],
                         "ret": np.diff(np.log(df.price.to_numpy())),
                         "reversal": side[1:] != side[:-1],
                         "usd_min": np.minimum(usd[1:], usd[:-1]),
                         "usd_max": np.maximum(usd[1:], usd[:-1])})


def round_trip(pairs: pd.DataFrame) -> dict:
    rev = pairs.ret[pairs.reversal].abs()
    same = pairs.ret[~pairs.reversal].abs()
    rt = float(rev.median()) if len(rev) else np.nan
    return {"n_reversal": len(rev), "n_same_side": len(same),
            "round_trip_pct": 100 * rt,
            "same_side_pct": 100 * float(same.median()) if len(same) else np.nan,
            "implied_fee_pct": 100 * (1 - np.exp(-rt / 2))}


def round_trip_rows(df: pd.DataFrame, key: str, sample: str, quarterly: bool) -> list[dict]:
    pairs = trade_pairs(df, key)
    base = {"pool": key, "venue": VENUE[key], "sample": sample,
            "declared_fee_pct": DECLARED_FEE_PCT.get(key, np.nan)}
    rules = [(f"both >= ${f:g}", pairs.usd_min >= f) for f in SIZE_FLOORS_USD]
    rules += [(f"both < ${c:g}", pairs.usd_max < c) for c in SIZE_CEILINGS_USD]
    rows = [{**base, "size_rule": rule, "period": "all", **round_trip(pairs[mask])}
            for rule, mask in rules]
    periods = [("year", pairs.timestamp.dt.year.astype(str))]
    if quarterly:
        periods.append(("quarter", pairs.timestamp.dt.tz_localize(None).dt.to_period("Q").astype(str)))
    for _, label in periods:
        for period, g in pairs.groupby(label):
            rows.append({**base, "size_rule": rules[0][0], "period": period, **round_trip(g)})
    return rows


# --- windows --------------------------------------------------------------

def lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    den = float(np.dot(x, x))
    return float(np.dot(x[1:], x[:-1]) / den) if den > 0 else 0.0


def window_table(ret: np.ndarray, stamps: pd.Series, window: int, group: str) -> pd.DataFrame:
    """Raw and prewhitened positive-q estimates on non-overlapping windows."""
    ret = np.asarray(ret, float)
    raw = trades.windows(ret, window=window, stamps=stamps)
    if raw.empty:
        return raw
    phi_all = lag1(ret)
    whole = ret.copy()
    whole[1:] = ret[1:] - phi_all * ret[:-1]
    glob = trades.windows(whole, window=window)[["window", "width_pos"]]
    pw = []
    for k, start in zip(raw.window, raw.start):
        seg = ret[start:start + window]
        prev = ret[start - 1:start + window - 1] if start > 0 else np.r_[0.0, seg[:-1]]
        phi = lag1(seg)
        res = positive_spectrum(seg - phi * prev, trades.S_MAX_DIV)
        pw.append({"window": k, "phi": phi,
                   "width_pos_pw": res["width_pos"] if res else np.nan})
    out = (raw.merge(pd.DataFrame(pw), on="window", how="left")
              .merge(glob.rename(columns={"width_pos": "width_pos_pw_global"}),
                     on="window", how="left"))
    out.insert(0, "group", group)
    out["window_len"] = window
    out["phi_global"] = phi_all
    out["span_days"] = (out.t_end - out.t_start).dt.total_seconds() / 86400
    out["t_mid"] = out.t_start + (out.t_end - out.t_start) / 2
    out["year"] = out.t_mid.dt.year
    return out


def trade_windows(df: pd.DataFrame, window: int, group: str) -> pd.DataFrame:
    r = trades.returns(df)
    return window_table(r.ret.to_numpy(), r.timestamp, window, group)


def hourly_windows(df: pd.DataFrame, group: str) -> tuple[pd.DataFrame, float]:
    cal = trades.calendar(df, "1h")
    w = window_table(cal.ret.to_numpy(), pd.Series(cal.index), HOURLY_WINDOW, group)
    return w, float(cal.no_trade.mean())


COLUMNS = {"raw": "width_pos", "prewhitened": "width_pos_pw",
           "prewhitened (full-series AR(1))": "width_pos_pw_global"}


def compare(a: pd.DataFrame, b: pd.DataFrame, design: str, test: str, spec: str,
            draws: int) -> dict:
    col = COLUMNS[spec]
    x, y = a[col].dropna().to_numpy(), b[col].dropna().to_numpy()
    row = {"design": design, "comparison": test, "spec": spec,
           "group_a": a.group.iloc[0], "group_b": b.group.iloc[0],
           "n_a": len(x), "n_b": len(y),
           "median_a": np.median(x) if len(x) else np.nan,
           "median_b": np.median(y) if len(y) else np.nan}
    if len(x) and len(y):
        res = stats.permutation_test(x, y, draws=draws)
        row.update(gap=res["gap"], p=res["p"], exact=res["exact"], mc_se=res["mc_se"])
    print(f"  {design:5s} {test:24s} {spec:32s} {row['n_a']:4d} vs {row['n_b']:<4d} "
          f"gap {row.get('gap', np.nan):+.3f}  p={row.get('p', np.nan):.3f}")
    return row


def record(out: dict, prefix: str, row: dict) -> None:
    for field in ("n_a", "n_b", "median_a", "median_b", "gap", "p"):
        out[f"{prefix}.{field}"] = row.get(field)


def dispersion(w: pd.Series) -> dict:
    q1, q3 = w.quantile([0.25, 0.75])
    return {"min": w.min(), "q1": q1, "median": w.median(), "q3": q3,
            "max": w.max(), "iqr": q3 - q1}


# --- activity -------------------------------------------------------------

def activity(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Swaps per hour of covered calendar time in each month, ISO week and day."""
    first, end = df.timestamp.iloc[0], trades.SAMPLE_END
    naive = df.timestamp.dt.tz_localize(None)
    rows = []
    for unit, code in (("month", "M"), ("week", "W"), ("day", "D")):
        period = naive.dt.to_period(code)
        counts = period.value_counts()
        for p in pd.period_range(period.iloc[0], (end - pd.Timedelta("1s")).tz_localize(None), freq=code):
            lo = max(p.start_time.tz_localize("UTC"), first)
            hi = min((p.end_time + pd.Timedelta("1ns")).tz_localize("UTC"), end)
            hours = (hi - lo).total_seconds() / 3600
            full = (p.end_time + pd.Timedelta("1ns") - p.start_time).total_seconds() / 3600
            rows.append({"group": group, "unit": unit, "period": str(p),
                         "swaps": int(counts.get(p, 0)), "hours": hours,
                         "complete": bool(np.isclose(hours, full)),
                         "swaps_per_hour": counts.get(p, 0) / hours if hours > 0 else np.nan})
    return pd.DataFrame(rows)


# --- power ----------------------------------------------------------------

def tag(label: str) -> int:
    return zlib.crc32(label.encode())


def _resample(strata, rng) -> np.ndarray:
    return np.concatenate([rng.choice(v, size=n, replace=True) + rng.normal(0.0, h, n)
                           for v, n, h in strata])


def _bootstrap_chunk(task):
    strata_a, strata_b, shift, reps, draws, key = task
    gaps, ps = [], []
    for rep in reps:
        rng = np.random.default_rng([stats.SEED, key, rep])
        a = _resample(strata_a, rng)
        b = _resample(strata_b, rng) + shift
        res = stats.permutation_test(a, b, draws=draws, seed=stats.SEED + rep)
        gaps.append(res["gap"])
        ps.append((res["p"], res["p"] - rng.uniform() / (res["draws"] + 1)))
    return gaps, ps


def _mrw_widths(n_windows: int, window: int, lambda2: float, rng) -> np.ndarray:
    return trades.windows(mrw(n_windows * window, lambda2, rng), window=window).width_pos.to_numpy()


def _mrw_chunk(task):
    n_a, n_b, window, lam_a, lam_b, reps, draws, key = task
    gaps, ps = [], []
    for rep in reps:
        rng = np.random.default_rng([stats.SEED, key, rep])
        wa, wb = _mrw_widths(n_a, window, lam_a, rng), _mrw_widths(n_b, window, lam_b, rng)
        if len(wa) == 0 or len(wb) == 0:
            continue
        res = stats.permutation_test(wa, wb, draws=draws, seed=stats.SEED + rep)
        gaps.append(res["gap"])
        ps.append((res["p"], res["p"] - rng.uniform() / (res["draws"] + 1)))
    return gaps, ps


def _calibration_point(task):
    window, lambda2, n_windows = task
    rng = np.random.default_rng([stats.SEED, tag(f"calibrate {window}"), int(lambda2 * 1e6)])
    return float(np.median(_mrw_widths(n_windows, window, lambda2, rng)))


_POOL: ProcessPoolExecutor | None = None


def run_chunks(fn, tasks, jobs: int):
    """Map over worker processes; one pool serves the whole run, since starting one costs seconds."""
    global _POOL
    if jobs <= 1:
        return [fn(t) for t in tasks]
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=jobs)
    return list(_POOL.map(fn, tasks))


def split(reps: int, parts: int) -> list[range]:
    edges = np.linspace(0, reps, min(parts, reps) + 1).astype(int)
    return [range(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]


def smoothed(values: np.ndarray, count: int) -> tuple[np.ndarray, int, float]:
    """One bootstrap stratum: centred values, number of draws, Gaussian bandwidth.

    The values are shifted so that the median of the jittered distribution,
    not of the values themselves, is zero; with few values the two differ.
    """
    c = values - np.median(values)
    if len(c) < 2:
        return c, count, 0.0
    q1, q3 = np.percentile(c, [25, 75])
    sd = c.std(ddof=1)
    h = 0.9 * (min(sd, (q3 - q1) / 1.34) if q3 > q1 else sd) * len(c) ** -0.2
    rng = np.random.default_rng([stats.SEED, len(c)])
    centre = np.median(rng.choice(c, 200_000) + rng.normal(0.0, h, 200_000))
    return c - centre, count, float(h)


def strata(w: pd.DataFrame, scale: float = 1.0, by_year: bool = False) -> list:
    groups = w.groupby("year") if by_year else [(None, w)]
    return [smoothed(g.width_pos.to_numpy(), max(1, int(round(scale * len(g))))) for _, g in groups]


def critical_p(null_ps) -> float:
    """Largest p threshold whose rejection rate at zero shift is at most ALPHA."""
    null = np.asarray(null_ps)
    ok = [v for v in np.unique(null) if np.mean(null <= v) <= ALPHA]
    return float(ok[-1]) if ok else -1.0


def summarise(rows: list[dict], ps: list[list[tuple[float, float]]]) -> list[dict]:
    """
    Nominal power from the p-values as the test reports them, and power at
    the critical p that holds size to ALPHA. The critical p is found on
    p-values randomised within their resolution, 1 / (draws + 1): a Monte
    Carlo p cannot fall below that floor, so replications that tie there
    would otherwise leave no threshold with size at most ALPHA.
    """
    null = [q for _, q in next(p for r, p in zip(rows, ps) if r["shift"] == 0)]
    # Fewer than 1 / ALPHA null replications cannot place a 5% threshold.
    crit = critical_p(null) if len(null) >= 1 / ALPHA else np.nan
    for r, p in zip(rows, ps):
        nominal = np.array([x for x, _ in p])
        randomised = np.array([q for _, q in p])
        adjusted = float(np.mean(randomised <= crit)) if len(p) and np.isfinite(crit) else np.nan
        r.update(power=float(np.mean(nominal < ALPHA)) if len(p) else np.nan,
                 critical_p=crit, power_size_adjusted=adjusted)
    return rows


def bootstrap_power(strata_a, strata_b, shifts, reps, draws, label, jobs) -> list[dict]:
    rows, ps = [], []
    for i, shift in enumerate(shifts):
        key = tag(f"{label} {i}")
        tasks = [(strata_a, strata_b, shift, r, draws, key) for r in split(reps, jobs)]
        res = run_chunks(_bootstrap_chunk, tasks, jobs)
        gaps = [g for chunk, _ in res for g in chunk]
        ps.append([p for _, chunk in res for p in chunk])
        rows.append({"shift": shift, "true_gap": shift, "mean_gap": np.mean(gaps),
                     "reps": len(gaps), "n_a": sum(s[1] for s in strata_a),
                     "n_b": sum(s[1] for s in strata_b)})
    return summarise(rows, ps)


def calibrate(window: int, n_windows: int, jobs: int) -> tuple[np.ndarray, np.ndarray]:
    widths = run_chunks(_calibration_point, [(window, lam, n_windows) for lam in MRW_LAMBDA2], jobs)
    return np.array(MRW_LAMBDA2), np.maximum.accumulate(np.array(widths))


def invert(lams: np.ndarray, widths: np.ndarray, target: float) -> tuple[float, bool]:
    """lambda2 whose calibrated median width is `target`, and whether it was clipped."""
    if target <= widths[0]:
        return float(lams[0]), target < widths[0]
    if target >= widths[-1]:
        return float(lams[-1]), target > widths[-1]
    rising = widths + np.arange(len(widths)) * 1e-12
    return float(np.interp(target, rising, lams)), False


def mrw_power(n_a, n_b, window, median_a, shifts, reps, draws, calib, label, jobs) -> list[dict]:
    lams, widths = calib
    lam_a, clip_a = invert(lams, widths, median_a)
    base_width = float(np.interp(lam_a, lams, widths))
    rows, ps = [], []
    for i, shift in enumerate(shifts):
        lam_b, clip_b = invert(lams, widths, base_width + shift)
        key = tag(f"{label} {i}")
        tasks = [(n_a, n_b, window, lam_a, lam_b, r, draws, key) for r in split(reps, jobs)]
        res = run_chunks(_mrw_chunk, tasks, jobs)
        gaps = [g for chunk, _ in res for g in chunk]
        ps.append([p for _, chunk in res for p in chunk])
        rows.append({"shift": shift, "true_gap": float(np.interp(lam_b, lams, widths)) - base_width,
                     "mean_gap": np.mean(gaps) if gaps else np.nan,
                     "reps": len(gaps), "n_a": n_a, "n_b": n_b,
                     "lambda2_a": lam_a, "lambda2_b": lam_b, "clipped": clip_a or clip_b})
    return summarise(rows, ps)


def mde(rows: list[dict], field: str = "power", target: float = POWER_TARGET) -> float | None:
    """Smallest true gap at which linearly interpolated power reaches `target`."""
    pts = sorted((r["true_gap"], r[field]) for r in rows if r["shift"] > 0)
    prev = (0.0, next((r[field] for r in rows if r["shift"] == 0), 0.0))
    for gap, power in pts:
        if power >= target:
            if power == prev[1]:
                return gap
            return prev[0] + (target - prev[1]) * (gap - prev[0]) / (power - prev[1])
        prev = (gap, power)
    return None


def power_at(rows: list[dict], shift: float, field: str = "power") -> float | None:
    return next((r[field] for r in rows if np.isclose(r["shift"], shift)), None)


# --- sections -------------------------------------------------------------

def section_roundtrip(frames, overlap, out) -> pd.DataFrame:
    print("\nRound-trip cost")
    rows = []
    lo, hi = overlap
    for name in ("BCT_polygon", "BCT_base", "NCT"):
        key = POOLS[name]
        rows += round_trip_rows(between(frames[name], lo, hi), key, "BCT venue overlap",
                                quarterly=name.startswith("BCT"))
    for name in POOLS:
        rows += round_trip_rows(frames[name], POOLS[name], "estimation sample", quarterly=False)
    rt = pd.DataFrame(rows)

    for sample, slug in (("BCT venue overlap", "overlap"), ("estimation sample", "sample")):
        for name, key in POOLS.items():
            g = rt[(rt["sample"] == sample) & (rt.pool == key) & (rt.period == "all")]
            if g.empty:
                continue
            main = g.iloc[0]
            pre = f"venues.roundtrip.{slug}.{name}"
            out[pre + ".round_trip_pct"] = main.round_trip_pct
            out[pre + ".same_side_pct"] = main.same_side_pct
            out[pre + ".implied_fee_pct"] = main.implied_fee_pct
            out[pre + ".n_reversal"] = main.n_reversal
            floors = g[g.size_rule.str.startswith("both >=")]
            out[pre + ".round_trip_pct_range_over_size_floors"] = [floors.round_trip_pct.min(),
                                                                   floors.round_trip_pct.max()]
            out[pre + ".round_trip_pct_range_over_size_rules"] = [g.round_trip_pct.min(),
                                                                  g.round_trip_pct.max()]
            print(f"  {sample:18s} {name:15s} round trip {main.round_trip_pct:.3f}%  "
                  f"same-side {main.same_side_pct:.3f}%  implied fee {main.implied_fee_pct:.3f}%  "
                  f"size rules {g.round_trip_pct.min():.3f}-{g.round_trip_pct.max():.3f}%")
    q = rt[(rt["sample"] == "BCT venue overlap") & (rt.pool == "base_BCT") & rt.period.str.contains("Q")]
    out["venues.roundtrip.overlap.BCT_base.quarterly_implied_fee_pct_range"] = \
        [q.implied_fee_pct.min(), q.implied_fee_pct.max()] if len(q) else None
    return rt


def section_same_asset(poly, base, overlap, out) -> None:
    a = poly.set_index("timestamp").price.resample("1D").median()
    b = base.set_index("timestamp").price.resample("1D").median()
    j = pd.concat([a.rename("polygon"), b.rename("base")], axis=1).dropna()
    gap = np.log(j.polygon / j.base)
    consecutive = j.index.to_series().diff().eq(pd.Timedelta("1D"))
    changes = np.log(j).diff()[consecutive]
    out.update({"venues.overlap.start": overlap[0], "venues.overlap.end": overlap[1],
                "venues.overlap.calendar_days": (overlap[1] - overlap[0]).total_seconds() / 86400,
                "venues.same_asset.common_days": len(j),
                "venues.same_asset.price_corr": j.polygon.corr(j.base),
                "venues.same_asset.daily_change_corr": changes.polygon.corr(changes.base),
                "venues.same_asset.daily_change_days": len(changes),
                "venues.same_asset.median_abs_log_gap_pct": 100 * gap.abs().median(),
                "venues.same_asset.median_log_gap_pct": 100 * gap.median()})
    print(f"\nSame asset: {len(j)} common days, price corr {j.polygon.corr(j.base):.4f}, "
          f"daily change corr {changes.polygon.corr(changes.base):.3f}, "
          f"median |gap| {100 * gap.abs().median():.2f}%")


def section_bct(frames, overlap, draws, out):
    print(f"\nBCT: venue and asset, {BCT_WINDOW:,}-return windows")
    lo, hi = overlap
    poly_ov, base_ov = between(frames["BCT_polygon"], lo, hi), between(frames["BCT_base"], lo, hi)
    nct_ov = between(frames["NCT"], lo, hi)
    out["venues.bct.window"] = BCT_WINDOW
    for venue, df in (("polygon", poly_ov), ("base", base_ov)):
        out[f"venues.bct.overlap_trades.{venue}"] = len(df)
        out[f"venues.bct.{venue}_windows_at_4000"] = max(len(df) - 1, 0) // 4000

    groups = {"BCT Polygon (overlap)": poly_ov, "BCT Base (overlap)": base_ov,
              "BCT Polygon (sample)": frames["BCT_polygon"], "NCT Polygon (sample)": frames["NCT"],
              "NCT Polygon (overlap)": nct_ov}
    w = {g: trade_windows(df, BCT_WINDOW, g) for g, df in groups.items()}
    rows = []
    for spec in COLUMNS:
        slug = {"raw": "raw", "prewhitened": "prewhitened",
                "prewhitened (full-series AR(1))": "prewhitened_global"}[spec]
        r = compare(w["BCT Polygon (overlap)"], w["BCT Base (overlap)"], "BCT", "venue (Base - Polygon)",
                    spec, draws)
        rows.append(r)
        record(out, f"venues.bct.venue.{slug}", r)
        r = compare(w["BCT Polygon (sample)"], w["NCT Polygon (sample)"], "BCT", "asset (NCT - BCT)",
                    spec, draws)
        rows.append(r)
        record(out, f"venues.bct.asset.{slug}", r)
    if not w["NCT Polygon (overlap)"].empty:
        r = compare(w["BCT Polygon (overlap)"], w["NCT Polygon (overlap)"], "BCT",
                    "asset over overlap", "raw", draws)
        rows.append(r)
        record(out, "venues.bct.asset_overlap.raw", r)
    venue, asset = out.get("venues.bct.venue.raw.gap"), out.get("venues.bct.asset.raw.gap")
    out["venues.bct.venue_to_asset_ratio"] = abs(venue / asset) if venue is not None and asset else None
    for name, g in w.items():
        if not g.empty:
            slug = name.replace(" (", "_").replace(")", "").replace(" ", "_")
            out[f"venues.bct.phi_median.{slug}"] = g.phi.median()
    windows = pd.concat([g for g in w.values() if not g.empty], ignore_index=True)
    return windows, rows, w


def section_weth(frames, draws, out):
    print(f"\nWETH: Sushiswap and Quickswap, {WETH_WINDOW:,}-return windows")
    names = ("WETH_sushiswap", "WETH_quickswap")
    for n in names:
        out[f"venues.weth.{n}.trades"] = len(frames[n])
        out[f"venues.weth.{n}.first"] = frames[n].timestamp.iloc[0]
        out[f"venues.weth.{n}.last"] = frames[n].timestamp.iloc[-1]
        print(f"  {n:15s} {len(frames[n]):,} cleaned trades in the estimation sample")
    w = {n: trade_windows(frames[n], WETH_WINDOW, n) for n in names}
    rows = []
    for spec in COLUMNS:
        slug = {"raw": "raw", "prewhitened": "prewhitened",
                "prewhitened (full-series AR(1))": "prewhitened_global"}[spec]
        r = compare(w[names[0]], w[names[1]], "WETH", "venue (Quick - Sushi)", spec, draws)
        rows.append(r)
        record(out, f"venues.weth.venue.{slug}", r)
    for n in names:
        for stat, v in dispersion(w[n].width_pos).items():
            out[f"venues.weth.{n}.width_{stat}"] = v
        out[f"venues.weth.{n}.phi_median"] = w[n].phi.median()

    years = []
    for year in sorted(set(w[names[0]].year) | set(w[names[1]].year)):
        a, b = (w[n][w[n].year == year] for n in names)
        row = {"year": year, "n_sushiswap": len(a), "n_quickswap": len(b)}
        for spec, suffix in (("raw", ""), ("prewhitened", "_pw")):
            col = COLUMNS[spec]
            row[f"median_sushiswap{suffix}"] = a[col].median()
            row[f"median_quickswap{suffix}"] = b[col].median()
            if len(a[col].dropna()) and len(b[col].dropna()):
                res = stats.permutation_test(a[col].dropna(), b[col].dropna(), draws=draws)
                row.update({f"gap{suffix}": res["gap"], f"p{suffix}": res["p"],
                            f"exact{suffix}": res["exact"]})
        years.append(row)
        for field in ("n_sushiswap", "n_quickswap", "median_sushiswap", "median_quickswap",
                      "gap", "p", "gap_pw", "p_pw"):
            out[f"venues.weth.year.{year}.{field}"] = row.get(field)
        print(f"  {year}  {len(a):4d} vs {len(b):<4d} {row['median_sushiswap']:.3f} "
              f"{row['median_quickswap']:.3f}  gap {row.get('gap', np.nan):+.3f} "
              f"(p={row.get('p', np.nan):.3f})  prewhitened {row.get('gap_pw', np.nan):+.3f} "
              f"(p={row.get('p_pw', np.nan):.3f})")

    hourly = {}
    for n in names:
        hourly[n], no_trade = hourly_windows(frames[n], n + " hourly")
        out[f"venues.weth.hourly.{n}.no_trade_share"] = no_trade
    r = compare(hourly[names[0]], hourly[names[1]], "WETH", "venue, hourly clock", "raw", draws)
    rows.append(r)
    record(out, "venues.weth.hourly.raw", r)
    r = compare(hourly[names[0]], hourly[names[1]], "WETH", "venue, hourly clock", "prewhitened", draws)
    rows.append(r)
    record(out, "venues.weth.hourly.prewhitened", r)

    act = pd.concat([activity(frames[n], n) for n in names], ignore_index=True)
    for n in names:
        for unit in ("month", "week", "day"):
            g = act[(act.group == n) & (act.unit == unit)]
            full = g[g.complete]
            low = full.loc[full.swaps_per_hour.idxmin()] if len(full) else None
            out[f"venues.weth.activity.{n}.min_swaps_per_hour_{unit}"] = \
                low.swaps_per_hour if low is not None else None
            out[f"venues.weth.activity.{n}.min_{unit}"] = low.period if low is not None else None
            out[f"venues.weth.activity.{n}.min_swaps_per_hour_{unit}_incl_partial"] = g.swaps_per_hour.min()
        gaps = frames[n].timestamp.diff().dt.total_seconds() / 3600
        out[f"venues.weth.activity.{n}.longest_gap_hours"] = gaps.max()
        out[f"venues.weth.activity.{n}.longest_gap_ends"] = frames[n].timestamp.iloc[int(gaps.idxmax())]
        print(f"  {n:15s} min swaps/hour: month "
              f"{out[f'venues.weth.activity.{n}.min_swaps_per_hour_month']:.2f} "
              f"({out[f'venues.weth.activity.{n}.min_month']}), week "
              f"{out[f'venues.weth.activity.{n}.min_swaps_per_hour_week']:.2f}; "
              f"longest gap {gaps.max():.1f} h")

    windows = pd.concat([w[names[0]].assign(clock="trade"), w[names[1]].assign(clock="trade"),
                         hourly[names[0]].assign(clock="hour"), hourly[names[1]].assign(clock="hour")],
                        ignore_index=True)
    return windows, rows, pd.DataFrame(years), act, w


def section_power(bct_w, weth_w, bct_gap, args, out) -> pd.DataFrame:
    shifts = sorted({round(s, 4) for s in FIXED_SHIFTS} |
                    {round(m * abs(bct_gap), 4) for m in GAP_MULTIPLES})
    at_gap = round(abs(bct_gap), 4)
    out["venues.power.shifts"] = shifts
    out["venues.power.bct_gap_shift"] = at_gap
    out["venues.power.reps"] = args.power_reps
    out["venues.power.draws"] = args.power_draws
    out["venues.power.mrw_reps"] = args.mrw_reps
    bct_a, bct_b = bct_w["BCT Polygon (overlap)"], bct_w["BCT Base (overlap)"]
    weth_a, weth_b = weth_w["WETH_sushiswap"], weth_w["WETH_quickswap"]
    n_bct, n_weth = (len(bct_a), len(bct_b)), (len(weth_a), len(weth_b))

    designs = [
        ("BCT", "bootstrap, within group", "BCT", n_bct, strata(bct_a), strata(bct_b)),
        ("WETH", "bootstrap, within group", "WETH", n_weth, strata(weth_a), strata(weth_b)),
        ("WETH", "bootstrap, within group x year", "WETH", n_weth,
         strata(weth_a, by_year=True), strata(weth_b, by_year=True)),
        ("WETH dispersion at BCT counts", "bootstrap, within group", "WETH", n_bct,
         strata(weth_a, n_bct[0] / n_weth[0]), strata(weth_b, n_bct[1] / n_weth[1])),
        ("BCT dispersion at WETH counts", "bootstrap, within group", "BCT", n_weth,
         strata(bct_a, n_weth[0] / n_bct[0]), strata(bct_b, n_weth[1] / n_bct[1])),
        ("WETH dispersion at 4x WETH counts", "bootstrap, within group", "WETH",
         tuple(4 * n for n in n_weth), strata(weth_a, 4.0), strata(weth_b, 4.0)),
    ]
    rows = []
    for design, method, source, counts, sa, sb in designs:
        print(f"\n  power: {design}, {method}, {counts[0]} vs {counts[1]} windows")
        res = bootstrap_power(sa, sb, shifts, args.power_reps, args.power_draws,
                              f"{design} {method}", args.jobs)
        rows += [{"design": design, "method": method, "dispersion_from": source, **r} for r in res]
        for r in res:
            print(f"    shift {r['shift']:.4f}  power {r['power']:6.1%}  "
                  f"size-adjusted {r['power_size_adjusted']:6.1%}")

    if args.mrw_reps > 0:
        for design, window, (a, b) in (("BCT", BCT_WINDOW, (bct_a, bct_b)),
                                       ("WETH", WETH_WINDOW, (weth_a, weth_b))):
            calib = calibrate(window, args.calibration_windows, args.jobs)
            out[f"venues.power.mrw.{design}.calibration"] = [list(calib[0]), list(calib[1])]
            print(f"\n  power: {design}, MRW, {len(a)} vs {len(b)} windows of {window}")
            res = mrw_power(len(a), len(b), window, a.width_pos.median(), shifts,
                            args.mrw_reps, args.power_draws, calib, f"{design} mrw", args.jobs)
            rows += [{"design": design, "method": "MRW simulation", "dispersion_from": "MRW", **r}
                     for r in res]
            for r in res:
                print(f"    shift {r['shift']:.4f}  realised {r['true_gap']:+.3f}  "
                      f"lambda2 {r['lambda2_a']:.3f}/{r['lambda2_b']:.3f}  power {r['power']:6.1%}  "
                      f"size-adjusted {r['power_size_adjusted']:6.1%}"
                      f"{'  clipped' if r['clipped'] else ''}")

    power = pd.DataFrame(rows)
    for (design, method), g in power.groupby(["design", "method"], sort=False):
        recs = g.to_dict("records")
        pre = (f"venues.power.{design.replace(' ', '_')}."
               f"{method.replace(', ', '_').replace(' ', '_')}")
        out[pre + ".n_a"], out[pre + ".n_b"] = int(g.n_a.iloc[0]), int(g.n_b.iloc[0])
        out[pre + ".size"] = power_at(recs, 0.0)
        out[pre + ".critical_p"] = recs[0]["critical_p"]
        out[pre + ".power_at_bct_gap"] = power_at(recs, at_gap)
        out[pre + ".power_at_bct_gap_size_adjusted"] = power_at(recs, at_gap, "power_size_adjusted")
        out[pre + ".mde80"] = mde(recs)
        out[pre + ".mde80_size_adjusted"] = mde(recs, "power_size_adjusted")
        out[pre + ".ladder"] = [[r["shift"], r["true_gap"], r["power"], r["power_size_adjusted"]]
                                for r in recs]
    return power


# --- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draws", type=int, default=stats.DRAWS,
                    help="permutation draws for the reported tests")
    ap.add_argument("--power-reps", type=int, default=POWER_REPS,
                    help="bootstrap replications per shift")
    ap.add_argument("--power-draws", type=int, default=POWER_DRAWS,
                    help="permutation draws inside each power replication")
    ap.add_argument("--mrw-reps", type=int, default=MRW_REPS,
                    help="MRW replications per shift; 0 skips the MRW cross-check")
    ap.add_argument("--calibration-windows", type=int, default=CALIBRATION_WINDOWS,
                    help="MRW windows per lambda2 when calibrating width")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                    help="worker processes for the power simulations")
    ap.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS),
                    help="run only these parts (power also runs bct and weth)")
    ap.add_argument("--quick", action="store_true",
                    help="999 draws, 40 bootstrap and 8 MRW replications, outputs suffixed _quick")
    args = ap.parse_args()
    if args.quick:
        args.draws, args.power_reps, args.power_draws = 999, 40, 199
        args.mrw_reps, args.calibration_windows = min(args.mrw_reps, 8), 30
    sections = set(args.sections)
    if "power" in sections:
        sections |= {"bct", "weth"}
    suffix = "_quick" if args.quick else ("" if sections == set(SECTIONS) else "_partial")
    needed = sorted({n for s in sections for n in NEEDS[s]})
    require(needed)

    frames = {}
    for n in needed:
        frames[n] = trades.trades(POOLS[n])
        print(f"{n:15s} {len(frames[n]):>10,} cleaned trades  "
              f"{frames[n].timestamp.iloc[0]:%Y-%m-%d} to {frames[n].timestamp.iloc[-1]:%Y-%m-%d}")

    out = {"venues.draws": args.draws, "venues.sample_end": trades.SAMPLE_END}
    overlap = None
    if "BCT_base" in frames:
        overlap = (max(frames["BCT_polygon"].timestamp.iloc[0], frames["BCT_base"].timestamp.iloc[0]),
                   min(frames["BCT_polygon"].timestamp.iloc[-1], frames["BCT_base"].timestamp.iloc[-1]))
    comparisons = []

    if "roundtrip" in sections:
        table("venues_roundtrip" + suffix, section_roundtrip(frames, overlap, out))
    if "bct" in sections:
        section_same_asset(between(frames["BCT_polygon"], *overlap),
                           between(frames["BCT_base"], *overlap), overlap, out)
        bct_windows, rows, bct_w = section_bct(frames, overlap, args.draws, out)
        comparisons += rows
        table("venues_bct_windows" + suffix, bct_windows)
    if "weth" in sections:
        weth_windows, rows, years, act, weth_w = section_weth(frames, args.draws, out)
        comparisons += rows
        table("venues_weth_windows" + suffix, weth_windows)
        table("venues_weth_year" + suffix, years)
        table("venues_weth_activity" + suffix, act)
        out["venues.weth.windows_total"] = len(weth_w["WETH_sushiswap"]) + len(weth_w["WETH_quickswap"])
    if comparisons:
        table("venues_comparisons" + suffix, pd.DataFrame(comparisons))
    if "power" in sections:
        bct_gap = out["venues.bct.venue.raw.gap"]
        if not np.isfinite(bct_gap) or len(bct_w["BCT Base (overlap)"]) < 2:
            raise SystemExit("the BCT venue comparison has too few windows for a power analysis")
        out["venues.power.window_ratio_weth_to_bct"] = \
            out["venues.weth.windows_total"] / (out["venues.bct.venue.raw.n_a"] + out["venues.bct.venue.raw.n_b"])
        table("venues_power" + suffix, section_power(bct_w, weth_w, bct_gap, args, out))
    print("wrote", numbers("venues" + suffix, out))


if __name__ == "__main__":
    main()
