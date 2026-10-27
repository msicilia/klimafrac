"""
Calendar-time estimates on the reconstructed pools and on the aggregator's
series: the paper's Tables tab:daily and tab:hourly and the aggregator
staleness of its data section.

For each carbon token the cleaned trades of the estimation sample are put on
a daily and an hourly grid (trades.calendar: last price carried forward,
no_trade observed from the trades), and the DefiLlama series is placed on
the same grid, a vendor bar at instant t being the price at t. Widths: the
full range, q in [-5, 5] (daily s_max = N/4, hourly N/10; flat segments
excluded from the negative moments, the *_ro columns keep them), and the
positive side, each against 1,000 shuffled and 1,000 phase-randomised
surrogates (mfdfa.surrogate_null). Vendor staleness compares, hour by hour,
the hours with a zero vendor return with the hours in which the pool did
not trade.

Outputs (data/results/): calendar_daily.csv, calendar_hourly.csv,
calendar_vendor_staleness.csv, calendar.json; figures fig1_staleness and
fig2_hq, not in the paper. --quick suffixes outputs _quick.

Usage:
    uv run python -m klimafrac.analysis.calendar [--quick --tokens NCT MCO2]
"""

import argparse
import json
import os
import time
import zlib
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd

from klimafrac import trades
from klimafrac.mfdfa import QS, QS_POS, full_spectrum, mfdfa, phase_randomise, positive_spectrum
from klimafrac.paths import data, figure
from klimafrac.plotstyle import INK, INK2, MUTED, SERIES, plt, save, style
from klimafrac.results import numbers, table

TOKENS = {"BCT": "polygon_BCT", "NCT": "polygon_NCT",
          "MCO2": "polygon_MCO2", "KLIMA": "polygon_KLIMA"}
FREQS = {"daily": "1D", "hourly": "1h"}
VENDOR_FILE = {"daily": "prices_full.csv", "hourly": "hourly_prices_full.csv"}
FULL_S_MAX_DIV = {"daily": 4, "hourly": 10}
POS_S_MAX_DIV = 10
SOURCES = ("vendor", "reconstructed")
SURROGATES = 1000
QUICK_SURROGATES = 100


# --- series ----------------------------------------------------------------

def require(path: str, hint: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: {hint}")


@lru_cache(maxsize=None)
def pool_grid(token: str, freq: str) -> pd.DataFrame:
    key = TOKENS[token]
    require(data(f"swaps_{key}.csv.gz"), f"run klimafrac.fetch.swaps --pool {key}")
    return trades.calendar(trades.trades(key), FREQS[freq])


@lru_cache(maxsize=None)
def vendor_prices(freq: str) -> pd.DataFrame:
    path = data(VENDOR_FILE[freq])
    require(path, f"run klimafrac.fetch.{freq}_prices")
    v = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    v.index = v.index.tz_localize("UTC")
    return v.where(v > 0)


def vendor_returns(token: str, freq: str, index: pd.DatetimeIndex,
                   aligned: bool = True) -> pd.Series:
    """Vendor log return into each pool interval; NaN where a bar is missing."""
    v = vendor_prices(freq)
    if token not in v.columns:
        raise SystemExit(f"{VENDOR_FILE[freq]} has no column {token}")
    step = pd.Timedelta(FREQS[freq])
    close, prev = (index + step, index) if aligned else (index, index - step)
    s = v[token]
    return pd.Series(np.log(s.reindex(close).to_numpy()) - np.log(s.reindex(prev).to_numpy()),
                     index=index)


def series(token: str, freq: str, source: str) -> pd.Series:
    grid = pool_grid(token, freq)
    if source == "reconstructed":
        return grid.ret
    return vendor_returns(token, freq, grid.index).dropna()


def seed_for(freq: str, source: str, token: str) -> int:
    return zlib.crc32(f"{freq}/{source}/{token}".encode())


# --- estimation ------------------------------------------------------------

def estimate(x: np.ndarray, freq: str) -> dict:
    full = full_spectrum(x, FULL_S_MAX_DIV[freq])
    ro = full_spectrum(x, FULL_S_MAX_DIV[freq], exclude_flat=False)
    pos = positive_spectrum(x, POS_S_MAX_DIV)
    if full is None or ro is None or pos is None:
        raise SystemExit(f"series of length {len(x)} too short to estimate")
    return {"n": len(x),
            "width": full["width"], "alpha0": full["alpha0"],
            "h2": full["h2"], "h_neg5": full["h_neg5"],
            "width_ro": ro["width"], "alpha0_ro": ro["alpha0"], "h_neg5_ro": ro["h_neg5"],
            "h2_pos": pos["h2"], "width_pos": pos["width_pos"]}


def surrogate_task(task: tuple) -> tuple:
    """Draws of mfdfa.surrogate_null, each evaluated on both widths."""
    label, x, freq, n_iter, seed = task
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("shuffled", "shuffled_pos", "phase", "phase_pos")}

    def evaluate(draw, name):
        r = full_spectrum(draw, FULL_S_MAX_DIV[freq])
        if r is not None:
            out[name].append(r["width"])
        r = positive_spectrum(draw, POS_S_MAX_DIV)
        if r is not None:
            out[name + "_pos"].append(r["width_pos"])

    for _ in range(n_iter):
        evaluate(rng.permutation(x), "shuffled")
        evaluate(phase_randomise(x, rng), "phase")
    return label, {k: np.array(v) for k, v in out.items()}


def p95(values: np.ndarray) -> float:
    return float(np.percentile(values, 95)) if len(values) else np.nan


def estimates(tokens, n_surrogates: int, workers: int) -> pd.DataFrame:
    rows, tasks = {}, []
    for freq in FREQS:
        for source in SOURCES:
            for token in tokens:
                s = series(token, freq, source)
                grid = pool_grid(token, freq)
                x = s.to_numpy()
                label = (freq, source, token)
                seed = seed_for(*label)
                rows[label] = {
                    "freq": freq, "source": source, "token": token,
                    "window_start": grid.index[0], "window_end": grid.index[-1],
                    "grid_intervals": len(grid),
                    "vendor_missing_intervals": len(grid) - len(s) if source == "vendor" else 0,
                    "zero_return_pct": 100 * float(np.mean(x == 0)),
                    "no_trade_pct": 100 * float(grid.no_trade.mean()) if source == "reconstructed" else np.nan,
                    **estimate(x, freq), "surrogate_seed": seed}
                tasks.append((label, x, freq, n_surrogates, seed))

    # Hourly series are ~25x longer than daily ones; start them first.
    tasks.sort(key=lambda t: -len(t[1]))
    with Pool(workers) as pool:
        for label, null in pool.imap_unordered(surrogate_task, tasks):
            r = rows[label]
            r["width_shuffled_p95"] = p95(null["shuffled"])
            r["width_phase_p95"] = p95(null["phase"])
            r["width_pos_shuffled_p95"] = p95(null["shuffled_pos"])
            r["width_pos_phase_p95"] = p95(null["phase_pos"])
            r["surrogates_shuffled"] = len(null["shuffled"])
            r["surrogates_phase"] = len(null["phase"])
            print(f"  {label[0]:6s} {label[1]:13s} {label[2]:5s} N={r['n']:,} "
                  f"width {r['width']:.3f} (shuffled p95 {r['width_shuffled_p95']:.3f}, "
                  f"phase p95 {r['width_phase_p95']:.3f})", flush=True)

    df = pd.DataFrame([rows[(f, s, t)] for f in FREQS for s in SOURCES for t in tokens])
    df["exceeds_shuffled"] = df.width > df.width_shuffled_p95
    df["exceeds_phase"] = df.width > df.width_phase_p95
    df["width_pos_exceeds_shuffled"] = df.width_pos > df.width_pos_shuffled_p95
    df["width_pos_exceeds_phase"] = df.width_pos > df.width_pos_phase_p95
    bct = df[df.token == "BCT"].set_index(["freq", "source"]).width
    df["width_rel_bct"] = [r.width / bct.get((r.freq, r.source), np.nan) for r in df.itertuples()]
    return df


# --- vendor staleness ------------------------------------------------------

def staleness_counts(token: str) -> dict:
    grid = pool_grid(token, "hourly")
    nt = grid.no_trade.to_numpy()
    ret = vendor_returns(token, "hourly", grid.index).to_numpy()
    raw = vendor_returns(token, "hourly", grid.index, aligned=False).to_numpy()
    live, live_raw = ~np.isnan(ret), ~np.isnan(raw)
    return {
        "token": token, "grid_start": grid.index[0], "grid_end": grid.index[-1],
        "hours": len(grid), "no_trade_hours": int(nt.sum()),
        "vendor_return_hours": int(live.sum()),
        "vendor_zero_hours": int((live & (ret == 0)).sum()),
        "matched_no_trade_hours": int((live & nt).sum()),
        "vendor_moves_in_no_trade_hours": int((live & nt & (ret != 0)).sum()),
        "matched_traded_hours": int((live & ~nt).sum()),
        "vendor_zero_in_traded_hours": int((live & ~nt & (ret == 0)).sum()),
        "unaligned_no_trade_hours": int((live_raw & nt).sum()),
        "unaligned_moves_in_no_trade_hours": int((live_raw & nt & (raw != 0)).sum()),
    }


def staleness_table(tokens) -> pd.DataFrame:
    df = pd.DataFrame([staleness_counts(t) for t in tokens])
    if len(tokens) > 1:
        total = {"token": "all", "grid_start": df.grid_start.min(), "grid_end": df.grid_end.max(),
                 **df.drop(columns=["token", "grid_start", "grid_end"]).sum()}
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    df["no_trade_pct"] = 100 * df.no_trade_hours / df.hours
    df["vendor_missing_hours"] = df.hours - df.vendor_return_hours
    df["vendor_zero_pct"] = 100 * df.vendor_zero_hours / df.vendor_return_hours
    df["matched_no_trade_pct"] = 100 * df.matched_no_trade_hours / df.vendor_return_hours
    df["understatement_ratio"] = df.matched_no_trade_pct / df.vendor_zero_pct
    df["vendor_moves_given_no_trade_pct"] = 100 * df.vendor_moves_in_no_trade_hours / df.matched_no_trade_hours
    df["staleness_deficit_pct"] = 100 * (1 - df.vendor_zero_hours / df.matched_no_trade_hours)
    df["vendor_zero_given_trade_pct"] = 100 * df.vendor_zero_in_traded_hours / df.matched_traded_hours
    df["unaligned_vendor_moves_given_no_trade_pct"] = (
        100 * df.unaligned_moves_in_no_trade_hours / df.unaligned_no_trade_hours)
    return df


# --- figures ---------------------------------------------------------------

def fig_staleness(st: pd.DataFrame, path: str) -> None:
    st = st[st.token != "all"].reset_index(drop=True)
    x = np.arange(len(st))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - 0.19, st.vendor_zero_pct, 0.36, color=MUTED,
           label="aggregated series: no price change")
    ax.bar(x + 0.19, st.matched_no_trade_pct, 0.36,
           color=[SERIES[t] for t in st.token], label="pool: no transaction")
    top = float(st[["vendor_zero_pct", "matched_no_trade_pct"]].to_numpy().max())
    for i, r in st.iterrows():
        ax.text(i - 0.19, r.vendor_zero_pct + top * 0.015, f"{r.vendor_zero_pct:.1f}%",
                ha="center", va="bottom", fontsize=9, color=INK2)
        ax.text(i + 0.19, r.matched_no_trade_pct + top * 0.015, f"{r.matched_no_trade_pct:.1f}%",
                ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(st.token)
    ax.set_ylim(0, top * 1.18)
    style(ax, "Apparent and true staleness, hourly",
          f"Hours with both vendor bars present; the pool's no-trade share is "
          f"{st.understatement_ratio.min():.1f} to {st.understatement_ratio.max():.1f} "
          f"times the vendor's zero-change share",
          "% of hours")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    save(fig, path)


def fig_hq(tokens, path: str) -> None:
    """h(q) on hourly data, corrected estimator only; the uncorrected variant
    (exclude_flat=False, the *_ro columns) is computed and stored, not drawn."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for token in tokens:
        x = series(token, "hourly", "reconstructed").to_numpy()
        s_max = len(x) // POS_S_MAX_DIV
        _, h = mfdfa(x, s_max=s_max, qs=QS)
        _, h_pos = mfdfa(x, s_max=s_max, qs=QS_POS)
        axes[0].plot(QS, h, color=SERIES[token], lw=2, label=token)
        axes[1].plot(QS_POS, h_pos, color=SERIES[token], lw=2, label=token)
    for ax in axes:
        ax.axhline(0.5, color=MUTED, lw=1, ls=":")
        ax.set_xlabel("q")
    style(axes[0], "Full range, q in [-5, 5]", "dotted line: h = 0.5", "h(q)")
    style(axes[1], "Positive branch, q in [0.5, 5]", "dotted line: h = 0.5")
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, path)


# --- cited numbers ---------------------------------------------------------

def span(df: pd.DataFrame, col: str) -> dict:
    lo, hi = df.loc[df[col].idxmin()], df.loc[df[col].idxmax()]
    return {"min": lo[col], "min_token": lo.token, "max": hi[col], "max_token": hi.token}


def cited_numbers(est: pd.DataFrame, st: pd.DataFrame, meta: dict) -> dict:
    out = {f"design.{k}": v for k, v in meta.items()}
    per_row = [c for c in est.columns if c not in ("freq", "source", "token")]
    for r in est.itertuples(index=False):
        for c in per_row:
            out[f"{r.freq}.{r.source}.{r.token}.{c}"] = getattr(r, c)

    for (freq, source), g in est.groupby(["freq", "source"]):
        g = g.reset_index(drop=True)
        pre = f"{freq}.{source}"
        for col in ("n", "width", "width_ro", "h_neg5", "h_neg5_ro", "h2_pos", "width_pos",
                    "zero_return_pct", "no_trade_pct", "alpha0"):
            if g[col].notna().any():
                for k, v in span(g, col).items():
                    out[f"{pre}.range.{col}.{k}"] = v
        out[f"{pre}.width_pos_spread"] = g.width_pos.max() - g.width_pos.min()
        out[f"{pre}.width_max_over_min"] = g.width.max() / g.width.min()
        out[f"{pre}.width_ro_max_over_min"] = g.width_ro.max() / g.width_ro.min()
        out[f"{pre}.order_by_width"] = list(g.sort_values("width", ascending=False).token)
        out[f"{pre}.fails_shuffled_null"] = list(g[~g.exceeds_shuffled].token)
        out[f"{pre}.fails_phase_null"] = list(g[~g.exceeds_phase].token)
        out[f"{pre}.width_pos_fails_shuffled_null"] = list(g[~g.width_pos_exceeds_shuffled].token)
        out[f"{pre}.width_pos_fails_phase_null"] = list(g[~g.width_pos_exceeds_phase].token)
        if {"BCT", "NCT"} <= set(g.token):
            w = g.set_index("token").width
            out[f"{pre}.NCT_over_BCT_width"] = w["NCT"] / w["BCT"]
            out[f"{pre}.NCT_minus_BCT_width_pct"] = 100 * (w["NCT"] / w["BCT"] - 1)
        thin = g[g.token.isin(["BCT", "NCT", "MCO2"])]
        if len(thin):
            for col in ("width", "width_ro"):
                out[f"{pre}.BCT_NCT_MCO2.{col}_min"] = thin[col].min()
                out[f"{pre}.BCT_NCT_MCO2.{col}_max"] = thin[col].max()

    for r in st.itertuples(index=False):
        for c in st.columns:
            if c != "token":
                out[f"staleness.{r.token}.{c}"] = getattr(r, c)
    tok = st[st.token != "all"].reset_index(drop=True)
    for col in ("understatement_ratio", "vendor_moves_given_no_trade_pct",
                "staleness_deficit_pct", "vendor_zero_pct", "matched_no_trade_pct"):
        for k, v in span(tok, col).items():
            out[f"staleness.range.{col}.{k}"] = v
    return out


# --- main ------------------------------------------------------------------

def print_tables(est: pd.DataFrame, st: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for freq in FREQS:
        g = est[est.freq == freq]
        print(f"\nTable {'tab:daily' if freq == 'daily' else 'tab:hourly (and full range)'}"
              f" -- {freq}, full range s_max = N/{FULL_S_MAX_DIV[freq]}, positive q s_max = N/10")
        print(g[["source", "token", "window_start", "n", "zero_return_pct", "no_trade_pct",
                 "width", "alpha0", "width_shuffled_p95", "width_phase_p95", "width_rel_bct",
                 "h_neg5", "width_ro", "h_neg5_ro", "h2_pos", "width_pos",
                 "width_pos_shuffled_p95", "width_pos_phase_p95"]].round(3).to_string(index=False))
    print("\nVendor staleness, hourly, on each pool's grid")
    print(st[["token", "hours", "no_trade_pct", "vendor_return_hours", "vendor_zero_pct",
              "matched_no_trade_pct", "understatement_ratio", "matched_no_trade_hours",
              "vendor_moves_given_no_trade_pct", "staleness_deficit_pct",
              "vendor_zero_given_trade_pct", "unaligned_vendor_moves_given_no_trade_pct"]]
          .round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--surrogates", type=int, default=None,
                    help=f"surrogates of each kind per series (default {SURROGATES}, "
                         f"{QUICK_SURROGATES} with --quick)")
    ap.add_argument("--quick", action="store_true", help="outputs suffixed _quick")
    ap.add_argument("--tokens", nargs="+", choices=list(TOKENS), default=list(TOKENS),
                    help="subset of tokens; only with --quick")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    if not args.quick and args.tokens != list(TOKENS):
        ap.error("--tokens needs --quick: the paper outputs cover all four tokens")
    n_surrogates = args.surrogates or (QUICK_SURROGATES if args.quick else SURROGATES)
    suffix = "_quick" if args.quick else ""
    tokens = [t for t in TOKENS if t in args.tokens]

    t0 = time.time()
    print(f"{len(tokens)} tokens, {n_surrogates} surrogates of each kind, {args.workers} workers")
    est = estimates(tokens, n_surrogates, args.workers)
    st = staleness_table(tokens)
    fig1, fig2 = figure(f"fig1_staleness{suffix}.png"), figure(f"fig2_hq{suffix}.png")
    fig_staleness(st, fig1)
    fig_hq(tokens, fig2)
    seconds = time.time() - t0

    table("calendar_daily" + suffix, est[est.freq == "daily"].drop(columns="freq"))
    table("calendar_hourly" + suffix, est[est.freq == "hourly"].drop(columns="freq"))
    table("calendar_vendor_staleness" + suffix, st)
    print_tables(est, st)

    manifest = json.loads(open(data("vendor_manifest.json")).read())
    meta = {"tokens": tokens, "surrogates_per_kind": n_surrogates,
            "sample_end_exclusive": trades.SAMPLE_END,
            "full_s_max_div_daily": FULL_S_MAX_DIV["daily"],
            "full_s_max_div_hourly": FULL_S_MAX_DIV["hourly"],
            "positive_s_max_div": POS_S_MAX_DIV,
            "vendor_daily_retrieved": manifest[VENDOR_FILE["daily"]]["retrieved"],
            "vendor_hourly_retrieved": manifest[VENDOR_FILE["hourly"]]["retrieved"],
            "seed_rule": "crc32('<freq>/<source>/<token>')",
            "runtime_seconds": round(seconds)}
    print("\nwrote", numbers("calendar" + suffix, cited_numbers(est, st, meta)))
    print("wrote", fig1, fig2, f"\nruntime {seconds:.0f}s")


if __name__ == "__main__":
    main()
