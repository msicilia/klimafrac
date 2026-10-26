"""
Calibration of the staleness bias: what carrying the last price forward does
to the estimated singularity spectrum when the generating process is known
and held fixed: the paper's Tables tab:bias and tab:informativeness and the
staleness threshold of its measurement practices. Figure fig3_staleness_bias
is drawn but not in the paper.

Process. Increments of a multifractal random walk (synthetic.mrw) with
intermittency lambda2 and integral scale synthetic.INTEGRAL_SCALE, fixed
across series lengths so that every N is a draw from the same process, on
the grid lambda2 in {0.05, 0.10, 0.15} with base 0.10. Its true spectrum
gives h(q) = 1/2 + lambda2 - lambda2 q / 2, so the positive-q width has a
known target.

Staleness. Each interval contains a trade or not; a traded interval returns
the whole price change accumulated since the previous trade and an untraded
one returns exactly zero, so the price path is preserved and only the
sampling degrades. The last interval always trades. Two inactivity schemes
with the same stationary no-trade fraction p:
  iid        each interval untraded independently with probability p
  markov<L>  a two-state Markov chain whose untraded runs have mean length
             L (leave-quiet probability 1/L, enter-quiet probability
             p / (L (1 - p))), started from its stationary distribution.
             Defined only for p <= 1 - 1/L, where the chain is at least as
             clustered as iid (i.i.d. runs have mean 1 / (1 - p)); larger p
             would make trades, not inactivity, the clustered state.
Clustering matters because a segment is flat -- all its returns identical --
only when all its intervals are untraded, and clustered inactivity produces
flat segments at much lower average p.

Estimators (mfdfa.py, s_min = 10, s_max = N/10):
  width       full-range width, q in [-5, 5], flat segments excluded from
              the negative moments (the paper's estimator)
  width_pos   positive-side width, q in [0.5, 5]
  h_neg5      h(-5)
  width_ro    full-range width with exclude_flat=False, which keeps the
              floating-point round-off residual of flat segments. Reported
              only to document that the old pathology is a function of
              machine precision, not a property of the data.

Knee. For each (N, lambda2, clustering) the inflation curve is median width
at p over median width at p = 0 (the p = 0 series have no staleness, so one
baseline serves all clusterings). The knee is the smallest grid p at which
inflation exceeds KNEE_TOL = 1.10, with a linearly interpolated crossing and
a 90% bootstrap interval (replications resampled within each p); the knee
at 1.50 is reported as well. `safe_p` is the largest grid p up to which
inflation never exceeds the tolerance. The whole curve is written, so the
knee is a summary, not the result.

Predicted knee. mfdfa calls a segment of length s flat when its last s - 1
returns are identical (a change at its first position only shifts the level
of the profile, which the linear detrend removes). Under staleness that means
s - 1 consecutive untraded intervals: probability p^(s-1) for iid and
p (1 - 1/L)^(s-2) for markov<L>. With 2 floor(N/s) segments (both scan
directions), the expected number of flat segments at s_min is
  iid        2 floor(N/s_min) p^(s_min - 1)
  markov<L>  2 floor(N/s_min) p (1 - 1/L)^(s_min - 2)
and the predicted knee is the p at which that count reaches one. The
simulated flat-segment count at s_min is recorded to check the count, and
the predicted knee is set against both estimators' simulated knees.

Informativeness. On clean series (p = 0), width_pos against lambda2 in
{0, 0.025, ..., 0.15} at every N: medians, interquartile range, the
theoretical width, correlation with lambda2 (of medians and pooled over
replications) and monotonicity.

Replication. REPS = 100 per cell everywhere. Every draw comes from
np.random.SeedSequence(SEED, spawn_key=(N, 1000 lambda2, 1000 p, L, rep)),
L = 0 for iid, so no two cells share random numbers; the p = 0 cells are
reused by the informativeness check, which is the same draw by definition.

Outputs (via klimafrac.results):
  data/results/staleness_replications.csv   one row per simulated series
  data/results/staleness_curves.csv         medians and inflation per cell
  data/results/staleness_knees.csv          knee, interval and prediction
                                            per (N, lambda2, clustering)
  data/results/staleness_bias_table.csv     Table tab:bias (N = 20,000,
                                            lambda2 = 0.10, iid)
  data/results/staleness_informativeness.csv
  data/results/staleness.json               every number the paper cites
  figures/fig3_staleness_bias.png
--quick runs 10 replications on a coarse p grid and writes the same files
with a _quick suffix, so a test run never replaces paper outputs. The full
run is 56,100 series, dominated by the N = 50,000 cells, and takes roughly
an hour to an hour and a half on the default four worker processes;
--quick takes about two minutes.

Usage:
    uv run python -m klimafrac.analysis.staleness
    uv run python -m klimafrac.analysis.staleness --reps 200 --workers 8
    uv run python -m klimafrac.analysis.staleness --quick
"""

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from klimafrac.mfdfa import QS_POS, S_MIN, full_spectrum, positive_spectrum, segment_fluctuations
from klimafrac.paths import figure
from klimafrac.results import numbers, table
from klimafrac.synthetic import INTEGRAL_SCALE, mrw

SEED = 20260726
REPS = 100
S_MAX_DIV = 10
N_GRID = (5_000, 20_000, 50_000)
LAMBDA_GRID = (0.05, 0.10, 0.15)
LAMBDA_INFO = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15)
# 0.01 and 0.02 resolve the knee under clustered inactivity, which comes
# before p = 0.05.
P_GRID = (0.0, 0.01, 0.02, *np.round(np.arange(0.05, 0.951, 0.05), 2))
P_GRID_QUICK = (0.0, 0.02, *np.round(np.arange(0.1, 0.91, 0.1), 2))
P_TABLE = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RUN_LENGTHS = (5, 20)
BASE_N = 20_000
BASE_LAMBDA = 0.10
KNEE_TOL = 1.10
KNEE_TOL_LOOSE = 1.50
BOOT_DRAWS = 1000
# Hourly no-trade fractions of BCT and MCO2, shaded in the figure as the
# range the empirical series occupy.
OBSERVED_NOTRADE = (0.478, 0.736)


def label(run_length: int) -> str:
    return "iid" if run_length == 0 else f"markov{run_length}"


def feasible(p: float, run_length: int) -> bool:
    return run_length == 0 or p <= 1.0 - 1.0 / run_length + 1e-12


def traded_mask(n: int, p: float, run_length: int, rng: np.random.Generator) -> np.ndarray:
    """True where the interval contains a trade; last interval always trades."""
    if p <= 0:
        return np.ones(n, dtype=bool)
    if run_length == 0:
        traded = rng.random(n) >= p
    else:
        leave_quiet = 1.0 / run_length
        enter_quiet = p / (run_length * (1.0 - p))
        start_quiet = rng.random() < p
        states, total = [], 0
        while total < n:
            cycles = int(1.2 * n * p / run_length) + 64
            quiet = rng.geometric(leave_quiet, cycles)
            active = rng.geometric(enter_quiet, cycles) if enter_quiet < 1 else np.ones(cycles, int)
            first, second = (quiet, active) if start_quiet else (active, quiet)
            lengths = np.column_stack([first, second]).ravel()
            flags = np.tile([not start_quiet, start_quiet], cycles)
            states.append(np.repeat(flags, lengths))
            total += int(lengths.sum())
        traded = np.concatenate(states)[:n]
    traded[-1] = True
    return traded


def stale(r: np.ndarray, traded: np.ndarray) -> np.ndarray:
    """Last trade carried forward: accumulated change at trades, exact zeros between."""
    out = np.zeros(len(r))
    idx = np.flatnonzero(traded)
    level = np.cumsum(r)[idx]
    out[idx] = np.diff(level, prepend=0.0)
    return out


def expected_flat(n: int, p: float, run_length: int, s: int = S_MIN) -> float:
    segments = 2 * (n // s)
    if run_length == 0:
        return segments * p ** (s - 1)
    return segments * p * (1.0 - 1.0 / run_length) ** (s - 2)


def predicted_knee(n: int, run_length: int, s: int = S_MIN) -> float:
    """p at which the expected number of flat segments at s_min reaches one."""
    segments = 2 * (n // s)
    if run_length == 0:
        return segments ** (-1.0 / (s - 1))
    p = (1.0 - 1.0 / run_length) ** (-(s - 2)) / segments
    return p if feasible(p, run_length) else np.nan


def simulate(task: tuple) -> dict:
    n, lam, p, run_length, rep = task
    ss = np.random.SeedSequence(SEED, spawn_key=(n, round(lam * 1000), round(p * 1000),
                                                 run_length, rep))
    rng = np.random.default_rng(ss)
    x = stale(mrw(n, lam, rng), traded_mask(n, p, run_length, rng))
    full = full_spectrum(x, S_MAX_DIV)
    # Without flat segments exclude_flat is irrelevant, so p = 0 is not recomputed.
    rounded = full_spectrum(x, S_MAX_DIV, exclude_flat=False) if p > 0 else full
    pos = positive_spectrum(x, S_MAX_DIV)
    _, flat = segment_fluctuations(x, np.cumsum(x - x.mean()), S_MIN)
    nan = float("nan")
    return {"n": n, "lambda2": lam, "p": p, "clustering": label(run_length), "rep": rep,
            "zero_frac": float(np.mean(x == 0)),
            "flat_segments_smin": int(flat.sum()),
            "width": full["width"] if full else nan,
            "h_neg5": full["h_neg5"] if full else nan,
            "h2": full["h2"] if full else nan,
            "width_ro": rounded["width"] if rounded else nan,
            "h_neg5_ro": rounded["h_neg5"] if rounded else nan,
            "width_pos": pos["width_pos"] if pos else nan}


def tasks(ns, lambdas, ps, reps) -> list[tuple]:
    out = set()
    for n in ns:
        for lam in lambdas:
            for rep in range(reps):
                out.add((n, lam, 0.0, 0, rep))
                for p in ps:
                    for run_length in (0, *RUN_LENGTHS):
                        if p > 0 and feasible(p, run_length):
                            out.add((n, lam, p, run_length, rep))
        for lam in LAMBDA_INFO:
            for rep in range(reps):
                out.add((n, lam, 0.0, 0, rep))
    return sorted(out)


# --- summaries ---------------------------------------------------------------

def with_baseline(reps_df: pd.DataFrame) -> pd.DataFrame:
    """Copy the p = 0 series into every clustering, since they are the same series."""
    base = reps_df[reps_df.p == 0]
    extra = [base.assign(clustering=label(L)) for L in RUN_LENGTHS]
    return pd.concat([reps_df, *extra], ignore_index=True)


def curves(reps_df: pd.DataFrame) -> pd.DataFrame:
    df = with_baseline(reps_df[reps_df.lambda2.isin(LAMBDA_GRID)])
    rows = []
    for (n, lam, clust, p), g in df.groupby(["n", "lambda2", "clustering", "p"]):
        run_length = 0 if clust == "iid" else int(clust.removeprefix("markov"))
        rows.append({
            "n": n, "lambda2": lam, "clustering": clust, "p": p, "reps": len(g),
            "valid_full": int(g.width.notna().sum()),
            "zero_frac": g.zero_frac.mean(),
            "flat_segments_smin": g.flat_segments_smin.mean(),
            "flat_segments_smin_expected": expected_flat(n, p, run_length),
            "width": g.width.median(),
            "width_q25": g.width.quantile(0.25), "width_q75": g.width.quantile(0.75),
            "width_pos": g.width_pos.median(),
            "width_pos_q25": g.width_pos.quantile(0.25),
            "width_pos_q75": g.width_pos.quantile(0.75),
            "h_neg5": g.h_neg5.median(),
            "width_ro": g.width_ro.median(), "h_neg5_ro": g.h_neg5_ro.median()})
    out = pd.DataFrame(rows)
    keys = ["n", "lambda2", "clustering"]
    for col, name in (("width", "inflation"), ("width_ro", "inflation_ro"),
                      ("width_pos", "inflation_pos")):
        base = out[out.p == 0].set_index(keys)[col]
        out[name] = out[col].to_numpy() / base.loc[list(out[keys].itertuples(index=False))].to_numpy()
    return out


def knee_from_curve(ps: np.ndarray, infl: np.ndarray, tol: float) -> tuple[float, float]:
    """Smallest grid p with inflation above tol, and the interpolated crossing."""
    above = np.flatnonzero(infl > tol)
    if len(above) == 0:
        return np.nan, np.nan
    i = above[0]
    if i == 0:
        return float(ps[0]), float(ps[0])
    p0, p1, y0, y1 = ps[i - 1], ps[i], infl[i - 1], infl[i]
    return float(ps[i]), float(p0 + (tol - y0) * (p1 - p0) / (y1 - y0))


def knees(reps_df: pd.DataFrame, curve_df: pd.DataFrame) -> pd.DataFrame:
    df = with_baseline(reps_df[reps_df.lambda2.isin(LAMBDA_GRID)])
    rows = []
    for (n, lam, clust), c in curve_df.groupby(["n", "lambda2", "clustering"]):
        c = c.sort_values("p")
        run_length = 0 if clust == "iid" else int(clust.removeprefix("markov"))
        ps = c.p.to_numpy()
        row = {"n": n, "lambda2": lam, "clustering": clust, "p_max": ps.max()}
        for col, suffix in (("inflation", ""), ("inflation_ro", "_ro")):
            infl = c[col].to_numpy()
            for tol, tag in ((KNEE_TOL, "110"), (KNEE_TOL_LOOSE, "150")):
                grid, interp = knee_from_curve(ps, infl, tol)
                row[f"knee{tag}{suffix}"] = grid
                row[f"knee{tag}{suffix}_interp"] = interp
            within = np.cumprod(infl <= KNEE_TOL).astype(bool)
            row[f"safe_p{suffix}"] = float(ps[within][-1])
            row[f"max_inflation_to_p030{suffix}"] = float(infl[ps <= 0.30 + 1e-9].max())
            for p_at in (0.40, 0.50, 0.70, 0.90):
                hit = np.isclose(ps, p_at)
                row[f"inflation_p{round(p_at * 100):03d}{suffix}"] = (
                    float(infl[hit][0]) if hit.any() else np.nan)

        # Bootstrap the knee: resample replications within each p.
        g = df[(df.n == n) & (df.lambda2 == lam) & (df.clustering == clust)]
        rng = np.random.default_rng(np.random.SeedSequence(
            SEED, spawn_key=(n, round(lam * 1000), run_length, 99)))
        boot = np.empty((BOOT_DRAWS, len(ps)))
        for j, p in enumerate(ps):
            w = g.loc[np.isclose(g.p, p), "width"].dropna().to_numpy()
            boot[:, j] = np.median(rng.choice(w, (BOOT_DRAWS, len(w))), axis=1)
        infl_boot = boot / boot[:, [0]]
        draws = np.array([knee_from_curve(ps, b, KNEE_TOL)[1] for b in infl_boot])
        # A draw that never crosses counts as beyond the grid.
        draws = np.where(np.isnan(draws), np.inf, draws)
        row["knee110_interp_lo"], row["knee110_interp_hi"] = np.quantile(draws, [0.05, 0.95])
        row["knee110_never_share"] = float(np.isinf(draws).mean())
        row["knee_predicted"] = predicted_knee(n, run_length)
        rows.append(row)
    return pd.DataFrame(rows)


def informativeness(reps_df: pd.DataFrame) -> pd.DataFrame:
    clean = reps_df[(reps_df.p == 0) & reps_df.lambda2.isin(LAMBDA_INFO)]
    rows = []
    for (n, lam), g in clean.groupby(["n", "lambda2"]):
        h = 0.5 + lam - lam * QS_POS / 2.0
        alpha = np.gradient(QS_POS * h - 1.0, QS_POS)
        theory = float(alpha.max() - alpha.min())
        rows.append({"n": n, "lambda2": lam, "reps": len(g),
                     "width_pos": g.width_pos.median(),
                     "width_pos_q25": g.width_pos.quantile(0.25),
                     "width_pos_q75": g.width_pos.quantile(0.75),
                     "width_pos_theory": theory,
                     "width_pos_mean": g.width_pos.mean(),
                     "width_pos_sd": g.width_pos.std(ddof=1),
                     "width_pos_bias": g.width_pos.mean() - theory,
                     "width_pos_rmse": float(np.sqrt(((g.width_pos - theory) ** 2).mean())),
                     "width": g.width.median(), "h2": g.h2.median()})
    return pd.DataFrame(rows)


# --- figure ------------------------------------------------------------------

def plot(curve_df: pd.DataFrame, info_df: pd.DataFrame, path: str) -> None:
    from klimafrac.plotstyle import ACCENT, INK2, MUTED, plt, style

    ink2, muted = INK2, MUTED
    blue, orange, green = ACCENT[:3]
    colours = {"iid": blue, "markov5": orange, "markov20": green}
    names = {"iid": "independent", "markov5": "clustered, mean quiet run 5",
             "markov20": "clustered, mean quiet run 20"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    base = curve_df[(curve_df.n == BASE_N) & np.isclose(curve_df.lambda2, BASE_LAMBDA)]
    ax.axvspan(*OBSERVED_NOTRADE, color=muted, alpha=0.12, lw=0)
    for clust in ("iid", "markov5", "markov20"):
        c = base[base.clustering == clust].sort_values("p")
        ax.plot(c.p, c.width, "-o", ms=3.5, lw=2, color=colours[clust],
                label=rf"$\Delta\alpha$, {names[clust]}")
        ax.plot(c.p, c.width_pos, "--", lw=1.4, color=colours[clust])
    p0 = base[(base.clustering == "iid") & (base.p == 0)]
    ax.axhline(float(p0.width.iloc[0]) * KNEE_TOL, color=muted, ls=":", lw=1.2)
    ax.plot([], [], "--", color=ink2, lw=1.4, label=r"$\Delta\alpha^{+}$ (dashed)")
    ax.set_xlabel("fraction of intervals with no trade, $p$")
    ax.set_xlim(-0.02, max(P_GRID) + 0.02)
    style(ax, "Spectrum width against no-trade fraction",
          rf"MRW, $\lambda^2={BASE_LAMBDA:.2f}$, $N={BASE_N:,}$, held fixed as $p$ rises;"
          "\nshaded: observed hourly staleness; dotted: "
          rf"{KNEE_TOL:.2f}$\times$ the $p=0$ width", r"median width")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    info = info_df[info_df.n == BASE_N].sort_values("lambda2")
    ax2.fill_between(info.lambda2, info.width_pos_q25, info.width_pos_q75,
                     color=blue, alpha=0.15, lw=0)
    ax2.plot(info.lambda2, info.width_pos, "-o", ms=4, lw=2, color=blue,
             label=r"estimated $\Delta\alpha^{+}$ (median, IQR)")
    ax2.plot(info.lambda2, info.width_pos_theory, "--", lw=1.4, color=ink2,
             label="true width of the process")
    ax2.set_xlabel(r"true intermittency $\lambda^2$")
    style(ax2, "Positive-$q$ width against true intermittency",
          rf"clean series ($p=0$), $N={BASE_N:,}$", None)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- main --------------------------------------------------------------------

def cell_key(n: int, lam: float, clust: str) -> str:
    return f"N{n}_lambda2_{lam:.2f}_{clust}"


def cited_numbers(reps: int, ps, curve_df, knee_df, info_df, bias_df, runtime) -> dict:
    out = {
        "design_reps_per_cell": reps, "design_seed": SEED, "design_s_min": S_MIN,
        "design_s_max_div": S_MAX_DIV, "design_integral_scale": INTEGRAL_SCALE,
        "design_n_grid": list(N_GRID), "design_lambda2_grid": list(LAMBDA_GRID),
        "design_lambda2_base": BASE_LAMBDA, "design_n_base": BASE_N,
        "design_lambda2_informativeness": list(LAMBDA_INFO),
        "design_p_grid": list(ps), "design_p_table": list(P_TABLE),
        "design_markov_mean_quiet_runs": list(RUN_LENGTHS),
        "design_knee_tolerance": KNEE_TOL, "design_knee_tolerance_loose": KNEE_TOL_LOOSE,
        "design_bootstrap_draws": BOOT_DRAWS,
        "design_series_simulated": int(runtime["series"]),
        "runtime_seconds": runtime["seconds"],
    }

    for _, r in bias_df.iterrows():
        tag = f"p{round(r.p * 100):03d}"
        for col in ("width", "inflation", "width_pos", "h_neg5", "width_ro",
                    "inflation_ro", "h_neg5_ro", "zero_frac"):
            out[f"bias_table_{tag}_{col}"] = r[col]
    out["bias_table_width_pos_min"] = bias_df.width_pos.min()
    out["bias_table_width_pos_max"] = bias_df.width_pos.max()
    out["bias_table_inflation_max"] = bias_df.inflation.max()
    out["bias_table_inflation_ro_max"] = bias_df.inflation_ro.max()

    for _, r in knee_df.iterrows():
        k = cell_key(r.n, r.lambda2, r.clustering)
        for col in ("knee110", "knee110_interp", "knee110_interp_lo", "knee110_interp_hi",
                    "knee150", "knee110_ro", "knee110_ro_interp", "knee150_ro",
                    "safe_p", "safe_p_ro", "max_inflation_to_p030",
                    "max_inflation_to_p030_ro", "inflation_p040", "inflation_p050",
                    "inflation_p040_ro", "inflation_p050_ro", "knee_predicted", "p_max"):
            out[f"knee_{k}_{col}"] = r[col]

    for clust, g in knee_df.groupby("clustering"):
        out[f"worst_{clust}_knee110_min"] = g.knee110.min()
        out[f"worst_{clust}_knee110_ro_min"] = g.knee110_ro.min()
        out[f"worst_{clust}_safe_p_min"] = g.safe_p.min()
        out[f"worst_{clust}_safe_p_ro_min"] = g.safe_p_ro.min()
        out[f"worst_{clust}_max_inflation_to_p030"] = g.max_inflation_to_p030.max()
        out[f"worst_{clust}_max_inflation_to_p030_ro"] = g.max_inflation_to_p030_ro.max()
        out[f"worst_{clust}_cells_never_exceed_tol"] = int(g.knee110.isna().sum())
        out[f"worst_{clust}_cells"] = len(g)
        for n, gn in g.groupby("n"):
            out[f"knee_by_n_{clust}_N{n}_knee110_interp_mean"] = gn.knee110_interp.mean()
            out[f"knee_by_n_{clust}_N{n}_knee110_ro_interp_mean"] = gn.knee110_ro_interp.mean()
            out[f"knee_by_n_{clust}_N{n}_safe_p_min"] = gn.safe_p.min()
        falling = [bool(np.all(np.diff(gl.sort_values("n").knee110_interp.fillna(np.inf)) <= 0))
                   for _, gl in g.groupby("lambda2")]
        out[f"knee_by_n_{clust}_falls_with_n_all_lambda2"] = all(falling)
        falling_ro = [bool(np.all(np.diff(gl.sort_values("n").knee110_ro_interp.fillna(np.inf)) <= 0))
                      for _, gl in g.groupby("lambda2")]
        out[f"knee_by_n_{clust}_ro_falls_with_n_all_lambda2"] = all(falling_ro)

    iid = knee_df[knee_df.clustering == "iid"]
    for col in ("knee110_interp", "knee110_ro_interp"):
        ok = iid[[col, "knee_predicted"]].dropna()
        out[f"prediction_iid_corr_{col}"] = (float(np.corrcoef(ok[col], ok.knee_predicted)[0, 1])
                                            if len(ok) > 2 and ok[col].std() > 0 else np.nan)
        out[f"prediction_iid_mean_abs_error_{col}"] = float((ok[col] - ok.knee_predicted).abs().mean()) \
            if len(ok) else np.nan
    for n in N_GRID:
        out[f"prediction_iid_N{n}_knee_predicted"] = predicted_knee(n, 0)
    flat = curve_df[(curve_df.p > 0) & (curve_df.flat_segments_smin_expected > 1)]
    rel = (flat.flat_segments_smin / flat.flat_segments_smin_expected - 1).abs()
    out["prediction_flat_count_max_rel_error_where_expected_gt1"] = rel.max()

    base = curve_df[(curve_df.n == BASE_N) & np.isclose(curve_df.lambda2, BASE_LAMBDA)]
    for clust, c in base.groupby("clustering"):
        out[f"base_{clust}_width_pos_inflation_max_abs_dev"] = float((c.inflation_pos - 1).abs().max())
        for p_at in (0.30, 0.50, 0.70):
            hit = c[np.isclose(c.p, p_at)]
            if len(hit):
                out[f"base_{clust}_p{round(p_at * 100):03d}_inflation"] = hit.inflation.iloc[0]
                out[f"base_{clust}_p{round(p_at * 100):03d}_width"] = hit.width.iloc[0]
    obs = base[(base.clustering == "iid") & (base.p >= OBSERVED_NOTRADE[0] - 0.03)
               & (base.p <= OBSERVED_NOTRADE[1] + 0.03)]
    out["base_iid_observed_range_width_min"] = obs.width.min()
    out["base_iid_observed_range_width_max"] = obs.width.max()
    out["base_iid_observed_range_width_ro_min"] = obs.width_ro.min()
    out["base_iid_observed_range_width_ro_max"] = obs.width_ro.max()

    for n, g in info_df.groupby("n"):
        g = g.sort_values("lambda2")
        pre = f"informativeness_N{n}"
        out[f"{pre}_width_pos_at_lambda2_0"] = g.width_pos.iloc[0]
        out[f"{pre}_width_pos_at_lambda2_max"] = g.width_pos.iloc[-1]
        out[f"{pre}_width_pos_theory_at_lambda2_max"] = g.width_pos_theory.iloc[-1]
        out[f"{pre}_corr_medians"] = float(np.corrcoef(g.lambda2, g.width_pos)[0, 1])
        out[f"{pre}_monotone"] = bool(np.all(np.diff(g.width_pos) > 0))
        for _, r in g.iterrows():
            out[f"{pre}_lambda2_{r.lambda2:.3f}_width_pos"] = r.width_pos
            out[f"{pre}_lambda2_{r.lambda2:.3f}_width_pos_bias"] = r.width_pos_bias
            out[f"{pre}_lambda2_{r.lambda2:.3f}_width_pos_sd"] = r.width_pos_sd
            out[f"{pre}_lambda2_{r.lambda2:.3f}_width_pos_rmse"] = r.width_pos_rmse
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    # Measured: throughput peaks at about four processes and falls beyond
    # (memory bandwidth, not BLAS threads), so more workers is slower.
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--quick", action="store_true",
                    help="10 replications, 0.1-step p grid, outputs suffixed _quick")
    args = ap.parse_args()
    reps = 10 if args.quick else args.reps
    ps = P_GRID_QUICK if args.quick else P_GRID
    suffix = "_quick" if args.quick else ""

    todo = tasks(N_GRID, LAMBDA_GRID, ps, reps)
    print(f"{len(todo):,} series, {reps} replications per cell, {args.workers} workers")
    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(simulate, todo, chunksize=16), 1):
            rows.append(row)
            if i % 5000 == 0:
                print(f"  {i:,} / {len(todo):,}  {time.time() - t0:.0f}s", flush=True)
    seconds = time.time() - t0
    print(f"simulated in {seconds:.0f}s")

    reps_df = pd.DataFrame(rows).sort_values(["n", "lambda2", "clustering", "p", "rep"])
    curve_df = curves(reps_df)
    knee_df = knees(reps_df, curve_df)
    info_df = informativeness(reps_df)
    bias_df = curve_df[(curve_df.n == BASE_N) & np.isclose(curve_df.lambda2, BASE_LAMBDA)
                       & (curve_df.clustering == "iid")
                       & curve_df.p.round(2).isin(P_TABLE)].reset_index(drop=True)

    table("staleness_replications" + suffix, reps_df)
    table("staleness_curves" + suffix, curve_df)
    table("staleness_knees" + suffix, knee_df)
    table("staleness_bias_table" + suffix, bias_df)
    table("staleness_informativeness" + suffix, info_df)
    fig_path = figure(f"fig3_staleness_bias{suffix}.png")
    plot(curve_df, info_df, fig_path)

    pd.set_option("display.width", 200)
    print("\nTable tab:bias  (N = 20,000, lambda2 = 0.10, independent inactivity)")
    print(bias_df[["p", "zero_frac", "width", "inflation", "width_pos", "h_neg5",
                   "width_ro", "inflation_ro", "h_neg5_ro"]].round(3).to_string(index=False))
    print(f"\nKnee: smallest p with median inflation > {KNEE_TOL} (interpolated, 90% bootstrap)")
    print(knee_df[["n", "lambda2", "clustering", "knee110", "knee110_interp",
                   "knee110_interp_lo", "knee110_interp_hi", "knee_predicted",
                   "knee110_ro_interp", "safe_p", "max_inflation_to_p030",
                   "inflation_p040", "inflation_p050"]].round(3).to_string(index=False))
    print("\nPositive-q width on clean series")
    print(info_df.pivot(index="lambda2", columns="n", values="width_pos").round(3).to_string())

    out = cited_numbers(reps, ps, curve_df, knee_df, info_df, bias_df,
                        {"seconds": round(seconds), "series": len(todo)})
    print("\nwrote", numbers("staleness" + suffix, out))
    print("wrote", fig_path)


if __name__ == "__main__":
    main()
