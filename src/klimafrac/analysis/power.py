"""
Size of the overlapping-window inference and power of the design the paper
uses, simulated on multifractal random walks at the empirical window counts
of the NCT vs BCT and MCO2 vs BCT comparisons.

Each group is one MRW series (synthetic.mrw) as long as the token's
trade-to-trade return series. The base intermittency is calibrated so that
the median 4,000-return window width matches BCT's; the alternatives are
placed on the same calibration curve at each gap of GAP_LADDER, upward when
the curve allows and downward otherwise, and the true gap of every step is
measured on a separate population of windows.
  size   both groups at the base lambda2; windows of 4,000 advanced by 250,
         widths regressed on a group dummy with Newey-West errors (maxlags
         16); the same series then cut into non-overlapping windows and
         tested with stats.permutation_test
  power  the BCT-sized group at the base lambda2 and the other at each
         alternative; non-overlapping windows and the permutation test at 5%
Every replication is seeded from its (design, step, rep) identifier.

Outputs (data/results/): power_calibration.csv, power_ladder.csv,
power_size.csv, power_curve.csv, power.json. --quick shrinks every count and
suffixes outputs _quick. The full run takes a few core-hours.

Usage:
    uv run python -m klimafrac.analysis.power [--skip-size --designs NCT] [--quick]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

# One BLAS thread per worker process: the per-window least-squares fits are
# tiny, and multithreaded BLAS in every worker oversubscribes the cores.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from klimafrac import stats, trades
from klimafrac.analysis.tiers import BASE, OTHERS, TOKENS, require
from klimafrac.mfdfa import positive_spectrum
from klimafrac.results import numbers, table
from klimafrac.synthetic import INTEGRAL_SCALE, mrw

WINDOW = trades.WINDOW
S_MAX_DIV = trades.S_MAX_DIV
STEP = 250
HAC_LAGS = WINDOW // STEP
ALPHA = 0.05

SIZE_REPS = 1000
POWER_REPS = 500
DRAWS = stats.DRAWS

# Up to 0.5 in steps of 0.02, then on to 1.0 in steps of 0.05. The width a
# multifractal random walk yields flattens out above 0.5, so the upper part
# only serves a base width that sits close to that ceiling.
LAMBDA2_GRID = (*np.round(np.arange(0.0, 0.5001, 0.02), 3),
                *np.round(np.arange(0.55, 1.0001, 0.05), 3))
CAL_WINDOWS = 1000
POP_WINDOWS = 10_000
SERIES_WINDOWS = 50            # windows per simulated series in calibration and populations
GAP_TARGETS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.125, 0.15)
POWER_LEVELS = (1 / 3, 0.5, 0.8)
BOOTSTRAP = 500

DATA_STREAM = 1
DESIGN_CODE = {"calibration": 1, "population": 2,
               "size_NCT": 3, "size_MCO2": 4, "power_NCT": 5, "power_MCO2": 6}

QUICK = dict(size_reps=20, reps=20, draws=999, cal_windows=100, pop_windows=400,
             targets=(0.0, 0.03, 0.06, 0.10))


def rep_id(design: str, step: int, rep: int) -> int:
    if not (0 <= step < 1000 and 0 <= rep < 100_000):
        raise ValueError("step or rep out of the range the seed scheme reserves")
    return DESIGN_CODE[design] * 10**8 + step * 10**5 + rep


def generator(design: str, step: int, rep: int) -> np.random.Generator:
    return np.random.default_rng([DATA_STREAM, rep_id(design, step, rep)])


def rate(hits: np.ndarray) -> tuple[float, float]:
    """Rejection rate and its binomial standard error."""
    r = float(np.mean(hits))
    return r, float(np.sqrt(r * (1 - r) / len(hits)))


# --- tasks run in worker processes ----------------------------------------

def series_widths(task) -> np.ndarray:
    """Widths of the non-overlapping windows of one simulated series."""
    design, step, rep, lambda2 = task
    x = mrw(SERIES_WINDOWS * WINDOW, lambda2, generator(design, step, rep))
    return trades.windows(x).width_pos.to_numpy()


def overlapping_widths(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, len(x) - WINDOW + 1, STEP)
    widths = np.array([(positive_spectrum(x[s:s + WINDOW], S_MAX_DIV) or {"width_pos": np.nan})
                       ["width_pos"] for s in starts])
    ok = np.isfinite(widths)
    return starts[ok], widths[ok]


def size_replication(task) -> dict:
    token, n_base, n_other, lambda2, rep, draws = task
    design = f"size_{token}"
    rng = generator(design, 0, rep)
    sa, wa = overlapping_widths(mrw(n_base, lambda2, rng))
    sb, wb = overlapping_widths(mrw(n_other, lambda2, rng))

    y = np.concatenate([wa, wb])
    X = np.column_stack([np.ones(len(y)), np.r_[np.zeros(len(wa)), np.ones(len(wb))]])
    nw_rule = int(4 * (len(y) / 100) ** (2 / 9))
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    hac_rule = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_rule})

    na, nb = wa[sa % WINDOW == 0], wb[sb % WINDOW == 0]
    perm = stats.permutation_test(na, nb, draws=draws, seed=rep_id(design, 0, rep))
    return {"design": token, "rep": rep, "seed": rep_id(design, 0, rep),
            "overlap_windows_base": len(wa), "overlap_windows_other": len(wb),
            "hac_coef": hac.params[1], "hac_p": hac.pvalues[1],
            "hac_rule_lags": nw_rule, "hac_rule_p": hac_rule.pvalues[1],
            "windows_base": len(na), "windows_other": len(nb),
            "perm_gap": perm["gap"], "perm_p": perm["p"]}


def power_replication(task) -> dict:
    token, step, n_base, n_other, lam_base, lam_alt, rep, draws = task
    design = f"power_{token}"
    rng = generator(design, step, rep)
    wa = trades.windows(mrw(n_base, lam_base, rng)).width_pos.to_numpy()
    wb = trades.windows(mrw(n_other, lam_alt, rng)).width_pos.to_numpy()
    perm = stats.permutation_test(wa, wb, draws=draws, seed=rep_id(design, step, rep))
    return {"design": token, "step": step, "rep": rep, "gap": perm["gap"],
            "p": perm["p"], "windows_base": len(wa), "windows_other": len(wb)}


# --- steps ----------------------------------------------------------------

def empirical_inputs(args) -> tuple[dict, float | None]:
    """Return counts per token and BCT's median window width, from data unless given."""
    given = {"BCT": args.n_bct, "NCT": args.n_nct, "MCO2": args.n_mco2}
    needed = [BASE, *args.designs]
    counts = {t: given[t] for t in needed if given[t] is not None}
    target = args.target_width
    load = [t for t in needed if t not in counts]
    if target is None and args.lambda2 is None and BASE not in load:
        load.append(BASE)
    if load:
        require(*load)
    for t in load:
        r = trades.returns(trades.trades(TOKENS[t]))
        counts.setdefault(t, len(r))
        if t == BASE and target is None:
            target = float(trades.windows(r.ret.to_numpy()).width_pos.median())
    return counts, target


def simulate_widths(pool, design: str, lambdas: list[float], windows: int) -> list[np.ndarray]:
    """For each lambda2, the widths of `windows` windows grouped by series."""
    n_series = int(np.ceil(windows / SERIES_WINDOWS))
    tasks = [(design, i, k, lam) for i, lam in enumerate(lambdas) for k in range(n_series)]
    out = list(pool.map(series_widths, tasks, chunksize=1))
    return [out[i * n_series:(i + 1) * n_series] for i in range(len(lambdas))]


def invert(curve_lambda: np.ndarray, curve_width: np.ndarray, width: float) -> float:
    if width < curve_width[0] - 1e-12 or width > curve_width[-1] + 1e-12:
        raise SystemExit(f"median width {width:.3f} is outside the calibrated range "
                         f"[{curve_width[0]:.3f}, {curve_width[-1]:.3f}]; extend LAMBDA2_GRID")
    return float(np.interp(width, curve_width, curve_lambda))


def bootstrap_median_se(groups: list[np.ndarray], seed: int) -> float:
    rng = np.random.default_rng(seed)
    meds = [np.median(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]))
            for _ in range(BOOTSTRAP)]
    return float(np.std(meds, ddof=1))


def crossing(gaps: np.ndarray, power: np.ndarray, level: float) -> float | None:
    """True gap at which power first exceeds `level`, linear between steps."""
    above = np.flatnonzero(power > level)
    if not len(above):
        return None
    k = above[0]
    if k == 0:
        return float(gaps[0])
    g0, g1, p0, p1 = gaps[k - 1], gaps[k], power[k - 1], power[k]
    return float(g0 + (level - p0) * (g1 - g0) / (p1 - p0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--designs", nargs="+", default=list(OTHERS), choices=list(OTHERS))
    ap.add_argument("--size-reps", type=int, help=f"size replications per design ({SIZE_REPS})")
    ap.add_argument("--reps", type=int, help=f"power replications per step ({POWER_REPS})")
    ap.add_argument("--draws", type=int, help=f"permutation draws per test ({DRAWS})")
    ap.add_argument("--cal-windows", type=int, help=f"windows per calibration lambda2 ({CAL_WINDOWS})")
    ap.add_argument("--pop-windows", type=int, help=f"windows per population lambda2 ({POP_WINDOWS})")
    ap.add_argument("--n-bct", type=int, help="BCT return count (default: from data)")
    ap.add_argument("--n-nct", type=int, help="NCT return count (default: from data)")
    ap.add_argument("--n-mco2", type=int, help="MCO2 return count (default: from data)")
    ap.add_argument("--target-width", type=float, help="BCT median window width (default: from data)")
    ap.add_argument("--lambda2", type=float, help="base lambda2, bypassing the width target")
    ap.add_argument("--direction", default="auto", choices=("auto", "up", "down"),
                    help="alternatives wider than the base (up) or narrower (down)")
    ap.add_argument("--skip-size", action="store_true")
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--quick", action="store_true", help="small counts, outputs suffixed _quick")
    args = ap.parse_args()
    defaults = {"size_reps": SIZE_REPS, "reps": POWER_REPS, "draws": DRAWS,
                "cal_windows": CAL_WINDOWS, "pop_windows": POP_WINDOWS}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, QUICK[name] if args.quick else value)
    targets = QUICK["targets"] if args.quick else GAP_TARGETS
    suffix = "_quick" if args.quick else ""

    counts, target = empirical_inputs(args)
    out = {"power.window": WINDOW, "power.step_overlap": STEP, "power.hac_lags": HAC_LAGS,
           "power.integral_scale": INTEGRAL_SCALE, "power.draws": args.draws,
           "power.target_width": target,
           **{f"power.n_returns.{t}": n for t, n in counts.items()}}
    print("return counts", counts, " BCT median width", target)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        grid = np.array(LAMBDA2_GRID)
        cal = simulate_widths(pool, "calibration", list(grid), args.cal_windows)
        cal_median = np.array([np.median(np.concatenate(g)) for g in cal])
        monotone = np.maximum.accumulate(cal_median)
        table("power_calibration" + suffix,
              pd.DataFrame({"lambda2": grid, "median_width": cal_median,
                            "monotone_width": monotone,
                            "windows": [sum(map(len, g)) for g in cal]}))
        lam_base = args.lambda2 if args.lambda2 is not None else invert(grid, monotone, target)
        base_width = float(np.interp(lam_base, grid, monotone))
        # A gap is simulated only if the family can produce a width that far
        # from the base in the ladder's direction; the rest are counted and
        # left out.
        def within(w: float) -> bool:
            return monotone[0] - 1e-12 <= w <= monotone[-1] + 1e-12

        direction = args.direction
        if direction == "auto":
            direction = "up" if all(within(base_width + g) for g in targets) else "down"
        sign = 1.0 if direction == "up" else -1.0
        reachable = [g for g in targets if within(base_width + sign * g)]
        out["power.ladder.direction"] = direction
        out["power.ladder.gaps_out_of_reach"] = [g for g in targets if g not in reachable]
        out["power.calibration.widest_simulated"] = float(monotone[-1])
        out["power.calibration.narrowest_simulated"] = float(monotone[0])
        if len(reachable) < len(targets):
            print(f"gaps above {reachable[-1]:g} cannot be simulated {direction}wards: base "
                  f"width {base_width:.3f}, family yields {monotone[0]:.3f} to {monotone[-1]:.3f}")
        targets = tuple(reachable)
        ladder_lambdas = [lam_base if g == 0 else invert(grid, monotone, base_width + sign * g)
                          for g in targets]
        out.update({"power.calibration.lambda2_base": lam_base,
                    "power.calibration.windows_per_lambda2": args.cal_windows})
        print(f"lambda2 base {lam_base:.4f}, ladder {direction}  ({time.time() - t0:.0f}s)")

        pop = simulate_widths(pool, "population", ladder_lambdas, args.pop_windows)
        base_pop = np.median(np.concatenate(pop[0]))
        base_se = bootstrap_median_se(pop[0], seed=0)
        ladder = []
        for k, (g, lam) in enumerate(zip(targets, ladder_lambdas)):
            med = np.median(np.concatenate(pop[k]))
            se = 0.0 if k == 0 else float(np.hypot(base_se, bootstrap_median_se(pop[k], seed=k)))
            ladder.append({"step": k, "nominal_gap": sign * g, "lambda2_base": lam_base,
                           "lambda2_alt": lam, "population_median": med,
                           "true_gap": 0.0 if k == 0 else med - base_pop, "true_gap_se": se})
        ladder = pd.DataFrame(ladder)
        table("power_ladder" + suffix, ladder)
        out["power.population.windows_per_lambda2"] = args.pop_windows
        out["power.population.base_median_width"] = base_pop
        print(ladder.round(4).to_string(index=False), f"({time.time() - t0:.0f}s)")

        if not args.skip_size:
            size_rows = []
            for token in args.designs:
                tasks = [(token, counts[BASE], counts[token], lam_base, rep, args.draws)
                         for rep in range(args.size_reps)]
                rows = pd.DataFrame(pool.map(size_replication, tasks, chunksize=1))
                size_rows.append(rows)
                pre = f"power.size.{token}_vs_{BASE}."
                for name, col in (("hac", "hac_p"), ("hac_rule", "hac_rule_p"),
                                  ("nonoverlap_perm", "perm_p")):
                    r, se = rate(rows[col].to_numpy() < ALPHA)
                    out[pre + name + ".rejection"], out[pre + name + ".se"] = r, se
                out[pre + "reps"] = len(rows)
                out[pre + "hac_rule_lags"] = int(rows.hac_rule_lags.iloc[0])
                out[pre + "overlap_windows"] = [int(rows.overlap_windows_base.iloc[0]),
                                                int(rows.overlap_windows_other.iloc[0])]
                out[pre + "nonoverlap_windows"] = [int(rows.windows_base.iloc[0]),
                                                   int(rows.windows_other.iloc[0])]
                print(f"size {token} vs {BASE}: HAC({HAC_LAGS}) "
                      f"{out[pre + 'hac.rejection']:.3f}  HAC(rule) "
                      f"{out[pre + 'hac_rule.rejection']:.3f}  permutation "
                      f"{out[pre + 'nonoverlap_perm.rejection']:.3f}  "
                      f"({len(rows)} reps, {time.time() - t0:.0f}s)")
            table("power_size" + suffix, pd.concat(size_rows, ignore_index=True))

        if not args.skip_power:
            curve = []
            for token in args.designs:
                tasks = [(token, int(r.step), counts[BASE], counts[token], lam_base,
                          r.lambda2_alt, rep, args.draws)
                         for r in ladder.itertuples() for rep in range(args.reps)]
                reps = pd.DataFrame(pool.map(power_replication, tasks, chunksize=1))
                for r in ladder.itertuples():
                    g = reps[reps.step == r.step]
                    power, se = rate(g.p.to_numpy() < ALPHA)
                    curve.append({"design": f"{token} vs {BASE}", "step": r.step,
                                  "nominal_gap": r.nominal_gap, "lambda2_alt": r.lambda2_alt,
                                  "true_gap": r.true_gap, "true_gap_se": r.true_gap_se,
                                  "reps": len(g), "rejections": int((g.p < ALPHA).sum()),
                                  "power": power, "power_se": se,
                                  "gap_mean": g.gap.mean(), "gap_sd": g.gap.std(),
                                  "gap_q025": g.gap.quantile(0.025),
                                  "gap_q975": g.gap.quantile(0.975),
                                  "windows_base": int(g.windows_base.iloc[0]),
                                  "windows_other": int(g.windows_other.iloc[0])})
                c = pd.DataFrame([row for row in curve if row["design"] == f"{token} vs {BASE}"])
                pre = f"power.curve.{token}_vs_{BASE}."
                out[pre + "size"], out[pre + "size_se"] = c.power.iloc[0], c.power_se.iloc[0]
                out[pre + "null_gap_95"] = [c.gap_q025.iloc[0], c.gap_q975.iloc[0]]
                out[pre + "reps"] = args.reps
                out[pre + "windows"] = [int(c.windows_base.iloc[0]), int(c.windows_other.iloc[0])]
                for row in c.itertuples():
                    out[pre + f"step{row.step}.true_gap"] = row.true_gap
                    out[pre + f"step{row.step}.power"] = row.power
                    out[pre + f"step{row.step}.power_se"] = row.power_se
                for level, name in zip(POWER_LEVELS, ("one_third", "half", "0.8")):
                    out[pre + f"gap_at_power_{name}"] = crossing(
                        np.abs(c.true_gap.to_numpy()), c.power.to_numpy(), level)
                print(c[["step", "true_gap", "power", "power_se", "windows_base",
                         "windows_other"]].round(4).to_string(index=False),
                      f"({time.time() - t0:.0f}s)")
            table("power_curve" + suffix, pd.DataFrame(curve))

    out["power.seconds"] = time.time() - t0
    print("wrote", numbers("power" + suffix, out))


if __name__ == "__main__":
    main()
