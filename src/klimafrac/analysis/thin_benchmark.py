"""
Carbon tokens against conventional tokens that trade as thinly as they do.

The asset-class comparison (analysis/asset_class.py) sets the carbon tokens
against WETH and WBTC, the most active pools on the same exchanges, so a wider
carbon spectrum cannot be told apart from a wider thin-market spectrum. This
module adds the group that separates the two: the conventional tokens that
fetch/thin_candidates.py selects for being as thin as the carbon pools, by a
rule that reads only depth and idleness from the chain.

Three groups, every series rebuilt by the same path (trades.trades, the
hourly grid of trades.calendar, non-overlapping windows of trades.WINDOW
hourly returns, the positive-side width of mfdfa.positive_spectrum):

  carbon   BCT, NCT, MCO2, KLIMA
  thin     the pools registered with role "thin" in fetch/pools.py
  deep     WETH on Sushiswap, WETH and WBTC on Quickswap
  active   the pools registered with role "active": conventional tokens that
           trade more often than any carbon pool, chosen by the mirror image
           of the rule that chose the thin ones

Plan, fixed before any thin pool was analysed.

  Unit. The token. A token's value is the median width of its windows.

  Contrasts.
    asset     carbon against thin: the asset differs, thinness is matched
    thinness  thin against deep: thinness differs, both conventional
    both      carbon against deep: the comparison the paper already reports
    activity  thin against active: tokens of the same kind, traded rarely or
              often
    busy      thin against deep and active together
    majors    deep against active: whether WETH and WBTC differ from other
              tokens that trade often
  The active group and the last three contrasts were added after the first
  three had been run, and fixed before any active pool was analysed.

  Tests, for each contrast.
    token test    asset_class.token_test: whole tokens, windows attached, are
                  reassigned between the two groups in every possible way and
                  the difference of group median window widths is recomputed;
                  exact, two-sided on the absolute difference. Where the
                  reassignments exceed EXACT_LIMIT they are drawn at random
                  instead, and p = (hits + 1) / (draws + 1).
    rank test     the Mann-Whitney statistic on the token values, with its
                  exact null distribution, the Hodges-Lehmann shift (the
                  median of all pairwise differences between tokens of the two
                  groups) and the exact confidence interval that inverts the
                  test, at the attainable level nearest above 95%.
    windows       stats.permutation_test on the pooled windows, reported as
                  description: windows of one token are not independent.

  Dose. Across the conventional tokens, thin, deep and active together, the Spearman
  correlation between a token's share of hours with no trade and its value,
  with a permutation p-value. If thinness drives the width, it is positive.

Outputs (via klimafrac.results):
  data/results/thin_benchmark_windows.csv   one row per window of every token
  data/results/thin_benchmark_tokens.csv    one row per token
  data/results/thin_benchmark_tests.csv     one row per contrast
  data/results/thin_benchmark.json          the numbers for the manuscript

Usage:
    uv run python -m klimafrac.analysis.thin_benchmark
"""

import os
from math import comb

import numpy as np
import pandas as pd

from klimafrac import stats, trades
from klimafrac.analysis.asset_class import CARBON, RECONSTRUCTED, median_gap, token_test
from klimafrac.fetch.pools import POOLS
from klimafrac.paths import data
from klimafrac.results import numbers, table

DEEP = RECONSTRUCTED
THIN = {p.symbol: p.key for p in POOLS if p.role == "thin"}
ACTIVE = {p.symbol: p.key for p in POOLS if p.role == "active"}
GROUPS = {"carbon": CARBON, "thin": THIN, "deep": DEEP, "active": ACTIVE}
SIDES = {**{g: (g,) for g in GROUPS}, "busy": ("deep", "active")}   # a side is one or more groups
CONTRASTS = {"asset": ("thin", "carbon"), "thinness": ("deep", "thin"),
             "both": ("deep", "carbon"), "activity": ("active", "thin"),
             "busy": ("busy", "thin"),
             "majors": ("active", "deep")}     # (first, second): gap is second minus first
LEVEL = 0.95
DRAWS = stats.DRAWS
SEED = stats.SEED
EXACT_LIMIT = 200_000                          # reassignments enumerated; beyond it they are drawn


def side(name: str, available=None) -> list[str]:
    """The tokens on one side of a contrast, optionally only those available."""
    tokens = [t for g in SIDES[name] for t in GROUPS[g]]
    return tokens if available is None else [t for t in tokens if t in available]


def reassignment_test(values: dict, first: list, second: list, draws: int = DRAWS,
                      seed: int = SEED) -> dict:
    """asset_class.token_test, by enumeration where feasible and by random draws otherwise."""
    if comb(len(first) + len(second), len(second)) <= EXACT_LIMIT:
        return {**token_test(values, first, second), "exact": True}
    pool = first + second
    observed = median_gap(values, first, second)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws):
        chosen = set(rng.choice(len(pool), size=len(second), replace=False).tolist())
        gap = median_gap(values, [t for i, t in enumerate(pool) if i not in chosen],
                         [t for i, t in enumerate(pool) if i in chosen])
        hits += abs(gap) >= abs(observed) - 1e-12
    return {"tokens_a": len(first), "tokens_b": len(second),
            "assignments": comb(len(pool), len(second)), "gap": observed,
            "p_two_sided": (hits + 1) / (draws + 1),
            "min_attainable_p_two_sided": 1 / (draws + 1), "exact": False, "draws": draws}


def require() -> None:
    missing = [f"swaps_{key}.csv.gz" for tokens in GROUPS.values() for key in tokens.values()
               if not os.path.exists(data(f"swaps_{key}.csv.gz"))]
    if missing:
        raise SystemExit(f"missing input(s) in {data('')}: {', '.join(missing)}. "
                         "Build them with klimafrac.fetch.swaps first.")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every window of every token, and one row of facts per token."""
    windows, facts = [], []
    for group, tokens in GROUPS.items():
        for token, key in tokens.items():
            d = trades.trades(key)
            grid = trades.calendar(d, freq="1h")
            if grid.ret.isna().any():
                raise ValueError(f"{token}: hourly grid has undefined returns")
            w = trades.windows(grid.ret.to_numpy(), stamps=pd.Series(grid.index))
            flag = grid.no_trade.to_numpy()
            w["no_trade"] = [flag[s:s + trades.WINDOW].mean() for s in w.start]
            w.insert(0, "token", token)
            w.insert(1, "group", group)
            windows.append(w)
            facts.append({"token": token, "group": group, "pool": key, "trades": len(d),
                          "hours": len(grid), "first_hour": grid.index[0],
                          "no_trade": float(flag.mean()),
                          "windows_possible": len(grid) // trades.WINDOW, "windows": len(w),
                          "width_median": float(w.width_pos.median()) if len(w) else np.nan,
                          "width_min": float(w.width_pos.min()) if len(w) else np.nan,
                          "width_max": float(w.width_pos.max()) if len(w) else np.nan})
            print(f"  {group:6s} {token:9s} {len(d):8,} trades  {len(w):2d} windows  "
                  f"no-trade {100 * flag.mean():5.1f}%  median width {facts[-1]['width_median']:.3f}")
    return pd.concat(windows, ignore_index=True), pd.DataFrame(facts)


def u_null(m: int, n: int) -> np.ndarray:
    """Exact null distribution of the Mann-Whitney U of samples of sizes m and n."""
    counts = np.zeros((m + 1, n + 1, m * n + 1))
    counts[0, :, 0] = 1
    counts[:, 0, 0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # the largest value is in the first sample (it beats all j of the
            # second) or in the second (it adds nothing)
            counts[i, j, j:] += counts[i - 1, j, :m * n + 1 - j]
            counts[i, j] += counts[i, j - 1]
    return counts[m, n] / counts[m, n].sum()


def rank_test(first: np.ndarray, second: np.ndarray, level: float = LEVEL) -> dict:
    """Mann-Whitney on token values, with the shift and the interval it implies."""
    m, n = len(second), len(first)
    diffs = np.sort((second[:, None] - first[None, :]).ravel())
    if len(np.unique(np.concatenate([first, second]))) < m + n:
        raise ValueError("tied token values: the exact null distribution does not apply")
    u = int((diffs > 0).sum())
    null = u_null(m, n)
    tail = min(null[:u + 1].sum(), null[u:].sum())
    cdf = np.cumsum(null)
    # the widest k for which P(U < k) <= (1 - level) / 2: the interval runs
    # from the k-th smallest pairwise difference to the k-th largest
    k = int(np.searchsorted(cdf, (1 - level) / 2, side="right"))
    out = {"tokens_second": m, "tokens_first": n, "u": u, "u_max": m * n,
           "p_two_sided": float(min(1.0, 2 * tail)),
           "min_attainable_p_two_sided": float(2 * null[0]),
           "shift": float(np.median(diffs))}
    if k >= 1:
        out.update({"ci_low": float(diffs[k - 1]), "ci_high": float(diffs[m * n - k]),
                    "ci_level": float(1 - 2 * cdf[k - 1])})
    else:                      # too few tokens for any interval at this level
        out.update({"ci_low": float(diffs[0]), "ci_high": float(diffs[-1]),
                    "ci_level": float(1 - 2 * null[0])})
    return out


def spearman(x: np.ndarray, y: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> dict:
    rank = lambda v: pd.Series(v).rank().to_numpy()            # noqa: E731
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    hits = sum(abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(rho) - 1e-12
               for _ in range(draws))
    return {"tokens": len(x), "rho": rho, "p": (hits + 1) / (draws + 1), "draws": draws}


def contrasts(win: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    values = {t: g.width_pos.to_numpy() for t, g in win.groupby("token")}
    value = facts.set_index("token").width_median
    rows = []
    for name, (a, b) in CONTRASTS.items():
        first, second = side(a), side(b)
        token = reassignment_test(values, first, second)
        rank = rank_test(value[first].to_numpy(), value[second].to_numpy())
        pooled = stats.permutation_test(np.concatenate([values[t] for t in first]),
                                        np.concatenate([values[t] for t in second]))
        rows.append({"contrast": name, "first": a, "second": b,
                     "tokens_first": len(first), "tokens_second": len(second),
                     "median_first": float(np.median(np.concatenate([values[t] for t in first]))),
                     "median_second": float(np.median(np.concatenate([values[t] for t in second]))),
                     "gap": token["gap"], "token_p": token["p_two_sided"],
                     "token_min_p": token["min_attainable_p_two_sided"],
                     "token_assignments": token["assignments"], "token_exact": token["exact"],
                     "second_above_every_first": bool(value[second].min() > value[first].max()),
                     "tokens_of_second_above_first_median":
                         int((value[second] > value[first].median()).sum()),
                     **{f"rank_{k}": v for k, v in rank.items()},
                     "windows_first": int(sum(len(values[t]) for t in first)),
                     "windows_second": int(sum(len(values[t]) for t in second)),
                     "window_p": pooled["p"]})
    return pd.DataFrame(rows)


def main() -> None:
    require()
    print("windows:")
    win, facts = load()
    if facts.windows.eq(0).any():
        raise SystemExit(f"no window for {list(facts[facts.windows == 0].token)}")
    tests = contrasts(win, facts)
    conventional = facts[facts.group.isin(("thin", "deep", "active"))]
    dose = {"conventional": spearman(conventional.no_trade.to_numpy(),
                                     conventional.width_median.to_numpy()),
            "thin": spearman(facts[facts.group == "thin"].no_trade.to_numpy(),
                             facts[facts.group == "thin"].width_median.to_numpy()),
            "all": spearman(facts.no_trade.to_numpy(), facts.width_median.to_numpy())}

    out = {"thin_benchmark.window": trades.WINDOW}
    for group, g in facts.groupby("group"):
        out[f"thin_benchmark.{group}.tokens"] = len(g)
        out[f"thin_benchmark.{group}.windows"] = int(g.windows.sum())
        out[f"thin_benchmark.{group}.token_median_min"] = float(g.width_median.min())
        out[f"thin_benchmark.{group}.token_median_median"] = float(g.width_median.median())
        out[f"thin_benchmark.{group}.token_median_max"] = float(g.width_median.max())
        out[f"thin_benchmark.{group}.no_trade_min"] = float(g.no_trade.min())
        out[f"thin_benchmark.{group}.no_trade_median"] = float(g.no_trade.median())
        out[f"thin_benchmark.{group}.no_trade_max"] = float(g.no_trade.max())
        out[f"thin_benchmark.{group}.window_median"] = float(
            win[win.group == group].width_pos.median())
    for row in tests.to_dict("records"):
        for k, v in row.items():
            if k not in ("contrast", "first", "second"):
                out[f"thin_benchmark.{row['contrast']}.{k}"] = v
    for name, d in dose.items():
        for k, v in d.items():
            out[f"thin_benchmark.dose.{name}.{k}"] = v

    table("thin_benchmark_windows", win)
    table("thin_benchmark_tokens", facts)
    table("thin_benchmark_tests", tests)
    print("\nwrote", numbers("thin_benchmark", out))
    print("\ngroups (token medians):")
    print(facts.groupby("group").agg(tokens=("token", "size"), windows=("windows", "sum"),
                                     no_trade=("no_trade", "median"),
                                     lowest=("width_median", "min"),
                                     median=("width_median", "median"),
                                     highest=("width_median", "max")).round(3).to_string())
    print("\ncontrasts (gap = second minus first):")
    show = ["contrast", "tokens_first", "tokens_second", "gap", "token_p", "rank_p_two_sided",
            "rank_shift", "rank_ci_low", "rank_ci_high", "rank_ci_level", "window_p"]
    print(tests[show].round(4).to_string(index=False))
    for name, d in dose.items():
        print(f"dose, {name}: rho {d['rho']:+.3f}, p {d['p']:.4f}, {d['tokens']} tokens")


if __name__ == "__main__":
    main()
