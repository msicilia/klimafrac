"""
The benchmark comparison of analysis/thin_benchmark.py repeated under one
family of variants at a time, the rest at the paper's values: the paper's
Tables tab:benchmark_cleaning and tab:benchmark_variants.

  baseline   the paper's choices; must reproduce thin_benchmark.json
  deep       the deep group without its Quickswap WETH pool, and without
             its Sushiswap one
  cleaning   the 32 settings of analysis/tiers.py: dollar floor
             {0, 0.10, 1, 10}, base floor {0, 1e-4}, median filter
             {2, 3, 5, off}
  floor      the dollar floor alone: 0.25, 0.5, 2, 3, 5, 7.5 and 20
  endpoint   the whole data window, 2026-01-01 and 2026-04-01
  window     2,000 and 8,000 returns per window
  s_min      16 and 20;  s_max  N/5 and N/20;  scales  10 and 40
  order      detrending order 2 and 3
  q          [0.5, 3], [0.5, 4], [0.5, 5], [1, 5], [0.5, 6], less the baseline
  fit        windows whose fit has r2 >= 0.90, and >= 0.95, at every q
Widths under another q range, window or order are not on the baseline's
scale; the gap's sign and the tests are what is compared.

Outputs (data/results/): thin_robustness.csv, thin_robustness_groups.csv,
thin_robustness.json.

Usage:
    uv run python -m klimafrac.analysis.thin_robustness [--family cleaning]
"""

import argparse
import time
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import pandas as pd

from klimafrac import stats, trades
from klimafrac.analysis.thin_benchmark import (CONTRASTS, GROUPS, SIDES, rank_test,
                                               reassignment_test, spearman)
from klimafrac.analysis.tiers import GRID_MAX_DEVIATION, GRID_MIN_BASE, GRID_MIN_USD
from klimafrac.mfdfa import N_SCALES, POLY_ORDER, QS_POS, S_MIN, fluctuation_functions
from klimafrac.results import numbers, read_numbers, read_table, table

DATA_END = pd.Timestamp("2026-07-26", tz="UTC")
ALPHA = 0.05


@dataclass(frozen=True)
class Spec:
    family: str
    label: str
    min_usd: float = trades.MIN_USD
    min_base: float = trades.MIN_BASE
    max_deviation: float | None = trades.MAX_DEVIATION
    end: pd.Timestamp = trades.SAMPLE_END
    window: int = trades.WINDOW
    s_min: int = S_MIN
    s_max_div: int = trades.S_MAX_DIV
    n_scales: int = N_SCALES
    order: int = POLY_ORDER
    q_low: float = float(QS_POS[0])
    q_high: float = float(QS_POS[-1])
    drop: tuple = ()
    min_r2: float = 0.0

    @property
    def cleaning(self) -> tuple:
        return self.min_usd, self.min_base, self.max_deviation

    @property
    def estimator(self) -> tuple:
        return (self.window, self.s_min, self.s_max_div, self.n_scales, self.order,
                self.q_low, self.q_high)


BASE = Spec("baseline", "the paper's choices")


def specs() -> list[Spec]:
    out = [BASE,
           replace(BASE, family="deep", label="deep group without WETH on Quickswap",
                   drop=("WETH_Q",)),
           replace(BASE, family="deep", label="deep group without WETH on Sushiswap",
                   drop=("WETH_S",))]
    for usd, base, dev in product(GRID_MIN_USD, GRID_MIN_BASE, GRID_MAX_DEVIATION):
        if (usd, base, dev) != BASE.cleaning:
            out.append(replace(BASE, family="cleaning", min_usd=usd, min_base=base,
                               max_deviation=dev,
                               label=f"min_usd={usd:g} min_base={base:g} max_dev={dev or 'off'}"))
    out += [replace(BASE, family="floor", label=f"min_usd={usd:g}", min_usd=usd)
            for usd in (0.25, 0.5, 2.0, 3.0, 5.0, 7.5, 20.0)]
    out += [replace(BASE, family="endpoint", label="whole data window", end=DATA_END),
            replace(BASE, family="endpoint", label="to 2026-01-01",
                    end=pd.Timestamp("2026-01-01", tz="UTC")),
            replace(BASE, family="endpoint", label="to 2026-04-01",
                    end=pd.Timestamp("2026-04-01", tz="UTC"))]
    out += [replace(BASE, family="window", label=f"{n:,} returns per window", window=n)
            for n in (2000, 8000)]
    out += [replace(BASE, family="s_min", label=f"s_min={s}", s_min=s) for s in (16, 20)]
    out += [replace(BASE, family="s_max", label=f"s_max=N/{d}", s_max_div=d) for d in (5, 20)]
    out += [replace(BASE, family="scales", label=f"{n} scales", n_scales=n) for n in (10, 40)]
    out += [replace(BASE, family="order", label=f"detrending order {k}", order=k) for k in (2, 3)]
    out += [replace(BASE, family="q", label=f"q in [{a:g}, {b:g}]", q_low=a, q_high=b)
            for a, b in ((0.5, 4.0), (1.0, 5.0), (0.5, 3.0), (0.5, 5.0), (0.5, 6.0))
            if (a, b) != (BASE.q_low, BASE.q_high)]
    out += [replace(BASE, family="fit", label=f"windows with every r2 >= {r:g}", min_r2=r)
            for r in (0.90, 0.95)]
    return out


def load_raw() -> dict:
    """Every pool's swaps, only the columns cleaning and the hourly grid need."""
    return {token: trades.load(key)[["timestamp", "base_amt", "quote_amt", "price"]]
            for tokens in GROUPS.values() for token, key in tokens.items()}


def width(x: np.ndarray, spec: Spec) -> float:
    """The window's width, or NaN where it has none or its fits fall short of spec.min_r2."""
    qs = np.arange(spec.q_low, spec.q_high + 0.25, 0.5)
    s, f = fluctuation_functions(x, s_min=spec.s_min, s_max=len(x) // spec.s_max_div, qs=qs,
                                 order=spec.order, n_scales=spec.n_scales)
    if s is None:
        return np.nan
    logs, logf = np.log(s), np.log(f)
    h = np.array([np.polyfit(logs, logf[:, j], 1)[0] for j in range(len(qs))])
    if spec.min_r2 > 0:
        fitted = np.outer(logs - logs.mean(), h) + logf.mean(axis=0)
        r2 = 1 - ((logf - fitted) ** 2).sum(axis=0) / ((logf - logf.mean(axis=0)) ** 2).sum(axis=0)
        if r2.min() < spec.min_r2:
            return np.nan
    alpha = np.gradient(qs * h - 1.0, qs)
    return float(alpha.max() - alpha.min())


def grids(raw: dict, keys: dict, spec: Spec) -> dict:
    """Hourly returns of each token under the spec's cleaning and endpoint.

    Cleaning runs over the full data window and the endpoint is applied
    afterwards, in that order, as trades.trades does; the rolling-median
    filter therefore sees the same neighbours whatever the endpoint.
    """
    out = {}
    for token, df in raw.items():
        d = trades.clean(df, keys[token], *spec.cleaning)
        d = d[d.timestamp < spec.end].reset_index(drop=True)
        g = trades.calendar(d, freq="1h", end=spec.end)
        out[token] = {"ret": g.ret.to_numpy(), "no_trade": float(g.no_trade.mean()),
                      "raw": int((df.timestamp < spec.end).sum()), "kept": len(d)}
    return out


def evaluate(spec: Spec, series: dict) -> tuple[list[dict], list[dict]]:
    group_of = {t: g for g, tokens in GROUPS.items() for t in tokens if t not in spec.drop}
    values = {}
    for token in group_of:
        x = series[token]["ret"]
        w = np.array([width(x[s:s + spec.window], spec)
                      for s in range(0, len(x) - spec.window + 1, spec.window)])
        values[token] = w[np.isfinite(w)]
    used = {t: v for t, v in values.items() if len(v)}
    value = pd.Series({t: float(np.median(v)) for t, v in used.items()})
    members = {g: [t for t in used if group_of[t] == g] for g in GROUPS}
    common = {"family": spec.family, "spec": spec.label, "min_usd": spec.min_usd,
              "min_base": spec.min_base,
              "max_deviation": np.nan if spec.max_deviation is None else spec.max_deviation,
              "end": spec.end.strftime("%Y-%m-%d"), "window": spec.window, "s_min": spec.s_min,
              "s_max_div": spec.s_max_div, "n_scales": spec.n_scales, "order": spec.order,
              "q_low": spec.q_low, "q_high": spec.q_high, "min_r2": spec.min_r2}

    group_rows = []
    for g, tokens in members.items():
        every = [t for t in group_of if group_of[t] == g]
        removed = pd.Series({t: 1 - series[t]["kept"] / series[t]["raw"] for t in every})
        group_rows.append({
            **common, "group": g, "tokens": len(tokens), "tokens_without_a_window":
            len(every) - len(tokens), "windows": int(sum(len(used[t]) for t in tokens)),
            "trades_raw": int(sum(series[t]["raw"] for t in every)),
            "trades_kept": int(sum(series[t]["kept"] for t in every)),
            "removed_share": 1 - sum(series[t]["kept"] for t in every)
            / sum(series[t]["raw"] for t in every),
            "removed_share_token_min": float(removed.min()),
            "removed_share_token_median": float(removed.median()),
            "removed_share_token_max": float(removed.max()),
            "no_trade_median": float(np.median([series[t]["no_trade"] for t in every])),
            "window_median": float(np.median(np.concatenate([used[t] for t in tokens]))),
            "token_median_min": float(value[tokens].min()),
            "token_median_median": float(value[tokens].median()),
            "token_median_max": float(value[tokens].max())})

    conventional = members["thin"] + members["deep"] + members["active"]
    dose = spearman(np.array([series[t]["no_trade"] for t in conventional]),
                    value[conventional].to_numpy(), draws=2000)
    rows = []
    for name, (a, b) in CONTRASTS.items():
        first, second = ([t for g in SIDES[a] for t in members[g]],
                         [t for g in SIDES[b] for t in members[g]])
        token = reassignment_test(used, first, second)
        try:
            rank = rank_test(value[first].to_numpy(), value[second].to_numpy())
        except ValueError:                      # tied token values
            rank = {}
        pooled = stats.permutation_test(np.concatenate([used[t] for t in first]),
                                        np.concatenate([used[t] for t in second]))
        rows.append({**common, "contrast": name, "tokens_first": len(first),
                     "tokens_second": len(second), "gap": token["gap"],
                     "token_p": token["p_two_sided"],
                     "token_min_p": token["min_attainable_p_two_sided"],
                     "rank_p": rank.get("p_two_sided", np.nan),
                     "shift": rank.get("shift", np.nan), "ci_low": rank.get("ci_low", np.nan),
                     "ci_high": rank.get("ci_high", np.nan),
                     "ci_level": rank.get("ci_level", np.nan),
                     "second_above_every_first":
                         bool(value[second].min() > value[first].max()),
                     "window_p": pooled["p"],
                     "dose_rho_conventional": dose["rho"], "dose_p_conventional": dose["p"]})
    return rows, group_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--family", action="append",
                    help="run only this family (repeatable); the baseline always runs")
    ap.add_argument("--summarise", action="store_true",
                    help="rewrite thin_robustness.json from the stored table, running nothing")
    args = ap.parse_args()
    if args.summarise:
        summarise(read_table("thin_robustness"))
        return
    todo = [s for s in specs() if not args.family or s.family in args.family + ["baseline"]]
    keys = {t: k for tokens in GROUPS.values() for t, k in tokens.items()}
    started = time.time()
    raw = load_raw()
    print(f"loaded {len(raw)} pools, {sum(len(v) for v in raw.values()):,} swaps "
          f"({time.time() - started:.0f} s); {len(todo)} variants", flush=True)

    rows, group_rows, cache = [], [], {}
    for n, spec in enumerate(todo, 1):
        key = (spec.cleaning, spec.end)
        if key not in cache:
            cache = {key: grids(raw, keys, spec)}        # keep only the current grid
        r, g = evaluate(spec, cache[key])
        rows += r
        group_rows += g
        by = {x["contrast"]: x for x in r}
        print(f"[{n}/{len(todo)}] {spec.family:9s} {spec.label:42s} "
              + "  ".join(f"{c} {by[c]['gap']:+.3f} p={by[c]['token_p']:.3f}/"
                          f"{by[c]['rank_p']:.3f}" for c in CONTRASTS)
              + f"  ({time.time() - started:.0f} s)", flush=True)
    result, groups = pd.DataFrame(rows), pd.DataFrame(group_rows)
    suffix = "" if not args.family else "_" + "_".join(args.family)
    table("thin_robustness" + suffix, result)
    table("thin_robustness_groups" + suffix, groups)

    if not args.family:
        summarise(result)
    print(f"runtime {time.time() - started:.0f} s")


def summarise(result: pd.DataFrame) -> None:
    """The counts across variants, and every variant's own numbers, to the json."""
    base = result[result.family == "baseline"].set_index("contrast")
    reference = read_numbers("thin_benchmark")
    for c in CONTRASTS:
        for mine, theirs in (("gap", "gap"), ("token_p", "token_p"),
                             ("rank_p", "rank_p_two_sided")):
            if abs(base.loc[c, mine] - reference[f"thin_benchmark.{c}.{theirs}"]) > 1e-12:
                raise SystemExit(f"baseline {c} {mine} does not reproduce thin_benchmark.json")
    out = {"thin_robustness.variants": int(result.spec.nunique()),
           "thin_robustness.baseline_reproduces_thin_benchmark": True}
    for c, g in result.groupby("contrast"):
        out[f"thin_robustness.{c}.gap_positive"] = int((g.gap > 0).sum())
        out[f"thin_robustness.{c}.gap_min"] = float(g.gap.min())
        out[f"thin_robustness.{c}.gap_max"] = float(g.gap.max())
        out[f"thin_robustness.{c}.token_p_below_alpha"] = int((g.token_p < ALPHA).sum())
        out[f"thin_robustness.{c}.rank_p_below_alpha"] = int((g.rank_p < ALPHA).sum())
        out[f"thin_robustness.{c}.token_p_max"] = float(g.token_p.max())
        out[f"thin_robustness.{c}.rank_p_max"] = float(g.rank_p.max())
        out[f"thin_robustness.{c}.complete_separation"] = int(g.second_above_every_first.sum())
        # variants where either token-level test rejects, and how many of
        # them are dollar floors of 2 and above, both cited by the paper
        sig = g[(g.token_p < ALPHA) | (g.rank_p < ALPHA)]
        out[f"thin_robustness.{c}.significant_variants"] = int(len(sig))
        out[f"thin_robustness.{c}.significant_at_floor_ge_2"] = int((sig.min_usd >= 2).sum())
    for r in result.itertuples():
        pre = f"thin_robustness.{r.family}.{r.spec}.{r.contrast}."
        out[pre + "gap"], out[pre + "token_p"], out[pre + "rank_p"] = (
            float(r.gap), float(r.token_p), float(r.rank_p))
    print("\nwrote", numbers("thin_robustness", out))


if __name__ == "__main__":
    main()
