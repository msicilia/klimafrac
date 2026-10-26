"""
Thin pool or small asset? Width against idleness and against the asset's size.

analysis/thin_benchmark.py finds thin conventional tokens wider than busy
ones, but the narrowest busy pools hold the largest assets, so a pool's
idleness and its asset's size are entangled there. This module sets the two
side by side. Idleness is a property of the pool: the share of hours with no
trade. Size is a property of the asset wherever it trades, and is read from
outside the chain by fetch/token_size.py.

Plan, fixed before any token's size was set against its width.

  Tokens. The conventional tokens of thin_benchmark: thin, deep and active.
  The carbon tokens are reported beside them and enter no test.

  Size. The asset's rank by market value in the public ranking of 3 October
  2021, the last weekly snapshot before the first hourly series begins, so
  that size is measured before the trading it is set against. A larger rank
  number is a smaller asset. A token matched to no ranked asset takes the
  rank after the last ranked one, all such tokens tied. WETH, WBTC and WPOL
  take the rank of the coin they wrap.

  Test. No cut-off divides large from small. On ranks, across tokens:
    rho(width, idleness)            and its partial holding size fixed
    rho(width, smallness)           and its partial holding idleness fixed
    rho(idleness, smallness)        how entangled the two are
  Partials are correlations of rank residuals; p-values come from permuting
  the residuals of the variable of interest, two-sided.

  Checks, each repeating the test with one thing changed:
    volume      size by rank of worldwide 24-hour volume in the same ranking
    check       size by rank in the ranking of 13 February 2022, the first
                snapshot after the date by which the candidate pools had to
                exist, four months into the sample
    late        size by market value on the last day of the estimation
                sample, from a second, documented source; later than the
                trading it is set against, so a check and not the measure.
                A token the source does not value counts as the smallest.
    listed      only tokens matched to a ranked asset
    weth_once   without the second WETH pool
    mismatch_recoded
                tokens whose ranked asset the ranking's record misplaces
                (fetch/token_size.KNOWN_MISMATCH) coded as that asset; one
                token, TEL. A sensitivity, not part of the plan

  Description. Median width in the four cells of thin or busy pool by asset
  ranked in the first hundred or not. The cut is the conventional one and
  carries no test.

Outputs (via klimafrac.results):
  data/results/size_tokens.csv   one row per token: group, idleness, width, sizes
  data/results/size_tests.csv    one row per variant
  data/results/size.json         the numbers for the manuscript

Usage:
    uv run python -m klimafrac.analysis.size
"""

import numpy as np
import pandas as pd

from klimafrac import stats
from klimafrac.paths import data
from klimafrac.results import numbers, read_table, table

CONVENTIONAL = ("thin", "deep", "active")
BUSY = ("deep", "active")
LARGE_RANK = 100
DRAWS = stats.DRAWS
SEED = stats.SEED


def ranks(v) -> np.ndarray:
    return pd.Series(np.asarray(v, float)).rank().to_numpy()


def residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a with its linear dependence on b removed."""
    slope, intercept = np.polyfit(b, a, 1)
    return a - (slope * b + intercept)


def correlation(x: np.ndarray, y: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> tuple:
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    hits = sum(abs(np.corrcoef(rng.permutation(x), y)[0, 1]) >= abs(r) - 1e-12
               for _ in range(draws))
    return r, (hits + 1) / (draws + 1)


def test(width, idle, small, label: str) -> dict:
    w, i, s = ranks(width), ranks(idle), ranks(small)
    out = {"variant": label, "tokens": len(w)}
    for name, (a, b) in {"width_idle": (w, i), "width_small": (w, s), "idle_small": (i, s)}.items():
        out[f"rho_{name}"], out[f"p_{name}"] = correlation(a, b)
    out["partial_width_idle"], out["p_partial_width_idle"] = correlation(
        residual(i, s), residual(w, s))
    out["partial_width_small"], out["p_partial_width_small"] = correlation(
        residual(s, i), residual(w, i))
    return out


def main() -> None:
    facts = read_table("thin_benchmark_tokens")
    size = pd.read_csv(data("token_size.csv"))
    t = facts.merge(size, on="token", how="left", validate="one_to_one")
    if t.snapshot_rank_filled.isna().any():
        raise SystemExit(f"no size for {list(t[t.snapshot_rank_filled.isna()].token)}: "
                         "run klimafrac.fetch.token_size")
    conv = t[t.group.isin(CONVENTIONAL)].reset_index(drop=True)
    late = conv.late_market_cap.fillna(0.0)
    variants = [test(conv.width_median, conv.no_trade, conv.snapshot_rank_filled, "main")]
    variants.append(test(conv.width_median, conv.no_trade, -conv.snapshot_volume.fillna(0.0), "volume"))
    variants.append(test(conv.width_median, conv.no_trade, conv.check_rank_filled, "check"))
    variants.append(test(conv.width_median, conv.no_trade, -late, "late"))
    listed = conv[conv.snapshot_rank.notna()]
    variants.append(test(listed.width_median, listed.no_trade, listed.snapshot_rank, "listed"))
    once = conv[conv.token != "WETH_Q"]
    variants.append(test(once.width_median, once.no_trade, once.snapshot_rank_filled, "weth_once"))
    # Tokens the ranking's own records misplace, coded as the asset the second
    # source resolves their contract to; the rule stands, this reports what it
    # costs.
    recoded = conv.snapshot_rank_filled.where(conv.sensitivity_rank.isna(), conv.sensitivity_rank)
    variants.append(test(conv.width_median, conv.no_trade, recoded, "mismatch_recoded"))
    tests = pd.DataFrame(variants)
    out_extra = {"size.mismatch_recoded.tokens": list(conv.token[conv.sensitivity_rank.notna()])}

    conv = conv.assign(pool=np.where(conv.group.isin(BUSY), "busy", "thin"),
                       asset=np.where(conv.snapshot_rank_filled <= LARGE_RANK, "large", "small"))
    cells = (conv.groupby(["asset", "pool"]).agg(tokens=("token", "size"),
                                                 width=("width_median", "median"),
                                                 idle=("no_trade", "median")).reset_index())

    out = {"size.large_rank": LARGE_RANK, "size.tokens": len(conv),
           "size.unlisted": int(conv.snapshot_rank.isna().sum()),
           "size.unlisted_check": int(conv.check_rank.isna().sum()), **out_extra}
    for row in tests.to_dict("records"):
        for k, v in row.items():
            if k != "variant":
                out[f"size.{row['variant']}.{k}"] = v
    for row in cells.to_dict("records"):
        pre = f"size.cell.{row['asset']}_{row['pool']}."
        out[pre + "tokens"], out[pre + "width"], out[pre + "idle"] = (
            row["tokens"], row["width"], row["idle"])
    carbon = t[t.group == "carbon"]
    for row in carbon.itertuples():
        out[f"size.carbon.{row.token}.rank"] = row.snapshot_rank
        out[f"size.carbon.{row.token}.check_rank"] = row.check_rank
    table("size_tokens", t[["token", "group", "no_trade", "width_median", "snapshot_rank",
                            "snapshot_rank_filled", "snapshot_volume", "check_rank",
                            "check_rank_filled", "late_market_cap", "matched_by",
                            "sensitivity_rank"]])
    table("size_tests", tests)
    print("wrote", numbers("size", out))
    pd.set_option("display.width", 220)
    print(tests.round(3).to_string(index=False))
    print(cells.round(3).to_string(index=False))
    print("carbon ranks:", {r.token: r.snapshot_rank for r in carbon.itertuples()})


if __name__ == "__main__":
    main()
