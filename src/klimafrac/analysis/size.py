"""
Width against pool idleness and against the asset's size, across the
conventional tokens of analysis/thin_benchmark.py.

Idleness is a token's share of hours with no trade. Size is the asset's
rank by market value on 3 October 2021 (fetch/token_size.py), a larger rank
being a smaller asset; unlisted tokens take the rank after the last ranked
asset, tied, and the wrapped coins the rank of the coin they wrap. The
carbon tokens are reported but enter no test.

Across tokens, on ranks: the Spearman correlations of width with idleness
and with smallness, each partial on the other, with permutation p-values on
the residuals. Checks repeat the test with one thing changed: size by
24-hour volume; the ranking of 13 February 2022; market value at the end of
the estimation sample from a second source; listed tokens only; WETH
counted once; TEL recoded as the asset its contract resolves to. Median
width is also given in the four cells of thin or busy pool by asset ranked
in the first hundred or not.

Outputs (data/results/): size_tokens.csv, size_tests.csv, size.json.

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
