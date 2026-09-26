"""
How large was each carbon tier's market? Table 1 and the magnitudes of
Sections 3.1 and 3.2.

Swap counts, dollar volume, cleaning shares and the checks behind the $0.10
threshold. Trades come from trades.load and trades.clean over each pool's
full data window; a quantity is restricted to the estimation sample or to a
calendar year only where its name says so.

  TIERS       Table 1: raw and usable swaps, USD volume of the
              usable swaps, first and last usable trade, median usable trade
              size, and whether a stablecoin pair exists. Also the share of
              raw swaps each cleaning stage removes, and the share of usable
              swaps the estimation sample retains.
  YEARLY      raw swaps, usable swaps and USD volume by calendar year, from
              which the BCT 2021, best-year and 2025 comparisons are read.
  THRESHOLDS  the two checks behind the $0.10 threshold, on NCT: price level
              and dispersion by trade-size bucket, and lag-1 autocorrelation
              of trade-to-trade log returns as the size threshold falls.

Design choices
  - Dollar value is trades.usd_value: |quote_amt| for USDC-quoted pools, and
    for UBO and NBO the KLIMA leg at the nearest cleaned KLIMA price. Every
    dollar figure, the UBO and NBO median trade size included, is over usable
    swaps.
  - Cleaning shares follow the order in which trades.clean applies its
    filters and are all shares of raw swaps, so the two stages add up to the
    total. The size stage also drops swaps without a positive price; a swap
    failing both size and deviation filters counts under size.
  - Stable pair is read from data/market_existence.csv
    (fetch/market_existence.py): one row per factory lookup, with columns
    chain, tier, quote and pool (empty when no pool exists). A token has a
    stable pair if any Polygon row has a stablecoin quote and a pool address.
    Without the file the column is left out; with a different schema the
    module stops.
  - Price by size bucket uses NCT swaps in 2025, with a positive price and
    the MIN_BASE floor but no deviation filter, so the comparison is not
    shaped by the filter it is meant to justify. Rows with no filter at all
    are included beside it.
  - Autocorrelation uses NCT over the full data window, where the deviation
    filter is defined. At each threshold every other filter is as in
    trades.clean, recomputed on the swaps surviving that threshold; "no
    filter" is raw swaps with only a positive price, which a log return
    requires. Rows restricted to 2025 after cleaning are included beside it.

Outputs data/results/market_tiers.csv, market_yearly.csv,
market_thresholds.csv and market.json.

Usage:
    uv run python -m klimafrac.analysis.market                 # every token
    uv run python -m klimafrac.analysis.market --token NCT     # prints only
"""

import argparse
import os

import numpy as np
import pandas as pd

from klimafrac import results, trades
from klimafrac.paths import data

TOKENS = ["BCT", "NCT", "MCO2", "KLIMA", "UBO", "NBO"]
STABLES = {"USDC", "USDC.e", "USDT", "DAI"}

THRESHOLD_TOKEN = "NCT"
THRESHOLD_YEAR = 2025
PRICE_BUCKETS = [(0.0, 0.10), (0.10, 1.0), (1.0, 10.0), (10.0, np.inf)]
AUTOCORR_THRESHOLDS = {"10": 10.0, "1": 1.0, "0p10": 0.10}


def pool(token: str) -> str:
    return f"polygon_{token}"


def require(tokens: list[str]) -> None:
    """Stop before any work if a swap file the run needs is missing."""
    needed = {pool(t) for t in tokens}
    if needed & trades.KLIMA_QUOTED:
        needed.add("polygon_KLIMA")
    missing = sorted(data(f"swaps_{k}.csv.gz") for k in needed
                     if not os.path.exists(data(f"swaps_{k}.csv.gz")))
    if missing:
        raise SystemExit("missing swap files (run klimafrac.fetch.swaps):\n  "
                         + "\n  ".join(missing))


def stable_pairs() -> dict[str, bool] | None:
    path = data("market_existence.csv")
    if not os.path.exists(path):
        return None
    census = pd.read_csv(path)
    absent = {"chain", "tier", "quote", "pool"} - set(census.columns)
    if absent:
        raise SystemExit(f"{path} lacks columns {sorted(absent)}")
    stable = census[(census["chain"] == "polygon") & census["quote"].isin(STABLES)
                    & census["pool"].notna()]
    return {t: t in set(stable.tier) for t in TOKENS}


def token_swaps(token: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw and usable swaps, both carrying each swap's dollar value."""
    key = pool(token)
    raw = trades.load(key)
    raw["usd"] = trades.usd_value(raw, key)
    return raw, trades.clean(raw, key)


def tier_row(token: str, raw: pd.DataFrame, usable: pd.DataFrame) -> dict:
    valid = raw.price.notna() & (raw.price > 0)
    size_ok = (valid & (raw.base_amt.abs() >= trades.MIN_BASE)
               & (raw.usd >= trades.MIN_USD))
    removed_size = len(raw) - int(size_ok.sum())
    removed_deviation = int(size_ok.sum()) - len(usable)
    in_sample = len(trades.sample(usable))
    return {
        "token": token,
        "pool": pool(token),
        "raw_swaps": len(raw),
        "usable_swaps": len(usable),
        "usd_volume": float(usable.usd.sum()),
        "median_trade_usd": float(usable.usd.median()),
        "first_trade": usable.timestamp.min().date().isoformat(),
        "last_trade": usable.timestamp.max().date().isoformat(),
        "removed_no_price": int((~valid).sum()),
        "removed_below_min_usd": int((raw.usd < trades.MIN_USD).sum()),
        "removed_below_min_base": int((raw.base_amt.abs() < trades.MIN_BASE).sum()),
        "removed_size": removed_size,
        "removed_deviation": removed_deviation,
        "removed_share": (len(raw) - len(usable)) / len(raw),
        "removed_share_size": removed_size / len(raw),
        "removed_share_deviation": removed_deviation / len(raw),
        "sample_usable_swaps": in_sample,
        "sample_retention": in_sample / len(usable),
    }


def yearly(token: str, raw: pd.DataFrame, usable: pd.DataFrame) -> pd.DataFrame:
    counts = raw.groupby(raw.timestamp.dt.year).size().rename("raw_swaps")
    agg = usable.groupby(usable.timestamp.dt.year).usd.agg(
        usable_swaps="size", usd_volume="sum", median_trade_usd="median")
    out = pd.concat([counts, agg], axis=1)
    out[["usable_swaps", "usd_volume"]] = out[["usable_swaps", "usd_volume"]].fillna(0)
    out["usable_swaps"] = out.usable_swaps.astype(int)
    out.index.name = "year"
    return out.reset_index().assign(token=token)[
        ["token", "year", "raw_swaps", "usable_swaps", "usd_volume", "median_trade_usd"]]


def weeks_in_year(usable: pd.DataFrame, year: int) -> tuple[float, int]:
    """Weeks from the first usable trade to the year's end, and weeks with a trade."""
    t = usable.timestamp[usable.timestamp.dt.year == year]
    if t.empty:
        return 0.0, 0
    start = max(usable.timestamp.min(), pd.Timestamp(f"{year}-01-01", tz="UTC"))
    span = pd.Timestamp(f"{year + 1}-01-01", tz="UTC") - start
    active = t.dt.tz_localize(None).dt.to_period("W").nunique()
    return span / pd.Timedelta(weeks=1), int(active)


def price_by_size(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    y = raw[(raw.timestamp.dt.year == year) & (raw.price > 0)]
    bases = {"min_base": y[y.base_amt.abs() >= trades.MIN_BASE], "no_filter": y}
    rows = []
    for basis, d in bases.items():
        for lo, hi in PRICE_BUCKETS:
            p = d.price[(d.usd >= lo) & (d.usd < hi)]
            q25, q50, q75 = p.quantile([0.25, 0.5, 0.75])
            rows.append({"test": "price_by_size", "sample": str(year), "basis": basis,
                         "usd_from": lo, "usd_to": hi, "n": len(p),
                         "median_price": q50, "q25_price": q25, "q75_price": q75,
                         "iqr_price": q75 - q25})
    return pd.DataFrame(rows)


def autocorrelation(raw: pd.DataFrame, year: int, key: str) -> pd.DataFrame:
    variants = {f"min_usd_{label}": trades.clean(raw, key, min_usd=v)
                for label, v in AUTOCORR_THRESHOLDS.items()}
    variants["no_filter"] = raw[raw.price > 0].reset_index(drop=True)
    rows = []
    for basis, d in variants.items():
        for sample, s in (("full_window", d),
                          (str(year), d[d.timestamp.dt.year == year])):
            r = trades.returns(s).ret
            rows.append({"test": "lag1_autocorr", "sample": sample, "basis": basis,
                         "usd_from": AUTOCORR_THRESHOLDS.get(basis.removeprefix("min_usd_")),
                         "n": len(r), "lag1_autocorr": r.autocorr(1)})
    return pd.DataFrame(rows)


def cited(tiers: pd.DataFrame, years: pd.DataFrame, thresholds: pd.DataFrame,
          swaps: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for r in tiers.to_dict("records"):
        t = r["token"]
        for field in ("raw_swaps", "usable_swaps", "usd_volume", "median_trade_usd",
                      "first_trade", "last_trade", "stable_pair"):
            if field in r:
                out[f"tier_{t}_{field}"] = r[field]
        for field in ("removed_share", "removed_share_size", "removed_share_deviation"):
            out[f"cleaning_{t}_{field}"] = r[field]
        out[f"sample_{t}_retention"] = r["sample_retention"]

    y = years.set_index(["token", "year"])
    span, active = weeks_in_year(swaps["BCT"], 2021)
    out["BCT_usd_volume_2021"] = y.loc[("BCT", 2021), "usd_volume"]
    out["BCT_weeks_2021_from_first_trade"] = span
    out["BCT_active_weeks_2021"] = active
    for t in ("NCT", "MCO2"):
        best = years[years.token == t].sort_values("usd_volume").iloc[-1]
        out[f"{t}_best_year"] = int(best.year)
        out[f"{t}_best_year_usd_volume"] = best.usd_volume
    out["UBO_NBO_usd_volume"] = tiers.set_index("token").loc[["UBO", "NBO"], "usd_volume"].sum()
    out["NCT_usable_swaps_2025"] = y.loc[("NCT", 2025), "usable_swaps"]
    out["BCT_usable_swaps_2025"] = y.loc[("BCT", 2025), "usable_swaps"]
    out["BCT_NCT_usd_volume_2025"] = (y.loc[("BCT", 2025), "usd_volume"]
                                      + y.loc[("NCT", 2025), "usd_volume"])
    out.update(threshold_numbers(thresholds))
    return out


def threshold_numbers(thresholds: pd.DataFrame) -> dict:
    out = {}
    p = thresholds[(thresholds.test == "price_by_size") & (thresholds.basis == "min_base")]
    for label, lo in (("below_0p10", 0.0), ("0p10_to_1", 0.10), ("above_10", 10.0)):
        row = p[p.usd_from == lo].iloc[0]
        stem = f"threshold_{THRESHOLD_TOKEN}_{row['sample']}_price_{label}"
        out[f"{stem}_median"] = row.median_price
        out[f"{stem}_iqr"] = row.iqr_price
    a = thresholds[(thresholds.test == "lag1_autocorr")
                   & (thresholds["sample"] == "full_window")]
    for row in a.to_dict("records"):
        out[f"threshold_{THRESHOLD_TOKEN}_autocorr_{row['basis']}"] = row["lag1_autocorr"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--token", action="append", choices=TOKENS,
                        help="restrict to these tokens; prints only, writes nothing")
    args = parser.parse_args()
    tokens = [t for t in TOKENS if t in (args.token or TOKENS)]
    require(tokens)

    swaps, tier_rows, year_frames = {}, [], []
    raw_threshold = None
    for t in tokens:
        raw, usable = token_swaps(t)
        tier_rows.append(tier_row(t, raw, usable))
        year_frames.append(yearly(t, raw, usable))
        swaps[t] = usable
        if t == THRESHOLD_TOKEN:
            raw_threshold = raw
        print(f"  {t}: {len(raw):,} raw, {len(usable):,} usable")

    tiers = pd.DataFrame(tier_rows)
    census = stable_pairs()
    if census is None:
        print("  data/market_existence.csv absent: stable_pair column left out")
    else:
        tiers["stable_pair"] = tiers.token.map(census)
    years = pd.concat(year_frames, ignore_index=True)
    thresholds = None
    if raw_threshold is not None:
        thresholds = pd.concat([price_by_size(raw_threshold, THRESHOLD_YEAR),
                                autocorrelation(raw_threshold, THRESHOLD_YEAR,
                                                pool(THRESHOLD_TOKEN))],
                               ignore_index=True)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\nTiers (full data window, usable swaps)\n")
        print(tiers.drop(columns="pool").T.to_string())
        print("\nBy year\n")
        print(years.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        if thresholds is not None:
            print(f"\nThreshold checks, {THRESHOLD_TOKEN}\n")
            print(thresholds.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if args.token:
        if thresholds is not None:
            print("\n" + "\n".join(f"  {k}: {v}" for k, v in threshold_numbers(thresholds).items()))
        return
    print("\n" + results.table("market_tiers", tiers))
    print(results.table("market_yearly", years))
    print(results.table("market_thresholds", thresholds))
    print(results.numbers("market", cited(tiers, years, thresholds, swaps)))


if __name__ == "__main__":
    main()
