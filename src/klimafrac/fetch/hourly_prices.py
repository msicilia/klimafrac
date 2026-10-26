"""
DefiLlama's hourly USD prices for BCT, NCT, MCO2 and KLIMA on Polygon.

The hourly counterpart of fetch/daily_prices.py, and like it an object of
study rather than an input: it supplies the aggregator's apparent staleness
and the aggregated rows of the hourly calendar-time table.

Source: https://coins.llama.fi/chart/polygon:<token address>, stored verbatim
by fetch/defillama.py, which documents the bar convention.

Pinned window: from each token's launch -- the first whole hour at or after
its Polygon deployment block, but not before 2021-10-08 00:00 UTC -- to
2026-07-26 00:00 UTC exclusive. The four Polygon series only: the Base BCT
series that fetch/daily_prices.py collects as evidence is a daily question
and is not asked hourly.

After the fetch the hourly and daily snapshots are compared at the instants
they share, and the comparison is recorded in data/vendor_manifest.json.

Outputs:
  data/hourly_prices_full.csv  date, BCT, NCT, MCO2, KLIMA (price at each hour)
  data/vendor_manifest.json    snapshot used, coverage, missing bars, the
                               hourly-against-daily comparison
Raw: data/raw/defillama/chart_1h_carbon/<retrieved>/

Usage:
    uv run python -m klimafrac.fetch.hourly_prices [--refresh | --snapshot YYYY-MM-DD]
"""

from klimafrac.fetch.daily_prices import HOURLY, launch_series, spent
from klimafrac.fetch.defillama import build_chart_csv, cli, hourly_daily_check


def main() -> None:
    args = cli(__doc__.strip().split("\n")[0])
    build_chart_csv("chart_1h_carbon", "1h", lambda: launch_series("1h", HOURLY),
                    "hourly_prices_full.csv", args.snapshot, args.refresh)
    hourly_daily_check("prices_full.csv", "hourly_prices_full.csv")
    spent("hourly_prices")


if __name__ == "__main__":
    main()
