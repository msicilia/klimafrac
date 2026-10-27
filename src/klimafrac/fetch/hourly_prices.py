"""
DefiLlama's hourly USD prices for BCT, NCT, MCO2 and KLIMA on Polygon: the
aggregator's apparent staleness and the aggregated rows of the hourly table.

Source https://coins.llama.fi/chart/polygon:<token address>, stored verbatim
by fetch/defillama.py. Window: from each token's Polygon deployment, not
before 2021-10-08 00:00 UTC, to 2026-07-26 00:00 UTC exclusive. The hourly
and daily snapshots are compared at the instants they share.

Outputs: data/hourly_prices_full.csv (date, BCT, NCT, MCO2, KLIMA),
data/vendor_manifest.json. Raw under data/raw/defillama/chart_1h_carbon/.

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
