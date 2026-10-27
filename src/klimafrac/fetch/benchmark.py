"""
BTC and ETH prices from DefiLlama, the aggregated side of the construction
control in the asset-class comparison.

Source https://coins.llama.fi/chart/coingecko:{bitcoin,ethereum}, stored
verbatim by fetch/defillama.py. Window: 2021-10-08 00:00 UTC to 2026-07-26
00:00 UTC exclusive. The hourly and daily series are compared at the
instants they share.

Outputs: data/benchmark_daily.csv and data/benchmark_hourly.csv (date, BTC,
ETH), data/vendor_manifest.json. Raw responses under
data/raw/defillama/chart_1{d,h}_benchmark/<retrieved>/.

Usage:
    uv run python -m klimafrac.fetch.benchmark [--refresh | --snapshot YYYY-MM-DD]
"""

from klimafrac.fetch.chain import utc_ts
from klimafrac.fetch.defillama import build_chart_csv, cli, hourly_daily_check

START = "2021-10-08"
COINS = {"BTC": "coingecko:bitcoin", "ETH": "coingecko:ethereum"}


def series() -> dict:
    return {sym: {"coin": coin, "symbol": sym, "start": utc_ts(START)}
            for sym, coin in COINS.items()}


def main() -> None:
    args = cli(__doc__.strip().split("\n")[0])
    build_chart_csv("chart_1d_benchmark", "1d", series, "benchmark_daily.csv",
                    args.snapshot, args.refresh)
    build_chart_csv("chart_1h_benchmark", "1h", series, "benchmark_hourly.csv",
                    args.snapshot, args.refresh)
    hourly_daily_check("benchmark_daily.csv", "benchmark_hourly.csv")


if __name__ == "__main__":
    main()
