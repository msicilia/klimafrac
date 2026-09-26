"""
Conventional crypto benchmark: BTC and ETH prices from DefiLlama.

A globally traded asset has no single pool to reconstruct, so the vendor's
blended price is the instrument here. The hourly series are the aggregated
side of the asset-class comparison (Section 4.3, Table 4 and Figure 1); the
daily series is used only for the hourly-against-daily check.

Source: https://coins.llama.fi/chart/coingecko:{bitcoin,ethereum}, stored
verbatim by fetch/defillama.py, which documents the bar convention.

Pinned window: instants from 2021-10-08 00:00 UTC (the opening of the swap
data window, block 20,000,000) to 2026-07-26 00:00 UTC exclusive.

After both are built they are compared at the instants they share, and the
comparison is recorded in data/vendor_manifest.json.

Outputs:
  data/benchmark_daily.csv    date, BTC, ETH   (price at 00:00 UTC each day)
  data/benchmark_hourly.csv   date, BTC, ETH   (price at each hour)
  data/vendor_manifest.json   snapshot used, coverage, missing bars, the
                              hourly-against-daily comparison
Raw: data/raw/defillama/chart_1d_benchmark/<retrieved>/,
     data/raw/defillama/chart_1h_benchmark/<retrieved>/

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
