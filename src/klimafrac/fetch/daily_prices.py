"""
DefiLlama's daily USD prices for the carbon tokens on Polygon.

These aggregated series are compared with prices reconstructed from the
swaps (the aggregated rows of Table A1); no carbon-token result is estimated
from them.

Source: https://coins.llama.fi/chart/polygon:<token address>, stored verbatim
by fetch/defillama.py, which documents the bar convention.

Pinned window: from each series' start -- the first 00:00 UTC at or after
the token's Polygon deployment block, but not before 2021-10-08 -- to
2026-07-26 00:00 UTC exclusive. Deployment blocks are read from the chain
(Alchemy key needed the first time; cached in data/raw/blocks/).

Outputs:
  data/prices_full.csv        date, BCT, NCT, MCO2, KLIMA
                              (price at 00:00 UTC)
  data/vendor_manifest.json   snapshot used, coverage, missing bars
Raw: data/raw/defillama/chart_1d_carbon/<retrieved>/

Usage:
    uv run python -m klimafrac.fetch.daily_prices [--refresh | --snapshot YYYY-MM-DD]
"""

import math

from klimafrac.fetch.chain import Blocks, metered, record_spend, utc_ts
from klimafrac.fetch.defillama import STEP, build_chart_csv, cli

WINDOW_START = "2021-10-08"
TOKENS = {
    "BCT": ("polygon", "0x2f800db0fdb5223b3c3f354886d907a671414a7f", "BCT"),
    "NCT": ("polygon", "0xd838290e877e0188a4a44700463419ed96c16107", "NCT"),
    "MCO2": ("polygon", "0xaa7dbd1598251f856c12f63557a4c4397c253cea", "MCO2"),
    "KLIMA": ("polygon", "0x4e78011ce80ee02d2c3e649fb657e45898257815", "KLIMA"),
}
HOURLY = ("BCT", "NCT", "MCO2", "KLIMA")     # the staleness series
CLIENTS: dict = {}                           # one metered client per chain, reused


def blocks_for(chain: str) -> Blocks:
    if chain not in CLIENTS:
        CLIENTS[chain] = Blocks(metered(chain), chain)
    return CLIENTS[chain]


def launch_series(period: str, labels=tuple(TOKENS)) -> dict:
    """{label: {coin, symbol, start}} for the chart fetcher."""
    step, out = STEP[period], {}
    for label in labels:
        chain, address, symbol = TOKENS[label]
        b = blocks_for(chain)
        deployed = b.timestamp(b.deployment(address))
        floor = utc_ts(WINDOW_START)
        start = max(math.ceil(deployed / step) * step, floor)
        out[label] = {"coin": f"{chain}:{address}", "symbol": symbol, "start": start,
                      "deployment_timestamp": deployed}
    return out


def spent(module: str) -> None:
    for b in CLIENTS.values():
        record_spend(module, b.client)


def main() -> None:
    args = cli(__doc__.strip().split("\n")[0])
    build_chart_csv("chart_1d_carbon", "1d", lambda: launch_series("1d"),
                    "prices_full.csv", args.snapshot, args.refresh)
    spent("daily_prices")


if __name__ == "__main__":
    main()
