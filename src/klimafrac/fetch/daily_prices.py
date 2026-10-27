"""
DefiLlama's daily USD prices for the carbon tokens, an object of study
rather than an input: the aggregated rows of the calendar-time tables.

Source https://coins.llama.fi/chart/<chain>:<token address>, stored verbatim
by fetch/defillama.py. The BCT_base column requests the bridged BCT
contract on Base from the same start as the Polygon series, so the bars the
vendor returns for instants before Base existed are in the stored data.
Window: from each token's Polygon deployment, not before 2021-10-08, to
2026-07-26 00:00 UTC exclusive.

Outputs: data/prices_full.csv (date, BCT, NCT, MCO2, KLIMA, BCT_base at
00:00 UTC), data/vendor_manifest.json. Raw under
data/raw/defillama/chart_1d_carbon/<retrieved>/.

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
    # Asked for from the same start as the Polygon series, deliberately: the
    # vendor's answer for instants before this chain existed is the evidence.
    "BCT_base": ("base", "0x576bca23dcb6d94ff8e537d88b0d3e1bead444a2", "BCT"),
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
        start = (max(math.ceil(deployed / step) * step, floor)
                 if chain == "polygon" else floor)
        out[label] = {"coin": f"{chain}:{address}", "symbol": symbol, "start": start,
                      "deployment_timestamp": deployed,
                      "chain_genesis_instant": b.timestamp(0)}
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
