"""The pools analysed and the pinned block range of the dataset.

The data window closes at the last block before 2026-07-26 00:00 UTC on each
chain, so any refetch reproduces the same history. On Polygon it opens at
block 20,000,000 (2021-10-08), ten days before the first BCT swap.

Token addresses are the pool's own token0/token1. The fetcher checks them,
and reads decimals, from the pair contract before fetching anything.
"""

from dataclasses import dataclass

POLYGON_START_BLOCK = 20_000_000
END_BLOCK = {"polygon": 90_879_334, "base": 49_117_326}

USDC_E_POLYGON = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"
KLIMA_POLYGON = "0x4e78011ce80ee02d2c3e649fb657e45898257815"

SUSHISWAP_POLYGON = "0xc35dadb65012ec5796536bd9864ed8773abc74c4"
QUICKSWAP_POLYGON = "0x5757371414417b8c6caad45baef941abc7d3ab32"


@dataclass(frozen=True)
class Pool:
    symbol: str
    chain: str
    dex: str
    factory: str
    pair: str
    base: str
    quote: str

    @property
    def key(self) -> str:
        return f"{self.chain}_{self.symbol}"


POOLS = [
    # Carbon pool tokens and the protocol token, each in its deepest pool.
    Pool("BCT", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x1e67124681b402064cd0abe8ed1b5c79d2e02f64",
         "0x2f800db0fdb5223b3c3f354886d907a671414a7f", USDC_E_POLYGON),
    Pool("NCT", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0xdb995f975f1bfc3b2157495c47e4efb31196b2ca",
         "0xd838290e877e0188a4a44700463419ed96c16107", USDC_E_POLYGON),
    Pool("MCO2", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x68ab4656736d48bb1de8661b9a323713104e24cf",
         "0xaa7dbd1598251f856c12f63557a4c4397c253cea", USDC_E_POLYGON),
    Pool("KLIMA", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x5786b267d35f9d011c4750e0b0ba584e1fdbead1",
         KLIMA_POLYGON, USDC_E_POLYGON),
    # UBO and NBO never had a stablecoin pair; their only market is against KLIMA.
    Pool("UBO", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x09053745a04a7e78492ee8fdb866c750594f2a60",
         "0x2b3ecb0991af0498ece9135bcd04013d7993110c", KLIMA_POLYGON),
    Pool("NBO", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0xe69e0e02297e2f3b4298f250ecfdc58207536327",
         "0x6bca3b77c1909ce1a4ba1a20d1103bde8d222e48", KLIMA_POLYGON),
    # Conventional controls on the same two factories, fetched by the same path.
    Pool("WETH_S", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x34965ba0ac2451a34a0471f04cca3f990b8dea27",
         "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619", USDC_E_POLYGON),
    Pool("WETH_Q", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x853ee4b2a13f8a742d64c8f088be7ba2131f670d",
         "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619", USDC_E_POLYGON),
    Pool("WBTC_Q", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xf6a637525402643b0654a54bead2cb9a83c8b498",
         "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6", USDC_E_POLYGON),
]

BY_KEY = {p.key: p for p in POOLS}
