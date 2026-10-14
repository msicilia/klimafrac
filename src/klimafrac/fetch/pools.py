"""The pools analysed and the pinned block range of the dataset.

The data window closes at the last block before 2026-07-26 00:00 UTC on each
chain, so any refetch reproduces the same history. On Polygon it opens at
block 20,000,000 (2021-10-08), ten days before the first BCT swap; on Base it
opens at each pool's deployment.

Token addresses are the pool's own token0/token1. The fetcher checks them,
and reads decimals, from the pair contract before fetching anything.
"""

from dataclasses import dataclass

POLYGON_START_BLOCK = 20_000_000
END_BLOCK = {"polygon": 90_879_334, "base": 49_117_326}

USDC_E_POLYGON = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"
USDC_BASE = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
KLIMA_POLYGON = "0x4e78011ce80ee02d2c3e649fb657e45898257815"

SUSHISWAP_POLYGON = "0xc35dadb65012ec5796536bd9864ed8773abc74c4"
QUICKSWAP_POLYGON = "0x5757371414417b8c6caad45baef941abc7d3ab32"
AERODROME_BASE = "0x420dd381b31aef6683db6b902084cb0ffece40da"


@dataclass(frozen=True)
class Pool:
    symbol: str
    chain: str
    dex: str
    factory: str
    pair: str
    base: str
    quote: str
    role: str

    @property
    def key(self) -> str:
        return f"{self.chain}_{self.symbol}"


POOLS = [
    # Carbon pool tokens and the protocol token, each in its deepest pool.
    Pool("BCT", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x1e67124681b402064cd0abe8ed1b5c79d2e02f64",
         "0x2f800db0fdb5223b3c3f354886d907a671414a7f", USDC_E_POLYGON, "carbon"),
    Pool("NCT", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0xdb995f975f1bfc3b2157495c47e4efb31196b2ca",
         "0xd838290e877e0188a4a44700463419ed96c16107", USDC_E_POLYGON, "carbon"),
    Pool("MCO2", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x68ab4656736d48bb1de8661b9a323713104e24cf",
         "0xaa7dbd1598251f856c12f63557a4c4397c253cea", USDC_E_POLYGON, "carbon"),
    Pool("KLIMA", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x5786b267d35f9d011c4750e0b0ba584e1fdbead1",
         KLIMA_POLYGON, USDC_E_POLYGON, "carbon"),
    # UBO and NBO never had a stablecoin pair; their only market is against KLIMA.
    Pool("UBO", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x09053745a04a7e78492ee8fdb866c750594f2a60",
         "0x2b3ecb0991af0498ece9135bcd04013d7993110c", KLIMA_POLYGON, "carbon"),
    Pool("NBO", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0xe69e0e02297e2f3b4298f250ecfdc58207536327",
         "0x6bca3b77c1909ce1a4ba1a20d1103bde8d222e48", KLIMA_POLYGON, "carbon"),
    # Conventional controls on the same two factories, fetched by the same path.
    Pool("WETH_S", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x34965ba0ac2451a34a0471f04cca3f990b8dea27",
         "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619", USDC_E_POLYGON, "control"),
    Pool("WETH_Q", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x853ee4b2a13f8a742d64c8f088be7ba2131f670d",
         "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619", USDC_E_POLYGON, "control"),
    Pool("WBTC_Q", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xf6a637525402643b0654a54bead2cb9a83c8b498",
         "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6", USDC_E_POLYGON, "control"),
    # Conventional tokens as thin as the carbon pools: the output of the rule in
    # fetch/thin_candidates.py, which refuses to run unless this list matches it.
    Pool("AAVE", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x7c76b6b3fe14831a39c0fec908da5f17180df677",
         "0xd6df932a45c0f255f85145f286ea0b292b21c90b", USDC_E_POLYGON, "thin"),
    Pool("ADS", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x85ba262be13329a2db5acf9aa46ac2345b5df4ff",
         "0x598e49f01befeb1753737934a5b11fea9119c796", USDC_E_POLYGON, "thin"),
    Pool("BLOK", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xd59898d9c951e36f7ee8782f137db919a8043429",
         "0x229b1b6c23ff8953d663c4cbb519717e323a0a84", USDC_E_POLYGON, "thin"),
    Pool("BNB", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x40a5df3e37152d4daf279e0450289af76472b02e",
         "0x5c4b7ccbf908e64f32e12c6650ec0c96d717f03f", USDC_E_POLYGON, "thin"),
    Pool("BOOTY", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xf2046f95621f349823cc1a12d914b2b5193abd93",
         "0xd12dc5319808bb31ba95ae5764def2627d5966ce", USDC_E_POLYGON, "thin"),
    Pool("DERC", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x0a8a3cb9a21c893a207826e76125ef6faaad99ec",
         "0xb35fcbcf1fd489fce02ee146599e893fdcdc60e6", USDC_E_POLYGON, "thin"),
    Pool("FISH", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x0df9e46c0eaedf41b9d4bbe2cea2af6e8181b033",
         "0x3a3df212b7aa91aa0402b9035b098891d276572b", USDC_E_POLYGON, "thin"),
    Pool("FLAME", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x27bb3fdfac9109d0b541a401717bf45ed6f56de8",
         "0x22e3f02f86bc8ea0d73718a2ae8851854e62adc5", USDC_E_POLYGON, "thin"),
    Pool("FYN", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xb46a13fd188976f8c3cd439ebfbf155ba443efd9",
         "0x3b56a704c01d650147ade2b8cee594066b3f9421", USDC_E_POLYGON, "thin"),
    Pool("GAJ", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x99dadf889520b9cebd9959d3ca1866dc2517ea1d",
         "0xf4b0903774532aee5ee567c02aab681a81539e92", USDC_E_POLYGON, "thin"),
    Pool("GFI", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xb1f3555a7c3753ab4e6df1d66cfdb25477a36ce7",
         "0x874e178a2f3f3f9d34db862453cd756e7eab0381", USDC_E_POLYGON, "thin"),
    Pool("GLM", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x04e2dc96afecdf72221882e1cee039cab4d443e0",
         "0x0b220b82f3ea3b7f6d9a1d8ab58930c064a2b5bf", USDC_E_POLYGON, "thin"),
    Pool("GTON", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xf01a0a0424bda0acdd044a61af88a34636e0001c",
         "0xf480f38c366daac4305dc484b2ad7a496ff00cea", USDC_E_POLYGON, "thin"),
    Pool("HANU", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xd255b96a6ab9334ca5a721afcafca50f9b94dc51",
         "0x709a4b6217584188ddb93c82f5d716d969acce1c", USDC_E_POLYGON, "thin"),
    Pool("ICE", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x9e3880647c07ba13e65663de29783ecd96ec21de",
         "0xc6c855ad634dcdad23e64da71ba85b8c51e5ad7c", USDC_E_POLYGON, "thin"),
    Pool("LUNA", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x74946a114ca2b26267983ef51ad8664cabde4034",
         "0x9cd6746665d9557e1b9a775819625711d0693439", USDC_E_POLYGON, "thin"),
    Pool("MASK", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x253d637068fbf11b18d0f2a1bf3b167d37802687",
         "0x2b9e7ccdf0f4e5b24757c1e1a80e311e34cb10c7", USDC_E_POLYGON, "thin"),
    Pool("MM", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x5e06e1da9b7cb3ddd0df596003ad4cb852f51955",
         "0x5647fe4281f8f6f01e84bce775ad4b828a7b8927", USDC_E_POLYGON, "thin"),
    Pool("MOCA", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xfd0d4cba23ed8307d4e92f8c4468ecb6f976774e",
         "0xce899f26928a2b21c6a2fddd393ef37c61dba918", USDC_E_POLYGON, "thin"),
    Pool("oBNB", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xe55cb144e02cffb6fbd65d9a4bd62378998bc267",
         "0x7e9928afe96fefb820b85b4ce6597b8f660fe4f4", USDC_E_POLYGON, "thin"),
    Pool("OMEN", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x50409de292f5f821888702e9538bf15fa273dfe6",
         "0x76e63a3e7ba1e2e61d3da86a87479f983de89a7e", USDC_E_POLYGON, "thin"),
    Pool("ORBS", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xb2b6d423e535b57aad06e9866803b95fb66152ea",
         "0x614389eaae0a6821dc49062d56bda3d9d45fa2ff", USDC_E_POLYGON, "thin"),
    Pool("PLOT", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x78fd833ec4464d3d6d470e5853dbfc2de0ca6f5b",
         "0xe82808eaa78339b06a691fd92e1be79671cad8d3", USDC_E_POLYGON, "thin"),
    Pool("PolyDoge", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x7b3e67e63906d8576466c2f48158a30be0a9e36c",
         "0x8a953cfe442c5e8855cc6c61b1293fa648bae472", USDC_E_POLYGON, "thin"),
    Pool("REVV", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xe4139dbf19e9c8d880f915711c8674022979d432",
         "0x70c006878a5a50ed185ac4c87d837633923de296", USDC_E_POLYGON, "thin"),
    Pool("ADDY", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x5fcb390b4422f4ff7940c23618a62bf5f69658a8",
         "0xc3fdbadc7c795ef1d6ba111e06ff8f16a20ea539", USDC_E_POLYGON, "thin"),
    Pool("AURUM", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0xabee7668a96c49d27886d1a8914a54a5f9805041",
         "0x34d4ab47bee066f361fa52d792e69ac7bd05ee23", USDC_E_POLYGON, "thin"),
    Pool("BIFI", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x180237bd326d5245d0898336f54b3c8012c5c62f",
         "0xfbdd194376de19a88118e84e279b977f165d01b8", USDC_E_POLYGON, "thin"),
    Pool("DINO", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x3324af8417844e70b81555a6d1568d78f4d4bf1f",
         "0xaa9654becca45b5bdfa5ac646c939c62b527d394", USDC_E_POLYGON, "thin"),
    Pool("DYDX", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x30933831b71710d0551b518b74032242b2a71d2c",
         "0x4c3bf0a3de9524af68327d1d2558a3b70d17d42a", USDC_E_POLYGON, "thin"),
    Pool("FLEATO", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0xe66f274542fb624f5489ebb9f6e953e91a9908d1",
         "0x6804b07d883d0169c05233332ccf17aa956424c5", USDC_E_POLYGON, "thin"),
    Pool("GAIA", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0xacd50fedc46fefee64790eab404eb52dec9e3ad9",
         "0x723b17718289a91af252d616de2c77944962d122", USDC_E_POLYGON, "thin"),
    Pool("MANA", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0xe1c99c7481421e299bf3945e5e91817e6d60c91b",
         "0xa1c57f48f0deb89f569dfbe6e2b7f46d33606fd4", USDC_E_POLYGON, "thin"),
    Pool("oAUTO", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0xd8d51ea2edca2c2ea15a053a5730f559d79a1570",
         "0x7f426f6dc648e50464a0392e60e1bb465a67e9cf", USDC_E_POLYGON, "thin"),
    Pool("SUSHI", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0xe15e9d2a5af5c1d3524bbc594ddc4a7d80ad27cd",
         "0x0b3f868e0be5597d5db7feb59e1cadbb0fdda50a", USDC_E_POLYGON, "thin"),
    Pool("UNI", "polygon", "sushiswap", SUSHISWAP_POLYGON,
         "0x7c5b028ea6a2a123bb1c8be5d3d2935d4532961b",
         "0xb33eaad8d922b1083446dc23f610c2567fb5180f", USDC_E_POLYGON, "thin"),
    # Conventional tokens that trade more often than any carbon pool: the output of
    # the mirror rule in fetch/thin_candidates.py, under the same registry check.
    Pool("GHST", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x096c5ccb33cfc5732bcd1f3195c13dbefc4c82f4",
         "0x385eeac5cb85a38a9a07a70c73e0a3271cfb54a7", USDC_E_POLYGON, "active"),
    Pool("MYST", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x7e2eaa9514c7a56b4d556ff0d1009cb9c7f2b89d",
         "0x1379e8886a944d2d9d440b3d88df536aea08d9f3", USDC_E_POLYGON, "active"),
    Pool("NAKA", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x9eb792c780d3e3714f245c596c8c8993bf6fc605",
         "0x311434160d7537be358930def317afb606c0d737", USDC_E_POLYGON, "active"),
    Pool("QUICK", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x1f1e4c845183ef6d50e9609f16f6f9cae43bc9cb",
         "0x831753dd7087cac61ab5644b308642cc1c33dc13", USDC_E_POLYGON, "active"),
    Pool("TEL", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0xa5cabfc725dfa129f618d527e93702d10412f039",
         "0xdf7837de1f2fa4631d716cf2502f8b230f1dcc32", USDC_E_POLYGON, "active"),
    Pool("WPOL", "polygon", "quickswap", QUICKSWAP_POLYGON,
         "0x6e7a5fafcec6bb1e78bae2a1f0b612012bf14827",
         "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270", USDC_E_POLYGON, "active"),
    # BCT's second venue, for the cross-chain comparison.
    Pool("BCT", "base", "aerodrome", AERODROME_BASE,
         "0x77e09a5043820820390904357463bfb739a76104",
         "0x576bca23dcb6d94ff8e537d88b0d3e1bead444a2", USDC_BASE, "venue"),
]

BY_KEY = {p.key: p for p in POOLS}
