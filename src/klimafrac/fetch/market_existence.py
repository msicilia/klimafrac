"""
Which markets each carbon token ever had, read from the chain.

Everything is pinned at END_BLOCK on each chain, the last block before
2026-07-26 00:00 UTC: Polygon 90,879,334, Base 49,117,326. Four censuses.

1. Transfer logs. Every Transfer event of every token in scope, read with
   eth_getLogs over the token's own address and stored verbatim under
   data/raw/market_existence/logs/. This is the primitive record from which
   every count below is derived, so a count can be re-audited without asking
   the node again, and it is also the complete holder set of each token. Each
   token's log stream is reconciled against totalSupply() at END_BLOCK --
   minted minus burned must equal it to the unit -- which is what establishes
   that no Transfer is missing. Where a mint comes from is a property of the
   token: the carbon tokens issue from the zero address, KLIMA from its own
   address, as Olympus forks do (see MINTS_FROM_SELF).

2. Factory census. Every pairing of each token with each quote is looked up
   in every factory at END_BLOCK. Factories never remove pools, so a pool
   created at any time up to END_BLOCK is found. Each factory is first
   verified on-chain: it must resolve a reference pair to a contract whose
   factory(), token0() and token1() match; every pool found is checked the
   same way. For each pool the transfers of the carbon token into and out of
   it are counted from the stored logs (exactly, over the whole chain, not
   sampled); whether it ever received the quote token is read from the chain;
   and, for constant-product and Solidly pairs, its LP totalSupply() at
   END_BLOCK, non-zero if and only if liquidity was ever added (the first
   mint locks MINIMUM_LIQUIDITY for good).

     Polygon quotes    USDC.e, USDC (native), USDT, DAI, WETH, WMATIC, and the
                       six carbon tokens against one another
     Polygon venues    getPair: Sushiswap, Quickswap, Uniswap V2, ApeSwap,
                       Dfyn, JetSwap, Polycat, MeshSwap, ComethSwap, Elk,
                       RadioShack; getPair(stable): Dystopia; getPool with fee
                       tiers 100/500/3000/10000: Uniswap V3, Sushiswap V3,
                       Retro; poolByPair: Quickswap V3 (Algebra)
     Base quotes       USDC, WETH, for bridged BCT
     Base venues       getPool(stable): Aerodrome; getPool(tickSpacing)
                       1/50/100/200/2000: Aerodrome Slipstream; getPool with
                       fee tiers: Uniswap V3 Base; getPair: SushiSwap Base,
                       BaseSwap

3. Registry and singleton venues, which have no per-pair factory. The Curve
   registry is asked for a pool holding each pair of coins, and the Uniswap V4
   PoolManager's Initialize logs are scanned on both chains for a pool whose
   currency0 or currency1 is a token in scope.

4. Holder census. Every counterparty of every token, taken from the stored
   logs, classified by code and pool interface (token0/token1/symbol). A
   token that traded on a venue with no per-pair factory -- Balancer, DODO, a
   fork not listed above -- must have a contract of that venue among its
   counterparties, so this is what makes the factory census's blind spots
   observable. For UBO and NBO the receipt of every transaction that moved
   the token is read as well: a contract whose net flow of the token and net
   flow of a stablecoin have opposite signs within one transaction exchanged
   the one for the other directly, whatever venue it belongs to. Opposite
   flows are not proof of a trade (a vault deposit of one token and
   withdrawal of another in one transaction looks the same), so the events
   are listed for inspection rather than counted as markets.

Outputs:
  data/market_existence.csv          one row per (chain, token, quote, venue,
                                     variant) lookup, pool or not: status,
                                     pool, transfers_in/out, first/last
                                     transfer block, quote_ever_received, LP
                                     supply, END_BLOCK balances
  data/market_existence_holders.csv  every counterparty of every token:
                                     address, transfers, is_contract, token0,
                                     token1, symbol, stablecoin_pool
  data/market_existence.json         scope, factory verification, log
                                     reconciliation, per-token summary,
                                     registry and singleton venues,
                                     direct-exchange candidates
Raw: data/raw/market_existence/ (logs, per-pool reads, caches, manifest.json,
     rpc_calls_<chain>.json)

Usage:
    uv run python -m klimafrac.fetch.market_existence
    uv run python -m klimafrac.fetch.market_existence --estimate
    uv run python -m klimafrac.fetch.market_existence --receipts UBO --receipts NBO
"""

import argparse
import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from klimafrac.fetch.chain import (BATCH, CU_PRICE, LOG_COLUMNS, TRANSFER_TOPIC,
                                   UNANSWERED, ZERO, Blocks, as_address,
                                   as_string, as_uint,
                                   calldata, calls_allowing_reverts, chunks,
                                   decode_transfer_logs, eth_calls, event_topic,
                                   logs_in_range, metered, raw_manifest,
                                   record_spend, say, topic_address, write_csv,
                                   write_json)
from klimafrac.fetch.pools import END_BLOCK
from klimafrac.paths import data, raw

MODULE = "market_existence"
CHUNK = 5_000_000            # blocks per stored log file
RECEIPT_PART = 500           # transactions per stored receipt file
CU_CODE, CU_CALL = 20, 26    # for the cost estimate reported before classifying

POLYGON_TOKENS = {
    "BCT": "0x2f800db0fdb5223b3c3f354886d907a671414a7f",
    "NCT": "0xd838290e877e0188a4a44700463419ed96c16107",
    "MCO2": "0xaa7dbd1598251f856c12f63557a4c4397c253cea",
    "UBO": "0x2b3ecb0991af0498ece9135bcd04013d7993110c",
    "NBO": "0x6bca3b77c1909ce1a4ba1a20d1103bde8d222e48",
    "KLIMA": "0x4e78011ce80ee02d2c3e649fb657e45898257815",
}
POLYGON_STABLECOINS = {
    "USDC.e": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
    "USDC": "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",
    "USDT": "0xc2132d05d31c914a87c6611c10748aeb04b58e8f",
    "DAI": "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063",
}
POLYGON_OTHER = {
    "WETH": "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619",
    "WMATIC": "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270",
}
BASE_TOKENS = {"BCT": "0x576bca23dcb6d94ff8e537d88b0d3e1bead444a2"}
BASE_STABLECOINS = {"USDC": "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"}
BASE_OTHER = {"WETH": "0x4200000000000000000000000000000000000006"}

RECEIPT_CENSUS = ("UBO", "NBO")

# Where a token's new supply comes from. ERC-20 mints from the zero address,
# and the carbon tokens all do. KLIMA is an Olympus fork, whose _mint emits
# Transfer(address(this), to, amount) instead, so its supply appears to leave
# the token contract. Declared rather than inferred, because a token whose
# mints were quietly read as ordinary transfers would pass every check here
# while counting its own issuance as trading. The reconciliation below still
# has to come out to the unit either way, which is what proves the entry
# right: for KLIMA it does so over 2.8 million logs, and would not if the
# zero address were used.
MINTS_FROM_SELF = frozenset({"KLIMA"})


@dataclass(frozen=True)
class Venue:
    name: str
    kind: str          # v2 | solidly | solidly_pool | v3 | cl | algebra
    factory: str


# Lookup signature and the variants each family indexes its pools by.
LOOKUPS = {
    "v2": ("getPair(address,address)", [""]),
    "solidly": ("getPair(address,address,bool)", ["volatile", "stable"]),
    "solidly_pool": ("getPool(address,address,bool)", ["volatile", "stable"]),
    "v3": ("getPool(address,address,uint24)", ["100", "500", "3000", "10000"]),
    "cl": ("getPool(address,address,int24)", ["1", "50", "100", "200", "2000"]),
    "algebra": ("poolByPair(address,address)", [""]),
}

POLYGON_VENUES = [
    Venue("sushiswap", "v2", "0xc35dadb65012ec5796536bd9864ed8773abc74c4"),
    Venue("quickswap", "v2", "0x5757371414417b8c6caad45baef941abc7d3ab32"),
    Venue("uniswap_v2", "v2", "0x9e5a52f57b3038f1b8eee45f28b3c1967e22799c"),
    Venue("apeswap", "v2", "0xcf083be4164828f00cae704ec15a36d711491284"),
    Venue("dfyn", "v2", "0xe7fb3e833efe5f9c441105eb65ef8b261266423b"),
    Venue("jetswap", "v2", "0x668ad0ed2622c62e24f0d5ab6b6ac1b9d2cd4ac7"),
    Venue("polycat", "v2", "0x477ce834ae6b7ab003cce4bc4d8697763ff456fa"),
    Venue("meshswap", "v2", "0x9f3044f7f9fc8bc9ed615d54845b4577b833282d"),
    Venue("comethswap", "v2", "0x800b052609c355ca8103e06f022aa30647ead60a"),
    Venue("elk", "v2", "0xe3bd06c7ac7e1ceb17bdd2e5ba83e40d1515af2a"),
    Venue("radioshack", "v2", "0xb581d0a3b7ea5cdc029260e989f768ae167ef39b"),
    Venue("dystopia", "solidly", "0x1d21db6cde1b18c7e47b0f7f42f4b3f68b9beec9"),
    Venue("uniswap_v3", "v3", "0x1f98431c8ad98523631ae4a59f267346ea31f984"),
    Venue("sushiswap_v3", "v3", "0x917933899c6a5f8e37f31e19f92cdbff7e8ff0e2"),
    Venue("retro", "v3", "0x91e1b99072f238352f59e58de875691e20dc19c1"),
    Venue("quickswap_v3", "algebra", "0x411b0facc3489691f28ad58c47006af5e3ab3a28"),
]
BASE_VENUES = [
    Venue("aerodrome", "solidly_pool", "0x420dd381b31aef6683db6b902084cb0ffece40da"),
    Venue("aerodrome_slipstream", "cl", "0x5e7bb104d84c7cb9b682aac2f3d509f5f406809a"),
    Venue("uniswap_v3_base", "v3", "0x33128a8fc17869897dce68ed026d694621f6fdfd"),
    Venue("sushiswap_base", "v2", "0x71524b4f93c58fcbf659783284e38825f0622859"),
    Venue("baseswap", "v2", "0xfda619b6d20975be80a10332cd39b9a4b0faa8bb"),
]

# Venues with no per-pair factory, reached by their own registry or by the
# events of a singleton contract.
CURVE_ADDRESS_PROVIDER = "0x0000000022d53366457f9d5e68ec105046fc4383"
CURVE_REGISTRY_ID = 0
UNISWAP_V4 = {"polygon": "0x67366782805870060151383f4bbff9dab53e5cd6",
              "base": "0x498581ff718922c3f8e6a244956af099b2652b2b"}
V4_INITIALIZE = ("Initialize(bytes32,address,address,uint24,int24,address,"
                 "uint160,int24)")


@dataclass(frozen=True)
class Scope:
    """One chain's tokens, quotes and venues, all pinned at its END_BLOCK."""
    name: str
    tokens: dict[str, str]
    stablecoins: dict[str, str]
    other: dict[str, str]
    venues: list[Venue]
    probe: tuple[str, str]
    curve: bool = False
    v4: bool = True
    receipts: tuple[str, ...] = RECEIPT_CENSUS

    @property
    def end(self) -> int:
        return END_BLOCK[self.name]

    @property
    def addresses(self) -> dict[str, str]:
        return {**self.tokens, **self.stablecoins, **self.other}

    @property
    def symbol_of(self) -> dict[str, str]:
        return {a: s for s, a in self.addresses.items()}

    def quotes(self) -> list[str]:
        return [*self.stablecoins, *self.other, *self.tokens]

    def pairings(self) -> list[tuple[str, str]]:
        return [(t, q) for t in self.tokens for q in self.quotes() if t != q]

    def quote_kind(self, sym: str) -> str:
        return ("stablecoin" if sym in self.stablecoins
                else "carbon" if sym in self.tokens else "other")


SCOPES = {
    "polygon": Scope("polygon", POLYGON_TOKENS, POLYGON_STABLECOINS, POLYGON_OTHER,
                     POLYGON_VENUES, ("WETH", "USDC.e"), curve=True),
    "base": Scope("base", BASE_TOKENS, BASE_STABLECOINS, BASE_OTHER,
                  BASE_VENUES, ("WETH", "USDC"), receipts=()),
}


def cache(*parts: str):
    return raw(MODULE, *parts)


def load_cache(path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# --- 1. Transfer logs ------------------------------------------------------

def log_chunks(scope: Scope, sym: str, start: int):
    for a, b in chunks(start, scope.end, CHUNK):
        yield a, b, cache("logs", scope.name, sym, f"{a}_{b}.csv.gz")


def fetch_token_logs(client, scope: Scope, sym: str, start: int) -> dict:
    """Every Transfer log of the token, stored one file per block chunk."""
    manifest_path = cache("logs", scope.name, sym, "manifest.json")
    manifest = load_cache(manifest_path)
    manifest.update(chain=scope.name, token=sym, address=scope.tokens[sym],
                    topic0=TRANSFER_TOPIC, first_block=start, end_block=scope.end,
                    chunk_blocks=CHUNK)
    parts = manifest.setdefault("chunks", {})
    for a, b, path in log_chunks(scope, sym, start):
        key = f"{a}_{b}"
        if path.exists() and key in parts:
            continue
        logs = logs_in_range(client, {"address": scope.tokens[sym],
                                      "topics": [TRANSFER_TOPIC]}, a, b)
        rows = decode_transfer_logs(logs, scope.tokens[sym])
        df = pd.DataFrame(rows, columns=LOG_COLUMNS)
        if df.duplicated(["block", "log_index"]).any():
            raise ValueError(f"{sym} {key}: duplicated (block, log_index)")
        write_csv(df.sort_values(["block", "log_index"]), path)
        parts[key] = {"from_block": a, "to_block": b, "rows": len(df),
                      "file": path.name}
        write_json(manifest, manifest_path)
        say(f"  {scope.name} {sym} logs {a:,}-{b:,}: {len(df):,}")
    write_json(manifest, manifest_path)
    return manifest


def load_token_logs(scope: Scope, sym: str, start: int) -> pd.DataFrame:
    frames = []
    for a, b, path in log_chunks(scope, sym, start):
        if not path.exists():
            raise FileNotFoundError(f"transfer logs incomplete: {path}")
        frames.append(pd.read_csv(path, dtype={"raw_value": str, "tx_hash": str,
                                               "from": str, "to": str}))
    df = pd.concat(frames, ignore_index=True)
    if df.duplicated(["block", "log_index"]).any():
        raise ValueError(f"{sym}: duplicated (block, log_index) across chunks")
    return df.sort_values(["block", "log_index"]).reset_index(drop=True)


def mint_source(scope: Scope, sym: str) -> str:
    """The address this token's new supply is emitted as coming from."""
    return scope.tokens[sym] if sym in MINTS_FROM_SELF else ZERO


def reconcile_supply(client, scope: Scope, sym: str, logs: pd.DataFrame) -> dict:
    """Minted minus burned must equal totalSupply() at END_BLOCK, to the unit.

    This is what makes the stored logs usable as a count: an eth_getLogs range
    that silently returned fewer logs than the chain holds would break it.
    """
    source = mint_source(scope, sym)
    # Python integers throughout: an 18-decimal total overruns int64, and a
    # numpy sum of these would wrap round silently.
    minted = sum(int(v) for v, f in zip(logs.raw_value, logs["from"]) if f == source)
    burned = sum(int(v) for v, t in zip(logs.raw_value, logs["to"]) if t == ZERO)
    supply = as_uint(client.eth_call(scope.tokens[sym],
                                     calldata("totalSupply()"), scope.end))
    if minted - burned != supply:
        raise ValueError(f"{scope.name} {sym}: Transfer logs give supply "
                         f"{minted - burned}, totalSupply() at END_BLOCK is "
                         f"{supply}: the log stream is incomplete")
    return {"transfers": int(len(logs)),
            "zero_value_transfers": int((logs.raw_value == "0").sum()),
            "minted_raw": str(minted), "burned_raw": str(burned),
            "mint_source": source,
            "total_supply_end_raw": str(supply),
            "first_block": int(logs.block.min()), "last_block": int(logs.block.max()),
            "counterparties": int(len(holder_index(logs, source)))}


def holder_index(logs: pd.DataFrame, source: str = ZERO) -> pd.Index:
    """Every address the token ever touched that is a holder rather than a
    place supply comes from or goes to: the zero address, and for a token that
    issues from itself its own address, are both left out."""
    both = pd.concat([logs["from"], logs["to"]]).unique()
    skip = {ZERO, source}
    return pd.Index(sorted(a for a in both if a not in skip))


def address_counts(logs: pd.DataFrame) -> pd.DataFrame:
    """Per address: how often the token reached it and left it, and when.

    transfers_in / transfers_out exclude zero-value transfers, as they did
    when these counts came from the vendor's transfer endpoint, so the census
    column keeps its meaning; the _all columns count every Transfer, and the
    block range covers every Transfer, because a zero-value transfer is still
    an interaction and the holder census must not miss one. The zero address
    is in the index -- it is where mints and burns come from and go to -- and
    is dropped where a counterparty is meant.
    """
    nz = logs[logs.raw_value != "0"]
    parts = []
    for column, direction in (("to", "in"), ("from", "out")):
        g = logs.groupby(column).block.agg(["size", "min", "max"])
        g.columns = [f"transfers_{direction}_all", f"first_block_{direction}",
                     f"last_block_{direction}"]
        counted = nz.groupby(column).block.size().rename(f"transfers_{direction}")
        g = g.join(counted, how="left")
        g.index.name = "address"
        for total, rows in ((f"transfers_{direction}_all", len(logs)),
                            (f"transfers_{direction}", len(nz))):
            if int(g[total].fillna(0).sum()) != rows:
                raise ValueError(f"per-address {total} sums to "
                                 f"{int(g[total].fillna(0).sum())}, {rows} rows "
                                 f"were stored")
        parts.append(g)
    out = parts[0].join(parts[1], how="outer")
    for c in out.columns:
        out[c] = out[c].astype("Int64")
    return out.fillna({"transfers_in": 0, "transfers_out": 0,
                       "transfers_in_all": 0, "transfers_out_all": 0})


def pool_activity(logs: pd.DataFrame, counts: pd.DataFrame, pool: str) -> dict:
    """One pool's counts, re-derived from the logs and checked against the table.

    The two derivations are independent -- a scan of the log rows against the
    groupby that produced `counts` -- so a mistake in either is loud.
    """
    nz = logs[logs.raw_value != "0"]
    blocks_in = nz.block[nz["to"] == pool]
    blocks_out = nz.block[nz["from"] == pool]
    touched = pd.concat([blocks_in, blocks_out])
    rec = {"transfers_in": int(len(blocks_in)), "transfers_out": int(len(blocks_out)),
           "first_transfer_block": int(touched.min()) if len(touched) else None,
           "last_transfer_block": int(touched.max()) if len(touched) else None}
    row = counts.loc[pool] if pool in counts.index else None
    tabled = (int(row.transfers_in), int(row.transfers_out)) if row is not None else (0, 0)
    if tabled != (rec["transfers_in"], rec["transfers_out"]):
        raise ValueError(f"pool {pool}: per-address table says {tabled}, the "
                         f"logs say {(rec['transfers_in'], rec['transfers_out'])}")
    return rec


# --- 2. factory census -----------------------------------------------------

def lookup(venue: Venue, variant: str, a: str, b: str) -> str:
    signature, _ = LOOKUPS[venue.kind]
    if venue.kind in ("solidly", "solidly_pool"):
        return calldata(signature, a, b, variant == "stable")
    if venue.kind in ("v3", "cl"):
        return calldata(signature, a, b, int(variant))
    return calldata(signature, a, b)


def identity(client, scope: Scope, pool: str) -> dict:
    f, t0, t1 = eth_calls(client, [(pool, calldata(s), scope.end)
                                   for s in ("factory()", "token0()", "token1()")])
    return {"factory": as_address(f), "token0": as_address(t0), "token1": as_address(t1)}


def check_pool(client, scope: Scope, venue: Venue, pool: str, a: str, b: str) -> dict:
    ident = identity(client, scope, pool)
    if ident["factory"] != venue.factory or {ident["token0"], ident["token1"]} != {a, b}:
        raise ValueError(f"{venue.name}: pool {pool} reports {ident}, "
                         f"expected factory {venue.factory} and tokens {a}/{b}")
    return ident


def verify_venues(client, scope: Scope) -> dict:
    """Every factory must resolve a reference pair to a pool that owns up to it."""
    path = cache("venues.json")
    store = load_cache(path)
    if scope.name in store:
        return store[scope.name]
    a, b = scope.other[scope.probe[0]], scope.stablecoins[scope.probe[1]]
    out = {}
    for v in scope.venues:
        if client.call("eth_getCode", [v.factory, hex(scope.end)]) in ("0x", ""):
            raise ValueError(f"{v.name}: no code at factory {v.factory}")
        variants = LOOKUPS[v.kind][1]
        found = {}
        for variant, res in zip(variants, eth_calls(
                client, [(v.factory, lookup(v, x, a, b), scope.end) for x in variants])):
            pool = as_address(res)
            if pool != ZERO:
                found[variant or "-"] = {"pool": pool,
                                         **check_pool(client, scope, v, pool, a, b)}
        if not found:
            raise ValueError(f"{v.name}: factory does not resolve "
                             f"{'/'.join(scope.probe)}; not verified")
        out[v.name] = {"chain": scope.name, "kind": v.kind, "factory": v.factory,
                       "lookup": LOOKUPS[v.kind][0], "probe": "/".join(scope.probe),
                       "resolved": found}
        say(f"  verified {scope.name}/{v.name:22s} {v.factory}  "
            f"({len(found)} probe pools)")
    store[scope.name] = out
    write_json(store, path)
    return out


def factory_lookups(client, scope: Scope) -> dict[str, str | None]:
    """Every (token, quote, venue, variant) -> pool, cached: 1,716 calls on Polygon."""
    path = cache("factory_lookups.json")
    store = load_cache(path)
    known = store.setdefault(scope.name, {})
    for v in scope.venues:
        calls, keys = [], []
        for t, q in scope.pairings():
            for variant in LOOKUPS[v.kind][1]:
                key = f"{t}|{q}|{v.name}|{variant}"
                if key in known:
                    continue
                calls.append((v.factory, lookup(v, variant, scope.tokens[t],
                                                scope.addresses[q]), scope.end))
                keys.append(key)
        if not calls:
            continue
        for key, res in zip(keys, eth_calls(client, calls)):
            pool = as_address(res)
            known[key] = None if pool == ZERO else pool
        write_json(store, path)
        say(f"  {scope.name}/{v.name}: {len(calls)} lookups")
    return known


def found_pools(scope: Scope, known: dict[str, str | None], sym: str) -> list[str]:
    return sorted({pool for key, pool in known.items()
                   if pool and key.split("|")[0] == sym})


def factory_census(client, scope: Scope, known: dict[str, str | None],
                   activity: dict[tuple[str, str], dict]) -> pd.DataFrame:
    """The lookup table, with each pool's transfer counts already derived."""
    by_venue = {v.name: v for v in scope.venues}
    # Only what the current registry asks for: a cache written for a venue or
    # a pairing since removed must not appear in the census.
    wanted = [f"{t}|{q}|{v.name}|{variant}" for v in scope.venues
              for t, q in scope.pairings() for variant in LOOKUPS[v.kind][1]]
    absent = [k for k in wanted if k not in known]
    if absent:
        raise ValueError(f"{scope.name}: {len(absent)} lookups were not read "
                         f"(first {absent[0]})")
    rows = []
    for key in wanted:
        pool = known[key]
        t, q, venue, variant = key.split("|")
        rows.append({"chain": scope.name, "tier": t, "quote": q,
                     "quote_kind": scope.quote_kind(q), "venue": venue,
                     "kind": by_venue[venue].kind, "variant": variant,
                     "factory": by_venue[venue].factory, "pool": pool,
                     "status": "pool" if pool else "no_pool"})
    df = pd.DataFrame(rows)
    found = df[df.status == "pool"]
    say(f"  {scope.name}: {len(df):,} lookups, {len(found)} pools")

    extra = []
    for r in found.itertuples():
        act = activity[(r.tier, r.pool)]
        path = cache("pools", f"{scope.name}_{r.pool}_{r.tier}.json")
        if path.exists():
            rec = json.loads(path.read_text())
        else:
            token, quote = scope.tokens[r.tier], scope.addresses[r.quote]
            check_pool(client, scope, by_venue[r.venue], r.pool, token, quote)
            balances = eth_calls(client, [
                (token, calldata("balanceOf(address)", r.pool), scope.end),
                (quote, calldata("balanceOf(address)", r.pool), scope.end)])
            lp = (str(as_uint(client.eth_call(r.pool, calldata("totalSupply()"),
                                              scope.end)))
                  if r.kind in ("v2", "solidly", "solidly_pool") else None)
            rec = {"pool": r.pool, "tier": r.tier,
                   "quote_ever_received":
                       first_transfer(client, scope, quote, "toAddress", r.pool)
                       is not None,
                   "lp_total_supply_raw": lp,
                   "tier_balance_end_raw": str(as_uint(balances[0])),
                   "quote_balance_end_raw": str(as_uint(balances[1]))}
            write_json(rec, path)
        extra.append({**rec, **act})
        say(f"  {r.tier:5s}/{r.quote:6s} {r.venue:22s} {r.variant:8s} {r.pool}  "
            f"in {act['transfers_in']:,} out {act['transfers_out']:,}")
    if extra:
        df = df.merge(pd.DataFrame(extra), on=["pool", "tier"], how="left")
    for col in ("transfers_in", "transfers_out"):
        df[col] = df[col].fillna(0).astype(int) if col in df else 0
    for col in ("first_transfer_block", "last_transfer_block"):
        df[col] = df[col].astype("Int64") if col in df else pd.NA
    return df


def first_transfer(client, scope: Scope, token: str, side: str, pool: str) -> int | None:
    """The first block in which `pool` was on `side` of a transfer of `token`."""
    res = client.call("alchemy_getAssetTransfers", [{
        "fromBlock": "0x0", "toBlock": hex(scope.end), "contractAddresses": [token],
        "category": ["erc20"], "excludeZeroValue": True, "maxCount": "0x1",
        "order": "asc", side: pool}])
    return int(res["transfers"][0]["blockNum"], 16) if res["transfers"] else None


# --- 3. venues without a per-pair factory ----------------------------------

def curve_census(client, scope: Scope) -> dict:
    """The Curve registry's own pool lookup for each pair, at END_BLOCK."""
    path = cache("curve.json")
    store = load_cache(path)
    coins = sorted(scope.addresses.values())
    if store.get("end_block") == scope.end and store.get("coins") == coins:
        return store
    registry = as_address(client.eth_call(
        CURVE_ADDRESS_PROVIDER, calldata("get_address(uint256)", CURVE_REGISTRY_ID),
        scope.end))
    if client.call("eth_getCode", [registry, hex(scope.end)]) in ("0x", ""):
        raise ValueError(f"Curve registry {registry} has no code at END_BLOCK")
    pairs = sorted({tuple(sorted((t, q))) for t, q in scope.pairings()})
    results = calls_allowing_reverts(client, [
        (registry, calldata("find_pool_for_coins(address,address,uint256)",
                            scope.addresses[a], scope.addresses[b], 0), scope.end)
        for a, b in pairs])
    found = {}
    for (a, b), res in zip(pairs, results):
        if res is None:
            continue                       # the registry reverts on an unknown pair
        pool = as_address(res)
        if pool != ZERO:
            found[f"{a}|{b}"] = pool
    store = {"chain": scope.name, "address_provider": CURVE_ADDRESS_PROVIDER,
             "registry": registry, "end_block": scope.end, "coins": coins,
             "lookup": "find_pool_for_coins(address,address,uint256)",
             "pairs_checked": len(pairs), "pools": found}
    write_json(store, path)
    say(f"  curve registry {registry}: {len(found)} of {len(pairs)} pairs resolve")
    return store


def v4_census(client, scope: Scope, blocks: Blocks) -> dict:
    """Uniswap V4 pools are events of one contract, not contracts of a factory.

    Initialize indexes currency0 and currency1, so a topic filter finds every
    pool holding a token in scope without reading the rest of the chain.
    """
    path = cache(f"uniswap_v4_{scope.name}.json")
    store = load_cache(path)
    watched = sorted(scope.tokens.values())
    if store.get("end_block") == scope.end and store.get("tokens") == watched:
        return store
    manager = UNISWAP_V4[scope.name]
    if client.call("eth_getCode", [manager, hex(scope.end)]) in ("0x", ""):
        raise ValueError(f"Uniswap V4 PoolManager {manager} has no code at END_BLOCK")
    init = event_topic(V4_INITIALIZE)
    start = blocks.deployment(manager)
    tokens = ["0x" + a[2:].rjust(64, "0") for a in scope.tokens.values()]
    pools = []
    for position, name in ((2, "currency0"), (3, "currency1")):
        topics: list = [init, None, None, None][:position] + [tokens]
        for lg in logs_in_range(client, {"address": manager, "topics": topics},
                                start, scope.end):
            pools.append({"pool_id": lg["topics"][1], "role": name,
                          "currency0": topic_address(lg["topics"][2]),
                          "currency1": topic_address(lg["topics"][3]),
                          "block": int(lg["blockNumber"], 16),
                          "tx_hash": lg["transactionHash"].lower()})
    store = {"chain": scope.name, "pool_manager": manager, "event": V4_INITIALIZE,
             "topic0": init, "from_block": start, "end_block": scope.end,
             "tokens": watched, "pools": pools}
    write_json(store, path)
    say(f"  uniswap v4 {scope.name}: {len(pools)} pools hold a token in scope")
    return store


# --- 4. holder census ------------------------------------------------------

def classify(client, scope: Scope, addresses: list[str]) -> dict:
    """Code, pool interface and symbol of every counterparty, at END_BLOCK.

    A revert is an answer here -- it says the contract has no such getter --
    so the probes travel in the same request as the calls that must succeed.
    """
    path = cache("holders", f"addresses_{scope.name}.json")
    known = load_cache(path)
    todo = [a for a in addresses if a not in known]
    if todo:
        say(f"  {scope.name}: classifying {len(todo):,} counterparties")
    # Classifying tens of thousands of addresses takes long enough that a
    # failure part-way through is likely, so whatever has been settled is
    # written before the error travels on: a rerun then resumes instead of
    # paying for the same answers again.
    try:
        for i in range(0, len(todo), BATCH):
            part = todo[i:i + BATCH]
            codes = client.batch([("eth_getCode", [a, hex(scope.end)]) for a in part])
            contracts = [a for a, code in zip(part, codes) if code not in ("0x", "")]
            probes = calls_allowing_reverts(client, [
                (a, calldata(sig), scope.end) for a in contracts
                for sig in ("token0()", "token1()", "symbol()")])
            got = {a: probes[3 * k:3 * k + 3] for k, a in enumerate(contracts)}
            for a, code in zip(part, codes):
                rec = {"is_contract": code not in ("0x", "")}
                if rec["is_contract"]:
                    t0, t1, sym = got[a]
                    rec.update(token0=probe_address(t0), token1=probe_address(t1),
                               symbol=probe_string(sym))
                    rec["stablecoin_pool"] = bool(
                        rec["token0"] and rec["token1"]
                        and {rec["token0"], rec["token1"]}
                        & set(scope.stablecoins.values()))
                known[a] = rec
            if i % (BATCH * 10) == 0:
                write_json(known, path)
    finally:
        write_json(known, path)
    return {a: known[a] for a in addresses}


def probe_address(result: str | None) -> str | None:
    if result is None or len(result) != 66 or int(result[2:26], 16):
        return None
    return "0x" + result[-40:].lower()


def probe_string(result: str | None) -> str | None:
    try:
        return as_string(result) if result and len(result) >= 66 else None
    except (ValueError, UnicodeDecodeError):
        return None


def receipts(client, scope: Scope, sym: str, hashes: list[str]) -> list[dict]:
    out = []
    for k, start in enumerate(range(0, len(hashes), RECEIPT_PART)):
        path = cache("holders", scope.name, sym, f"receipts_{k:05d}.jsonl.gz")
        part = hashes[start:start + RECEIPT_PART]
        if not path.exists():
            lines = []
            for i in range(0, len(part), BATCH):
                batch = part[i:i + BATCH]
                for h, rc in zip(batch, client.batch([("eth_getTransactionReceipt", [h])
                                                      for h in batch])):
                    if rc is None:
                        raise ValueError(f"receipt missing: {h}")
                    lines.append(json.dumps({
                        "tx_hash": h, "block": int(rc["blockNumber"], 16),
                        "status": int(rc["status"], 16),
                        "transfers": [[lg["address"].lower(),
                                       topic_address(lg["topics"][1]),
                                       topic_address(lg["topics"][2]),
                                       str(int(lg["data"], 16))]
                                      for lg in rc["logs"]
                                      if len(lg["topics"]) == 3
                                      and lg["topics"][0] == TRANSFER_TOPIC
                                      and len(lg["data"]) == 66]}))
            tmp = path.with_name(path.name + ".tmp")
            with gzip.open(tmp, "wt") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(tmp, path)
            say(f"  {sym} receipts part {k}: {len(part):,}")
        with gzip.open(path, "rt") as f:
            got = [json.loads(line) for line in f if line.strip()]
        if [g["tx_hash"] for g in got] != part:
            raise ValueError(f"{path}: receipts do not match the transaction list")
        out += got
    return out


def holder_frame(scope: Scope, sym: str, counts: pd.DataFrame,
                 cls: dict) -> pd.DataFrame:
    rows = []
    for address, r in counts.iterrows():
        rows.append({"chain": scope.name, "tier": sym, "address": address,
                     "transfers": int(r.transfers_in_all) + int(r.transfers_out_all),
                     "transfers_in": int(r.transfers_in),
                     "transfers_out": int(r.transfers_out),
                     "first_block": int(min(x for x in (r.first_block_in,
                                                        r.first_block_out)
                                            if pd.notna(x))),
                     "last_block": int(max(x for x in (r.last_block_in,
                                                       r.last_block_out)
                                           if pd.notna(x))),
                     **cls[address]})
    return pd.DataFrame(rows)


def direct_exchanges(scope: Scope, sym: str, logs: pd.DataFrame, rcs: list[dict],
                     cls: dict) -> dict:
    """Transactions in which one address's token flow and stablecoin flow oppose."""
    token = scope.tokens[sym]
    stables = {v: k for k, v in scope.stablecoins.items()}
    failed, with_stable, candidates = 0, 0, []
    for rc in rcs:
        if rc["status"] != 1:
            failed += 1
            continue
        net_tier: dict = defaultdict(int)
        net_stable: dict = defaultdict(lambda: defaultdict(int))
        for tok, frm, to, value in rc["transfers"]:
            v = int(value)
            if tok == token:
                net_tier[frm] -= v
                net_tier[to] += v
            elif tok in stables:
                net_stable[stables[tok]][frm] -= v
                net_stable[stables[tok]][to] += v
        with_stable += bool(net_stable)
        other_tokens = sorted({scope.symbol_of.get(tok, tok) for tok, *_ in rc["transfers"]
                               if tok != token and tok not in stables})
        for a, nt in net_tier.items():
            if nt == 0 or a == ZERO:
                continue
            for s, flows in net_stable.items():
                ns = flows.get(a, 0)
                if not ns or (ns > 0) == (nt > 0):
                    continue
                tier_cp = sorted({to if frm == a else frm
                                  for tok, frm, to, _ in rc["transfers"]
                                  if tok == token and a in (frm, to)})
                stable_cp = sorted({to if frm == a else frm
                                    for tok, frm, to, _ in rc["transfers"]
                                    if tok == scope.stablecoins[s] and a in (frm, to)})
                candidates.append({
                    "tx_hash": rc["tx_hash"], "block": rc["block"], "address": a,
                    # Every address that moved the token is in the stored logs,
                    # so a missing classification is a bug, not an unknown.
                    "is_contract": cls[a]["is_contract"],
                    "tier_net_raw": str(nt), "stablecoin": s,
                    "stablecoin_net_raw": str(ns),
                    "tier_counterparties": tier_cp,
                    "stablecoin_counterparties": stable_cp,
                    "other_tokens_moved": other_tokens,
                    # Only the carbon token and stablecoins moved: nothing was
                    # routed through KLIMA or any other intermediate asset.
                    "route": "direct" if not other_tokens else "routed"})
    direct = [c for c in candidates if c["is_contract"]]
    unrouted = [c for c in candidates if c["route"] == "direct"]
    return {
        "transactions": len(rcs), "transactions_failed_status": failed,
        "transactions_with_a_stablecoin_transfer": with_stable,
        "direct_exchange_contracts": sorted({c["address"] for c in direct}),
        "direct_exchange_events": direct,
        "eoa_opposite_flow_events": sum(1 for c in candidates if not c["is_contract"]),
        "unrouted_stablecoin_exchange_transactions": len({c["tx_hash"] for c in unrouted}),
        "unrouted_stablecoin_exchange_addresses": sorted({c["address"] for c in unrouted}),
        "unrouted_stablecoin_exchange_blocks": [
            min((c["block"] for c in unrouted), default=None),
            max((c["block"] for c in unrouted), default=None)],
        "unrouted_stablecoin_exchange_token_raw": str(sum(abs(int(c["tier_net_raw"]))
                                                          for c in unrouted
                                                          if c["is_contract"])),
        "logs_covered": int(len(logs)),
    }


# --- driver ----------------------------------------------------------------

SCOPE_TEXT = (
    "Polygon blocks 0-{polygon_end} and Base blocks 0-{base_end}, each the last "
    "block before 2026-07-26 00:00 UTC. Every Transfer log of every token in "
    "scope is stored and reconciled against totalSupply(), and every count "
    "below is derived from those logs rather than from a vendor endpoint. "
    "Factory census: each token paired with every quote in its chain's scope, "
    "looked up in {n} factories verified on-chain. Registry and singleton "
    "venues: the Curve registry's find_pool_for_coins on Polygon, and the "
    "Uniswap V4 PoolManager's Initialize logs on both chains. Holder census: "
    "every counterparty of every token classified by code and token0/token1; "
    "for {receipts} the receipt of every transaction that moved the token is "
    "read as well and tested for a contract exchanging the token directly "
    "against a stablecoin."
)
NOT_COVERED = (
    "Venues with no per-pair factory, no registry entry and no singleton event "
    "scanned here -- Balancer, DODO, Kyber, and Solidly or Algebra forks not "
    "listed -- are reached only through the holder census, which observes them "
    "as counterparties of the token but cannot name the venue. Pools against "
    "quote tokens outside each chain's scope are not looked up. Chains other "
    "than Polygon and Base are out of scope."
)


def run(receipt_tokens: tuple[str, ...] | None = None, estimate: bool = False) -> None:
    clients = {name: metered(name) for name in SCOPES}
    recon, census_frames, holder_frames = {}, [], []
    holders, curve, v4, venues = {}, {}, {}, {}

    for name, scope in SCOPES.items():
        client = clients[name]
        blocks = Blocks(client, name)
        for sym, address in scope.tokens.items():
            got = as_string(client.eth_call(address, calldata("symbol()"), scope.end))
            if got != sym:
                raise ValueError(f"{name}: {address} reports symbol {got!r}, "
                                 f"expected {sym}")
        starts = {sym: blocks.deployment(a) for sym, a in scope.tokens.items()}
        recon[name] = {}
        if not estimate:
            venues[name] = verify_venues(client, scope)
        # The pools are known before the logs are read, so each token's logs
        # can be summarised and released instead of all six being held at once.
        known = factory_lookups(client, scope)
        wanted = scope.receipts if receipt_tokens is None else receipt_tokens
        activity = {}

        for sym in scope.tokens:
            fetch_token_logs(client, scope, sym, starts[sym])
            frame = load_token_logs(scope, sym, starts[sym])
            table = address_counts(frame)
            recon[name][sym] = reconcile_supply(client, scope, sym, frame)
            say(f"  {name} {sym}: {recon[name][sym]['transfers']:,} transfers, "
                f"{recon[name][sym]['counterparties']:,} counterparties, "
                f"supply reconciled at END_BLOCK")
            index = holder_index(frame, mint_source(scope, sym))
            say_classification_cost(name, sym, index)
            if estimate:
                del frame, table
                continue
            for pool in found_pools(scope, known, sym):
                activity[(sym, pool)] = pool_activity(frame, table, pool)

            table = table.reindex(index)
            cls = classify(client, scope, list(index))
            holder_frames.append(holder_frame(scope, sym, table, cls))
            summary = {**recon[name][sym],
                       "contracts": sum(1 for c in cls.values() if c["is_contract"]),
                       "pool_contracts": [
                           {"address": a, "token0": scope.symbol_of.get(c["token0"], c["token0"]),
                            "token1": scope.symbol_of.get(c["token1"], c["token1"]),
                            "symbol": c.get("symbol"),
                            "transfers": int(table.loc[a, "transfers_in_all"])
                                         + int(table.loc[a, "transfers_out_all"])}
                           for a, c in cls.items()
                           if c["is_contract"] and c.get("token0") and c.get("token1")],
                       "stablecoin_pool_contracts": sum(
                           1 for c in cls.values()
                           if c["is_contract"] and c.get("stablecoin_pool")),
                       "receipts_read": False}
            if sym in wanted:
                hashes = list(dict.fromkeys(frame.tx_hash))
                summary.update(receipts_read=True,
                               **direct_exchanges(scope, sym, frame,
                                                  receipts(client, scope, sym, hashes),
                                                  cls))
            holders[f"{name}:{sym}"] = summary
            del frame, table, cls

        if estimate:
            continue
        census_frames.append(factory_census(client, scope, known, activity))
        if scope.curve:
            curve[name] = curve_census(client, scope)
        if scope.v4:
            v4[name] = v4_census(client, scope, blocks)

    if estimate:
        return

    census = pd.concat(census_frames, ignore_index=True)
    write_csv(census, data("market_existence.csv"))
    write_csv(pd.concat(holder_frames, ignore_index=True),
              data("market_existence_holders.csv"))

    found = census[census.status == "pool"]
    tokens = {}
    for name, scope in SCOPES.items():
        for sym in scope.tokens:
            f = found[(found.chain == name) & (found.tier == sym)]
            st = f[f.quote_kind == "stablecoin"]
            columns = ["quote", "venue", "variant", "pool", "transfers_in",
                       "transfers_out"]
            tokens[f"{name}:{sym}"] = {
                "pools": int(len(f)),
                "stablecoin_pools": int(len(st)),
                "stablecoin_pools_with_token_transfers":
                    int(((st.transfers_in + st.transfers_out) > 0).sum()),
                "stablecoin_pools_list": st[columns].to_dict("records"),
                "non_stablecoin_pools_list":
                    f[f.quote_kind != "stablecoin"][columns].to_dict("records"),
            }
    out = {"end_block": {name: s.end for name, s in SCOPES.items()},
           "scope": SCOPE_TEXT.format(
               polygon_end=END_BLOCK["polygon"], base_end=END_BLOCK["base"],
               n=sum(len(s.venues) for s in SCOPES.values()),
               receipts=", ".join(RECEIPT_CENSUS)),
           "not_covered": NOT_COVERED,
           "venues": venues, "transfer_logs": recon, "tokens": tokens,
           "curve_registry": curve, "uniswap_v4": v4, "holder_census": holders,
           "lookups": int(len(census)),
           # A handful of contracts have getters that cannot be evaluated
           # inside the node's gas cap. They are counted rather than dropped,
           # because a census that quietly discards what it could not read
           # overstates how much of the market it saw.
           "probes_unanswered": sorted(UNANSWERED)}
    write_json(out, data("market_existence.json"))

    say("\nStablecoin pools (any venue in scope):")
    for key, t in tokens.items():
        h = holders.get(key, {})
        say(f"  {key:14s} {t['stablecoin_pools']} stablecoin pools "
            f"({t['stablecoin_pools_with_token_transfers']} with transfers); "
            f"{h.get('counterparties', 0):,} counterparties, "
            f"{h.get('contracts', 0):,} contracts, "
            f"{len(h.get('pool_contracts', []))} pool-like, "
            f"{h.get('stablecoin_pool_contracts', 0)} of them holding a stablecoin")
    for name, client in clients.items():
        record_spend(MODULE, client)
    raw_manifest(MODULE, {"end_block": {n: s.end for n, s in SCOPES.items()}})


def say_classification_cost(chain: str, sym: str, index: pd.Index) -> None:
    """What classifying this token's unclassified counterparties would cost.

    The cache is shared by every token on the chain, so an address that holds
    two of them is read once and is counted here only for the first.
    """
    known = set(load_cache(cache("holders", f"addresses_{chain}.json")))
    fresh = len(set(index) - known)
    # eth_getCode for every address, three probes for those with code; the
    # share with code is unknown before the call, so both bounds are given.
    low, high = fresh * CU_CODE, fresh * (CU_CODE + 3 * CU_CALL)
    say(f"    {chain}:{sym:6s} {len(index):>8,} counterparties, {fresh:>8,} to "
        f"classify: {low:,}-{high:,} compute units "
        f"(${low * CU_PRICE:,.2f}-{high * CU_PRICE:,.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--receipts", action="append",
                    help="token whose transaction receipts to read, repeatable; "
                         f"default {', '.join(RECEIPT_CENSUS)}")
    ap.add_argument("--estimate", action="store_true",
                    help="fetch the Transfer logs, then report what classifying "
                         "their counterparties would cost, and stop")
    args = ap.parse_args()
    run(tuple(args.receipts) if args.receipts else None, args.estimate)


if __name__ == "__main__":
    main()
