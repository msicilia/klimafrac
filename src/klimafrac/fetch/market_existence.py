"""
Which markets each carbon token ever had, read from the chain.

Everything is pinned at END_BLOCK on each chain, the last block before
2026-07-26 00:00 UTC: Polygon 90,879,334, Base 49,117,326.

1. Transfer logs. Every Transfer event of every token in scope, read with
   eth_getLogs over the token's own address and stored verbatim under
   data/raw/market_existence/logs/, so every count below can be re-audited
   without asking the node again. Each token's log stream is reconciled
   against totalSupply() at END_BLOCK -- minted minus burned must equal it to
   the unit -- which is what establishes that no Transfer is missing. Where a
   mint comes from is a property of the token: the carbon tokens issue from
   the zero address, KLIMA from its own address, as Olympus forks do (see
   MINTS_FROM_SELF).

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

Output:
  data/market_existence.csv  one row per (chain, token, quote, venue, variant)
                             lookup, pool or not: status, pool,
                             transfers_in/out, first/last transfer block,
                             quote_ever_received, LP supply, END_BLOCK balances
Raw: data/raw/market_existence/ (logs, per-pool reads, caches, manifest.json,
     rpc_calls_<chain>.json)

Usage:
    uv run python -m klimafrac.fetch.market_existence
"""

import argparse
import json
from dataclasses import dataclass

import pandas as pd

from klimafrac.fetch.chain import (LOG_COLUMNS, TRANSFER_TOPIC, ZERO, Blocks,
                                   as_address, as_string, as_uint, calldata,
                                   chunks, decode_transfer_logs, eth_calls,
                                   logs_in_range, metered, raw_manifest,
                                   record_spend, say, write_csv, write_json)
from klimafrac.fetch.pools import END_BLOCK
from klimafrac.paths import data, raw

MODULE = "market_existence"
CHUNK = 5_000_000            # blocks per stored log file

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

# Where a token's new supply comes from. ERC-20 mints from the zero address,
# and the carbon tokens all do. KLIMA is an Olympus fork, whose _mint emits
# Transfer(address(this), to, amount) instead. Declared rather than inferred;
# the supply reconciliation fails if an entry here is wrong.
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

@dataclass(frozen=True)
class Scope:
    """One chain's tokens, quotes and venues, all pinned at its END_BLOCK."""
    name: str
    tokens: dict[str, str]
    stablecoins: dict[str, str]
    other: dict[str, str]
    venues: list[Venue]
    probe: tuple[str, str]

    @property
    def end(self) -> int:
        return END_BLOCK[self.name]

    @property
    def addresses(self) -> dict[str, str]:
        return {**self.tokens, **self.stablecoins, **self.other}

    def quotes(self) -> list[str]:
        return [*self.stablecoins, *self.other, *self.tokens]

    def pairings(self) -> list[tuple[str, str]]:
        return [(t, q) for t in self.tokens for q in self.quotes() if t != q]

    def quote_kind(self, sym: str) -> str:
        return ("stablecoin" if sym in self.stablecoins
                else "carbon" if sym in self.tokens else "other")


SCOPES = {
    "polygon": Scope("polygon", POLYGON_TOKENS, POLYGON_STABLECOINS, POLYGON_OTHER,
                     POLYGON_VENUES, ("WETH", "USDC.e")),
    "base": Scope("base", BASE_TOKENS, BASE_STABLECOINS, BASE_OTHER,
                  BASE_VENUES, ("WETH", "USDC")),
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


def reconcile_supply(client, scope: Scope, sym: str, logs: pd.DataFrame) -> None:
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


def address_counts(logs: pd.DataFrame) -> pd.DataFrame:
    """Per address: how many non-zero-value transfers reached it and left it."""
    nz = logs[logs.raw_value != "0"]
    parts = []
    for column, direction in (("to", "in"), ("from", "out")):
        g = nz.groupby(column).block.size().rename(f"transfers_{direction}")
        g.index.name = "address"
        if int(g.sum()) != len(nz):
            raise ValueError(f"per-address transfers_{direction} sums to "
                             f"{int(g.sum())}, {len(nz)} rows were stored")
        parts.append(g)
    return pd.concat(parts, axis=1).fillna(0).astype(int)


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
    # Only the lookups the venue list and pairings ask for: cached entries for
    # anything else are ignored.
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


# --- driver ----------------------------------------------------------------

def run() -> None:
    clients = {name: metered(name) for name in SCOPES}
    census_frames = []

    for name, scope in SCOPES.items():
        client = clients[name]
        blocks = Blocks(client, name)
        for sym, address in scope.tokens.items():
            got = as_string(client.eth_call(address, calldata("symbol()"), scope.end))
            if got != sym:
                raise ValueError(f"{name}: {address} reports symbol {got!r}, "
                                 f"expected {sym}")
        starts = {sym: blocks.deployment(a) for sym, a in scope.tokens.items()}
        verify_venues(client, scope)
        # The pools are known before the logs are read, so each token's logs
        # can be summarised and released instead of all six being held at once.
        known = factory_lookups(client, scope)
        activity = {}

        for sym in scope.tokens:
            fetch_token_logs(client, scope, sym, starts[sym])
            frame = load_token_logs(scope, sym, starts[sym])
            table = address_counts(frame)
            reconcile_supply(client, scope, sym, frame)
            say(f"  {name} {sym}: {len(frame):,} transfers, "
                f"supply reconciled at END_BLOCK")
            for pool in found_pools(scope, known, sym):
                activity[(sym, pool)] = pool_activity(frame, table, pool)
            del frame, table

        census_frames.append(factory_census(client, scope, known, activity))

    census = pd.concat(census_frames, ignore_index=True)
    write_csv(census, data("market_existence.csv"))

    found = census[census.status == "pool"]
    say("\nStablecoin pools (any venue in scope):")
    for name, scope in SCOPES.items():
        for sym in scope.tokens:
            st = found[(found.chain == name) & (found.tier == sym)
                       & (found.quote_kind == "stablecoin")]
            active = int(((st.transfers_in + st.transfers_out) > 0).sum())
            say(f"  {name}:{sym:9s} {len(st)} stablecoin pools "
                f"({active} with transfers)")
    for name, client in clients.items():
        record_spend(MODULE, client)
    raw_manifest(MODULE, {"end_block": {n: s.end for n, s in SCOPES.items()}})


def main() -> None:
    argparse.ArgumentParser(description=__doc__.strip().split("\n")[0]).parse_args()
    run()


if __name__ == "__main__":
    main()
