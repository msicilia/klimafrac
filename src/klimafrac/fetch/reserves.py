"""
Pool reserves at every daily boundary, and the pool value they imply.

Figure 7's value panel is read from the chain, so the value has to come from
the pool's own balance sheet rather than from an aggregator. For each pool in
fetch/pools.py this module calls getReserves() at the last block before 00:00
UTC on every day of the window and records both reserves exactly.

Value. A constant-product or Solidly pair holds both legs, so
    value = base_reserve x price + quote_reserve x quote_usd
where price is the pool's own spot, quote_reserve / base_reserve adjusted for
decimals -- the price the pool would quote for an infinitesimal trade, taken
from the same reading as the reserves, so no external series enters. The only
assumption is what a unit of the quote is worth in dollars: 1.0 for USDC,
USDC.e and other stablecoins, and for KLIMA-quoted pools (UBO, NBO) the spot
of the KLIMA/USDC.e pool at the same block, which is itself read here. Every
row records `quote_usd` and where it came from, so an alternative convention
can be applied to the stored reserves without refetching.

Pinned range: the last block before 00:00 UTC on each day from 2021-10-01 to
2026-07-26 inclusive, on Polygon and on Base, the boundary blocks being the
same ones fetch/inventory.py and fetch/treasury.py use. A pool contributes
rows only from the day its pair contract has code; before that its status is
not_deployed. A pool holding no reserves is `empty`: its value is zero and
its price is undefined, which is a reading rather than a failure.

Outputs:
  data/reserves_daily.csv   date, as_of_instant, chain, pool, pair, block,
                            status, reserve0_raw, reserve1_raw,
                            base_reserve, quote_reserve, price,
                            quote_symbol, quote_usd, value_usd
  data/reserves_summary.json  per pool: pair identity, decimals, first and
                            last day with reserves, peak value and when it
                            was reached, value at the window end
Raw: data/raw/reserves/<pool>/<year>.json, pools.json, manifest.json,
     rpc_calls_<chain>.json

Usage:
    uv run python -m klimafrac.fetch.reserves
    uv run python -m klimafrac.fetch.reserves --pool polygon_BCT
"""

import argparse
import json

import pandas as pd

from klimafrac.fetch.chain import (BATCH, WINDOW_END, Blocks, as_address, as_uint,
                                   as_uints, calldata, eth_calls, iso, metered,
                                   raw_manifest, record_spend, say, utc_ts,
                                   write_csv, write_json)
from klimafrac.fetch.pools import BY_KEY, END_BLOCK, POOLS
from klimafrac.paths import data, raw

MODULE = "reserves"
FIRST_DATE = "2021-10-01"
STABLE_USD = {                       # a unit of these is taken to be one dollar
    "0x2791bca1f2de4661ed88a30c99a7a9449aa84174": "USDC.e",
    "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913": "USDC",
}
KLIMA = "0x4e78011ce80ee02d2c3e649fb657e45898257815"
KLIMA_POOL = "polygon_KLIMA"         # the pool that prices the KLIMA quote
COLUMNS = ["date", "as_of_instant", "chain", "pool", "pair", "block", "status",
           "reserve0_raw", "reserve1_raw", "reserves_timestamp_last",
           "base_reserve", "quote_reserve", "price", "quote_symbol",
           "quote_usd", "quote_usd_source", "value_usd"]


def describe(client, blocks: Blocks, pool) -> dict:
    """The pair's identity and decimals, checked against the registry once."""
    path = raw(MODULE, "pools.json")
    store = json.loads(path.read_text()) if path.exists() else {}
    if pool.key in store:
        return store[pool.key]
    end = END_BLOCK[pool.chain]
    token0, token1, factory = (as_address(r) for r in eth_calls(client, [
        (pool.pair, calldata(s), end) for s in ("token0()", "token1()", "factory()")]))
    if {token0, token1} != {pool.base, pool.quote}:
        raise ValueError(f"{pool.key}: pair holds {token0}/{token1}, registry says "
                         f"{pool.base}/{pool.quote}")
    if factory != pool.factory:
        raise ValueError(f"{pool.key}: factory {factory}, registry {pool.factory}")
    decimals = {t: as_uint(r) for t, r in zip(
        (token0, token1), eth_calls(client, [(t, calldata("decimals()"), end)
                                             for t in (token0, token1)]))}
    store[pool.key] = {"pool": pool.key, "chain": pool.chain, "pair": pool.pair,
                       "base": pool.base, "quote": pool.quote,
                       "base_is_token0": token0 == pool.base,
                       "base_decimals": decimals[pool.base],
                       "quote_decimals": decimals[pool.quote],
                       "deploy_block": blocks.deployment(pool.pair)}
    write_json(store, path)
    return store[pool.key]


def daily_instants() -> list[int]:
    days = pd.date_range(FIRST_DATE, WINDOW_END.tz_localize(None), freq="D", tz="UTC")
    return [utc_ts(d) for d in days]


def fetch_reserves(client, pool, meta: dict, at: dict[int, int]) -> dict[str, dict]:
    """getReserves() at each daily boundary block, one stored file per year."""
    by_year: dict[str, list] = {}
    for t in daily_instants():
        if t in at and at[t] >= meta["deploy_block"]:
            by_year.setdefault(iso(t)[:4], []).append(t)
    out: dict[str, dict] = {}
    for year, instants in sorted(by_year.items()):
        path = raw(MODULE, pool.key, f"{year}.json")
        if path.exists():
            record = json.loads(path.read_text())
            if sorted(record) == sorted(iso(t) for t in instants):
                out.update(record)
                continue
        record = {}
        for i in range(0, len(instants), BATCH):
            part = instants[i:i + BATCH]
            for t, res in zip(part, eth_calls(
                    client, [(pool.pair, calldata("getReserves()"), at[t]) for t in part])):
                r0, r1, last = as_uints(res, 3)
                record[iso(t)] = {"block": at[t], "reserve0_raw": str(r0),
                                  "reserve1_raw": str(r1),
                                  "reserves_timestamp_last": last}
        write_json(record, path)
        say(f"  {pool.key} {year}: {len(record)} daily readings")
        out.update(record)
    return out


def rows_for(pool, meta: dict, readings: dict, at: dict[int, int],
             klima_price: dict[str, float] | None) -> list[dict]:
    quote_symbol = STABLE_USD.get(pool.quote, "KLIMA" if pool.quote == KLIMA else None)
    if quote_symbol is None:
        raise ValueError(f"{pool.key}: no dollar convention for quote {pool.quote}")
    base_unit = 10 ** meta["base_decimals"]
    quote_unit = 10 ** meta["quote_decimals"]
    rows = []
    for t in daily_instants():
        instant = iso(t)
        row = {"date": instant[:10], "as_of_instant": instant, "chain": pool.chain,
               "pool": pool.key, "pair": pool.pair, "block": at.get(t),
               "quote_symbol": quote_symbol}
        reading = readings.get(instant)
        if reading is None:
            # Base opened in 2023, after the window did, so its pools have no
            # block to read at all on the early days: that is a different
            # statement from a pool that did not yet exist on a live chain.
            status = "chain_not_launched" if t not in at else "not_deployed"
            row.update(status=status, reserve0_raw=None, reserve1_raw=None,
                       reserves_timestamp_last=None, base_reserve=0.0,
                       quote_reserve=0.0, price=float("nan"),
                       quote_usd=float("nan"), quote_usd_source=status,
                       value_usd=float("nan"))
            rows.append(row)
            continue
        if reading["block"] != at[t]:
            raise ValueError(f"{pool.key}: the stored reading for {instant} is at "
                             f"block {reading['block']}, this run pins {at[t]}")
        r0, r1 = int(reading["reserve0_raw"]), int(reading["reserve1_raw"])
        base_raw, quote_raw = (r0, r1) if meta["base_is_token0"] else (r1, r0)
        base, quote = base_raw / base_unit, quote_raw / quote_unit
        if quote_symbol == "KLIMA":
            quote_usd = (klima_price or {}).get(instant, float("nan"))
            source = KLIMA_POOL
        else:
            quote_usd, source = 1.0, f"{quote_symbol} taken as one dollar"
        price = quote / base if base_raw and quote_raw else float("nan")
        row.update(status="ok" if base_raw and quote_raw else "empty",
                   reserve0_raw=str(r0), reserve1_raw=str(r1),
                   reserves_timestamp_last=reading["reserves_timestamp_last"],
                   base_reserve=base, quote_reserve=quote, price=price,
                   quote_usd=quote_usd, quote_usd_source=source,
                   value_usd=(base * price + quote) * quote_usd
                   if base_raw and quote_raw else 0.0)
        rows.append(row)
    return rows


def summarise(frame: pd.DataFrame, meta: dict) -> dict:
    priced = frame[frame.value_usd.notna() & (frame.value_usd > 0)]
    if priced.empty:
        return {**meta, "days_with_reserves": 0}
    top = priced.loc[priced.value_usd.idxmax()]
    last = priced.iloc[-1]
    return {**meta,
            "days_with_reserves": int(len(priced)),
            "first_day_with_reserves": priced.date.iloc[0],
            "last_day_with_reserves": priced.date.iloc[-1],
            "peak_value_usd": float(top.value_usd), "peak_as_of": top.as_of_instant,
            "peak_price": float(top.price),
            "last_value_usd": float(last.value_usd), "last_as_of": last.as_of_instant,
            "days_empty": int((frame.status == "empty").sum()),
            "days_not_deployed": int((frame.status == "not_deployed").sum()),
            "days_before_chain_launched": int((frame.status == "chain_not_launched").sum())}


def run(keys: list[str]) -> None:
    pools = [BY_KEY[k] for k in keys]
    klima_pool = BY_KEY[KLIMA_POOL]
    needs_klima = any(p.quote == KLIMA for p in pools)
    chains = sorted({p.chain for p in pools}
                    | ({klima_pool.chain} if needs_klima else set()))
    clients = {c: metered(c) for c in chains}
    blocks = {c: Blocks(clients[c], c) for c in chains}
    instants = daily_instants()
    # A chain has no block before its own genesis, so those days are asked of
    # nothing rather than answered with the first block it ever had.
    at = {c: blocks[c].block_before([t for t in instants
                                     if t > blocks[c].timestamp(0)])
          for c in chains}
    for c in chains:
        if at[c][utc_ts(WINDOW_END)] != END_BLOCK[c]:
            raise ValueError(f"{c}: the window end maps to block "
                             f"{at[c][utc_ts(WINDOW_END)]}, END_BLOCK is {END_BLOCK[c]}")

    # KLIMA-quoted pools need the KLIMA price at the same blocks, so that pool
    # is read first whether or not it was asked for.
    klima_price = None
    if needs_klima:
        client = clients[klima_pool.chain]
        meta = describe(client, blocks[klima_pool.chain], klima_pool)
        readings = fetch_reserves(client, klima_pool, meta, at[klima_pool.chain])
        klima_rows = rows_for(klima_pool, meta, readings, at[klima_pool.chain], None)
        klima_price = {r["as_of_instant"]: r["price"] for r in klima_rows
                       if r["status"] == "ok"}

    frames, summary = [], {}
    for pool in pools:
        client = clients[pool.chain]
        meta = describe(client, blocks[pool.chain], pool)
        readings = fetch_reserves(client, pool, meta, at[pool.chain])
        frame = pd.DataFrame(rows_for(pool, meta, readings, at[pool.chain], klima_price),
                             columns=COLUMNS)
        missing = frame[(frame.status == "ok") & frame.value_usd.isna()]
        if len(missing):
            raise ValueError(f"{pool.key}: {len(missing)} days have reserves but no "
                             f"value (first {missing.date.iloc[0]}): the quote price "
                             f"is missing, which must not be recorded as a zero")
        frames.append(frame)
        summary[pool.key] = summarise(frame, meta)
        s = summary[pool.key]
        say(f"  {pool.key:16s} {s.get('days_with_reserves', 0):,} days with reserves; "
            f"peak ${s.get('peak_value_usd', 0):,.0f} at {s.get('peak_as_of')}; "
            f"end ${s.get('last_value_usd', 0):,.0f}")

    write_csv(pd.concat(frames, ignore_index=True), data("reserves_daily.csv"))
    write_json({"first_date": FIRST_DATE,
                "window_end_exclusive": iso(utc_ts(WINDOW_END)),
                "end_block": {c: END_BLOCK[c] for c in chains},
                "quote_convention": {"stablecoins": sorted(STABLE_USD.values()),
                                     "klima_priced_by": KLIMA_POOL},
                "pools": summary}, data("reserves_summary.json"))
    for client in clients.values():
        record_spend(MODULE, client)
    raw_manifest(MODULE, {"pools": sorted(summary)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--pool", action="append", choices=sorted(BY_KEY),
                    help="pool key, repeatable; default every pool")
    args = ap.parse_args()
    run(args.pool or [p.key for p in POOLS])


if __name__ == "__main__":
    main()
