"""
Census of candidate pools for a conventional benchmark as thin as the carbon pools.

The asset-class comparison sets the carbon tokens against WETH and WBTC, the
most active pools on the same two exchanges, so it cannot tell a carbon effect
from a thin-market effect. This module lists the pools from which a matched
benchmark can be drawn, using nothing but the chain and nothing about their
price dynamics, so the choice cannot be tuned to a result.

Universe. Every constant-product pair that the Sushiswap and Quickswap
factories on Polygon ever created against USDC.e -- the quote of every pool
already analysed -- read from the factories' PairCreated logs up to END_BLOCK.
Factories never remove pairs, so the list is complete.

Screen. A pair can supply the same windows as the carbon pools only if it
existed when the last of them started trading and still held reserves when
the estimation sample ends. getReserves() is therefore read at three anchor
instants (ANCHORS) for every pair created before the first one.

Yardstick. For the pairs that pass, getReserves() is read at the last block
before 00:00 UTC on every fifth day of the estimation sample, the boundary
fetch/reserves.py uses daily for the pools already analysed. Each reading carries the quote reserve
(depth) and blockTimestampLast, the time of the pair's last swap, mint, burn
or sync. The share of readings at which that is more than an hour old
estimates the share of hours with no trade: read daily on the seven pools
whose swaps are known, it lands within a few points of the exact figure.

Rule. A candidate is kept when it is as thin as the carbon pools on every
measure, each range being the one the four carbon pools quoted in USDC.e
(BCT, NCT, MCO2, KLIMA) span on the same readings:
  1. the share of readings idle for more than an hour lies in their range;
  2. the share idle for more than a day does not exceed their largest;
  3. its price is not pegged: the pool price moved by at least PEG_RANGE
     between its lowest and highest reading (stablecoins move by less than
     1.7, the next candidate by 2.9, WETH by 4.5);
  4. one pool per token, the one with the larger median quote reserve;
  5. its median quote reserve lies in the carbon pools' range;
  6. its swap history can be built. One pool that meets the other five cannot
     (UNBUILDABLE): the transfer index fetch/swaps.py takes block times from
     omits a block in which the pair swapped, so the history fails the
     fetcher's completeness check. It was set aside before any of these
     pools was analysed.
The pools kept are registered in fetch/pools.py with role "thin", and `select`
fails unless that registry is exactly the rule's output.

Active pools. The same readings give the mirror image, a benchmark of pools
that trade more often than any carbon pool: the share of readings idle for
more than an hour lies below the carbon pools' range, and criteria 2, 3, 4
and 6 apply unchanged. Depth is not constrained, since idleness is what the
groups are matched or contrasted on. These pools are registered with role
"active" under the same check. The rule was added after the thin pools had
been analysed and before any active pool was.

Verification. Once a registered pool's logs are fetched, `verify` sets them
against the readings above, which came from a different method: the reserves
in the last Sync log at or before each reading's block must equal what
getReserves() returned at that block, to the unit. Agreement shows the logs
belong to that pair and reproduce its state at about three hundred points
spread over the sample.

Outputs:
  data/thin_candidates_pairs.csv     every USDC.e pair, with its creation block
  data/thin_candidates_screen.csv    anchor readings for pairs old enough
  data/thin_candidates_daily.csv     depth and idleness of pairs that pass
  data/thin_candidates_measures.csv  one row per pair that passes, with each
                                     measure and each criterion of the rule
  data/thin_candidates_selected.csv  the pools the rule keeps
  data/thin_candidates_active.csv    the pools the mirror rule keeps
  data/thin_candidates_verified.csv  logs against readings, per fetched pool
Raw: data/raw/thin_candidates/

Usage:
    uv run python -m klimafrac.fetch.thin_candidates --stage pairs
    uv run python -m klimafrac.fetch.thin_candidates --stage screen
    uv run python -m klimafrac.fetch.thin_candidates --stage daily
    uv run python -m klimafrac.fetch.thin_candidates --stage select
    uv run python -m klimafrac.fetch.thin_candidates --stage verify
"""

import argparse
import json

import pandas as pd

from klimafrac.fetch.chain import (BATCH, Blocks, as_string, as_uint, as_uints, calldata,
                                   calls_allowing_reverts, eth_calls, event_topic, iso,
                                   logs_in_range, metered, record_spend, say, topic_address,
                                   utc_ts, write_csv, write_json)
from klimafrac.fetch.pools import (END_BLOCK, POOLS, QUICKSWAP_POLYGON, SUSHISWAP_POLYGON,
                                   USDC_E_POLYGON)
from klimafrac.fetch.swaps import load_logs, pool_path, words
from klimafrac.paths import data, raw
from klimafrac.trades import SAMPLE_END

MODULE = "thin_candidates"
CHAIN = "polygon"
FACTORIES = {"sushiswap": SUSHISWAP_POLYGON, "quickswap": QUICKSWAP_POLYGON}
PAIR_CREATED = event_topic("PairCreated(address,address,address,uint256)")
QUOTE_DECIMALS = 6
# The day after the first NCT swap, NCT being the last carbon pool to start;
# a mid-sample date; and the end of the estimation sample.
ANCHORS = ("2022-02-11", "2024-01-01", SAMPLE_END.strftime("%Y-%m-%d"))
MIN_QUOTE = 50.0                   # USDC.e a pair must hold at every anchor
FIRST_DAY = ANCHORS[0]
DAY_STEP = 5                       # every fifth day, so every weekday is sampled
PEG_RANGE = 2.0                    # highest over lowest pool price, below which a token is pegged
THIN_ROLE = "thin"
ACTIVE_ROLE = "active"
# pair: why fetch/swaps.py cannot build its history
UNBUILDABLE = {
    "0x4deb2d750b790f2a1a81ed51d9694675ae8628b8":
        "KASTA: alchemy_getAssetTransfers returns no transfer in block 64,169,525, where "
        "transaction 0x342cbe3d4ca25cd066c9244c34c8a68b06fa3feadced8df0829e42e217a62209 "
        "swaps through the pair and the chain's own logs show both token transfers",
}


def _topic(address: str) -> str:
    return "0x" + address[2:].lower().rjust(64, "0")


def pairs(client, blocks: Blocks) -> pd.DataFrame:
    """Every pair either factory created with USDC.e on one side."""
    rows = []
    for dex, factory in FACTORIES.items():
        path = raw(MODULE, f"pair_created_{dex}.json")
        if path.exists():
            logs = json.loads(path.read_text())
        else:
            start, logs = blocks.deployment(factory), []
            for topics in ([PAIR_CREATED, _topic(USDC_E_POLYGON)],
                           [PAIR_CREATED, None, _topic(USDC_E_POLYGON)]):
                logs += logs_in_range(client, {"address": factory, "topics": topics},
                                      start, END_BLOCK[CHAIN])
            write_json(logs, path)
        say(f"{dex}: {len(logs)} USDC.e pairs created")
        for lg in logs:
            token0, token1 = topic_address(lg["topics"][1]), topic_address(lg["topics"][2])
            rows.append({"dex": dex, "factory": factory,
                         "pair": "0x" + lg["data"][26:66].lower(),
                         "base": token1 if token0 == USDC_E_POLYGON else token0,
                         "base_is_token0": token0 != USDC_E_POLYGON,
                         "created_block": int(lg["blockNumber"], 16)})
    out = pd.DataFrame(rows).sort_values(["dex", "created_block"]).reset_index(drop=True)
    if out.pair.duplicated().any():
        raise ValueError("a pair appears twice in the factory logs")
    return out


def _reserves(client, pair_blocks: list[tuple[str, int]]) -> list[tuple[int, int, int]]:
    out = []
    for i in range(0, len(pair_blocks), BATCH):
        part = pair_blocks[i:i + BATCH]
        out += [tuple(as_uints(r, 3)) for r in eth_calls(
            client, [(pair, calldata("getReserves()"), block) for pair, block in part])]
    return out


def screen(client, blocks: Blocks, universe: pd.DataFrame) -> pd.DataFrame:
    """Quote reserve at each anchor, for pairs created before the first one."""
    at = blocks.block_before([utc_ts(pd.Timestamp(a, tz="UTC")) for a in ANCHORS])
    anchor_block = {a: at[utc_ts(pd.Timestamp(a, tz="UTC"))] for a in ANCHORS}
    old = universe[universe.created_block <= anchor_block[ANCHORS[0]]].copy()
    path = raw(MODULE, "screen.json")
    store = json.loads(path.read_text()) if path.exists() else {}
    todo = [(p, a) for p in old.pair for a in ANCHORS if f"{p}@{a}" not in store]
    for i in range(0, len(todo), 400):
        part = todo[i:i + 400]
        for (p, a), r in zip(part, _reserves(client, [(p, anchor_block[a]) for p, a in part])):
            store[f"{p}@{a}"] = [str(r[0]), str(r[1]), r[2]]
        write_json(store, path)
        say(f"  screen: {min(i + 400, len(todo))}/{len(todo)} readings")
    for a in ANCHORS:
        quote = [int(store[f"{p}@{a}"][0 if b0 else 1]) / 10 ** QUOTE_DECIMALS
                 for p, b0 in zip(old.pair, ~old.base_is_token0)]
        old[f"quote_{a}"] = quote
    old["passes"] = (old[[f"quote_{a}" for a in ANCHORS]] >= MIN_QUOTE).all(axis=1)
    return old


def describe(client, passed: pd.DataFrame) -> pd.DataFrame:
    """Symbol and decimals of each base token, where the token reports them."""
    end = END_BLOCK[CHAIN]
    symbols = calls_allowing_reverts(client, [(t, calldata("symbol()"), end) for t in passed.base])
    decimals = calls_allowing_reverts(client, [(t, calldata("decimals()"), end)
                                               for t in passed.base])
    out = passed.copy()
    names = []
    for s in symbols:
        try:
            names.append(as_string(s) if s else "")
        except (ValueError, UnicodeDecodeError):
            names.append("")
    out["symbol"] = names
    out["decimals"] = [as_uint(d) if d else -1 for d in decimals]
    return out


def daily(client, blocks: Blocks, passed: pd.DataFrame) -> pd.DataFrame:
    """Depth and idleness at every DAY_STEP-th daily boundary of the sample."""
    days = pd.date_range(FIRST_DAY, SAMPLE_END.tz_localize(None), freq=f"{DAY_STEP}D",
                         tz="UTC")
    instants = [utc_ts(d) for d in days]
    at = blocks.block_before(instants)
    rows = []
    for n, row in enumerate(passed.itertuples(), 1):
        path = raw(MODULE, "daily", f"{row.pair}.json")
        if path.exists():
            record = json.loads(path.read_text())
        else:
            readings = _reserves(client, [(row.pair, at[t]) for t in instants])
            record = {iso(t): [str(r[0]), str(r[1]), r[2]] for t, r in zip(instants, readings)}
            write_json(record, path)
            say(f"  daily: {n}/{len(passed)} {row.dex} {row.symbol or row.base}")
        for t in instants:
            r0, r1, last = record[iso(t)]
            base, quote = (int(r0), int(r1)) if row.base_is_token0 else (int(r1), int(r0))
            rows.append({"pair": row.pair, "instant": iso(t), "block": at[t],
                         "reserve0_raw": r0, "reserve1_raw": r1,
                         "quote_reserve": quote / 10 ** QUOTE_DECIMALS,
                         # quote per base in raw units: its level depends on
                         # the tokens' decimals, its range over time does not
                         "price_raw": quote / base if base and quote else float("nan"),
                         "idle_hours": (t - int(last)) / 3600})
    return pd.DataFrame(rows)


def measures(screened: pd.DataFrame, readings: pd.DataFrame) -> pd.DataFrame:
    """Each pair's thinness measures, and where it stands on each criterion."""
    g = readings.groupby("pair")
    m = pd.DataFrame({
        "readings": g.size(),
        "idle_1h": g.idle_hours.apply(lambda x: float((x > 1).mean())),
        "idle_24h": g.idle_hours.apply(lambda x: float((x > 24).mean())),
        "depth_median": g.quote_reserve.median(),
        "price_range": g.price_raw.max() / g.price_raw.min()}).reset_index()
    m = screened[screened.passes].drop(columns="passes").merge(m, on="pair")
    role = {p.pair: p.role for p in POOLS if p.chain == CHAIN and p.quote == USDC_E_POLYGON}
    key = {p.pair: p.key for p in POOLS if p.chain == CHAIN}
    m["analysed_as"] = m.pair.map(
        lambda x: key.get(x, "") if role.get(x) not in (THIN_ROLE, ACTIVE_ROLE) else "")
    carbon = m[m.pair.map(role) == "carbon"]
    if len(carbon) != 4:
        raise ValueError(f"expected the four carbon pools among the readings, found {len(carbon)}")
    m["candidate"] = m.analysed_as == ""
    m["ok_idle_1h"] = m.idle_1h.between(carbon.idle_1h.min(), carbon.idle_1h.max())
    m["ok_idle_24h"] = m.idle_24h <= carbon.idle_24h.max()
    m["ok_not_pegged"] = m.price_range >= PEG_RANGE
    upto = m.candidate & m.ok_idle_1h & m.ok_idle_24h & m.ok_not_pegged
    deepest = m[upto].sort_values("depth_median", ascending=False).drop_duplicates("base").pair
    m["ok_one_per_token"] = m.pair.isin(deepest)
    m["ok_depth"] = m.depth_median.between(carbon.depth_median.min(), carbon.depth_median.max())
    m["ok_buildable"] = ~m.pair.isin(UNBUILDABLE)
    m["selected"] = upto & m.ok_one_per_token & m.ok_depth & m.ok_buildable
    busier = (m.candidate & (m.idle_1h < carbon.idle_1h.min()) & m.ok_idle_24h
              & m.ok_not_pegged & m.ok_buildable)
    deepest = m[busier].sort_values("depth_median", ascending=False).drop_duplicates("base").pair
    m["selected_active"] = busier & m.pair.isin(deepest)
    steps, keep = [], m.candidate.copy()
    steps.append(("held reserves at every anchor, not already analysed", int(keep.sum())))
    for column in ("ok_idle_1h", "ok_idle_24h", "ok_not_pegged", "ok_one_per_token", "ok_depth",
                   "ok_buildable"):
        keep &= m[column]
        steps.append((column, int(keep.sum())))
    for name, n in steps:
        say(f"  {name}: {n}")
    say(f"  active: {int(busier.sum())} busier than every carbon pool, not pegged, "
        f"{int(m.selected_active.sum())} after one pool per token")
    say("  carbon ranges: idle over an hour "
        f"{carbon.idle_1h.min():.3f}-{carbon.idle_1h.max():.3f}, over a day up to "
        f"{carbon.idle_24h.max():.3f}, median depth {carbon.depth_median.min():,.0f}-"
        f"{carbon.depth_median.max():,.0f} USDC.e")
    return m.sort_values(["selected", "idle_1h"], ascending=[False, True])


def check_registry(selected: pd.DataFrame, role: str = THIN_ROLE) -> None:
    """The pools registered under a role must be exactly the ones its rule keeps."""
    want = {(r.pair, r.base, r.factory) for r in selected.itertuples()}
    have = {(p.pair, p.base, p.factory) for p in POOLS if p.role == role}
    if want != have:
        raise SystemExit(f"fetch/pools.py registers {len(have)} {role} pools, the rule keeps "
                         f"{len(want)}: {len(want - have)} missing, {len(have - want)} extra")
    say(f"  registry agrees with the rule: {len(have)} {role} pools")


def verify(readings: pd.DataFrame) -> pd.DataFrame:
    """Fetched logs against the independent getReserves() readings."""
    rows = []
    for pool in (p for p in POOLS if p.role in (THIN_ROLE, ACTIVE_ROLE)):
        meta_path = pool_path(pool, "pool.json")
        mine = readings[readings.pair == pool.pair].sort_values("block")
        try:
            logs = load_logs(pool, json.loads(meta_path.read_text()))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        sync = logs[logs.event == "sync"]
        at = sync.block.to_numpy().searchsorted(mine.block.to_numpy(), side="right") - 1
        agree = differ = before_window = 0
        for i, r in zip(at, mine.itertuples()):
            if i < 0:
                before_window += 1           # no Sync yet inside the fetched window
                continue
            got = tuple(str(w) for w in words(sync.data.iloc[i])[:2])
            if got == (str(r.reserve0_raw), str(r.reserve1_raw)):
                agree += 1
            else:
                differ += 1
        rows.append({"pool": pool.key, "pair": pool.pair, "readings": len(mine), "agree": agree,
                     "differ": differ, "before_first_sync": before_window,
                     "sync_logs": len(sync), "swap_logs": int((logs.event == "swap").sum())})
        say(f"  {pool.key}: {agree}/{len(mine)} readings reproduced by the logs, "
            f"{differ} differ, {before_window} before the first Sync")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--stage", choices=("pairs", "screen", "daily", "select", "verify"),
                    required=True)
    args = ap.parse_args()
    client = metered(CHAIN)
    blocks = Blocks(client, CHAIN)
    try:
        universe = pairs(client, blocks)
        write_csv(universe, data("thin_candidates_pairs.csv"))
        say(f"{len(universe)} USDC.e pairs in all")
        if args.stage == "pairs":
            return
        screened = screen(client, blocks, universe)
        passed = describe(client, screened[screened.passes])
        screened = screened.merge(passed[["pair", "symbol", "decimals"]], on="pair", how="left")
        write_csv(screened, data("thin_candidates_screen.csv"))
        say(f"{len(screened)} pairs old enough, {int(screened.passes.sum())} hold "
            f">= {MIN_QUOTE:g} USDC.e at every anchor")
        if args.stage == "screen":
            return
        readings = daily(client, blocks, passed)
        write_csv(readings, data("thin_candidates_daily.csv"))
        if args.stage == "daily":
            return
        table = measures(screened, readings)
        write_csv(table, data("thin_candidates_measures.csv"))
        write_csv(table[table.selected], data("thin_candidates_selected.csv"))
        write_csv(table[table.selected_active], data("thin_candidates_active.csv"))
        if args.stage == "select":
            if any(p.role == THIN_ROLE for p in POOLS):
                check_registry(table[table.selected])
            if any(p.role == ACTIVE_ROLE for p in POOLS):
                check_registry(table[table.selected_active], ACTIVE_ROLE)
            return
        check_registry(table[table.selected])
        check_registry(table[table.selected_active], ACTIVE_ROLE)
        checked = verify(readings)
        write_csv(checked, data("thin_candidates_verified.csv"))
        if len(checked) and int(checked.differ.sum()):
            raise SystemExit(f"{int((checked.differ > 0).sum())} pools have logs that "
                             "disagree with the chain")
    finally:
        blocks.save()
        record_spend(MODULE, client)
        say(f"spent {client.meter.summary()}")


if __name__ == "__main__":
    main()
