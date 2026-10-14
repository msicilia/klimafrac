"""
The KlimaDAO treasury, and who holds the carbon pool tokens.

The Olympus-fork treasury contract at TREASURY describes itself on-chain:
manager() returns the account that controls it, KLIMA() the token it backs,
reserveTokens(i) enumerates its registered reserve assets (an index past the
end reverts), and totalReserves() / excessReserves() give its booked reserves
in KLIMA's 9-decimal units, which is checked against KLIMA.decimals() before
anything is scaled.

The set of tokens read is not fixed in advance. The registry is enumerated at
every sampled block first, and the union of everything it ever returned --
together with the tokens named below, which the paper reports -- is the set
whose totalSupply() and balances are then read at every block. A registered
address that would fall outside that set stops the run rather than being
skipped, so no reserve asset can be dropped by being absent from a list.

For every token in that set the module reads totalSupply() and the balances
held by the treasury and by the treasury's manager at that block.

Pinned blocks: the last block before 00:00 UTC on the first of each month,
2021-10-01 to 2026-07-01 (fetch/chain.py, shared with fetch/inventory.py),
and END_BLOCK, the last Polygon block before 2026-07-26 00:00 UTC.

Explicit statuses instead of missing values: a contract with no code yet is
not_deployed; a getter the contract refuses is recorded as reverted, a
refusal being the node's execution-reverted code and nothing else.

Outputs:
  data/treasury.csv        one row per month per reserve token:
                           date, as_of_instant, block, token, token_status,
                           registered, supply, treasury, share, manager,
                           manager_balance, manager_share
  data/treasury_state.csv  one row per month: date, as_of_instant, block,
                           status, manager, klima, reserve_tokens,
                           total_reserves, excess_reserves,
                           excess_reserves_status, klima_supply
  data/treasury_cross.csv  the per-token rows at END_BLOCK
Raw: data/raw/treasury/{registry,month}/<date>.json, one file per sampled
     block per stage, so an interrupted run resumes where it stopped

Usage:
    uv run python -m klimafrac.fetch.treasury
"""

import json

import pandas as pd

from klimafrac.fetch.chain import (WINDOW_END, Blocks, as_address, as_string,
                                   as_uint, calls_allowing_reverts, calldata,
                                   eth_calls, iso, metered, raw_manifest,
                                   record_spend, say, utc_ts, write_csv,
                                   write_json)
from klimafrac.fetch.pools import END_BLOCK
from klimafrac.paths import data, raw

MODULE = "treasury"
TREASURY = "0x7dd4f0b986f032a44f913bf92c9e8b7c17d77ad7"
MANAGER = "0x65a5076c0ba74e5f3e069995dc3dab9d197d995c"   # expected manager(); checked
KLIMA = "0x4e78011ce80ee02d2c3e649fb657e45898257815"
KLIMA_DECIMALS = 9                  # the unit of totalReserves(); checked on-chain
RESERVES = {                        # reported in the paper; the registry may hold more
    "BCT": "0x2f800db0fdb5223b3c3f354886d907a671414a7f",
    "MCO2": "0xaa7dbd1598251f856c12f63557a4c4397c253cea",
    "UBO": "0x2b3ecb0991af0498ece9135bcd04013d7993110c",
    "NBO": "0x6bca3b77c1909ce1a4ba1a20d1103bde8d222e48",
    "NCT": "0xd838290e877e0188a4a44700463419ed96c16107",
}
FIRST_DATE = "2021-10-01"
LAST_MONTH = "2026-07-01"
MAX_RESERVES = 64
PROBE_STEP = 8                      # reserveTokens indices probed per request
END = END_BLOCK["polygon"]

COLUMNS = ["date", "as_of_instant", "block", "token", "token_status",
           "registered", "supply", "treasury", "share", "manager",
           "manager_balance", "manager_share"]


def cached(stage: str, label: str, build, valid=lambda record: True):
    """Read data/raw/treasury/<stage>/<label>.json, or build and store it.

    A stored record that no longer answers the question being asked -- built
    for a different token set, say -- is rebuilt rather than reused.
    """
    path = raw(MODULE, stage, f"{label}.json")
    if path.exists():
        record = json.loads(path.read_text())
        if valid(record):
            return record
    record = build()
    write_json(record, path)
    return record


def reserve_tokens(client, block: int) -> list[str]:
    """reserveTokens(0..n-1) at one block; the first index past the end reverts.

    The indices travel in one request: a revert is an expected answer here, so
    it shares the request with the calls that must succeed.
    """
    out: list[str] = []
    while len(out) <= MAX_RESERVES:
        part = calls_allowing_reverts(client, [
            (TREASURY, calldata("reserveTokens(uint256)", i), block)
            for i in range(len(out), len(out) + PROBE_STEP)])
        for k, res in enumerate(part):
            if res is None:
                if any(x is not None for x in part[k + 1:]):
                    raise ValueError(f"block {block}: reserveTokens reverts at index "
                                     f"{len(out)} but answers a later one")
                return out
            out.append(as_address(res))
    raise ValueError(f"more than {MAX_RESERVES} reserve tokens at block {block}")


def read_registry(client, deploy: dict, label: str, instant: int, block: int) -> dict:
    """What the treasury says about itself at one block, cached per block."""
    head = {"date": label, "as_of_instant": iso(instant), "block": block}

    def build() -> dict:
        if block < deploy[TREASURY]:
            return {**head, "status": "not_deployed", "manager": None, "klima": None,
                    "registered": [], "total_reserves_raw": "0",
                    "excess_reserves_raw": None,
                    "excess_reserves_status": "not_deployed"}
        mgr, klima, reserves = eth_calls(client, [
            (TREASURY, calldata("manager()"), block),
            (TREASURY, calldata("KLIMA()"), block),
            (TREASURY, calldata("totalReserves()"), block)])
        excess = calls_allowing_reverts(
            client, [(TREASURY, calldata("excessReserves()"), block)])[0]
        return {**head, "status": "deployed", "manager": as_address(mgr),
                "klima": as_address(klima),
                "registered": reserve_tokens(client, block),
                "total_reserves_raw": str(as_uint(reserves)),
                "excess_reserves_raw": None if excess is None else str(as_uint(excess)),
                "excess_reserves_status": "reverted" if excess is None else "ok"}
    return cached("registry", label, build,
                  valid=lambda r: (r.get("block") == block
                                   and r.get("as_of_instant") == head["as_of_instant"]))


def token_universe(client, blocks: Blocks, registered_ever: set[str]) -> dict:
    """Every token to read: the reported set, plus everything ever registered."""
    path = raw(MODULE, "tokens.json")
    if path.exists():
        universe = json.loads(path.read_text())
        if set(registered_ever) <= {t["address"] for t in universe.values()}:
            return universe
    by_address = {a: s for s, a in RESERVES.items()}
    discovered = sorted(set(registered_ever) - set(RESERVES.values()))
    universe = {}
    for address in [*RESERVES.values(), *discovered]:   # reported set first
        symbol = as_string(client.eth_call(address, calldata("symbol()"), END))
        named = by_address.get(address)
        if named and symbol != named:
            raise ValueError(f"{address} reports symbol {symbol!r}, expected {named}")
        key = named or symbol
        if key in universe:
            key = f"{symbol}:{address[:10]}"        # two tokens, one symbol
        universe[key] = {
            "address": address, "symbol": symbol, "in_paper": address in by_address,
            "decimals": as_uint(client.eth_call(address, calldata("decimals()"), END)),
            "deploy_block": blocks.deployment(address)}
    write_json(universe, path)
    return universe


def read_month(client, universe: dict, deploy: dict, registry: dict) -> tuple[dict, list]:
    """Supply and balances of every token in the universe at one block."""
    label, block, manager = registry["date"], registry["block"], registry["manager"]
    registered = set(registry["registered"])
    unknown = registered - {t["address"] for t in universe.values()}
    if unknown:
        raise ValueError(f"block {block}: registered reserve tokens {sorted(unknown)} "
                         f"are outside the set being read")

    def build() -> dict:
        live = [s for s, t in universe.items() if block >= t["deploy_block"]]
        calls = []
        for sym in live:
            a = universe[sym]["address"]
            calls += [(a, calldata("totalSupply()"), block),
                      (a, calldata("balanceOf(address)", TREASURY), block)]
            if manager:
                calls.append((a, calldata("balanceOf(address)", manager), block))
        results = iter(eth_calls(client, calls))
        raw_values = {}
        for sym in live:
            supply = as_uint(next(results))
            held = as_uint(next(results))
            mbal = as_uint(next(results)) if manager else 0
            raw_values[sym] = [str(supply), str(held), str(mbal)]
        klima_supply = (str(as_uint(client.eth_call(KLIMA, calldata("totalSupply()"),
                                                    block)))
                        if block >= deploy[KLIMA] else "0")
        return {"block": block, "tokens": sorted(universe), "live": live,
                "raw": raw_values, "klima_supply_raw": klima_supply,
                "manager": manager}
    record = cached("month", label, build,
                    valid=lambda r: (r.get("block") == block
                                     and r.get("tokens") == sorted(universe)
                                     and r.get("manager") == manager))

    scale = 10 ** KLIMA_DECIMALS
    excess = registry["excess_reserves_raw"]
    state = {"date": label, "as_of_instant": registry["as_of_instant"],
             "block": block, "status": registry["status"], "manager": manager,
             "klima": registry["klima"],
             "reserve_tokens": ";".join(
                 next((s for s, t in universe.items() if t["address"] == a), a)
                 for a in registry["registered"]),
             "total_reserves": int(registry["total_reserves_raw"]) / scale,
             "excess_reserves": None if excess is None else int(excess) / scale,
             "excess_reserves_status": registry["excess_reserves_status"],
             "klima_supply": int(record["klima_supply_raw"]) / scale}

    rows = []
    for sym, token in universe.items():
        row = {"date": label, "as_of_instant": registry["as_of_instant"],
               "block": block, "token": sym,
               "registered": token["address"] in registered, "manager": manager}
        if sym in record["live"]:
            unit = 10 ** token["decimals"]
            supply, held, mbal = (int(v) / unit for v in record["raw"][sym])
            row.update(token_status="deployed", supply=supply, treasury=held,
                       manager_balance=mbal,
                       share=held / supply if supply else float("nan"),
                       manager_share=mbal / supply if supply else float("nan"))
        else:
            row.update(token_status="not_deployed", supply=0.0, treasury=0.0,
                       manager_balance=0.0, share=float("nan"),
                       manager_share=float("nan"))
        rows.append(row)
    return state, rows


def run() -> None:
    client = metered("polygon")
    blocks = Blocks(client)
    deploy = {a: blocks.deployment(a) for a in (TREASURY, KLIMA)}
    decimals = as_uint(client.eth_call(KLIMA, calldata("decimals()"), END))
    if decimals != KLIMA_DECIMALS:
        raise ValueError(f"KLIMA.decimals() is {decimals}, not {KLIMA_DECIMALS}: "
                         f"totalReserves() would be scaled wrongly")

    months = pd.date_range(FIRST_DATE, LAST_MONTH, freq="MS", tz="UTC")
    instants = [utc_ts(m) for m in months] + [utc_ts(WINDOW_END)]
    at = blocks.block_before(instants)
    if at[utc_ts(WINDOW_END)] != END:
        raise ValueError("window end does not map to END_BLOCK")

    labels = [(m.strftime("%Y-%m-%d"), utc_ts(m)) for m in months]
    labels.append((WINDOW_END.strftime("%Y-%m-%d"), utc_ts(WINDOW_END)))
    registries = [read_registry(client, deploy, label, t, at[t]) for label, t in labels]
    registered_ever = {a for r in registries for a in r["registered"]}
    universe = token_universe(client, blocks, registered_ever)
    extra = sorted(set(universe) - set(RESERVES))
    if extra:
        say(f"  registry holds tokens beyond the reported set: {extra}")

    states, rows = [], []
    for reg in registries[:-1]:
        st, rs = read_month(client, universe, deploy, reg)
        states.append(st)
        rows += rs
    end_state, cross = read_month(client, universe, deploy, registries[-1])

    if end_state["manager"] != MANAGER:
        raise ValueError(f"manager() at END_BLOCK is {end_state['manager']}, "
                         f"expected {MANAGER}")
    if end_state["klima"] != KLIMA:
        raise ValueError(f"KLIMA() at END_BLOCK is {end_state['klima']}")

    write_csv(pd.DataFrame(rows, columns=COLUMNS), data("treasury.csv"))
    write_csv(pd.DataFrame(states + [end_state]), data("treasury_state.csv"))
    write_csv(pd.DataFrame(cross, columns=COLUMNS), data("treasury_cross.csv"))

    say(f"treasury {TREASURY} at END_BLOCK {END:,}: manager {end_state['manager']}, "
        f"reserve tokens {end_state['reserve_tokens']}, "
        f"totalReserves {end_state['total_reserves']:,.0f}, "
        f"KLIMA supply {end_state['klima_supply']:,.0f}")
    for r in cross:
        say(f"  {r['token']:12s} supply {r['supply']:14,.0f}  "
            f"treasury {r['treasury']:14,.0f} ({100 * r['share']:5.1f}%)  "
            f"manager {r['manager_balance']:12,.0f}")
    changes = pd.DataFrame(states)[["date", "manager"]].dropna()
    changes = changes[changes.manager != changes.manager.shift()]
    say("  manager() over time:\n" + changes.to_string(index=False))
    record_spend(MODULE, client)
    raw_manifest(MODULE, {"end_block": END, "tokens": sorted(universe)})


if __name__ == "__main__":
    run()
