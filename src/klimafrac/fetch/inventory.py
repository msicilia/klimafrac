"""
Pool inventory in tonnes, read from each pool token's own supply.

A Toucan (BCT, NCT) or C3 (UBO, NBO) pool token is minted 1:1 against the
credits deposited in the pool and burned when credits leave it, so its
totalSupply() is the tonnage the pool holds. No vendor data enter here.

Two independent readings, which must agree exactly:

  events  every mint (Transfer from the zero address) and burn (Transfer to
          the zero address) of the token from its deployment block to
          END_BLOCK, read with alchemy_getAssetTransfers. Summed, they give
          the supply after every block in which it changed.
  calls   totalSupply() by eth_call at the last block before 00:00 UTC on the
          first of each month, and at END_BLOCK.

The run fails unless the event sum equals totalSupply() to the unit at every
one of those blocks, so no supply change escapes the event stream. The event
path then gives the supply at any block, and the peak is exact rather than
whatever a monthly grid happens to sample.

Conventions. Both grids label an instant and report the state immediately
BEFORE it: the row carrying as_of_instant 2025-06-01T00:00:00Z is the supply
after the last block whose timestamp is earlier than that instant
(fetch/chain.py), so a peak reached late in May is sampled by the June row.
The grids are therefore a sample of the path, not the path: the exact peak
comes from data/inventory_supply_path.csv, which carries the supply after
every block in which it changed. Before a token's deployment its status is
not_deployed and its supply 0; the outputs have no missing values.

Pinned range: each token's deployment block to END_BLOCK, the last Polygon
block before 2026-07-26 00:00 UTC.

Outputs:
  data/inventory.csv           monthly, 2021-10-01 to 2026-07-01:
                               date, as_of_instant, block, <SYM> (tonnes),
                               <SYM>_status
  data/inventory_daily.csv     daily, 2021-10-01 to 2026-07-26:
                               date, as_of_instant, token, status,
                               supply_raw, tonnes
  data/inventory_supply_path.csv  every block in which a supply changed:
                               token, block, block_instant, net_raw,
                               supply_raw, tonnes
  data/inventory_summary.json  per token: deployment, event counts, exact
                               all-time peak (value, first block, when it
                               ended), supply at END_BLOCK, largest monthly
                               decline, and the reconciliation performed
Raw: data/raw/inventory/<SYM>/{mint,burn}_<from>_<to>.csv.gz, manifest.json,
     rpc_calls_polygon.json

Usage:
    uv run python -m klimafrac.fetch.inventory
"""

import bisect
import json

import pandas as pd

from klimafrac.fetch.chain import (TRANSFER_COLUMNS, WINDOW_END, ZERO, Blocks,
                                   as_string, as_uint, asset_transfers, calldata,
                                   chunks, eth_calls, iso, metered, raw_manifest,
                                   record_spend, say, utc_ts, write_csv,
                                   write_json)
from klimafrac.fetch.pools import END_BLOCK
from klimafrac.paths import data, raw

TOKENS = {
    "BCT": "0x2f800db0fdb5223b3c3f354886d907a671414a7f",   # Toucan
    "NCT": "0xd838290e877e0188a4a44700463419ed96c16107",   # Toucan
    "UBO": "0x2b3ecb0991af0498ece9135bcd04013d7993110c",   # C3
    "NBO": "0x6bca3b77c1909ce1a4ba1a20d1103bde8d222e48",   # C3
}
FIRST_DATE = "2021-10-01"
LAST_MONTH = "2026-07-01"
CHUNK = 5_000_000
KINDS = {"mint": "fromAddress", "burn": "toAddress"}
END = END_BLOCK["polygon"]


def describe(client, blocks: Blocks) -> dict:
    path = raw("inventory", "tokens.json")
    if path.exists():
        return json.loads(path.read_text())
    meta = {}
    for sym, address in TOKENS.items():
        symbol = as_string(client.eth_call(address, calldata("symbol()"), END))
        if symbol != sym:
            raise ValueError(f"{address} reports symbol {symbol!r}, expected {sym}")
        dep = blocks.deployment(address)
        meta[sym] = {"address": address, "symbol": symbol,
                     "decimals": as_uint(client.eth_call(address, calldata("decimals()"), END)),
                     "deploy_block": dep, "deploy_timestamp": iso(blocks.timestamp(dep))}
    write_json(meta, path)
    return meta


def event_files(sym: str, meta: dict):
    for kind in KINDS:
        for a, b in chunks(meta[sym]["deploy_block"], END, CHUNK):
            yield kind, a, b, raw("inventory", sym, f"{kind}_{a}_{b}.csv.gz")


def fetch_events(client, sym: str, meta: dict) -> None:
    token = meta[sym]["address"]
    for kind, a, b, path in event_files(sym, meta):
        if path.exists():
            continue
        rows = asset_transfers(client, token, a, b, **{KINDS[kind]: ZERO})
        zero_col = 5 if kind == "mint" else 6
        if any(r[zero_col] != ZERO for r in rows):
            raise ValueError(f"{sym} {kind}: transfer not involving the zero address")
        write_csv(pd.DataFrame(rows, columns=TRANSFER_COLUMNS), path)
        say(f"  {sym} {kind}s {a:,}-{b:,}: {len(rows):,}")


def load_events(sym: str) -> pd.DataFrame:
    """Mint and burn events of a token, signed, exact (python int deltas)."""
    meta = json.loads(raw("inventory", "tokens.json").read_text())
    frames = []
    for kind, a, b, path in event_files(sym, meta):
        if not path.exists():
            raise FileNotFoundError(f"inventory events incomplete: {path} "
                                    f"(run klimafrac.fetch.inventory)")
        f = pd.read_csv(path, dtype={"raw_value": str, "tx_hash": str,
                                     "from": str, "to": str})
        f["kind"] = kind
        frames.append(f)
    ev = pd.concat(frames, ignore_index=True)
    if ((ev["from"] == ZERO) & (ev["to"] == ZERO)).any():
        raise ValueError(f"{sym}: transfer from and to the zero address")
    if ev.duplicated(["tx_hash", "log_index"]).any():
        raise ValueError(f"{sym}: duplicated supply event")
    sign = ev.kind.map({"mint": 1, "burn": -1})
    ev["delta"] = [s * int(v) for s, v in zip(sign, ev.raw_value)]
    return ev.sort_values(["block", "log_index"]).reset_index(drop=True)


class SupplyPath:
    """Supply after every block in which it changed, exact."""

    def __init__(self, events: pd.DataFrame):
        net, when = {}, {}
        for b, t, d in zip(events.block, events.timestamp, events.delta):
            net[int(b)] = net.get(int(b), 0) + d
            when[int(b)] = int(t)
        self.blocks = sorted(net)
        self.times = [when[b] for b in self.blocks]
        self.net = [net[b] for b in self.blocks]
        if any(t1 < t0 for t0, t1 in zip(self.times, self.times[1:])):
            raise ValueError("event timestamps decrease along the chain")
        self.supply, s = [], 0
        for d in self.net:
            s += d
            if s < 0:
                raise ValueError("supply path goes negative: events incomplete")
            self.supply.append(s)

    def at_block(self, block: int) -> int:
        i = bisect.bisect_right(self.blocks, block)
        return self.supply[i - 1] if i else 0

    def before_instant(self, ts: int) -> int:
        i = bisect.bisect_left(self.times, ts)      # events with timestamp < ts
        return self.supply[i - 1] if i else 0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"block": self.blocks, "timestamp": self.times,
                             "net": self.net, "supply": self.supply})

    def table(self, sym: str, decimals: int) -> pd.DataFrame:
        """The path as it is stored: exact integers, and the block's own time."""
        f = self.frame()
        return pd.DataFrame({
            "token": sym, "block": f.block,
            "block_instant": [iso(t) for t in f.timestamp],
            "timestamp": f.timestamp,
            "net_raw": [str(v) for v in f.net],
            "supply_raw": [str(v) for v in f.supply],
            "tonnes": [int(v) / 10 ** decimals for v in f.supply]})


def peak(path: SupplyPath) -> dict:
    top = max(path.supply)
    i = path.supply.index(top)
    return {"supply_raw": str(top), "first_block": path.blocks[i],
            "first_timestamp": iso(path.times[i]),
            "ended_block": path.blocks[i + 1] if i + 1 < len(path.blocks) else None,
            "ended_timestamp": iso(path.times[i + 1]) if i + 1 < len(path.blocks) else None,
            "supply_before_raw": str(path.supply[i - 1] if i else 0)}


def run() -> None:
    client = metered("polygon")
    blocks = Blocks(client)
    meta = describe(client, blocks)
    for sym in TOKENS:
        fetch_events(client, sym, meta)
    events = {sym: load_events(sym) for sym in TOKENS}
    paths = {sym: SupplyPath(events[sym]) for sym in TOKENS}

    months = pd.date_range(FIRST_DATE, LAST_MONTH, freq="MS", tz="UTC")
    instants = [utc_ts(m) for m in months] + [utc_ts(WINDOW_END)]
    at = blocks.block_before(instants)
    if at[utc_ts(WINDOW_END)] != END:
        raise ValueError(f"window end maps to block {at[utc_ts(WINDOW_END)]}, "
                         f"END_BLOCK is {END}")

    # Reconcile the event path against totalSupply() at every sampled block.
    probes = [(t, sym) for t in instants for sym in TOKENS
              if at[t] >= meta[sym]["deploy_block"]]
    results = eth_calls(client, [(TOKENS[sym], calldata("totalSupply()"), at[t])
                                 for t, sym in probes])
    called = {(t, sym): as_uint(r) for (t, sym), r in zip(probes, results)}
    for t in instants:
        for sym in TOKENS:
            by_block = paths[sym].at_block(at[t])
            by_time = paths[sym].before_instant(t)
            expected = called.get((t, sym), 0)
            if not by_block == by_time == expected:
                raise ValueError(f"{sym} at block {at[t]} ({iso(t)}): totalSupply "
                                 f"{expected}, events {by_block} / {by_time}")
    say(f"  reconciled {len(probes)} totalSupply() calls against the event path")

    def status(sym, ts):
        deployed = utc_ts(meta[sym]["deploy_timestamp"]) < ts
        return "deployed" if deployed else "not_deployed"

    # Both grids carry the instant they label, because the value is the state
    # immediately before it and the date alone has misled readers into taking
    # the row as the state during that day.
    rows = []
    for m, t in zip(months, instants):
        row = {"date": m.strftime("%Y-%m-%d"), "as_of_instant": iso(t), "block": at[t]}
        for sym in TOKENS:
            row[sym] = called.get((t, sym), 0) / 10 ** meta[sym]["decimals"]
        for sym in TOKENS:
            row[f"{sym}_status"] = status(sym, t)
        rows.append(row)
    monthly = pd.DataFrame(rows)
    write_csv(monthly, data("inventory.csv"))

    days = pd.date_range(FIRST_DATE, WINDOW_END.tz_localize(None), freq="D", tz="UTC")
    daily = []
    for d in days:
        t = utc_ts(d)
        for sym in TOKENS:
            s = paths[sym].before_instant(t)
            daily.append((d.strftime("%Y-%m-%d"), iso(t), sym, status(sym, t), str(s),
                          s / 10 ** meta[sym]["decimals"]))
    daily = pd.DataFrame(daily, columns=["date", "as_of_instant", "token", "status",
                                         "supply_raw", "tonnes"])
    write_csv(daily, data("inventory_daily.csv"))

    supply_path = pd.concat([paths[sym].table(sym, meta[sym]["decimals"])
                             for sym in TOKENS], ignore_index=True)
    write_csv(supply_path, data("inventory_supply_path.csv"))

    # Exact peaks, each confirmed by eth_call on both sides of its first block.
    summary = {"window": {"end_block": END, "end_exclusive": iso(utc_ts(WINDOW_END))},
               "reconciliation": f"event path == totalSupply() at {len(probes)} "
                                 f"(block, token) pairs: monthly boundary blocks "
                                 f"{FIRST_DATE}..{LAST_MONTH} and END_BLOCK",
               "tokens": {}}
    for sym in TOKENS:
        p, dec = peak(paths[sym]), meta[sym]["decimals"]
        chk = eth_calls(client, [(TOKENS[sym], calldata("totalSupply()"), b)
                                 for b in (p["first_block"] - 1, p["first_block"])])
        if [as_uint(x) for x in chk] != [int(p["supply_before_raw"]), int(p["supply_raw"])]:
            raise ValueError(f"{sym}: peak not confirmed by totalSupply()")
        s = monthly[sym]
        drop = s.diff()
        k = int(drop.idxmin())
        tokdaily = daily[daily.token == sym]
        summary["tokens"][sym] = {
            **meta[sym],
            "mint_events": int((events[sym].kind == "mint").sum()),
            "burn_events": int((events[sym].kind == "burn").sum()),
            "blocks_with_supply_change": len(paths[sym].blocks),
            "peak": {**p, "tonnes": int(p["supply_raw"]) / 10 ** dec},
            # `as_of` is the instant each grid labels; the value is the supply
            # immediately before it, so a grid peak lands after the true one.
            "daily_peak": {"as_of": tokdaily.loc[tokdaily.tonnes.idxmax(),
                                                 "as_of_instant"],
                           "tonnes": float(tokdaily.tonnes.max())},
            "monthly_peak": {"as_of": monthly.loc[s.idxmax(), "as_of_instant"],
                             "tonnes": float(s.max())},
            "end_block_supply_raw": str(paths[sym].at_block(END)),
            "end_block_tonnes": paths[sym].at_block(END) / 10 ** dec,
            "largest_monthly_decline": {
                "from": monthly.loc[k - 1, "date"], "to": monthly.loc[k, "date"],
                "tonnes": float(drop.loc[k])} if k > 0 else None,
        }
        say(f"  {sym}: peak {summary['tokens'][sym]['peak']['tonnes']:,.0f} t "
            f"at block {p['first_block']:,} ({p['first_timestamp']}); "
            f"END_BLOCK {summary['tokens'][sym]['end_block_tonnes']:,.0f} t")
    write_json(summary, data("inventory_summary.json"))
    say(f"  wrote inventory.csv ({len(monthly)} months), inventory_daily.csv "
        f"({len(daily):,} rows), inventory_supply_path.csv "
        f"({len(supply_path):,} supply changes), inventory_summary.json")
    record_spend("inventory", client)
    raw_manifest("inventory", {"end_block": END, "tokens": sorted(TOKENS)})


if __name__ == "__main__":
    run()
