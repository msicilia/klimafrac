"""
The October 2025 BCT burn, derived from the chain.

Every number is recomputed from Polygon state; nothing is typed in. Not
used in the paper.

  1. Month.       S(a) and S(b), BCT totalSupply() at a and b, the last blocks
                  before 00:00 UTC on 1 October and 1 November 2025. The
                  month's decline is S(a) - S(b).
  2. Bisection.   Starting from [a, b], halve the interval and keep the half
                  across which totalSupply() falls further, until it spans a
                  single block B. S(B-1) - S(B) is the decrease in B.
  3. Cross-check. The reconciled mint and burn events of fetch/inventory.py
                  give the net supply change of every block in the month. B
                  must be the block with the largest net decrease, by the same
                  amount, and the month's events must sum to the decline.
  4. Block.       BCT Transfer logs in B (eth_getLogs over one block), the
                  transactions carrying the burns, and their receipts.
  5. Account.     The address whose BCT was burned. If it is a Gnosis Safe:
                  getOwners(), getThreshold() and VERSION() at B. Whether it
                  is manager() of the KlimaDAO treasury at B-1 and B, and what
                  the treasury's role registries (isReserveManager(), ...)
                  return for it. Where its BCT came from since July 2025.
  6. Markets.     Whether any log in the burn transactions was emitted by, or
                  moved tokens to or from, a pool in fetch/pools.py, and
                  whether any carries a swap event of a known DEX family.
  7. Credits.     The TCO2 tokens that left the BCT pool in those transactions,
                  where they went (to the zero address = retired in the same
                  transaction), and how much of each the burner still held at
                  B and at END_BLOCK.
  8. Peak.        The exact all-time peak of BCT supply from the event path,
                  confirmed by totalSupply() either side of its first block.
                  It is the peak of the path, not of a monthly grid: the grid
                  labels 00:00 on the first of a month and reports the supply
                  immediately before it, so a peak reached in late May is
                  sampled as June.

Each stage is cached as data/raw/burn/<stage>.json and skipped when present,
so an interrupted run resumes instead of redoing the chain reads. Exact
amounts are stored as decimal strings, never as floats.

Output data/burn.json, including a `claims` block comparing each paper
figure with the value computed.

Usage (after klimafrac.fetch.inventory and klimafrac.fetch.treasury exist):
    uv run python -m klimafrac.fetch.burn
"""

import json
from collections import defaultdict

import pandas as pd

from klimafrac.fetch.chain import (TRANSFER_TOPIC, ZERO, Blocks, as_address,
                                   as_string, as_uint, asset_transfers,
                                   call_or_revert, calldata, eth_calls,
                                   event_topic, iso, metered, raw_manifest,
                                   record_spend, say, utc_ts, write_json)
from klimafrac.fetch.inventory import TOKENS, SupplyPath, load_events, peak
from klimafrac.fetch.pools import END_BLOCK, POOLS
from klimafrac.fetch.treasury import KLIMA, KLIMA_DECIMALS, TREASURY
from klimafrac.paths import data, raw

BCT = TOKENS["BCT"]
MONTH = ("2025-10-01", "2025-11-01")
INFLOWS_FROM = "2025-07-01"
PAPER = {"tonnes_in_block": 4_789_946, "date": "2025-10-28", "multisig": "3-of-5",
         "share_of_supply_pct": 23, "share_of_month_decline_pct": 94.3,
         "manager_of_treasury": True, "no_swap": True,
         "peak_million_tonnes": 21.1, "peak_month": "2025-06"}
SWAP_EVENTS = {
    "UniswapV2-family Swap": "Swap(address,uint256,uint256,uint256,uint256,address)",
    "UniswapV3/Algebra Swap": "Swap(address,address,int256,int256,uint160,uint128,int24)",
    "Aerodrome Swap": "Swap(address,address,uint256,uint256,uint256,uint256)",
    "UniswapV4 Swap": "Swap(bytes32,address,int128,int128,uint160,uint128,int24,uint24)",
    "BalancerV2 Swap": "Swap(bytes32,address,address,uint256,uint256)",
    "Curve TokenExchange": "TokenExchange(address,int128,uint256,int128,uint256)",
    "Curve TokenExchangeUnderlying":
        "TokenExchangeUnderlying(address,int128,uint256,int128,uint256)",
    "DODO DODOSwap": "DODOSwap(address,address,uint256,uint256,address,address)",
}
SWAP_TOPICS = {event_topic(sig): name for name, sig in SWAP_EVENTS.items()}
EXECUTION_SUCCESS = event_topic("ExecutionSuccess(bytes32,uint256)")
ROLES = ["isReserveManager(address)", "isReserveSpender(address)",
         "isReserveDepositor(address)", "isLiquidityManager(address)",
         "isLiquidityDepositor(address)", "isRewardManager(address)",
         "isDebtor(address)"]
POOL_PAIRS = {p.pair: p.key for p in POOLS if p.chain == "polygon"}


EIP1967_IMPLEMENTATION_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"


MODULE = "burn"


def month_grid_label(timestamp: str) -> str:
    """The monthly-grid row that first reports a state reached at this instant.

    A row labelled 00:00 on the first of a month carries the supply
    immediately before that instant, so a change during May is first visible
    in the row dated 1 June.
    """
    t = pd.Timestamp(timestamp)
    return f"{t.year + (t.month == 12):04d}-{t.month % 12 + 1:02d}"


def stage(name: str, build):
    """Read data/raw/burn/<name>.json, or run the stage and store it."""
    path = raw(MODULE, f"{name}.json")
    if path.exists():
        return json.loads(path.read_text())
    record = build()
    write_json(record, path)
    say(f"  stage {name} -> {path}")
    return record


def topic_address(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def decode_transfers(logs: list[dict]) -> list[dict]:
    out = []
    for lg in logs:
        t = lg["topics"]
        if len(t) == 3 and t[0] == TRANSFER_TOPIC and len(lg["data"]) == 66:
            out.append({"token": lg["address"].lower(), "from": topic_address(t[1]),
                        "to": topic_address(t[2]), "value": int(lg["data"], 16),
                        "tx": lg["transactionHash"].lower(),
                        "log_index": int(lg["logIndex"], 16)})
    return out


def supply(client, block: int) -> int:
    return as_uint(client.eth_call(BCT, calldata("totalSupply()"), block))


def bisection(client, a: int, b: int):
    lo, hi, s_lo, s_hi, steps = a, b, supply(client, a), supply(client, b), 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        s_mid = supply(client, mid)
        steps += 1
        if s_lo - s_mid >= s_mid - s_hi:
            hi, s_hi = mid, s_mid
        else:
            lo, s_lo = mid, s_mid
    return hi, s_lo, s_hi, steps


def safe_info(client, address: str, block: int) -> dict:
    code = client.call("eth_getCode", [address, hex(block)])
    if code in ("0x", ""):
        return {"is_contract": False, "is_safe": False}
    th = call_or_revert(client, address, calldata("getThreshold()"), block)
    ow = call_or_revert(client, address, calldata("getOwners()"), block)
    if not th or not ow or len(th) != 66 or len(ow) < 130:
        return {"is_contract": True, "is_safe": False}
    n = int(ow[66:130], 16)
    owners = ["0x" + ow[130 + 64 * i:194 + 64 * i][-40:] for i in range(n)]
    version = call_or_revert(client, address, calldata("VERSION()"), block)
    return {"is_contract": True, "is_safe": True, "threshold": as_uint(th),
            "owners": owners,
            "version": as_string(version) if version and len(version) > 66 else None}


def analyse_tx(client, h: str, block: int) -> dict:
    tx, rc = client.batch([("eth_getTransactionByHash", [h]),
                           ("eth_getTransactionReceipt", [h])])
    if rc is None or int(rc["blockNumber"], 16) != block:
        raise ValueError(f"receipt missing or in another block: {h}")
    logs = rc["logs"]
    transfers = decode_transfers(logs)
    emitters = defaultdict(int)
    for lg in logs:
        emitters[lg["address"].lower()] += 1
    touched = sorted({a for a in emitters if a in POOL_PAIRS}
                     | {x[k] for x in transfers for k in ("from", "to")
                        if x[k] in POOL_PAIRS})
    swaps = [{"address": lg["address"].lower(), "event": SWAP_TOPICS[lg["topics"][0]],
              "log_index": int(lg["logIndex"], 16)}
             for lg in logs if lg["topics"] and lg["topics"][0] in SWAP_TOPICS]

    credits = [x for x in transfers if x["from"] == BCT and x["token"] != BCT]
    credit_tokens = sorted({x["token"] for x in credits})
    dec = dict(zip(credit_tokens, (as_uint(r) for r in eth_calls(
        client, [(t, calldata("decimals()"), block) for t in credit_tokens]))))
    destinations = defaultdict(float)
    for x in credits:
        destinations[x["to"]] += x["value"] / 10 ** dec[x["token"]]
    burned_credits = [x for x in transfers
                      if x["token"] in dec and x["to"] == ZERO]
    return {
        "hash": h, "status": int(rc["status"], 16),
        "from": tx["from"].lower(), "to": (tx["to"] or "").lower(),
        "input_selector": tx["input"][:10], "log_count": len(logs),
        "log_emitters": dict(sorted(emitters.items(), key=lambda kv: -kv[1])),
        "safe_execution_success_emitters": sorted(
            {lg["address"].lower() for lg in logs
             if lg["topics"] and lg["topics"][0] == EXECUTION_SUCCESS}),
        "bct_transfers": [{**x, "value": str(x["value"]), "tonnes": x["value"] / 1e18}
                          for x in transfers if x["token"] == BCT],
        "registry_pools_touched": [POOL_PAIRS[a] for a in touched],
        "swap_events": swaps,
        "credits_leaving_pool": {
            "transfers": len(credits), "distinct_tco2_tokens": len(credit_tokens),
            "tco2_tokens": credit_tokens,
            "tco2_decimals": dec,
            "tonnes": sum(x["value"] / 10 ** dec[x["token"]] for x in credits),
            "tonnes_by_destination": dict(destinations),
            "tonnes_burned_in_same_tx": sum(x["value"] / 10 ** dec[x["token"]]
                                            for x in burned_credits)},
    }


def run() -> None:
    client = metered("polygon")
    blocks = Blocks(client)
    end = END_BLOCK["polygon"]
    dec_bct, dec_klima = (as_uint(r) for r in eth_calls(client, [
        (BCT, calldata("decimals()"), end), (KLIMA, calldata("decimals()"), end)]))
    if dec_bct != 18:
        raise ValueError(f"BCT.decimals() is {dec_bct}, not 18")
    if dec_klima != KLIMA_DECIMALS:
        raise ValueError(f"KLIMA.decimals() is {dec_klima}, not {KLIMA_DECIMALS}: "
                         f"totalReserves() and KLIMA supply would be scaled wrongly")
    scale, klima_scale = 10 ** dec_bct, 10 ** dec_klima
    t0, t1, ti = utc_ts(MONTH[0]), utc_ts(MONTH[1]), utc_ts(INFLOWS_FROM)
    at = blocks.block_before([t0, t1, ti])
    a, b = at[t0], at[t1]

    # 1-2. month boundaries and bisection
    def find_block() -> dict:
        s_a, s_b = supply(client, a), supply(client, b)
        B, s_before, s_after, steps = bisection(client, a, b)
        if supply(client, B - 1) != s_before:
            raise ValueError("bisection bracket inconsistent")
        return {"block_month_start": a, "block_month_end": b,
                "supply_month_start_raw": str(s_a), "supply_month_end_raw": str(s_b),
                "block": B, "supply_before_raw": str(s_before),
                "supply_after_raw": str(s_after), "steps": steps}

    found = stage("bisection", find_block)
    if (found["block_month_start"], found["block_month_end"]) != (a, b):
        raise ValueError(f"data/raw/{MODULE}/bisection.json was built for blocks "
                         f"{found['block_month_start']}..{found['block_month_end']}, "
                         f"this run pins {a}..{b}; delete it and rerun")
    s_a, s_b = int(found["supply_month_start_raw"]), int(found["supply_month_end_raw"])
    B, steps = found["block"], found["steps"]
    s_before, s_after = int(found["supply_before_raw"]), int(found["supply_after_raw"])
    decline, drop = s_a - s_b, s_before - s_after
    say(f"  October 2025: blocks {a:,}..{b:,}, decline {decline / scale:,.3f} t; "
        f"largest single-block decrease at {B:,}: {drop / scale:,.3f} t ({steps} steps)")

    # 3. cross-check against the reconciled event path
    path = SupplyPath(load_events("BCT"))
    if path.at_block(a) != s_a or path.at_block(b) != s_b:
        raise ValueError("event path disagrees with totalSupply() at month boundaries")
    fr = path.frame()
    month = fr[(fr.block > a) & (fr.block <= b)]
    if sum(month.net) != s_b - s_a:
        raise ValueError("month's events do not sum to the decline")
    ranked = sorted(zip(month.net, month.block))
    if ranked[0][1] != B or -ranked[0][0] != drop:
        raise ValueError(f"events put the largest decrease at {ranked[0][1]}, "
                         f"bisection at {B}")

    # 4. the block
    def read_block() -> dict:
        blk = client.call("eth_getBlockByNumber", [hex(B), False])
        transfers = decode_transfers(client.call("eth_getLogs", [
            {"address": BCT, "fromBlock": hex(B), "toBlock": hex(B),
             "topics": [TRANSFER_TOPIC]}]))
        burns = [x for x in transfers if x["to"] == ZERO]
        mints = [x for x in transfers if x["from"] == ZERO]
        if sum(x["value"] for x in burns) - sum(x["value"] for x in mints) != drop:
            raise ValueError("Transfer logs in the block do not account for the decrease")
        totals: dict = defaultdict(int)
        for x in burns:
            totals[x["from"]] += x["value"]
        return {"number": B, "timestamp": int(blk["timestamp"], 16),
                "hash": blk["hash"], "bct_burn_logs": len(burns),
                "bct_mint_logs": len(mints),
                "burned_by_raw": {k: str(v) for k, v in totals.items()},
                "transaction_hashes": sorted({x["tx"] for x in burns})}

    block = stage("block", read_block)
    ts = block["timestamp"]
    by_burner = {k: int(v) for k, v in block["burned_by_raw"].items()}
    burner = max(by_burner, key=by_burner.get)
    txs = stage("transactions",
                lambda: [analyse_tx(client, h, B) for h in block["transaction_hashes"]])

    # 5. the account
    def read_account() -> dict:
        roles = {}
        for sig in ROLES:
            r = call_or_revert(client, TREASURY, calldata(sig, burner), B)
            roles[sig] = "reverted" if r is None else bool(as_uint(r))
        balances = [str(as_uint(r)) for r in eth_calls(client, [
            (BCT, calldata("balanceOf(address)", TREASURY), B - 1),
            (BCT, calldata("balanceOf(address)", TREASURY), B),
            (BCT, calldata("balanceOf(address)", burner), B - 1),
            (BCT, calldata("balanceOf(address)", burner), B),
            (TREASURY, calldata("totalReserves()"), B - 1),
            (TREASURY, calldata("totalReserves()"), B),
            (KLIMA, calldata("totalSupply()"), B - 1),
            (KLIMA, calldata("totalSupply()"), B)])]
        senders: dict = defaultdict(lambda: {"transfers": 0, "raw": 0, "blocks": []})
        for blk_n, _, _, _, value, frm, _ in asset_transfers(
                client, BCT, at[ti], B, toAddress=burner):
            s = senders[frm]
            s["transfers"] += 1
            s["raw"] += int(value)
            s["blocks"].append(blk_n)
        return {
            "burner": burner, "safe": safe_info(client, burner, B),
            "managers": [as_address(r) for r in eth_calls(
                client, [(TREASURY, calldata("manager()"), B - 1),
                         (TREASURY, calldata("manager()"), B)])],
            "roles": roles, "balances_raw": balances,
            "inflow_senders": {k: {"transfers": v["transfers"], "raw": str(v["raw"]),
                                   "first_block": min(v["blocks"]),
                                   "last_block": max(v["blocks"])}
                               for k, v in senders.items()}}

    account = stage("account", read_account)
    if account["burner"] != burner:
        raise ValueError(f"data/raw/{MODULE}/account.json is about "
                         f"{account['burner']}, this run found {burner}")
    safe, managers, roles = account["safe"], account["managers"], account["roles"]
    bal = [int(v) for v in account["balances_raw"]]
    senders = {k: {"transfers": v["transfers"], "tonnes": int(v["raw"]) / scale,
                   "first_block": v["first_block"], "last_block": v["last_block"],
                   "is_treasury": k == TREASURY}
               for k, v in account["inflow_senders"].items()}

    # 7. were the redeemed credits still held by the burner at the window end?
    decimals = {tok: t["credits_leaving_pool"]["tco2_decimals"][tok]
                for t in txs for tok in t["credits_leaving_pool"]["tco2_tokens"]}
    tokens = sorted(decimals)

    def read_balances() -> dict:
        out = {}
        for token in tokens:
            out[token] = [str(as_uint(r)) for r in eth_calls(client, [
                (token, calldata("balanceOf(address)", burner), B),
                (token, calldata("balanceOf(address)", burner), end)])]
        return out

    held = {tok: [int(v) / 10 ** decimals[tok] for v in raw_pair]
            for tok, raw_pair in stage("credit_balances", read_balances).items()}

    def credit_transfers(**filters) -> list[dict]:
        """Transfers of the redeemed credit tokens after the burn block, all pages."""
        rows = []
        for i in range(0, len(tokens), 100):
            params = {"fromBlock": hex(B + 1), "toBlock": hex(end),
                      "contractAddresses": tokens[i:i + 100], "category": ["erc20"],
                      "excludeZeroValue": False, "maxCount": hex(1000),
                      "order": "asc", **filters}
            key = None
            while True:
                res = client.call("alchemy_getAssetTransfers",
                                  [dict(params, pageKey=key) if key else params])
                rows += res["transfers"]
                key = res.get("pageKey")
                if not key:
                    break
        return rows

    def tonnes(x: dict) -> float:
        return int(x["rawContract"]["value"], 16) / 10 ** decimals[x["rawContract"]["address"].lower()]

    def read_outflows() -> dict:
        out: dict = defaultdict(lambda: {"transfers": 0, "tonnes": 0.0,
                                         "first_block": None, "last_block": None})
        for x in credit_transfers(fromAddress=burner):
            o = out[(x["to"] or "").lower()]
            o["transfers"] += 1
            o["tonnes"] += tonnes(x)
            blk_n = int(x["blockNum"], 16)
            o["first_block"] = blk_n if o["first_block"] is None else min(o["first_block"], blk_n)
            o["last_block"] = blk_n if o["last_block"] is None else max(o["last_block"], blk_n)
        return dict(out)

    outflows = stage("credit_outflows", read_outflows)

    # 7b. where the credits ended up: balance, code and retirements by the window end.
    # A TCO2 retirement burns the token, so a transfer to the zero address from the
    # destination is the on-chain record of retirement.
    def read_destinations() -> dict:
        out = {}
        for dest, flow in outflows.items():
            if dest == ZERO or flow["tonnes"] < 1:
                continue
            balances = [as_uint(r) for r in eth_calls(client, [
                (tok, calldata("balanceOf(address)", dest), end) for tok in tokens])]
            code = client.call("eth_getCode", [dest, hex(end)])
            slot = client.call("eth_getStorageAt",
                               [dest, EIP1967_IMPLEMENTATION_SLOT, hex(end)])
            retired = credit_transfers(fromAddress=dest, toAddress=ZERO)
            out[dest] = {
                "received_tonnes": flow["tonnes"],
                "tonnes_held_at_end_block": sum(bal / 10 ** decimals[tok]
                                                for tok, bal in zip(tokens, balances)),
                "is_contract": code not in ("0x", ""),
                "eip1967_implementation": ("0x" + slot[-40:]) if int(slot, 16) else None,
                "retired_tonnes_by_end_block": sum(tonnes(x) for x in retired),
                "retirement_transfers": len(retired)}
        return out

    destinations = stage("credit_destinations", read_destinations)

    credits_held = {"tokens": len(held),
                    "burner_tonnes_at_block": sum(v[0] for v in held.values()),
                    "burner_tonnes_at_end_block": sum(v[1] for v in held.values()),
                    "tokens_fully_held_at_end_block": sum(
                        1 for v in held.values() if v[1] >= v[0] > 0),
                    "outflows_after_block_by_destination": {
                        k: {**v, "is_zero_address": k == ZERO}
                        for k, v in sorted(outflows.items(), key=lambda kv: -kv[1]["tonnes"])},
                    "destinations_at_end_block": destinations}

    # 8. peak
    p = peak(path)
    confirm = stage("peak", lambda: {
        "first_block": p["first_block"],
        "at_raw": str(supply(client, p["first_block"])),
        "before_raw": str(supply(client, p["first_block"] - 1))})
    if confirm["first_block"] != p["first_block"] or \
            (confirm["at_raw"], confirm["before_raw"]) != (p["supply_raw"],
                                                           p["supply_before_raw"]):
        raise ValueError("peak not confirmed by totalSupply()")

    touched = sorted({k for t in txs for k in t["registry_pools_touched"]})
    swap_events = [e for t in txs for e in t["swap_events"]]
    computed = {
        "tonnes_in_block": round(drop / scale),
        "date": iso(ts)[:10],
        "multisig": (f"{safe['threshold']}-of-{len(safe['owners'])}"
                     if safe["is_safe"] else "not a Safe"),
        "share_of_supply_pct": round(100 * drop / s_before),
        "share_of_month_decline_pct": round(100 * drop / decline, 1),
        "manager_of_treasury": managers[0] == burner and managers[1] == burner,
        "no_swap": not touched and not swap_events,
        "peak_million_tonnes": round(int(p["supply_raw"]) / scale / 1e6, 1),
        "peak_month": p["first_timestamp"][:7],
        # Both labels are reported, to say which one a quoted month came from.
        "peak_month_on_the_monthly_grid": month_grid_label(p["first_timestamp"]),
    }
    out = {
        "token": BCT, "end_block": end,
        "month": {"start": {"instant": iso(t0), "block": a, "supply_raw": str(s_a),
                            "tonnes": s_a / scale},
                  "end": {"instant": iso(t1), "block": b, "supply_raw": str(s_b),
                          "tonnes": s_b / scale},
                  "decline_raw": str(decline), "decline_tonnes": decline / scale},
        "bisection": {"steps": steps, "block": B,
                      "supply_before_raw": str(s_before), "supply_after_raw": str(s_after),
                      "decrease_raw": str(drop), "decrease_tonnes": drop / scale},
        "event_cross_check": {
            "blocks_with_supply_change_in_month": len(month),
            "largest_net_decreases": [{"block": int(bk), "net_tonnes": n / scale}
                                      for n, bk in ranked[:5]]},
        "block": {"number": B, "timestamp": iso(ts), "hash": block["hash"],
                  "bct_burn_logs": block["bct_burn_logs"],
                  "bct_mint_logs": block["bct_mint_logs"],
                  "burned_by": {k: v / scale for k, v in by_burner.items()}},
        "shares": {"of_supply_before_block_pct": 100 * drop / s_before,
                   "of_supply_at_month_start_pct": 100 * drop / s_a,
                   "of_month_decline_pct": 100 * drop / decline},
        "burner": {"address": burner, **safe},
        "treasury": {"address": TREASURY, "manager_before_block": managers[0],
                     "manager_at_block": managers[1],
                     "roles_for_burner_at_block": roles,
                     "bct_balance_tonnes": {"before": bal[0] / scale, "after": bal[1] / scale},
                     "total_reserves": {"before": bal[4] / klima_scale,
                                        "after": bal[5] / klima_scale},
                     "klima_supply": {"before": bal[6] / klima_scale,
                                      "after": bal[7] / klima_scale}},
        "burner_bct_balance_tonnes": {"before": bal[2] / scale, "after": bal[3] / scale},
        "burner_bct_inflows_since": {"from_instant": iso(ti), "from_block": at[ti],
                                     "to_block": B, "by_sender": senders},
        "transactions": txs,
        "redeemed_credits_held_by_burner": credits_held,
        "markets": {"registry_pools_touched": touched, "swap_events": swap_events,
                    "swap_signatures_checked": SWAP_EVENTS},
        "peak": {**p, "tonnes": int(p["supply_raw"]) / scale},
        "computed": computed,
        "claims": {k: {"paper": PAPER[k], "computed": computed[k],
                       "match": PAPER[k] == computed[k]} for k in PAPER},
    }
    write_json(out, data("burn.json"))
    for k, v in out["claims"].items():
        say(f"  {k:28s} paper {v['paper']!s:>12}  computed {v['computed']!s:>12}  "
            f"{'ok' if v['match'] else 'MISMATCH'}")
    say(f"  wrote {data('burn.json')}")
    record_spend(MODULE, client)
    raw_manifest(MODULE, {"end_block": end, "burn_block": B, "burner": burner})


if __name__ == "__main__":
    run()
