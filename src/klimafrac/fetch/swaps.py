"""
Swap-level trade history, decoded from each pool's own events.

Every reconstructed price starts here. One output row is one Swap event, and
rows are ordered by (block, log_index), which is execution order.

Three stages per pool, each resumable and pinned to a fixed end block:

  1. logs        every log the pair contract emitted over the window, read
                 with eth_getLogs. Not only Swap: Sync carries the reserves
                 after each trade, Mint and Burn the liquidity operations, so
                 the whole history of the pool is captured once and never has
                 to be fetched again. Ranges adapt to the node's 10,000-log
                 response limit, aiming just under it.
  2. transfers   a log carries no time, so block times come from the pair's
                 ERC-20 transfers (alchemy_getAssetTransfers returns a
                 timestamp per transfer and costs far less than one block call
                 per block). The transfers are also evidence in their own
                 right: every swap moves both tokens, so they cross-check the
                 amounts and catch a missing log.
  3. build       Swap logs are decoded, timed, reconciled against the reserve
                 path, cross-checked against the transfers, and written.

Two checks make the result verifiable rather than merely plausible:

  reserve path   a constant-product pair emits Sync with its reserves before
                 each Swap, Mint or Burn, so the change in reserves between
                 consecutive Syncs must equal the amounts of the event between
                 them. Walking the whole log stream that way proves no event
                 is missing anywhere in the window, which sampling cannot.
                 A burn is the one event that can legitimately fall short of
                 the identity, because it pays out a share of tokens sent to
                 the pair since the last Sync that it does not report; those
                 are accepted only when the tokens' own transfers account for
                 the reserve change exactly, and are counted.
  transfers      for a transaction holding one swap, the tokens that moved
                 must equal the event's amounts. On Aerodrome the fee leaves
                 the pool during the swap, so the fee is added back before
                 comparing; the fraction of swaps actually covered is recorded
                 and a floor is enforced, because a check that silently covers
                 nothing is worse than no check at all.

Amounts come from the Swap event, which is what the pair booked. Where a pair
already held surplus tokens -- a donation, or the remains of an earlier
transaction -- it credits that surplus to the next swap, so the event can
report a larger input than the transaction paid; `build` counts those.

Output data/swaps_<chain>_<SYMBOL>.csv.gz:
  block, tx_index, log_index   position in the chain: execution order
  tx_hash, timestamp           transaction and the time of its block
  base_amt, quote_amt          net amounts from the pool's side, decimal
                               adjusted (+ = the pool received the token)
  price                        |quote_amt| / |base_amt|, quote per base
  side                         sell if the trader sold base to the pool
  source                       always "log"
  base_raw, quote_raw          the same net amounts as exact integers

data/swaps_manifest.json records, per pool, the pool identity, block range,
fetch parameters, check counts, checksums and RPC spend.

Under data/raw/<chain>/<SYMBOL>/: logs/ and transfers/ chunks, pool.json,
params.json (the fetch parameters and code hash a chunk was written under; a
chunk written under different parameters is refused rather than reused).

Usage:
    uv run python -m klimafrac.fetch.swaps                    # every pool
    uv run python -m klimafrac.fetch.swaps --pool polygon_NCT --jobs 4
    uv run python -m klimafrac.fetch.swaps --estimate         # cost, no fetch
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from Crypto.Hash import keccak

from klimafrac.fetch.pools import BY_KEY, END_BLOCK, POLYGON_START_BLOCK, POOLS, Pool
from klimafrac.fetch.rpc import Client, RpcError
from klimafrac.paths import RAW_DIR, data


def keccak_hex(text: str) -> str:
    h = keccak.new(digest_bits=256)
    h.update(text.encode())
    return "0x" + h.hexdigest()


V2_SWAP = keccak_hex("Swap(address,uint256,uint256,uint256,uint256,address)")
AERODROME_SWAP = keccak_hex("Swap(address,address,uint256,uint256,uint256,uint256)")
V2_SYNC = keccak_hex("Sync(uint112,uint112)")
AERODROME_SYNC = keccak_hex("Sync(uint256,uint256)")
V2_BURN = keccak_hex("Burn(address,uint256,uint256,address)")
AERODROME_BURN = keccak_hex("Burn(address,address,uint256,uint256)")
MINT = keccak_hex("Mint(address,uint256,uint256)")
FEES = keccak_hex("Fees(address,uint256,uint256)")

EVENT_NAMES = {
    V2_SWAP: "swap", AERODROME_SWAP: "swap",
    V2_SYNC: "sync", AERODROME_SYNC: "sync",
    MINT: "mint", V2_BURN: "burn", AERODROME_BURN: "burn", FEES: "fees",
    keccak_hex("Transfer(address,address,uint256)"): "lp_transfer",
    keccak_hex("Approval(address,address,uint256)"): "lp_approval",
    keccak_hex("Claim(address,address,uint256,uint256)"): "claim",
}

SCHEMA = 2
LOG_CHUNK = 1_000_000            # blocks per stored log file
LOG_SPAN_START = 200_000         # first span tried inside a chunk
LOG_SPAN_MIN = 1                 # a single block is the smallest fetchable unit
LOG_LIMIT = 10_000               # the node's response cap
LOG_TARGET = 8_000               # logs aimed for per call, just under the cap
TRANSFER_CHUNK = 5_000_000       # blocks per stored transfer file
TRANSFER_PAGE = 1_000            # the API maximum
COVERAGE_FLOOR = 0.9             # of swap transactions the transfer check covers
CU = {"eth_getLogs": 60, "alchemy_getAssetTransfers": 120, "eth_call": 26,
      "eth_getCode": 20, "eth_blockNumber": 10}
CU_PRICE = 0.525 / 1e6           # dollars per compute unit, pay as you go

STREAMS = {                      # name: (leg, filter on the pair)
    "base_in": ("base", "toAddress"),
    "base_out": ("base", "fromAddress"),
    "quote_in": ("quote", "toAddress"),
    "quote_out": ("quote", "fromAddress"),
}
LOG_COLUMNS = ["block", "tx_index", "log_index", "tx_hash", "event", "topics", "data"]
LOG_DTYPES = {"block": "int64", "tx_index": "int64", "log_index": "int64",
              "tx_hash": str, "event": str, "topics": str, "data": str}
TRANSFER_COLUMNS = ["tx_hash", "block", "log_index", "timestamp", "raw_value",
                    "from", "to"]
TRANSFER_DTYPES = {"tx_hash": str, "block": "int64", "log_index": "int64",
                   "timestamp": str, "raw_value": str, "from": str, "to": str}
OUTPUT_COLUMNS = ["block", "tx_index", "log_index", "tx_hash", "timestamp",
                  "base_amt", "quote_amt", "price", "side", "source",
                  "base_raw", "quote_raw"]


class Meter:
    """Counts calls per method so a run can report what it spent."""

    def __init__(self):
        self.calls = defaultdict(int)
        self.lock = threading.Lock()

    def add(self, method: str, n: int = 1) -> None:
        with self.lock:
            self.calls[method] += n

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.calls)

    @staticmethod
    def cost(calls: dict) -> dict:
        units = sum(CU.get(m, 0) * n for m, n in calls.items())
        return {"calls": calls, "compute_units": units,
                "usd": round(units * CU_PRICE, 4)}

    def since(self, before: dict) -> dict:
        now = self.snapshot()
        return self.cost({m: n - before.get(m, 0) for m, n in now.items()
                          if n - before.get(m, 0)})


METER = Meter()
_LOCAL = threading.local()


def thread_client(chain: str) -> Client:
    """One client per worker thread: an HTTP session is not thread-safe."""
    clients = getattr(_LOCAL, "clients", None)
    if clients is None:
        clients = _LOCAL.clients = {}
    if chain not in clients:
        clients[chain] = Client(chain)
    return clients[chain]


# --- helpers ---------------------------------------------------------------

def say(msg: str) -> None:
    print(msg, flush=True)


def code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def fetch_params() -> dict:
    """
    What a stored chunk holds, and what wrote it.

    Only the first group decides whether a chunk can be reused: these are the
    choices that change its contents. The code hashes are provenance -- every
    version that contributed chunks to this pool -- so that editing the module
    does not invalidate data already paid for, while the record still shows
    what produced it.
    """
    return {"schema": SCHEMA, "log_chunk": LOG_CHUNK, "log_limit": LOG_LIMIT,
            "transfer_chunk": TRANSFER_CHUNK, "transfer_page": TRANSFER_PAGE,
            "exclude_zero_value": False, "end_block": END_BLOCK,
            "code_sha": [code_sha()]}


DATA_KEYS = ("schema", "log_chunk", "log_limit", "transfer_chunk",
             "transfer_page", "exclude_zero_value", "end_block")


def check_params(pool: Pool) -> None:
    path = pool_path(pool, "params.json")
    current = fetch_params()
    if not path.exists():
        write_json_atomic(current, path)
        return
    stored = json.loads(path.read_text())
    differ = {k: (stored.get(k), current[k]) for k in DATA_KEYS
              if stored.get(k) != current[k]}
    if differ:
        raise SystemExit(
            f"{pool.key}: data/raw holds chunks fetched under different "
            f"parameters: {differ}\nDelete {pool_path(pool)} and refetch, or "
            f"restore the code that wrote them.")
    shas = stored.get("code_sha", [])
    if code_sha() not in shas:
        write_json_atomic({**stored, "code_sha": shas + [code_sha()]}, path)


def pool_path(pool: Pool, *parts: str) -> Path:
    path = (RAW_DIR / pool.chain / pool.symbol).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Deterministic bytes: the gzip header carries no timestamp, so the same
    data always hashes the same and a reviewer can compare checksums."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression={"method": "gzip", "mtime": 0})
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def chunks(lo: int, hi: int, size: int):
    start = lo
    while start <= hi:
        yield start, min(start + size - 1, hi)
        start += size


def words(hexdata: str) -> list[int]:
    body = hexdata[2:]
    return [int(body[i:i + 64], 16) for i in range(0, len(body), 64)]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def start_block(pool: Pool, meta: dict) -> int:
    floor = POLYGON_START_BLOCK if pool.chain == "polygon" else 0
    return max(floor, meta["deploy_block"])


# --- pool identity ---------------------------------------------------------

def describe_pool(pool: Pool, client: Client) -> dict:
    """
    Check the pair against the registry and read its decimals, every run.

    Only the deployment block is cached: it costs a bisection, while the
    identity checks cost four calls and are the thing most worth repeating
    before spending money on a pool.
    """
    path = pool_path(pool, "pool.json")
    cached = json.loads(path.read_text()) if path.exists() else {}
    end = END_BLOCK[pool.chain]

    def address(signature: str, target: str) -> str:
        METER.add("eth_call")
        return "0x" + client.eth_call(target, keccak_hex(signature)[:10], end)[-40:].lower()

    token0, token1 = address("token0()", pool.pair), address("token1()", pool.pair)
    factory = address("factory()", pool.pair)
    if {token0, token1} != {pool.base, pool.quote}:
        raise ValueError(f"{pool.key}: pair holds {token0}/{token1}, "
                         f"registry says {pool.base}/{pool.quote}")
    if factory != pool.factory:
        raise ValueError(f"{pool.key}: factory {factory}, registry {pool.factory}")
    decimals = {}
    for token in (token0, token1):
        METER.add("eth_call")
        decimals[token] = int(client.eth_call(token, keccak_hex("decimals()")[:10], end), 16)

    meta = {"pool": pool.__dict__, "token0": token0, "token1": token1,
            "base_is_token0": token0 == pool.base,
            "base_decimals": decimals[pool.base],
            "quote_decimals": decimals[pool.quote],
            "deploy_block": cached.get("deploy_block")
            or deployment_block(client, pool.pair, end),
            "identity_checked": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    write_json_atomic(meta, path)
    return meta


def deployment_block(client: Client, address: str, end: int) -> int:
    """First block at which the contract has code, by bisection."""
    METER.add("eth_getCode")
    if client.call("eth_getCode", [address, hex(end)]) in ("0x", ""):
        raise ValueError(f"no code at {address} by block {end}")
    lo, hi = 0, end
    while lo < hi:
        mid = (lo + hi) // 2
        METER.add("eth_getCode")
        if client.call("eth_getCode", [address, hex(mid)]) in ("0x", ""):
            lo = mid + 1
        else:
            hi = mid
    return lo


# --- stage 1: every log of the pair ----------------------------------------

def too_many(error: Exception) -> bool:
    text = str(error).lower()
    return ("response size exceeded" in text or "more than" in text
            or "query timeout" in text or "too large" in text)


def next_span(span: int, got: int) -> int:
    """Aim at LOG_TARGET logs per call, damped so one quiet stretch cannot
    blow the span up into an error on the next."""
    if got == 0:
        return min(span * 4, LOG_CHUNK)
    return max(LOG_SPAN_MIN, min(int(span * LOG_TARGET / got), span * 4, LOG_CHUNK))


def logs_in_range(client: Client, address: str, a: int, b: int,
                  span: int) -> tuple[list[dict], int]:
    """Every log in [a, b], in sub-ranges sized to the response limit."""
    out, at = [], a
    while at <= b:
        upto = min(at + span - 1, b)
        try:
            METER.add("eth_getLogs")
            logs = client.call("eth_getLogs", [{"address": address,
                                                "fromBlock": hex(at),
                                                "toBlock": hex(upto)}])
        except RpcError as e:
            if not too_many(e) or span <= LOG_SPAN_MIN:
                raise
            span = max(LOG_SPAN_MIN, span // 2)
            continue
        # This node errors rather than truncating, so a full response is
        # complete; the guard stands for providers that silently cut off.
        if len(logs) >= LOG_LIMIT and span > LOG_SPAN_MIN:
            span = max(LOG_SPAN_MIN, span // 2)
            continue
        out.extend(logs)
        at = upto + 1
        span = next_span(span, len(logs))
    return out, span


def log_rows(pool: Pool, logs: list[dict]) -> list[tuple]:
    rows = []
    for lg in logs:
        if lg["address"].lower() != pool.pair:
            raise ValueError(f"{pool.key}: log from {lg['address']}, not the pair")
        if lg.get("removed"):
            raise ValueError(f"{pool.key}: reorged log returned: {lg}")
        rows.append((int(lg["blockNumber"], 16), int(lg["transactionIndex"], 16),
                     int(lg["logIndex"], 16), lg["transactionHash"].lower(),
                     EVENT_NAMES.get(lg["topics"][0], "other"),
                     json.dumps(lg["topics"]), lg["data"]))
    return rows


def fetch_logs(pool: Pool, meta: dict, jobs: int) -> None:
    spans = list(chunks(start_block(pool, meta), END_BLOCK[pool.chain], LOG_CHUNK))
    todo = [(a, b) for a, b in spans
            if not pool_path(pool, "logs", f"{a}_{b}.csv.gz").exists()]
    if not todo:
        return
    say(f"  {pool.key}: {len(todo)} of {len(spans)} log chunks to fetch")

    def one(span: tuple[int, int]) -> int:
        a, b = span
        logs, _ = logs_in_range(thread_client(pool.chain), pool.pair, a, b,
                                LOG_SPAN_START)
        rows = log_rows(pool, logs)
        write_csv_atomic(pd.DataFrame(rows, columns=LOG_COLUMNS),
                         pool_path(pool, "logs", f"{a}_{b}.csv.gz"))
        return len(rows)

    with ThreadPoolExecutor(max_workers=jobs) as workers:
        for done, ((a, b), n) in enumerate(zip(todo, workers.map(one, todo)), 1):
            say(f"  {pool.key} logs {a:,}-{b:,}: {n:,} logs ({done}/{len(todo)})")


def load_logs(pool: Pool, meta: dict) -> pd.DataFrame:
    frames = []
    for a, b in chunks(start_block(pool, meta), END_BLOCK[pool.chain], LOG_CHUNK):
        path = pool_path(pool, "logs", f"{a}_{b}.csv.gz")
        if not path.exists():
            raise FileNotFoundError(f"logs incomplete: {path}")
        frames.append(pd.read_csv(path, dtype=LOG_DTYPES))
    df = pd.concat(frames, ignore_index=True)
    if df.duplicated(["block", "log_index"]).any():
        raise ValueError(f"{pool.key}: duplicated (block, log_index) in logs")
    return df.sort_values(["block", "log_index"]).reset_index(drop=True)


# --- stage 2: transfers, for block times and as a cross-check --------------

def transfer_pages(client: Client, pool: Pool, token: str, side: str,
                   a: int, b: int, decimals: int) -> list[tuple]:
    params = {"fromBlock": hex(a), "toBlock": hex(b),
              "contractAddresses": [token], "category": ["erc20"],
              side: pool.pair, "withMetadata": True,
              "excludeZeroValue": False, "maxCount": hex(TRANSFER_PAGE),
              "order": "asc"}
    rows, page_key = [], None
    while True:
        query = dict(params, pageKey=page_key) if page_key else params
        METER.add("alchemy_getAssetTransfers")
        res = client.call("alchemy_getAssetTransfers", [query])
        for t in res["transfers"]:
            contract = t["rawContract"]
            if contract["address"].lower() != token:
                raise ValueError(f"unexpected contract {contract['address']}")
            if contract.get("decimal") and int(contract["decimal"], 16) != decimals:
                raise ValueError(f"decimals mismatch for {token}")
            if contract.get("value") is None or t.get("to") is None:
                raise ValueError(f"transfer without value or recipient: {t['uniqueId']}")
            tx, kind, index = t["uniqueId"].split(":")
            if kind != "log" or tx.lower() != t["hash"].lower():
                raise ValueError(f"unexpected uniqueId {t['uniqueId']}")
            block = int(t["blockNum"], 16)
            if not a <= block <= b:
                raise ValueError(f"transfer outside [{a}, {b}]: block {block}")
            rows.append((tx.lower(), block, int(index),
                         t["metadata"]["blockTimestamp"],
                         str(int(contract["value"], 16)),
                         t["from"].lower(), t["to"].lower()))
        page_key = res.get("pageKey")
        if not page_key:
            return rows


def fetch_transfers(pool: Pool, meta: dict, jobs: int) -> None:
    spans = list(chunks(start_block(pool, meta), END_BLOCK[pool.chain], TRANSFER_CHUNK))
    todo = [(stream, a, b) for stream in STREAMS for a, b in spans
            if not pool_path(pool, "transfers", f"{stream}_{a}_{b}.csv.gz").exists()]
    if not todo:
        return
    say(f"  {pool.key}: {len(todo)} of {4 * len(spans)} transfer chunks to fetch")

    def one(job: tuple[str, int, int]) -> int:
        stream, a, b = job
        leg, side = STREAMS[stream]
        rows = transfer_pages(thread_client(pool.chain), pool, getattr(pool, leg),
                              side, a, b, meta[f"{leg}_decimals"])
        write_csv_atomic(pd.DataFrame(rows, columns=TRANSFER_COLUMNS),
                         pool_path(pool, "transfers", f"{stream}_{a}_{b}.csv.gz"))
        return len(rows)

    with ThreadPoolExecutor(max_workers=jobs) as workers:
        for done, ((stream, a, b), n) in enumerate(zip(todo, workers.map(one, todo)), 1):
            say(f"  {pool.key} {stream} {a:,}-{b:,}: {n:,} transfers "
                f"({done}/{len(todo)})")


def load_transfers(pool: Pool, meta: dict) -> pd.DataFrame:
    frames = []
    for stream, (leg, side) in STREAMS.items():
        for a, b in chunks(start_block(pool, meta), END_BLOCK[pool.chain], TRANSFER_CHUNK):
            path = pool_path(pool, "transfers", f"{stream}_{a}_{b}.csv.gz")
            if not path.exists():
                raise FileNotFoundError(f"transfers incomplete: {path}")
            f = pd.read_csv(path, dtype=TRANSFER_DTYPES)
            f["leg"] = leg
            f["inbound"] = side == "toAddress"
            frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    dup = df.duplicated(["tx_hash", "log_index", "inbound"])
    if dup.any():
        raise ValueError(f"{pool.key}: {int(dup.sum())} duplicated transfer rows")
    return df


# --- decoding --------------------------------------------------------------

def orient(net0: int, net1: int, base_is_token0: bool) -> tuple[int, int]:
    return (net0, net1) if base_is_token0 else (net1, net0)


def decode_swap(hexdata: str, base_is_token0: bool) -> tuple[int, int]:
    """Net (base, quote) from the pool's side: in minus out."""
    a0_in, a1_in, a0_out, a1_out = words(hexdata)[:4]
    return orient(a0_in - a0_out, a1_in - a1_out, base_is_token0)


def decode_pair(hexdata: str, base_is_token0: bool, sign: int = 1) -> tuple[int, int]:
    """Two-word events: Sync reserves, Mint and Burn amounts, Fees amounts."""
    a0, a1 = words(hexdata)[:2]
    return orient(sign * a0, sign * a1, base_is_token0)


def event_amounts(event: str, hexdata: str, base_is_token0: bool) -> tuple[int, int] | None:
    """How an event changed the pool's holdings, from the pool's side."""
    if event == "swap":
        return decode_swap(hexdata, base_is_token0)
    if event == "mint":
        return decode_pair(hexdata, base_is_token0, +1)
    if event == "burn":
        return decode_pair(hexdata, base_is_token0, -1)
    return None


# --- stage 3: build --------------------------------------------------------

def position(block, log_index):
    """A log's place in the chain, as one sortable number."""
    return (block << 32) + log_index


def token_flow(transfers: pd.DataFrame) -> dict:
    """The pair's token movements, ordered by log position."""
    t = transfers.sort_values(["block", "log_index"])
    return {"at": position(t.block.to_numpy("int64"), t.log_index.to_numpy("int64")),
            "leg": t.leg.to_numpy(), "inbound": t.inbound.to_numpy(bool),
            "value": [int(v) for v in t.raw_value]}


def net_flow(flow: dict, after: int, until: int) -> tuple[int, int]:
    """
    Net base and quote the tokens moved for the pair over (after, until].

    Summed in Python integers: an 18-decimal amount overruns int64, so numpy
    would wrap around rather than raise.
    """
    lo, hi = np.searchsorted(flow["at"], [after, until], side="right")
    total = {"base": 0, "quote": 0}
    for leg, inbound, value in zip(flow["leg"][lo:hi], flow["inbound"][lo:hi],
                                   flow["value"][lo:hi]):
        total[leg] += value if inbound else -value
    return total["base"], total["quote"]


def reconcile(logs: pd.DataFrame, transfers: pd.DataFrame, meta: dict) -> dict:
    """
    Walk the reserve path: reserves change by exactly the amounts of the event
    that follows each Sync. A missing log breaks the chain, so this proves
    completeness over the whole window rather than over a sample.

    On Aerodrome the fee leaves the pool inside the swap, so the fee of the
    same transaction is subtracted before comparing.

    One departure from that identity is genuine. burn() pays out a share of
    the pair's balance, which includes anything sent to the pair directly
    since the last Sync, while the Burn event reports only the payout; the
    reserves then fall by the payout less that donation. Such an event counts
    as reconciled only if the shortfall is positive in both tokens and the
    tokens themselves account for it: what they moved in and out of the pair
    between the two Syncs must equal the reserve change exactly. A swap
    absorbs a donation into its reported input and a mint into its reported
    deposit, so neither can depart from the identity this way.
    """
    base_is_token0 = meta["base_is_token0"]
    flow = token_flow(transfers)
    checked = breaks = sync_without_event = donations = 0
    donated = [0, 0]
    first_break = None
    previous = current = None                      # reserves before / after
    at_previous = at_current = 0                   # and where those Syncs are
    pending = False                                # a Sync awaiting its event
    fee = [0, 0]                                   # fees accrued since the last event
    for row in logs.itertuples(index=False):
        if row.event == "fees":
            # A pair that sends its fee out emits it just before the swap that
            # charged it, so it belongs to that swap and not to the others a
            # transaction may also contain.
            charged = decode_pair(row.data, base_is_token0, +1)
            fee = [fee[0] + charged[0], fee[1] + charged[1]]
            continue
        if row.event == "sync":
            if pending:
                sync_without_event += 1            # a donation, or skim/sync
            previous, current = current, decode_pair(row.data, base_is_token0)
            at_previous, at_current = at_current, position(int(row.block),
                                                           int(row.log_index))
            pending = True
            continue
        amounts = event_amounts(row.event, row.data, base_is_token0)
        if amounts is None or not pending:
            continue
        pending = False
        charged, fee = fee, [0, 0]
        if previous is None:
            continue                               # nothing to difference yet
        expected = (amounts[0] - charged[0], amounts[1] - charged[1])
        delta = (current[0] - previous[0], current[1] - previous[1])
        checked += 1
        if delta == expected:
            continue
        short = (delta[0] - expected[0], delta[1] - expected[1])
        here = position(int(row.block), int(row.log_index))
        if (row.event == "burn" and min(short) >= 0
                and net_flow(flow, at_previous, here) == delta):
            donations += 1
            donated = [donated[0] + short[0], donated[1] + short[1]]
            continue
        breaks += 1
        if first_break is None:
            first_break = {"block": int(row.block), "log_index": int(row.log_index),
                           "tx_hash": row.tx_hash, "event": row.event,
                           "reserve_delta": [str(d) for d in delta],
                           "event_amounts": [str(e) for e in expected],
                           "shortfall": [str(s) for s in short]}
    return {"reconciled": checked, "breaks": breaks,
            "sync_without_event": sync_without_event,
            "burns_after_donation": donations,
            "donated_raw": [str(d) for d in donated],
            "first_break": first_break}


def moved(transfers: pd.DataFrame) -> pd.DataFrame:
    """Transfers that actually moved tokens.

    The fetch keeps zero-value transfers so the record is complete, but they
    are not pool activity: a token that allows transferFrom with a zero
    allowance lets anyone emit one naming the pair, which address-poisoning
    spam does in bulk.
    """
    return transfers[transfers.raw_value != "0"]


def transfer_amounts(transfers: pd.DataFrame) -> dict[str, dict]:
    """
    Net base and quote per transaction, and how many legs it had.

    Kept in Python integers: an 18-decimal amount overruns float64 and even
    int64, so any arithmetic through numpy would lose the exactness this
    check depends on.
    """
    totals: dict[str, dict] = defaultdict(lambda: {"base": 0, "quote": 0, "legs": 0})
    for tx, leg, inbound, value in zip(transfers.tx_hash, transfers.leg,
                                       transfers.inbound, transfers.raw_value):
        amount = int(value)
        if amount == 0:
            continue                    # address-poisoning spam; see moved()
        entry = totals[tx]
        entry[leg] += amount if inbound else -amount
        entry["legs"] += 1
    return totals


def cross_check(df: pd.DataFrame, logs: pd.DataFrame, transfers: pd.DataFrame,
                meta: dict) -> dict:
    """
    Per transaction, what the pair's events say it gained against what the
    tokens did.

    Every event that moves the pair's holdings counts -- swaps, and the mints
    and burns they are often bundled with -- less the fee, which on Aerodrome
    leaves the pool inside the swap. So the comparison covers whole
    transactions rather than only the simple ones, which on some venues are a
    minority.
    """
    base_is_token0 = meta["base_is_token0"]
    expected: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in logs.itertuples(index=False):
        amounts = event_amounts(row.event, row.data, base_is_token0)
        if amounts is None and row.event != "fees":
            continue
        if row.event == "fees":
            amounts = decode_pair(row.data, base_is_token0, -1)
        entry = expected[row.tx_hash]
        entry[0] += amounts[0]
        entry[1] += amounts[1]

    swap_txs = set(df.tx_hash)
    actual = transfer_amounts(transfers[transfers.tx_hash.isin(swap_txs)])
    checked = agree = surplus = 0
    for tx in swap_txs:
        legs = actual.get(tx)
        if legs is None:
            continue
        checked += 1
        want = expected[tx]
        if want[0] == legs["base"] and want[1] == legs["quote"]:
            agree += 1
        else:
            surplus += 1
    coverage = checked / max(len(swap_txs), 1)
    if coverage < COVERAGE_FLOOR:
        raise ValueError(
            f"transfer cross-check covers only {coverage:.1%} of swap "
            f"transactions ({checked} of {len(swap_txs)}); a check that covers "
            f"little cannot be reported as a pass")
    return {"checked_against_transfers": checked, "coverage": round(coverage, 4),
            "agree_with_transfers": agree, "surplus_input_transactions": surplus,
            "surplus_input_rate": round(surplus / max(checked, 1), 6)}


def build(pool: Pool, meta: dict, logs: pd.DataFrame,
          transfers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    swaps = logs[logs.event == "swap"]
    if swaps.empty:
        raise ValueError(f"{pool.key}: no Swap events in the window")

    times = transfers.groupby("block")["timestamp"].first()
    missing = swaps.loc[~swaps.block.isin(times.index), "tx_hash"]
    if len(missing):
        raise ValueError(f"{pool.key}: {len(missing)} swap blocks have no transfer "
                         f"(first {missing.iloc[0]}) -- transfers are incomplete")
    # Tokens can also leave the pair with no event at all: skim() hands the
    # surplus over the reserves to whoever asks, transferring without emitting
    # anything and without touching the reserves. Those transactions are
    # counted rather than treated as a fault, because the reserve path already
    # proves no event is missing. Zero-value transfers are ignored throughout:
    # anyone may call transferFrom(pair, victim, 0) on a token that permits it,
    # and address-poisoning spam does exactly that.
    explained = set(logs.loc[logs.event.isin(("swap", "burn", "fees")), "tx_hash"])
    real = moved(transfers)
    unexplained = sorted(set(real.loc[~real.inbound, "tx_hash"]) - explained)

    decoded = [decode_swap(d, meta["base_is_token0"]) for d in swaps.data]
    df = pd.DataFrame({
        "block": swaps.block.to_numpy(), "tx_index": swaps.tx_index.to_numpy(),
        "log_index": swaps.log_index.to_numpy(), "tx_hash": swaps.tx_hash.to_numpy(),
        "timestamp": pd.to_datetime(swaps.block.map(times).to_numpy(), utc=True),
        "base_raw": [str(b) for b, _ in decoded],
        "quote_raw": [str(q) for _, q in decoded], "source": "log"})

    db, dq = 10 ** meta["base_decimals"], 10 ** meta["quote_decimals"]
    df["base_amt"] = [int(v) / db for v in df.base_raw]
    df["quote_amt"] = [int(v) / dq for v in df.quote_raw]
    priced = [int(b) != 0 and int(q) != 0 for b, q in zip(df.base_raw, df.quote_raw)]
    df["price"] = (df.quote_amt.abs() / df.base_amt.abs()).where(priced)
    df["side"] = ["sell" if int(v) > 0 else "buy" if int(v) < 0 else ""
                  for v in df.base_raw]
    df = df.sort_values(["block", "log_index"]).reset_index(drop=True)
    if df.duplicated(["block", "log_index"]).any():
        raise ValueError(f"{pool.key}: two swaps share a (block, log_index)")
    if not df.groupby("block").tx_index.is_monotonic_increasing.all():
        raise ValueError(f"{pool.key}: log order disagrees with transaction order")

    counts = {
        "logs": len(logs),
        **{f"logs_{name}": int(n) for name, n in logs.event.value_counts().items()},
        "transfers": len(transfers),
        "swaps": len(df), "swap_transactions": int(df.tx_hash.nunique()),
        "unpriced": int(df.price.isna().sum()),
        "skims": len(unexplained), "skim_sample": unexplained[:5],
        **reconcile(logs, real, meta),
        **cross_check(df, logs, transfers, meta),
        "first": str(df.timestamp.min()), "last": str(df.timestamp.max()),
    }
    if counts["breaks"]:
        raise ValueError(f"{pool.key}: reserve path breaks at {counts['first_break']}")
    return df[OUTPUT_COLUMNS], counts


# --- driver ----------------------------------------------------------------

def chunk_inventory(pool: Pool, meta: dict) -> list[dict]:
    out = []
    for kind, size in (("logs", LOG_CHUNK), ("transfers", TRANSFER_CHUNK)):
        for path in sorted(pool_path(pool, kind).glob("*.csv.gz")):
            out.append({"file": f"{kind}/{path.name}", "sha256": sha256(str(path)),
                        "rows": int(path.stat().st_size and
                                    len(pd.read_csv(path, usecols=[0])))})
    return out


def run(pool: Pool, jobs: int) -> None:
    say(f"\n{pool.key}: pair {pool.pair} on {pool.dex}")
    before = METER.snapshot()
    check_params(pool)
    meta = describe_pool(pool, thread_client(pool.chain))
    fetch_logs(pool, meta, jobs)
    fetch_transfers(pool, meta, jobs)
    logs, transfers = load_logs(pool, meta), load_transfers(pool, meta)
    swaps, counts = build(pool, meta, logs, transfers)

    out = data(f"swaps_{pool.key}.csv.gz")
    tmp = out + ".tmp"
    swaps.to_csv(tmp, index=False, compression={"method": "gzip", "mtime": 0})
    os.replace(tmp, out)
    plain = hashlib.sha256(swaps.to_csv(index=False).encode()).hexdigest()

    manifest_path = Path(data("swaps_manifest.json"))
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[pool.key] = {
        **meta, "start_block": start_block(pool, meta),
        "end_block": END_BLOCK[pool.chain],
        "window_note": ("history before the Polygon data window (block "
                        f"{POLYGON_START_BLOCK:,}) is deliberately excluded"
                        if pool.chain == "polygon"
                        and meta["deploy_block"] < POLYGON_START_BLOCK else
                        "the pool's whole history"),
        "params": fetch_params(), "endpoint": pool.chain,
        "counts": counts, "file": Path(out).name,
        "sha256": sha256(out), "sha256_uncompressed": plain,
        "chunks": chunk_inventory(pool, meta),
        "spent": METER.since(before),
        "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    write_json_atomic(manifest, manifest_path)
    say(f"  {pool.key}: {len(swaps):,} swaps -> {out}")
    say(f"    reserve path: {counts['reconciled']:,} events reconciled "
        f"({counts['burns_after_donation']} after a donation), "
        f"{counts['breaks']} breaks; transfers cover {counts['coverage']:.1%} "
        f"({counts['agree_with_transfers']:,} agree, "
        f"{counts['surplus_input_transactions']:,} surplus)")
    say(f"    spent: {json.dumps(METER.since(before))}")


def estimate(pool: Pool, client: Client) -> dict:
    """Project what a run costs, whether or not anything has been fetched."""
    path = pool_path(pool, "pool.json")
    meta = (json.loads(path.read_text()) if path.exists()
            else {"deploy_block": deployment_block(client, pool.pair, END_BLOCK[pool.chain])})
    lo, hi = start_block(pool, meta), END_BLOCK[pool.chain]
    log_chunks = [(a, b) for a, b in chunks(lo, hi, LOG_CHUNK)
                  if not pool_path(pool, "logs", f"{a}_{b}.csv.gz").exists()]
    transfer_chunks = [(s, a, b) for s in STREAMS for a, b in chunks(lo, hi, TRANSFER_CHUNK)
                       if not pool_path(pool, "transfers", f"{s}_{a}_{b}.csv.gz").exists()]
    # Probe the log density on a few spans to project calls and pages.
    probes, span = [], 100_000
    for a, _ in (log_chunks or [(lo, hi)])[:: max(1, len(log_chunks) // 4 or 1)][:4]:
        METER.add("eth_getLogs")
        probes.append(len(client.call("eth_getLogs", [
            {"address": pool.pair, "fromBlock": hex(a), "toBlock": hex(a + span - 1)}])))
    density = (sum(probes) / len(probes) / span) if probes else 0.0
    logs = density * sum(b - a + 1 for a, b in log_chunks)
    log_calls = len(log_chunks) + math.ceil(logs / LOG_TARGET)
    pages = math.ceil(logs * 0.9 / TRANSFER_PAGE) + len(transfer_chunks)
    calls = {"eth_getLogs": log_calls, "alchemy_getAssetTransfers": pages,
             "eth_call": 5, "eth_getCode": 0 if path.exists() else 28}
    return {"pool": pool.key, "log_chunks_left": len(log_chunks),
            "transfer_chunks_left": len(transfer_chunks),
            "logs_per_1k_blocks": round(1000 * density, 1),
            "projected": Meter.cost(calls)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", action="append", choices=sorted(BY_KEY),
                    help="pool key, repeatable; default every pool")
    ap.add_argument("--jobs", type=int, default=4,
                    help="parallel range fetches per pool (default 4)")
    ap.add_argument("--estimate", action="store_true",
                    help="project what is left to fetch and what it costs")
    args = ap.parse_args()
    keys = args.pool or [p.key for p in POOLS]
    if args.estimate:
        total = defaultdict(int)
        for key in keys:
            pool = BY_KEY[key]
            row = estimate(pool, thread_client(pool.chain))
            say(json.dumps(row))
            for method, n in row["projected"]["calls"].items():
                total[method] += n
        say(f"total {json.dumps(Meter.cost(dict(total)))}")
        return
    for key in keys:
        run(BY_KEY[key], args.jobs)
    say(f"\nspent this run: {json.dumps(Meter.cost(METER.snapshot()))}")


if __name__ == "__main__":
    main()
