"""
Shared on-chain helpers for the non-swap fetchers.

  metered(chain)           a JSON-RPC client that counts its calls
  Blocks.block_before(ts)  the last block whose timestamp is before ts, by
                           bracketed interpolation search; the state "at" an
                           instant is the state after that block
  Blocks.deployment(addr)  the first block at which addr has code
  logs_in_range(...)       eth_getLogs over a span, split to the node's
                           10,000-log limit

Blocks are cached in data/raw/blocks/<chain>.json with their hash, chain id
and endpoint. Every helper raises on failure; an explicit revert (code 3)
is returned as None by `call_or_revert`, and a call the EVM could not
finish (a -32000 message listed in EVM_FAILURES) is recorded in UNANSWERED.
"""

import bisect
import datetime as dt
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

from klimafrac.fetch.pools import END_BLOCK
from klimafrac.fetch.rpc import Client, RpcError
from klimafrac.fetch.swaps import chunks, deployment_block, keccak_hex
from klimafrac.paths import RAW_DIR, raw

WINDOW_END = pd.Timestamp("2026-07-26", tz="UTC")     # exclusive
ZERO = "0x" + "0" * 40
ZERO_TOPIC = "0x" + "0" * 64
TRANSFER_TOPIC = keccak_hex("Transfer(address,address,uint256)")
BATCH = 20
LOG_LIMIT = 10_000                 # the node's cap on logs in one response
SAVE_EVERY = 50                    # block batches between cache writes
SEARCH_GROUP = 8                   # instants searched for at a time
CHAIN_ID = {"polygon": 137, "base": 8453}
TRANSFER_COLUMNS = ["block", "log_index", "tx_hash", "timestamp", "raw_value",
                    "from", "to"]
LOG_COLUMNS = ["block", "log_index", "tx_hash", "from", "to", "raw_value"]

# Alchemy compute units per method, pay-as-you-go price per unit.
CU = {"eth_getLogs": 60, "alchemy_getAssetTransfers": 120, "eth_call": 26,
      "eth_getTransactionReceipt": 20, "eth_getTransactionByHash": 17,
      "eth_getBlockByNumber": 20, "eth_getCode": 20, "eth_getStorageAt": 17,
      "eth_blockNumber": 10, "eth_chainId": 0}
CU_PRICE = 0.525 / 1e6

__all__ = ["BATCH", "CU", "CU_PRICE", "LOG_COLUMNS", "LOG_LIMIT",
           "TRANSFER_COLUMNS", "TRANSFER_TOPIC", "WINDOW_END", "ZERO",
           "ZERO_TOPIC", "Blocks", "Meter", "MeteredClient", "as_address",
           "as_string", "as_uint", "as_uints", "asset_transfers",
           "call_or_revert", "calldata", "calls_allowing_reverts", "chunks",
           "decode_transfer_logs", "eth_calls", "event_topic", "file_record",
           "EVM_FAILURES", "UNANSWERED", "is_execution_failure", "is_revert",
           "iso", "logs_in_range", "metered", "now", "range_hint",
           "raw_manifest", "record_spend", "say", "selector", "sha256_of",
           "topic_address", "utc_ts", "write_csv", "write_json"]


# --- small utilities -------------------------------------------------------

def say(msg: str) -> None:
    print(msg, flush=True)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_ts(value) -> int:
    t = pd.Timestamp(value)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.timestamp())


def iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(obj, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def write_csv(df: pd.DataFrame, path: Path | str, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=index,
              compression="gzip" if path.name.endswith(".gz") else None)
    os.replace(tmp, path)


def sha256_of(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- metering --------------------------------------------------------------

class Meter:
    """Counts calls per method so a run can report what it spent."""

    def __init__(self):
        self.calls: dict[str, int] = {}

    def add(self, method: str, n: int = 1) -> None:
        self.calls[method] = self.calls.get(method, 0) + n

    @property
    def units(self) -> int:
        return sum(CU.get(m, 0) * n for m, n in self.calls.items())

    def summary(self) -> dict:
        unpriced = sorted(m for m in self.calls if m not in CU)
        return {"calls": dict(sorted(self.calls.items())),
                "compute_units": self.units,
                "usd": round(self.units * CU_PRICE, 4),
                "methods_without_a_published_price": unpriced}


class MeteredClient:
    """A `Client` that counts every method it is asked for.

    The wrapper carries the whole client interface used here, so any helper
    that takes a `Client` takes one of these and the count follows the work
    wherever it happens.
    """

    def __init__(self, chain: str):
        self.chain = chain
        self.client = Client(chain)
        self.name = self.client.name
        self.meter = Meter()

    def batch(self, calls: list[tuple[str, list]], allow_errors: bool = False) -> list:
        for method, _ in calls:
            self.meter.add(method)
        return self.client.batch(calls, allow_errors=allow_errors)

    def call(self, method: str, params: list):
        return self.batch([(method, params)])[0]

    def eth_call(self, to: str, data: str, block: int | str = "latest") -> str:
        tag = hex(block) if isinstance(block, int) else block
        return self.call("eth_call", [{"to": to, "data": data}, tag])


def metered(chain: str) -> MeteredClient:
    return MeteredClient(chain)


def record_spend(module: str, client: MeteredClient, extra: dict | None = None) -> Path:
    """Write data/raw/<module>/rpc_calls_<chain>.json: calls, units, dollars."""
    path = raw(module, f"rpc_calls_{client.chain}.json")
    write_json({"module": module, "chain": client.chain, "endpoint": client.name,
                "recorded_at": now(), "price_usd_per_compute_unit": CU_PRICE,
                **client.meter.summary(), **(extra or {})}, path)
    s = client.meter.summary()
    say(f"  {module}: {s['compute_units']:,} compute units (${s['usd']:.4f}) "
        f"-> {path}")
    return path


# --- raw-file manifests ----------------------------------------------------

def file_record(path: Path, root: Path | None = None) -> dict:
    """sha256, size and, for a stored table, its row count and block range."""
    path = Path(path)
    rec = {"file": str(path.relative_to(root)) if root else path.name,
           "bytes": path.stat().st_size, "sha256": sha256_of(path)}
    if path.name.endswith(".json"):
        return rec
    if path.name.endswith((".csv", ".csv.gz")):
        columns = list(pd.read_csv(path, nrows=0).columns)
        # Only the block column is parsed: these tables run to millions of rows.
        use = ["block"] if "block" in columns else columns[:1]
        df = pd.read_csv(path, usecols=use)
        rec["rows"] = int(len(df))
        if "block" in columns and len(df):
            rec["first_block"] = int(df.block.min())
            rec["last_block"] = int(df.block.max())
    elif path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt") as f:
            rec["rows"] = sum(1 for line in f if line.strip())
    return rec


MANIFEST_PATTERNS = ("*.csv.gz", "*.csv", "*.jsonl.gz", "*.json")


def raw_manifest(module: str, extra: dict | None = None,
                 patterns: tuple[str, ...] = MANIFEST_PATTERNS) -> Path:
    """List every raw file a module wrote, with sha256, rows and block range.

    The manifest itself and the spending records are left out: they are
    written after the files they describe, so including them could never be
    made to agree with their own contents.
    """
    base = RAW_DIR / module
    files = sorted(p for pattern in patterns for p in base.rglob(pattern)
                   if not p.name.endswith(".tmp")
                   and p.name != "manifest.json"
                   and not p.name.startswith("rpc_calls"))
    path = base / "manifest.json"
    write_json({"module": module, "root": f"data/raw/{module}", "built_at": now(),
                "files": [file_record(p, base) for p in files],
                **(extra or {})}, path)
    return path


# --- ABI encoding and decoding ---------------------------------------------

def selector(signature: str) -> str:
    return keccak_hex(signature)[:10]


def event_topic(signature: str) -> str:
    return keccak_hex(signature)


def calldata(signature: str, *args) -> str:
    out = selector(signature)
    for a in args:
        if isinstance(a, str):
            if not (a.startswith("0x") and len(a) == 42):
                raise ValueError(f"not an address: {a}")
            out += a[2:].lower().rjust(64, "0")
        elif isinstance(a, bool):
            out += f"{int(a):064x}"
        elif isinstance(a, int) and a < 0:
            out += f"{a & (2 ** 256 - 1):064x}"          # two's complement
        else:
            out += f"{int(a):064x}"
    return out


def as_uint(result: str) -> int:
    if not isinstance(result, str) or len(result) < 66:
        raise ValueError(f"not a uint256 return value: {result!r}")
    return int(result[2:66], 16)


def as_uints(result: str, n: int) -> list[int]:
    if not isinstance(result, str) or len(result) < 2 + 64 * n:
        raise ValueError(f"not {n} uint256 return values: {result!r}")
    return [int(result[2 + 64 * i:66 + 64 * i], 16) for i in range(n)]


def as_address(result: str) -> str:
    if not isinstance(result, str) or len(result) != 66 or int(result[2:26], 16):
        raise ValueError(f"not an address return value: {result!r}")
    return "0x" + result[-40:].lower()


def as_string(result: str) -> str:
    body = bytes.fromhex(result[2:])
    if len(body) == 32:                          # bytes32 symbol
        return body.rstrip(b"\0").decode()
    if len(body) < 96:
        raise ValueError(f"not a string return value: {result!r}")
    offset = int.from_bytes(body[:32], "big")
    n = int.from_bytes(body[offset:offset + 32], "big")
    return body[offset + 32:offset + 32 + n].decode()


def topic_address(topic: str) -> str:
    if not isinstance(topic, str) or len(topic) != 66 or int(topic[2:26], 16):
        raise ValueError(f"not an address topic: {topic!r}")
    return "0x" + topic[-40:].lower()


# --- calls -----------------------------------------------------------------

UNANSWERED: set[str] = set()       # contracts whose getters exhausted the gas cap


def is_revert(error) -> bool:
    """The node executed the call and the contract refused it.

    JSON-RPC code 3 is the execution-reverted code. Nothing else counts: a
    node fault carrying the words "execution reverted" in some wrapper text
    must reach the caller as an error, never as a contract's answer.
    """
    return isinstance(error, dict) and error.get("code") == 3


# Ways the EVM itself can run and produce no value. Every one of them is
# decided by the contract's code and the state at the block asked about, so it
# is reproducible: asking again gets the same answer. They are listed rather
# than matched loosely because -32000 is also the code the node uses when it
# is merely unwell -- "header not found", "missing trie node", a rate limit --
# and those must stay retryable.
EVM_FAILURES = ("out of gas", "gas uint64 overflow", "invalid opcode",
                "invalid jump destination", "stack limit reached",
                "stack underflow", "write protection", "max code size exceeded",
                "contract creation code storage out of gas")


def is_execution_failure(error) -> bool:
    """The node ran the call and no value came back.

    A getter that cannot be evaluated -- it exhausts the gas cap, or trips an
    invalid opcode -- returns nothing however often it is asked, because the
    cap, the code and the state are all fixed by the block. So this is the
    call's outcome rather than a fault to retry, even though it arrives under
    -32000, the code that otherwise carries genuinely transient node trouble.
    The addresses it happens to are recorded in UNANSWERED, so a census can
    report how many of its probes went unanswered instead of silently counting
    them as contracts that answered nothing.
    """
    if not isinstance(error, dict) or error.get("code") != -32000:
        return False
    message = str(error.get("message", "")).lower()
    return any(failure in message for failure in EVM_FAILURES)


def _eth_call(to: str, data: str, block: int) -> tuple[str, list]:
    return ("eth_call", [{"to": to, "data": data}, hex(block)])


def calls_allowing_reverts(client, calls: list[tuple[str, str, int]]) -> list[str | None]:
    """Batched eth_call where a call that produced no answer is None.

    A request carrying `allow_errors` is answered once, without the retries a
    plain batch makes, so a call that came back with anything other than a
    revert or an exhausted gas cap is asked again through the retrying path: a
    transient fault is then retried there, and a permanent one raises.
    """
    out: list[str | None] = []
    for i in range(0, len(calls), BATCH):
        part = calls[i:i + BATCH]
        got = client.batch([_eth_call(*c) for c in part], allow_errors=True)
        answers: list[str | None] = []
        again = []
        for k, (result, error) in enumerate(got):
            if error is not None and not is_revert(error):
                if is_execution_failure(error):
                    UNANSWERED.add(part[k][0])
                else:
                    again.append(k)
            answers.append(None if error is not None else result)
        if again:
            for k, value in zip(again, client.batch([_eth_call(*part[k])
                                                     for k in again])):
                answers[k] = value
        out.extend(answers)
    return out


def call_or_revert(client, to: str, data: str, block: int) -> str | None:
    """eth_call that returns None on an explicit revert and raises otherwise."""
    return calls_allowing_reverts(client, [(to, data, block)])[0]


def eth_calls(client, calls: list[tuple[str, str, int]]) -> list[str]:
    """Batched eth_call; (to, data, block) in, results in order. Any error raises."""
    out = []
    for i in range(0, len(calls), BATCH):
        part = calls[i:i + BATCH]
        out.extend(client.batch([("eth_call", [{"to": to, "data": d}, hex(b)])
                                 for to, d, b in part]))
    return out


# --- logs ------------------------------------------------------------------

RANGE_HINT = re.compile(r"\[\s*(0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*\]")


def range_hint(error: Exception) -> tuple[int, int] | None:
    """The block range the node says would work, from a -32602 refusal.

    Alchemy answers an oversized eth_getLogs with
      "... this block range should work: [0x1406f40, 0x141328f]"
    which is a measurement of the log density just ahead. Following it costs
    one call per refusal; bisecting would cost several.
    """
    text = str(error)
    if "-32602" not in text and "response size exceeded" not in text.lower():
        return None
    m = RANGE_HINT.search(text)
    return (int(m.group(1), 16), int(m.group(2), 16)) if m else None


def logs_in_range(client, query: dict, a: int, b: int) -> list[dict]:
    """Every log matching `query` in blocks [a, b], inclusive.

    The span shrinks to whatever the node says it can serve and grows again
    from the size of the responses, so a quiet stretch of chain costs few
    calls and a busy one stays inside the response limit.
    """
    out, at, span = [], a, b - a + 1
    while at <= b:
        upto = min(at + span - 1, b)
        try:
            logs = client.call("eth_getLogs", [{**query, "fromBlock": hex(at),
                                                "toBlock": hex(upto)}])
        except RpcError as e:
            hint = range_hint(e)
            if hint is None:
                raise
            lo, hi = hint
            if lo != at or not at <= hi < upto:
                # The hint does not start where we asked or does not narrow
                # the range; fall back to halving rather than trust it.
                if span <= 1:
                    raise
                span = max(1, span // 2)
            else:
                span = hi - at + 1
            continue
        if len(logs) >= LOG_LIMIT and upto > at:
            # A response at the cap may have been truncated; redo it smaller.
            span = max(1, span // 2)
            continue
        for lg in logs:
            if lg.get("removed"):
                raise ValueError(f"reorged log returned: {lg}")
        out.extend(logs)
        served = upto - at + 1
        at = upto + 1
        # Aim the next span at 80% of the limit from the density just seen,
        # and never grow by more than 4x at once.
        want = served * LOG_LIMIT * 4 // (5 * len(logs)) if logs else 4 * served
        span = max(1, min(want, 4 * served, max(1, b - at + 1)))
    return out


def decode_transfer_logs(logs: list[dict], token: str) -> list[tuple]:
    """Transfer logs to LOG_COLUMNS rows, exact values as decimal strings."""
    rows = []
    for lg in logs:
        if lg["address"].lower() != token:
            raise ValueError(f"log from {lg['address']}, expected {token}")
        topics = lg["topics"]
        if topics[0] != TRANSFER_TOPIC:
            raise ValueError(f"unexpected topic0 {topics[0]}")
        if len(topics) != 3 or len(lg["data"]) != 66:
            # A non-standard Transfer (fewer indexed arguments, or a packed
            # payload) would be silently mis-decoded; stop instead.
            raise ValueError(f"non-standard Transfer log: {lg}")
        rows.append((int(lg["blockNumber"], 16), int(lg["logIndex"], 16),
                     lg["transactionHash"].lower(), topic_address(topics[1]),
                     topic_address(topics[2]), str(int(lg["data"], 16))))
    return rows


# --- transfers -------------------------------------------------------------

def asset_transfers(client, token: str, a: int, b: int,
                    exclude_zero: bool = False, **side) -> list[tuple]:
    """Every ERC-20 transfer of `token` in blocks [a, b], all pages.

    `side` is fromAddress=... and/or toAddress=... Rows follow
    TRANSFER_COLUMNS, raw_value an exact integer as a string.
    """
    params = {"fromBlock": hex(a), "toBlock": hex(b), "contractAddresses": [token],
              "category": ["erc20"], "withMetadata": True,
              "excludeZeroValue": exclude_zero, "maxCount": hex(1000),
              "order": "asc", **side}
    rows, page_key = [], None
    while True:
        query = dict(params, pageKey=page_key) if page_key else params
        res = client.call("alchemy_getAssetTransfers", [query])
        for t in res["transfers"]:
            contract = t["rawContract"]
            if (contract["address"] or "").lower() != token.lower():
                raise ValueError(f"unexpected contract {contract['address']}")
            if contract["value"] is None:
                raise ValueError(f"transfer without a raw value: {t['uniqueId']}")
            tx, kind, index = t["uniqueId"].split(":")
            if kind != "log" or tx.lower() != t["hash"].lower():
                raise ValueError(f"unexpected uniqueId {t['uniqueId']}")
            rows.append((int(t["blockNum"], 16), int(index), tx.lower(),
                         utc_ts(t["metadata"]["blockTimestamp"]),
                         str(int(contract["value"], 16)),
                         (t["from"] or "").lower(), (t["to"] or "").lower()))
        page_key = res.get("pageKey")
        if not page_key:
            return rows


# --- blocks ----------------------------------------------------------------

class Blocks:
    """Exact timestamp -> block lookups and deployment blocks, cached on disk."""

    def __init__(self, client, chain: str = "polygon"):
        self.client, self.chain = client, chain
        self.end = END_BLOCK[chain]
        self.path = raw("blocks", f"{chain}.json")
        self._ts: dict[int, int] = {}
        self._hash: dict[int, str] = {}
        self._before: dict[int, int] = {}
        self._deploy: dict[str, int] = {}
        self._load()
        self.chain_id = self._check_chain_id()

    def _check_chain_id(self) -> int:
        got = int(self.client.call("eth_chainId", []), 16)
        if got != CHAIN_ID[self.chain]:
            raise RpcError(f"endpoint {self.client.name} reports chain id {got}, "
                           f"expected {CHAIN_ID[self.chain]} for {self.chain}")
        return got

    def _load(self) -> None:
        if not self.path.exists():
            return
        d = json.loads(self.path.read_text())
        for k, v in d.get("blocks", {}).items():
            self._ts[int(k)] = int(v["timestamp"])
            if v.get("hash"):
                self._hash[int(k)] = v["hash"]
        for k, v in d.get("timestamp", {}).items():      # cache written before hashes
            self._ts.setdefault(int(k), int(v))
        self._before.update({int(k): int(v) for k, v in d["block_before"].items()})
        self._deploy.update(d["deployment"])

    def save(self) -> None:
        mine = (dict(self._ts), dict(self._hash), dict(self._before), dict(self._deploy))
        self._load()                         # merge anything another run added
        self._ts.update(mine[0])
        self._hash.update(mine[1])
        self._before.update(mine[2])
        self._deploy.update(mine[3])
        write_json({"chain": self.chain, "chain_id": CHAIN_ID[self.chain],
                    "endpoint": self.client.name, "end_block": self.end,
                    "retrieved_at": now(),
                    "blocks": {str(k): {"timestamp": self._ts[k],
                                        "hash": self._hash.get(k)}
                               for k in sorted(self._ts)},
                    "block_before": {str(k): self._before[k] for k in sorted(self._before)},
                    "deployment": dict(sorted(self._deploy.items()))}, self.path)

    def timestamps(self, blocks) -> dict[int, int]:
        need = sorted({int(b) for b in blocks} - self._ts.keys())
        for n, i in enumerate(range(0, len(need), BATCH)):
            part = need[i:i + BATCH]
            got = self.client.batch([("eth_getBlockByNumber", [hex(b), False])
                                     for b in part])
            for b, blk in zip(part, got):
                if blk is None or int(blk["number"], 16) != b:
                    raise RpcError(f"{self.chain}: block {b} not returned")
                self._ts[b] = int(blk["timestamp"], 16)
                self._hash[b] = blk["hash"]
            if n % SAVE_EVERY == SAVE_EVERY - 1:
                self.save()                  # a crash must not discard these
        if need:
            self.save()
        return {int(b): self._ts[int(b)] for b in blocks}

    def timestamp(self, block: int) -> int:
        return self.timestamps([block])[block]

    def hash(self, block: int) -> str:
        """The block's hash. Read again if the cache predates hashes being kept."""
        block = int(block)
        if block not in self._hash:
            blk = self.client.call("eth_getBlockByNumber", [hex(block), False])
            if blk is None or int(blk["number"], 16) != block:
                raise RpcError(f"{self.chain}: block {block} not returned")
            self._ts[block] = int(blk["timestamp"], 16)
            self._hash[block] = blk["hash"]
            self.save()
        return self._hash[block]

    def block_before(self, targets) -> dict[int, int]:
        """For each unix time t: the last block with timestamp < t.

        Targets must fall after genesis and no later than the timestamp of
        END_BLOCK + 1, so the answer never lies beyond the pinned range.

        Unresolved instants are searched for in ascending groups rather than
        all at once: each group's answers tighten the bracket the next group
        starts from, which costs about a third fewer block reads.
        """
        targets = sorted({int(t) for t in targets})
        todo = [t for t in targets if t not in self._before]
        if todo:
            hi0 = self.end + 1
            ts = self.timestamps([0, hi0])
            outside = [t for t in todo if not ts[0] < t <= ts[hi0]]
            if outside:
                raise ValueError(f"{self.chain}: instants outside the pinned range: "
                                 f"{[iso(t) for t in outside[:3]]}")
        for i in range(0, len(todo), SEARCH_GROUP):
            self._resolve(todo[i:i + SEARCH_GROUP])
        return {t: self._before[t] for t in targets}

    def _resolve(self, todo: list[int]) -> None:
        if todo:
            state = {t: self._seed(t) for t in todo}
            while True:
                probes = {}
                for t, (lo, lts, hi, hts) in state.items():
                    if hi - lo <= 1:
                        continue
                    g = lo + int((t - lts) * (hi - lo) / max(hts - lts, 1))
                    g = min(max(g, lo + 1), hi - 1)
                    probes[t] = {g, min(g + 1, hi - 1), max(g - 1, lo + 1),
                                 (lo + hi) // 2}
                if not probes:
                    break
                known = self.timestamps(set().union(*probes.values()))
                for t, cand in probes.items():
                    s = state[t]
                    for b in cand:
                        if known[b] < t and b > s[0]:
                            s[0], s[1] = b, known[b]
                        elif known[b] >= t and b < s[2]:
                            s[2], s[3] = b, known[b]
            for t, s in state.items():
                self._before[t] = s[0]
            self.save()

    def _seed(self, t: int) -> list[int]:
        """Tightest bracket [lo, hi] from timestamps already known."""
        blocks = sorted(b for b in self._ts if b <= self.end + 1)
        times = [self._ts[b] for b in blocks]
        i = bisect.bisect_left(times, t)          # first known block with ts >= t
        lo, hi = blocks[i - 1], blocks[i]
        return [lo, self._ts[lo], hi, self._ts[hi]]

    def deployment(self, address: str) -> int:
        address = address.lower()
        if address not in self._deploy:
            self._deploy[address] = deployment_block(self.client, address, self.end)
            self.save()
        return self._deploy[address]
