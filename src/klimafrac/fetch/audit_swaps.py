"""
Independent check of every swap file against the chain.

fetch/swaps.py builds the files from the pair's own logs and proves internal
completeness by reconciling the reserve path. This module checks the result
from outside, by routes that share as little with it as possible: it decodes
events with its own decoder rather than importing one, reads transactions
through receipts rather than through log queries, and verifies the timestamps
against the blocks themselves, which is the one step in the pipeline nothing
else corroborates.

  receipts     for a sample of rows stratified by year, the transaction
               receipt is fetched and every Swap log the pair emitted in it is
               decoded here and compared with the file's rows for that
               transaction: same order, same exact amounts, nothing missing,
               nothing extra, and every row's log_index belongs to the
               transaction. Any difference is a discrepancy; there are no
               expected ones.
  ranges       complete block ranges are read back with eth_getLogs and
               compared with every row in those blocks, in order. Half the
               ranges are drawn at random, and the rest sit on the boundaries
               between stored chunks, where an error in chunking would hide.
  timestamps   a sample of rows is checked against the timestamp of its own
               block.

Writes data/swaps_audit.json and exits non-zero if anything disagrees.

Usage:
    uv run python -m klimafrac.fetch.audit_swaps
    uv run python -m klimafrac.fetch.audit_swaps --pool polygon_NCT --receipts 2000
"""

import argparse
import json
import sys
import zlib

import numpy as np
import pandas as pd

from klimafrac.fetch.pools import BY_KEY, POOLS
from klimafrac.fetch.rpc import Client
from klimafrac.paths import data

RECEIPT_SAMPLE = 1_200
RANGES = 200
RANGE_BLOCKS = 2_000
TIMESTAMP_SAMPLE = 200
CHUNK_BOUNDARY = 1_000_000       # fetch/swaps.py LOG_CHUNK
SEED = 20260921
# Written out rather than imported, so a wrong topic or a wrong word offset in
# the fetcher cannot hide behind the same mistake here.
SWAP_TOPICS = {
    "sushiswap": "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822",
    "quickswap": "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822",
    "aerodrome": "0xb3e2773606abfd36b5bd91394b3a54d1398336c65005baf7bf7a05efeffaf75b",
}


def swap_amounts(data_hex: str, base_is_token0: bool) -> tuple[str, str]:
    """(base, quote) net of the pool, decoded from the four data words."""
    body = data_hex[2:]
    if len(body) < 256:
        raise ValueError(f"short Swap payload: {data_hex[:20]}...")
    amounts = [int(body[i * 64:(i + 1) * 64], 16) for i in range(4)]
    net0 = amounts[0] - amounts[2]
    net1 = amounts[1] - amounts[3]
    base, quote = (net0, net1) if base_is_token0 else (net1, net0)
    return str(base), str(quote)


def events(logs: list[dict], pool, base_is_token0: bool) -> list[tuple]:
    """The pair's Swap logs in execution order."""
    topic = SWAP_TOPICS[pool.dex]
    out = []
    for lg in sorted(logs, key=lambda x: (int(x["blockNumber"], 16),
                                          int(x["logIndex"], 16))):
        if lg["address"].lower() != pool.pair or lg["topics"][0] != topic:
            continue
        base, quote = swap_amounts(lg["data"], base_is_token0)
        out.append((int(lg["blockNumber"], 16), int(lg["logIndex"], 16),
                    lg["transactionHash"].lower(), base, quote))
    return out


def rows_of(frame: pd.DataFrame) -> list[tuple]:
    frame = frame.sort_values(["block", "log_index"])
    return [(int(b), int(i), h, x, y) for b, i, h, x, y in
            frame[["block", "log_index", "tx_hash", "base_raw", "quote_raw"]]
            .itertuples(index=False)]


def audit(pool, meta: dict, swaps: pd.DataFrame, receipts: int) -> dict:
    client = Client(pool.chain)
    base_is_token0 = meta["base_is_token0"]
    rng = np.random.default_rng([SEED, zlib.crc32(pool.key.encode())])
    problems = []

    # receipts
    years = pd.to_datetime(swaps.timestamp, utc=True).dt.year
    per_year = max(1, receipts // max(years.nunique(), 1))
    picked = [g.sample(min(per_year, len(g)), random_state=int(rng.integers(1 << 31)))
              for _, g in swaps.groupby(years)]
    hashes = list(dict.fromkeys(pd.concat(picked).tx_hash))
    by_tx = swaps[swaps.tx_hash.isin(hashes)].groupby("tx_hash")
    for i in range(0, len(hashes), 20):
        batch = hashes[i:i + 20]
        for h, rc in zip(batch, client.batch([("eth_getTransactionReceipt", [h])
                                              for h in batch])):
            mine = by_tx.get_group(h)
            in_tx = {int(lg["logIndex"], 16) for lg in rc["logs"]}
            outside = sorted(set(mine.log_index.astype(int)) - in_tx)
            chain_rows = events(rc["logs"], pool, base_is_token0)
            if chain_rows != rows_of(mine) or outside:
                problems.append({"check": "receipt", "tx_hash": h,
                                 "chain": chain_rows, "file": rows_of(mine),
                                 "log_index_outside_transaction": outside})

    # complete ranges: random, plus the seams between stored chunks. Chunks
    # start at the pool's start block, so a seam is start + k * CHUNK_BOUNDARY
    # (the first block of chunk k); a window centred there covers both sides.
    lo, hi = int(swaps.block.min()), int(swaps.block.max())
    start = int(meta["start_block"])
    seams = [b for b in range(start + CHUNK_BOUNDARY, hi + 1, CHUNK_BOUNDARY) if lo <= b <= hi]
    centres = ([int(x) for x in rng.choice(swaps.block.to_numpy(), RANGES // 2)]
               + [int(x) for x in rng.integers(lo, hi + 1, max(0, RANGES // 2 - len(seams)))]
               + seams)
    windows = sorted({(max(lo, c - RANGE_BLOCKS // 2),
                       max(lo, c - RANGE_BLOCKS // 2) + RANGE_BLOCKS - 1)
                      for c in centres})
    seen = 0
    for i in range(0, len(windows), 10):
        calls = [("eth_getLogs", [{"address": pool.pair, "topics": [SWAP_TOPICS[pool.dex]],
                                   "fromBlock": hex(a), "toBlock": hex(b)}])
                 for a, b in windows[i:i + 10]]
        for (a, b), logs in zip(windows[i:i + 10], client.batch(calls)):
            chain_rows = events(logs, pool, base_is_token0)
            seen += len(chain_rows)
            mine = rows_of(swaps[(swaps.block >= a) & (swaps.block <= b)])
            if chain_rows != mine:
                problems.append({
                    "check": "range", "blocks": [a, b],
                    "missing_from_file": sorted(set(chain_rows) - set(mine)),
                    "extra_in_file": sorted(set(mine) - set(chain_rows)),
                    "same_rows_wrong_order": set(chain_rows) == set(mine)})

    # timestamps
    sample = swaps.sample(min(TIMESTAMP_SAMPLE, len(swaps)),
                          random_state=int(rng.integers(1 << 31)))
    blocks = sorted({int(b) for b in sample.block})
    stamped = dict(zip(sample.block, pd.to_datetime(sample.timestamp, utc=True)))
    for i in range(0, len(blocks), 20):
        batch = blocks[i:i + 20]
        for b, blk in zip(batch, client.batch([("eth_getBlockByNumber", [hex(b), False])
                                               for b in batch])):
            chain_time = pd.Timestamp(int(blk["timestamp"], 16), unit="s", tz="UTC")
            if chain_time != stamped[b]:
                problems.append({"check": "timestamp", "block": b,
                                 "chain": str(chain_time), "file": str(stamped[b])})

    return {"receipt_transactions": len(hashes), "ranges": len(windows),
            "range_blocks": RANGE_BLOCKS, "range_events": seen,
            "chunk_seams_checked": len(seams),
            "timestamp_blocks": len(blocks),
            "n_problems": len(problems), "problems": problems[:50]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pool", action="append", choices=sorted(BY_KEY))
    ap.add_argument("--receipts", type=int, default=RECEIPT_SAMPLE)
    args = ap.parse_args()
    manifest = json.loads(open(data("swaps_manifest.json")).read())
    report_path = data("swaps_audit.json")
    try:
        report = json.loads(open(report_path).read())
    except FileNotFoundError:
        report = {}
    failed = False
    for key in args.pool or [p.key for p in POOLS]:
        swaps = pd.read_csv(data(f"swaps_{key}.csv.gz"),
                            dtype={"base_raw": str, "quote_raw": str, "tx_hash": str})
        result = audit(BY_KEY[key], manifest[key], swaps, args.receipts)
        report[key] = result
        failed |= result["n_problems"] > 0
        print(f"{key:14s} receipts {result['receipt_transactions']:5d}  "
              f"ranges {result['ranges']:4d} ({result['range_events']:6d} events, "
              f"{result['chunk_seams_checked']} seams)  "
              f"timestamps {result['timestamp_blocks']:4d}  "
              f"problems {result['n_problems']}", flush=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
