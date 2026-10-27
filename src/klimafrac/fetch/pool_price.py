"""
The pool's own price after every swap, from the logs already fetched.

The execution price, quote amount over base amount, includes the fee and
the rounding of small amounts. The Sync log the pair emits just before each
Swap carries the reserves after it; their ratio, adjusted for decimals, is
the price the pool quotes next. For every pool whose swap file is built,
each Swap is paired with the last Sync before it in the same transaction;
the module fails if any swap has none.

Outputs: data/pool_price_<chain>_<SYMBOL>.csv.gz (block, log_index,
base_reserve_raw, quote_reserve_raw, pool_price),
data/pool_price_manifest.json.

Usage:
    uv run python -m klimafrac.fetch.pool_price [--pool polygon_NCT]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from klimafrac.fetch.pools import BY_KEY, END_BLOCK, POOLS
from klimafrac.fetch.swaps import (LOG_CHUNK, LOG_DTYPES, chunks, pool_path, sha256,
                                   start_block, write_csv_atomic, write_json_atomic)
from klimafrac.paths import data

COLUMNS = ["block", "log_index", "base_reserve_raw", "quote_reserve_raw", "pool_price"]


def sync_logs(pool, meta: dict) -> pd.DataFrame:
    parts = []
    for a, b in chunks(start_block(pool, meta), END_BLOCK[pool.chain], LOG_CHUNK):
        path = pool_path(pool, "logs", f"{a}_{b}.csv.gz")
        if not path.exists():
            raise FileNotFoundError(f"logs incomplete: {path}")
        f = pd.read_csv(path, dtype=LOG_DTYPES, usecols=["block", "log_index", "tx_hash",
                                                         "event", "data"])
        parts.append(f[f.event == "sync"])
    return pd.concat(parts, ignore_index=True).sort_values(["block", "log_index"])


def build(pool) -> dict:
    meta = json.loads(pool_path(pool, "pool.json").read_text())
    swap_file = data(f"swaps_{pool.key}.csv.gz")
    swaps = pd.read_csv(swap_file, usecols=["block", "log_index", "tx_hash", "price"],
                        dtype={"tx_hash": str})
    sync = sync_logs(pool, meta)
    at = np.searchsorted((sync.block.to_numpy() << 32) + sync.log_index.to_numpy(),
                         (swaps.block.to_numpy() << 32) + swaps.log_index.to_numpy()) - 1
    if (at < 0).any():
        raise ValueError(f"{pool.key}: {int((at < 0).sum())} swaps precede every Sync")
    mine = sync.iloc[at]
    same = mine.tx_hash.to_numpy() == swaps.tx_hash.to_numpy()
    if not same.all():
        raise ValueError(f"{pool.key}: {int((~same).sum())} swaps have no Sync in their "
                         f"own transaction (first {swaps.tx_hash[~same].iloc[0]})")
    r0 = [int(d[2:66], 16) for d in mine.data]
    r1 = [int(d[66:130], 16) for d in mine.data]
    base, quote = (r0, r1) if meta["base_is_token0"] else (r1, r0)
    scale = 10.0 ** (meta["base_decimals"] - meta["quote_decimals"])
    out = pd.DataFrame({"block": swaps.block.to_numpy(), "log_index": swaps.log_index.to_numpy(),
                        "base_reserve_raw": [str(v) for v in base],
                        "quote_reserve_raw": [str(v) for v in quote],
                        "pool_price": [q / b * scale if b and q else np.nan
                                       for q, b in zip(quote, base)]}, columns=COLUMNS)
    if out.pool_price.isna().any():
        raise ValueError(f"{pool.key}: {int(out.pool_price.isna().sum())} swaps leave an "
                         "empty reserve")
    path = Path(data(f"pool_price_{pool.key}.csv.gz"))
    write_csv_atomic(out, path)
    gap = swaps.log_index.to_numpy() - mine.log_index.to_numpy()
    priced = swaps.price.notna().to_numpy()
    ratio = out.pool_price.to_numpy()[priced] / swaps.price.to_numpy()[priced]
    return {"rows": len(out), "file": path.name, "sha256": sha256(str(path)),
            "swap_file_sha256": sha256(swap_file),
            "logs_between_sync_and_swap_max": int(gap.max()) - 1,
            "pool_over_execution_price": {q: float(np.quantile(ratio, float(q)))
                                          for q in ("0.05", "0.5", "0.95")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--pool", action="append", choices=sorted(BY_KEY),
                    help="pool key, repeatable; default every pool with a swap file")
    args = ap.parse_args()
    keys = args.pool or [p.key for p in POOLS if os.path.exists(data(f"swaps_{p.key}.csv.gz"))]
    manifest_path = Path(data("pool_price_manifest.json"))
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key in keys:
        manifest[key] = build(BY_KEY[key])
        write_json_atomic(manifest, manifest_path)
        m = manifest[key]
        q = m["pool_over_execution_price"]
        print(f"{key:18s} {m['rows']:9,} swaps  pool/execution price: median {q['0.5']:.4f}, "
              f"5%-95% {q['0.05']:.4f}-{q['0.95']:.4f}  "
              f"(at most {m['logs_between_sync_and_swap_max']} logs between Sync and Swap)",
              flush=True)


if __name__ == "__main__":
    main()
