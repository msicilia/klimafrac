"""
Numbers that describe the data rather than analyse it: the census behind the
benchmark pools, the two prices, and one cleaning rule. Written so that the
manuscript's data section can be verified like every other.

  census.universe              USDC.e pairs the two factories ever created
  census.old_enough            of those, created by the first anchor date
  census.screened              holding at least MIN_QUOTE at the three anchors
  census.thin, census.active   pools the rule and its mirror keep
  census.unbuildable           pools kept by the rule whose history could not
                               be built (fetch/thin_candidates.UNBUILDABLE)
  census.proxy_abs_error_max   largest difference, over the pools whose trades
                               are known, between the idle-share proxy of the
                               census and the hourly no-trade fraction
  census.pool_over_execution.<pool>.<q>
                               quantiles of pool price over execution price
  census.same_sign_swaps       swaps in which the pool received or paid out
                               both tokens, dropped by trades.clean

Outputs: data/results/census.json

Usage:
    uv run python -m klimafrac.analysis.census
"""

import json

import numpy as np
import pandas as pd

from klimafrac import trades
from klimafrac.fetch.pools import POOLS
from klimafrac.fetch.thin_candidates import UNBUILDABLE
from klimafrac.paths import data
from klimafrac.results import numbers, read_table


def main() -> None:
    pairs = pd.read_csv(data("thin_candidates_pairs.csv"))
    screen = pd.read_csv(data("thin_candidates_screen.csv"))
    measures = pd.read_csv(data("thin_candidates_measures.csv"))
    tokens = read_table("thin_benchmark_tokens")
    out = {"census.universe": len(pairs), "census.old_enough": len(screen),
           "census.screened": int(screen.passes.sum()),
           "census.thin": int(measures.selected.sum()),
           "census.active": int(measures.selected_active.sum()),
           "census.unbuildable": len(UNBUILDABLE)}

    # The proxy against the exact hourly no-trade fraction, pool by pool.
    by_pair = measures.set_index("pair").idle_1h
    pair_of = {p.symbol: p.pair for p in POOLS}
    errors = {t: float(by_pair[pair_of[t]] - r) for t, r in zip(tokens.token, tokens.no_trade)
              if pair_of.get(t) in by_pair.index}
    out["census.proxy_abs_error_max"] = max(abs(e) for e in errors.values())
    out["census.proxy_abs_error_mean"] = float(np.mean([abs(e) for e in errors.values()]))
    out["census.proxy_pools"] = len(errors)

    manifest = json.loads(open(data("pool_price_manifest.json")).read())
    for key, m in manifest.items():
        for q, v in m["pool_over_execution_price"].items():
            out[f"census.pool_over_execution.{key}.{q}"] = v

    same = 0
    for p in POOLS:
        try:
            df = trades.load(p.key)
        except (FileNotFoundError, SystemExit):
            continue
        same += int((np.sign(df.base_amt) == np.sign(df.quote_amt)).sum())
    out["census.same_sign_swaps"] = same
    print("wrote", numbers("census", out))
    print({k: v for k, v in out.items() if "pool_over" not in k})


if __name__ == "__main__":
    main()
