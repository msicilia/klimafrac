# klimafrac

Code for *Thin Trading and the Singularity Spectrum: Evidence from Tokenized
Carbon Markets* (A. Ballesteros-Rodríguez, L. González-García and
M.-A. Sicilia).

Every price analysed is reconstructed from the chain: one row per Swap event
of one pool, ordered as the chain executed them. The carbon tokens are
KlimaDAO's pool tokens on Polygon; the conventional benchmarks are WETH and
WBTC pools on the same exchanges, built by the same code. A price aggregator
is used only as the object of a data-quality comparison.

## Layout

```
src/klimafrac/
  trades.py       loading, cleaning, the estimation sample, returns, windows,
                  calendar grids, used by every analysis
  mfdfa.py        the estimator, with an exact rule for flat segments
  synthetic.py    fractional Brownian motion and the multifractal random walk
  stats.py        the permutation test (exact when enumerable)
  results.py      where tables and numbers are written
  plotstyle.py    palette and axis treatment for every figure
  paths.py        data and figure locations (KLIMAFRAC_DATA_DIR to relocate)

  fetch/                data acquisition, resumable and pinned to fixed blocks
    pools.py            the pools, their pairs and the block range
    rpc.py              JSON-RPC access that raises rather than return partial data
    chain.py            block lookup, calldata, atomic writes
    swaps.py            swap-level history from each pool's own events
    audit_swaps.py      independent check of every swap file against the chain
    market_existence.py factory census: which pairs ever existed
    defillama.py        vendor snapshots, stored verbatim with a manifest
    daily_prices.py, hourly_prices.py, benchmark.py

  analysis/
    market.py       Table 1: swaps, volumes, cleaning shares
    calendar.py     daily and hourly estimates by data source (Table 3, Table A1)
    staleness.py    bias of the estimator under inactivity (Table 2)
    tiers.py        comparison within the quality ladder (Table A2)
    support.py      window coverage in staleness
    power.py        size and power of this design and of the customary one
    asset_class.py  carbon against conventional crypto (Table 4, Figure 1)
```

## Setup

```bash
uv sync
cp .env.example .env     # then put a free Alchemy key in it
```

The key is needed only to fetch. Once `data/` is populated the analysis runs
offline.

## Reproducing

```bash
make fetch      # on-chain and vendor data: hours, resumable, needs the key
make audit      # every swap file against receipts and complete block ranges
make analysis   # tables, numbers (data/results/) and figures (figures/)
```

`make all` runs the three in order. Each fetch stage writes completed chunks
atomically under `data/raw/` and skips what it already has, so re-running
after an interruption resumes where it stopped. If the free tier's quota runs
out, the fetch stops with an error and continues when quota returns.

No data is included in this repository. The window ends at the last block
before 2026-07-26 00:00 UTC (Polygon block 90,879,334). Every estimate uses
the estimation sample, which ends at 2026-05-27 00:00 UTC.

## How the swap history is built

For each pool, every log the pair contract emitted over the window is read
with `eth_getLogs`, and block times are taken from the pair's token transfers.
Each Swap event gives one row, with the amounts the pool booked, in execution
order. Two checks make the result verifiable: the reserves reported by
consecutive Sync events must change by exactly the amounts of the event
between them, which proves no event is missing; and for transactions holding
a single swap, the tokens transferred must match the event's amounts.

`make audit` then checks each swap file by routes that share no code with the
fetcher: transaction receipts, every Swap event in complete block ranges, and
block timestamps. It exits non-zero on any discrepancy.

## Notes on the estimator

**The negative-q branch is not usable on stale calendar-time series.** Where a
segment contains no trade its detrended fluctuation is exactly zero and its
negative moments are undefined. `mfdfa.py` detects such segments from the
returns and excludes them; `exclude_flat=False` keeps the floating-point
residual instead, as implementations usually do. Use the positive-side width
on any series with material inactivity.

**Spectrum width needs a length-matched null.** `analysis/calendar.py` sets
every width against shuffled and phase-randomised surrogates of the same
length.

**Report the size and power of the design you use.** `analysis/power.py`
simulates both the design used here and the overlapping-window regression
customary in this literature, at the group sizes these data have.

## License

MIT; see [LICENSE](LICENSE).
