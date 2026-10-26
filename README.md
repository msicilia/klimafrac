# klimafrac

Code for *Thin Trading and the Singularity Spectrum: Evidence from Tokenized
Carbon Markets*.

Every price analysed is reconstructed from the chain: one row per Swap event
of one pool, ordered as the chain executed them, and priced from the pool's
own reserves after the swap. Nothing in the analysis depends on a price
aggregator; the aggregator's series appear only as the object of a
data-quality comparison.

## What is here

```
src/klimafrac/
  trades.py       loading, cleaning, the estimation sample, returns, windows,
                  calendar grids -- one implementation, used by every analysis
  mfdfa.py        the estimator, with an exact rule for flat segments
  synthetic.py    fractional Brownian motion and the multifractal random walk
  stats.py        the permutation test (exact when enumerable)
  results.py      where tables and the numbers the paper cites are written
  plotstyle.py    one palette and axis treatment for every figure
  paths.py        data and figure locations, and the two configuration
                  switches (price definition, top of the q range)

  fetch/          data acquisition, all of it resumable and pinned
    pools.py          the pools, their pairs, roles and the pinned block range
    rpc.py            JSON-RPC access that raises rather than return partial data
    chain.py          block lookup, calldata, atomic writes
    swaps.py          swap-level history from each pool's own Swap, Sync, Mint
                      and Burn logs, reconciled along the reserve path
    pool_price.py     the pool's reserve ratio after every swap
    audit_swaps.py    independent check of every swap file against the chain
    thin_candidates.py  census and rule that select the conventional benchmark
                      pools (as thin as the carbon pools, and busier than them)
    token_size.py     each asset's market-capitalisation rank, from outside the chain
    market_existence.py  factory census: which pairs ever existed
    reserves.py, depth.py  reserves and depth over time
    defillama.py      vendor snapshots, stored verbatim with a manifest
    daily_prices.py, hourly_prices.py, benchmark.py  vendor price series
    inventory.py, treasury.py, burn.py  pool inventory (not used in the paper)

  analysis/
    market.py       Table 1, volumes, cleaning shares, threshold checks
    calendar.py     daily and hourly tables, aggregator staleness
    staleness.py    the bias of the estimator under inactivity, and the
                    bias, dispersion and RMSE of the positive-side width
    tiers.py        the comparison within the quality ladder, robustness
    support.py      window coverage in staleness
    power.py        size of the overlapping-window inference and power of
                    the design used
    asset_class.py  carbon against the deep conventional pools, and the
                    construction control
    thin_benchmark.py  carbon against conventional tokens as thin as the
                    carbon pools, thin against busy, size against idleness
    thin_robustness.py  the benchmark comparison under 60 variants of
                    cleaning, endpoint, window, scales, detrending and q range
    scaling.py      log-log fit diagnostics per window and q
    size.py         width against asset size and pool idleness
    verify_paper.py compares the manuscript with data/results/
    venues.py, inventory.py  not used in the paper
```

## Setup

```bash
uv sync
cp .env.example .env     # then put a free Alchemy key in it
```

The key is needed only to fetch. Once `data/` is populated the analysis runs
offline.

## Reproducing

To reproduce the paper from nothing: clone, `uv sync`, put a free Alchemy key
in `.env`, then run the four targets below or `make all`.

```bash
make fetch      # on-chain and vendor data: hours, resumable, needs the key
make audit      # every swap file against receipts and complete block ranges
make analysis   # tables, cited numbers (data/results/) and figures
make verify     # the manuscript against data/results/
```

`make all` runs the four in order. Each fetch stage writes completed block
chunks atomically under `data/raw/` and skips what it already has, so
re-running after an interruption resumes where it stopped. The free Alchemy
tier is enough; a month's quota may not be, in which case the fetch stops with
an error and continues when quota returns.

## Configuration

Two switches in `src/klimafrac/paths.py`, read from the environment, select
the price definition and the top of the positive q range. The defaults, the
pool price and q up to 3, are the paper's and write to `data/results/` and
`figures/`. Any other combination writes to folders named by both switches:

```bash
KLIMAFRAC_PRICE=execution KLIMAFRAC_QMAX=5 make analysis   # execution price, q up to 5
```

## Data

No data is committed. Everything the paper uses is public and `make fetch`
retrieves it: the Swap, Sync, Mint and Burn logs of every pool from the
Polygon and Base chains, through a free Alchemy key; the aggregated price
series from the DefiLlama API; and the market-capitalisation ranking from
CoinMarketCap's historical snapshots. The on-chain fetch is pinned to fixed
block ranges, so a run today retrieves exactly the swaps the paper analysed,
and it writes `data/swaps_manifest.json` with the block range, the counts at
each stage and the SHA-256 of every swap file; `make audit` writes
`data/swaps_audit.json` with the result of the independent check against the
chain. The aggregator series are the one input the vendor can revise, so a
refetch is stored as a dated snapshot rather than overwritten.

The window ends at the last block before 2026-07-26 00:00 UTC: Polygon
90,879,334 and Base 49,117,326. Every estimate uses the estimation sample,
which ends at 2026-05-27 00:00 UTC, immediately before the liquidation of 27
May 2026.

## How the swap history is built

Every log a pair contract emitted over the window is read with `eth_getLogs`
in ranges sized to the node's response cap, and stored in block chunks. Block
times come from the pair's ERC-20 transfers (`alchemy_getAssetTransfers`
returns a timestamp per transfer). Swap logs are decoded, timed and
reconciled along the reserve path: the pair emits a Sync with the reserves
that result from each Swap, Mint or Burn, so the change in reserves between
consecutive Syncs must equal the amounts of the event between them. Walking
the whole stream proves that no Sync is missing and that every event between
two Syncs is the one the reserves record; reserves that moved between two
Syncs with no event between them would indicate a missing log and are
counted, and are zero in every pool. The transfers cross-check the amounts
of single-swap transactions.

`make audit` then checks the result against the chain by two routes that
share no code with the fetcher: transaction receipts, and every Swap event in
complete short block ranges, half of them placed on the seams between stored
chunks. It compares amounts, ordering and completeness, and exits non-zero on
any discrepancy.

## Three things to know before reading the results

**The negative-q branch is not usable on stale calendar-time series.** Where a
segment contains no trade its detrended fluctuation is exactly zero and its
negative moments are undefined. `mfdfa.py` detects such segments from the
returns and excludes them. Keeping the floating-point residual instead, as
implementations usually do, makes h(q<0) a function of machine precision;
`exclude_flat=False` reproduces that behaviour and is used only to quantify
it. Use the positive-side width on any series with material inactivity, and
check the log-log fits (`analysis/scaling.py`) before trusting the top of the
q range.

**Spectrum width means nothing without a length-matched null.** Finite samples
manufacture apparent multifractality in monofractal series.
`mfdfa.surrogate_null` supplies shuffled and phase-randomised comparisons, and
`analysis/calendar.py` reports both for every width.

**Report the size and power of the design you use.** `analysis/power.py`
simulates both the design used here and a regression on overlapping windows
with a HAC correction, at the group sizes these data have.
