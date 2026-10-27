# klimafrac

Code for *Thin Trading and the Singularity Spectrum: Evidence from Tokenized
Carbon Markets* (Fractal and Fractional, 2026). It fetches every swap of the
pools studied from the chain, rebuilds the price series, and produces every
table, number and figure in the paper.

## Requirements

Python 3.13 and [uv](https://docs.astral.sh/uv/). A free Alchemy key, needed
only to fetch the on-chain data.

## Reproducing the paper

```bash
uv sync
cp .env.example .env      # put the Alchemy key in it
make fetch                # on-chain and vendor data: hours, resumable
make audit                # checks every swap file against the chain
make analysis             # tables, cited numbers (data/results/) and figures/
make verify               # every number in the manuscript against data/results/
```

`make all` runs the four targets in order. A fetch interrupted by an error or
by the API quota resumes where it stopped when run again.

No data is committed. The on-chain fetch is pinned to fixed block ranges, so
it retrieves exactly the swaps the paper analysed; the aggregated price series
(DefiLlama) and the market-capitalisation ranking (CoinMarketCap) come from
their public APIs.

## Configuration

The defaults, the pool price and q up to 3, are the paper's. Two environment
variables select the alternatives and write to their own folders:

```bash
KLIMAFRAC_PRICE=execution KLIMAFRAC_QMAX=5 make analysis
```

## Layout

```
src/klimafrac/
  trades.py, mfdfa.py, synthetic.py, stats.py   cleaning, estimator, simulations, tests
  fetch/                                        data acquisition, resumable and pinned
  analysis/                                     one module per table or comparison;
                                                verify_paper.py checks the manuscript
```
