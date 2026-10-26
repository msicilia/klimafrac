# Reproduce every number and figure in the paper.
#
#   make fetch      on-chain and vendor data (needs ALCHEMY_KEY; hours, resumable)
#   make audit      check every swap file against receipts and block ranges
#   make analysis   all tables, cited numbers (data/results/) and figures
#   make verify     compare the manuscript with data/results/
#   make all        everything above, in order
#
# The paper's configuration is the default: the pool price and q up to 3
# (src/klimafrac/paths.py). Any other configuration is selected with the
# environment, e.g. `KLIMAFRAC_PRICE=execution KLIMAFRAC_QMAX=5 make analysis`
# writes to data/results_execution_q5/ and figures_execution_q5/.
#
# Every fetch step writes completed chunks atomically and resumes where it
# stopped, so re-running `make fetch` after a failure is safe.

PY := uv run python -m klimafrac

# Vendor snapshots are dated, and the fetchers reuse the newest one by default.
# A from-scratch run must ask for its own, so `vendor` passes --refresh: today's
# snapshot is created, and re-running the target resumes it rather than starting
# another. Pass REFRESH= to resume the newest existing snapshot instead.
REFRESH ?= --refresh

.PHONY: all fetch swaps candidates pool-price size-data chain vendor audit \
        analysis verify validate extra paper-figures

all: fetch audit analysis verify

fetch: candidates swaps pool-price size-data chain vendor

# The census of candidate benchmark pools, run before their swaps are fetched;
# `select` refuses to run unless fetch/pools.py lists exactly the pools the
# rule keeps.
candidates:
	$(PY).fetch.thin_candidates --stage pairs
	$(PY).fetch.thin_candidates --stage screen
	$(PY).fetch.thin_candidates --stage daily
	$(PY).fetch.thin_candidates --stage select

# Every pool in fetch/pools.py: carbon, control, thin and active.
swaps:
	$(PY).fetch.swaps
	$(PY).fetch.thin_candidates --stage verify

pool-price:
	$(PY).fetch.pool_price

size-data:
	$(PY).fetch.token_size

chain:
	$(PY).fetch.market_existence
	$(PY).fetch.reserves

vendor:
	$(PY).fetch.daily_prices $(REFRESH)
	$(PY).fetch.hourly_prices $(REFRESH)
	$(PY).fetch.benchmark $(REFRESH)
	$(PY).fetch.depth $(REFRESH)

audit:
	$(PY).fetch.audit_swaps

validate:
	$(PY).mfdfa

# In dependency order: size reads thin_benchmark's token table, and
# thin_robustness checks that its baseline reproduces thin_benchmark.
analysis: validate
	$(PY).analysis.staleness
	$(PY).analysis.market
	$(PY).analysis.calendar
	$(PY).analysis.tiers
	$(PY).analysis.support
	$(PY).analysis.power
	$(PY).analysis.asset_class
	$(PY).analysis.thin_benchmark
	$(PY).analysis.thin_robustness
	$(PY).analysis.scaling
	$(PY).analysis.size
	$(PY).analysis.census
	$(PY).analysis.price_definition
	$(PY).analysis.benchmark_figures
	$(PY).analysis.appendix_tables

verify:
	$(PY).analysis.verify_paper

# Analyses not used in the paper (venues across chains; pool inventory
# and the October 2025 burn). They need `make chain-extra` first.
chain-extra:
	$(PY).fetch.inventory
	$(PY).fetch.treasury
	$(PY).fetch.burn

extra:
	$(PY).analysis.venues
	$(PY).analysis.inventory

paper-figures:
	mkdir -p ../paper/figures && cp figures/fig*.png ../paper/figures/
