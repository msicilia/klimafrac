# Reproduce every number and figure in the paper.
#
#   make fetch      on-chain and vendor data (needs ALCHEMY_KEY; hours, resumable)
#   make audit      check every swap file against receipts and block ranges
#   make analysis   all tables, numbers (data/results/) and figures
#   make all        everything above, in order
#
# Every fetch step writes completed chunks atomically and resumes where it
# stopped, so re-running `make fetch` after a failure is safe.

PY := uv run python -m klimafrac

# Vendor snapshots are dated, and the fetchers reuse the newest one by default.
# A from-scratch run must ask for its own, so `vendor` passes --refresh: today's
# snapshot is created, and re-running the target resumes it rather than starting
# another. Pass REFRESH= to resume the newest existing snapshot instead.
REFRESH ?= --refresh

POOLS := polygon_BCT polygon_NCT polygon_MCO2 polygon_KLIMA polygon_UBO polygon_NBO \
         polygon_WETH_S polygon_WETH_Q polygon_WBTC_Q

.PHONY: all fetch swaps chain vendor audit analysis validate

all: fetch audit analysis

fetch: swaps chain vendor

swaps:
	@for pool in $(POOLS); do $(PY).fetch.swaps --pool $$pool || exit 1; done

chain:
	$(PY).fetch.market_existence

vendor:
	$(PY).fetch.daily_prices $(REFRESH)
	$(PY).fetch.hourly_prices $(REFRESH)
	$(PY).fetch.benchmark $(REFRESH)

audit:
	$(PY).fetch.audit_swaps

validate:
	$(PY).mfdfa

analysis: validate
	$(PY).analysis.staleness
	$(PY).analysis.market
	$(PY).analysis.calendar
	$(PY).analysis.tiers
	$(PY).analysis.support
	$(PY).analysis.power
	$(PY).analysis.asset_class
