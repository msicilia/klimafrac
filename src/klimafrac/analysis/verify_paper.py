"""
Check every quantitative claim in paper/paper_mdpi.tex against the pipeline's
own output.

Each analysis module writes the numbers the manuscript cites to
data/results/<module>.json under stable keys (klimafrac.results.numbers). This
module holds the other half of that contract: a table of every number the
manuscript asserts, where it appears, which result key it has to equal, and
how close is close enough. Nothing here recomputes anything and nothing is
read from a figure or a printed log, so a disagreement is a disagreement
between the text and the pipeline, and cannot be a second implementation
drifting from the first.

The failure this is built to avoid is silent agreement. A claim whose result
file or key is absent is reported MISSING rather than skipped, so a manuscript
number that no analysis produces shows up as a gap in the pipeline instead of
as a pass, and a renamed key fails loudly rather than quietly. Every row
records the value the manuscript currently asserts, so after a refetch the
list of disagreements is the edit list for the next draft.

Comparison kinds:
  int          exact integer equality
  abs, rel     absolute or relative tolerance on a number
  p            a p-value: within tolerance, and on the same side of 0.05
  gt lt ge le  direction only, against the stated threshold
  between      the computed value lies inside the stated interval
  pair         a stored [lo, hi] matches the stated interval
  bool, str    identity, or exact string; prefix for timestamps and dates
  set          the same members, order ignored

Result files hold their numbers under dotted keys, some as literal flat keys
and some nested, so a key is resolved by trying the longest literal match at
each level before descending.

Usage:
    uv run python -m klimafrac.analysis.verify_paper
    uv run python -m klimafrac.analysis.verify_paper --section 4.3
    uv run python -m klimafrac.analysis.verify_paper --json
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass

from klimafrac.results import RESULTS_DIR, numbers

ALPHA = 0.05
DEFAULT_TOL = {"abs": 5e-4, "rel": 5e-3, "p": 5e-4, "pair": 5e-4}
DIRECTIONS = {"gt": lambda x, t: x > t, "ge": lambda x, t: x >= t,
              "lt": lambda x, t: x < t, "le": lambda x, t: x <= t}

# Every row is (description, "<result file>:<key>", value asserted by the
# manuscript, comparison kind, tolerance). The tolerance defaults to half a
# unit of the last digit the manuscript quotes for that kind. Sections run in
# manuscript order; a number stated twice is declared once, at the place that
# states it most precisely.
CLAIMS: list[tuple[str, list[tuple]]] = [

    ("3.1 Data -- Table 1 (tab:tiers)", [
        ("BCT raw swaps", "market:tier_BCT_raw_swaps", 476_404, "int"),
        ("BCT usable swaps", "market:tier_BCT_usable_swaps", 399_691, "int"),
        ("BCT usable USD volume", "market:tier_BCT_usd_volume", 1.8967e9, "abs", 5e4),
        ("BCT has a stablecoin pair", "market:tier_BCT_stable_pair", True, "bool"),
        ("NCT raw swaps", "market:tier_NCT_raw_swaps", 50_760, "int"),
        ("NCT usable swaps", "market:tier_NCT_usable_swaps", 40_938, "int"),
        ("NCT usable USD volume", "market:tier_NCT_usd_volume", 18.6e6, "abs", 5e4),
        ("NCT has a stablecoin pair", "market:tier_NCT_stable_pair", True, "bool"),
        ("MCO2 raw swaps", "market:tier_MCO2_raw_swaps", 61_477, "int"),
        ("MCO2 usable swaps", "market:tier_MCO2_usable_swaps", 60_708, "int"),
        ("MCO2 usable USD volume", "market:tier_MCO2_usd_volume", 20.6e6, "abs", 5e4),
        ("MCO2 has a stablecoin pair", "market:tier_MCO2_stable_pair", True, "bool"),
        ("KLIMA raw swaps", "market:tier_KLIMA_raw_swaps", 630_175, "int"),
        ("KLIMA usable swaps", "market:tier_KLIMA_usable_swaps", 594_369, "int"),
        ("KLIMA usable USD volume", "market:tier_KLIMA_usd_volume", 293.5e6, "abs", 5e4),
        ("KLIMA has a stablecoin pair", "market:tier_KLIMA_stable_pair", True, "bool"),
        ("UBO raw swaps", "market:tier_UBO_raw_swaps", 219, "int"),
        ("UBO usable swaps", "market:tier_UBO_usable_swaps", 156, "int"),
        ("UBO usable USD volume", "market:tier_UBO_usd_volume", 7_000.0, "abs", 500),
        ("UBO has no stablecoin pair", "market:tier_UBO_stable_pair", False, "bool"),
        ("NBO raw swaps", "market:tier_NBO_raw_swaps", 314, "int"),
        ("NBO usable swaps", "market:tier_NBO_usable_swaps", 304, "int"),
        ("NBO usable USD volume", "market:tier_NBO_usd_volume", 6_000.0, "abs", 500),
        ("NBO has no stablecoin pair", "market:tier_NBO_stable_pair", False, "bool"),
        ("data window opens", "market:tier_BCT_first_trade", "2021-10-18", "prefix"),
        ("data window closes", "market:tier_KLIMA_last_trade", "2026-07-25", "prefix"),
    ]),

    ("3.2 Reconstruction -- construction, cleaning and sample", [
        ("cleaning removes a sixth to a fifth, BCT", "market:cleaning_BCT_removed_share", (0.15, 0.20), "between"),
        ("cleaning removes a sixth to a fifth, NCT", "market:cleaning_NCT_removed_share", (0.15, 0.20), "between"),
        ("cleaning removes almost none of MCO2", "market:cleaning_MCO2_removed_share", 0.02, "lt"),
        ("BCT usable volume, data window", "market:tier_BCT_usd_volume", 1.90e9, "abs", 5e6),
        ("UBO and NBO lifetime volume", "market:UBO_NBO_usd_volume", 13_000.0, "abs", 500),
        ("estimation sample retains nearly all, BCT", "market:sample_BCT_retention", 0.97, "ge"),
        ("estimation sample retains nearly all, MCO2", "market:sample_MCO2_retention", 0.97, "ge"),
        ("estimation sample retains nearly all, NCT", "market:sample_NCT_retention", 0.97, "ge"),
        ("estimation sample ends", "calendar:design.sample_end_exclusive",
         "2026-05-27", "prefix"),
        ("aggregator staleness, BCT hourly", "calendar:staleness.BCT.vendor_zero_pct",
         3.8, "abs", 5e-2),
        ("true no-trade share, BCT hourly", "calendar:staleness.BCT.matched_no_trade_pct",
         43.5, "abs", 5e-2),
        ("Fig 1: understatement, smallest", "calendar:staleness.range.understatement_ratio.min",
         1.0, "abs", 0.5),
        ("Fig 1: understatement, largest", "calendar:staleness.range.understatement_ratio.max",
         11.3, "abs", 0.5),
        ("fifteen same-sign swaps discarded", "census:census.same_sign_swaps", 15, "int"),
        ("census: pairs the factories ever created", "census:census.universe", 6_875, "int"),
        ("census: pairs old enough", "census:census.old_enough", 5_016, "int"),
        ("census: pairs holding 50 USDC.e at the three anchors", "census:census.screened", 302, "int"),
        ("census: thin tokens", "census:census.thin", 36, "int"),
        ("census: one unbuildable", "census:census.unbuildable", 1, "int"),
        ("census: active tokens", "census:census.active", 6, "int"),
        ("idle-share proxy: about four points off on average",
         "census:census.proxy_abs_error_mean", 0.04, "abs", 0.01),
        ("45 conventional tokens", "size:size.tokens", 45, "int"),
        ("17 of the 45 unlisted in October 2021", "size:size.unlisted", 17, "int"),
        ("carbon tokens near rank 3,000 in February 2022, BCT",
         "size:size.carbon.BCT.check_rank", (2_500, 3_500), "between"),
        ("carbon tokens near rank 3,000 in February 2022, KLIMA",
         "size:size.carbon.KLIMA.check_rank", (2_500, 3_500), "between"),
        ("carbon tokens near rank 3,000 in February 2022, MCO2",
         "size:size.carbon.MCO2.check_rank", (2_500, 3_500), "between"),
    ]),

    ("3.3 Estimation and inference", [
        ("positive-q scale divisor, s_max = N/10", "calendar:design.positive_s_max_div",
         10, "int"),
        ("full-range daily divisor, s_max = N/4", "calendar:design.full_s_max_div_daily",
         4, "int"),
        ("fBm validation recovers H = 0.3 within 0.02", "mfdfa_validation:h2_at_H_0.3", (0.28, 0.32), "between"),
        ("fBm validation recovers H = 0.5 within 0.02", "mfdfa_validation:h2_at_H_0.5", (0.48, 0.52), "between"),
        ("fBm validation recovers H = 0.7 within 0.02", "mfdfa_validation:h2_at_H_0.7", (0.68, 0.72), "between"),
        ("shuffled surrogates", "calendar:design.surrogates_per_kind", 1_000, "int"),
        ("rolling window length, trades", "tiers:tiers.window", 4_000, "int"),
        ("permutation relabellings", "tiers:tiers.draws", 20_000, "int"),
        ("hourly no-trade: a fifth to three quarters, least stale",
         "calendar:hourly.reconstructed.range.no_trade_pct.min", (20.0, 25.0), "between"),
        ("hourly no-trade: a fifth to three quarters, most stale",
         "calendar:hourly.reconstructed.range.no_trade_pct.max", (70.0, 75.0), "between"),
    ]),

    ("Appendix A Daily estimates -- Table A1 (tab:daily)", [
        ("aggregated BCT zero %", "calendar:daily.vendor.BCT.zero_return_pct", 0.4, "abs", 5e-2),
        ("aggregated BCT width", "calendar:daily.vendor.BCT.width", 0.974, "abs"),
        ("aggregated NCT zero %", "calendar:daily.vendor.NCT.zero_return_pct", 4.2, "abs", 5e-2),
        ("aggregated NCT width", "calendar:daily.vendor.NCT.width", 1.444, "abs"),
        ("aggregated MCO2 zero %", "calendar:daily.vendor.MCO2.zero_return_pct",
         2.0, "abs", 5e-2),
        ("aggregated MCO2 width", "calendar:daily.vendor.MCO2.width", 0.802, "abs"),
        ("aggregated KLIMA zero %", "calendar:daily.vendor.KLIMA.zero_return_pct",
         3.0, "abs", 5e-2),
        ("aggregated KLIMA width", "calendar:daily.vendor.KLIMA.width", 0.724, "abs"),
        ("reconstructed BCT zero %", "calendar:daily.reconstructed.BCT.zero_return_pct",
         1.6, "abs", 5e-2),
        ("reconstructed BCT width", "calendar:daily.reconstructed.BCT.width", 1.712, "abs"),
        ("reconstructed NCT zero %", "calendar:daily.reconstructed.NCT.zero_return_pct",
         1.7, "abs", 5e-2),
        ("reconstructed NCT width", "calendar:daily.reconstructed.NCT.width", 2.073, "abs"),
        ("reconstructed MCO2 zero %", "calendar:daily.reconstructed.MCO2.zero_return_pct",
         5.5, "abs", 5e-2),
        ("reconstructed MCO2 width", "calendar:daily.reconstructed.MCO2.width", 0.808, "abs"),
        ("reconstructed KLIMA zero %", "calendar:daily.reconstructed.KLIMA.zero_return_pct",
         1.4, "abs", 5e-2),
        ("reconstructed KLIMA width", "calendar:daily.reconstructed.KLIMA.width", 0.885, "abs"),
        ("aggregated BCT width+", "calendar:daily.vendor.BCT.width_pos", 0.288, "abs"),
        ("aggregated BCT width+ shuffled p95",
         "calendar:daily.vendor.BCT.width_pos_shuffled_p95", 0.374, "abs"),
        ("aggregated NCT width+", "calendar:daily.vendor.NCT.width_pos", 0.913, "abs"),
        ("aggregated NCT width+ shuffled p95",
         "calendar:daily.vendor.NCT.width_pos_shuffled_p95", 0.759, "abs"),
        ("aggregated MCO2 width+", "calendar:daily.vendor.MCO2.width_pos", 0.357, "abs"),
        ("aggregated MCO2 width+ shuffled p95",
         "calendar:daily.vendor.MCO2.width_pos_shuffled_p95", 0.331, "abs"),
        ("aggregated KLIMA width+", "calendar:daily.vendor.KLIMA.width_pos", 0.234, "abs"),
        ("aggregated KLIMA width+ shuffled p95",
         "calendar:daily.vendor.KLIMA.width_pos_shuffled_p95", 0.206, "abs"),
        ("reconstructed BCT width+ shuffled p95",
         "calendar:daily.reconstructed.BCT.width_pos_shuffled_p95", 0.387, "abs"),
        ("reconstructed NCT width+ shuffled p95",
         "calendar:daily.reconstructed.NCT.width_pos_shuffled_p95", 0.525, "abs"),
        ("reconstructed MCO2 width+ shuffled p95",
         "calendar:daily.reconstructed.MCO2.width_pos_shuffled_p95", 0.418, "abs"),
        ("reconstructed KLIMA width+ shuffled p95",
         "calendar:daily.reconstructed.KLIMA.width_pos_shuffled_p95", 0.160, "abs"),
    ]),

    ("4.1 Calendar-time estimates -- prose", [
        ("NCT is 1.48x BCT, aggregated, full range", "calendar:daily.vendor.NCT_over_BCT_width",
         1.48, "abs", 5e-3),
        ("positive side: NCT, MCO2 and KLIMA pass the shuffled null, aggregated",
         "calendar:daily.vendor.width_pos_fails_shuffled_null", ["BCT"], "set"),
        ("positive side: only KLIMA passes the shuffled null, reconstructed",
         "calendar:daily.reconstructed.width_pos_fails_shuffled_null", ["BCT", "NCT", "MCO2"], "set"),
        ("aggregated NCT more than three times BCT, positive side",
         "calendar:daily.vendor.NCT.width_pos", 3 * 0.288, "gt"),
        ("own trades: NCT a quarter wider than BCT, positive side",
         "calendar:daily.reconstructed.NCT.width_pos", (1.20 * 0.345, 1.30 * 0.345), "between"),
        ("price: deep pools by execution, 0.08 to 0.10, WETH_S",
         "price_definition:price_definition.WETH_S.execution", (0.075, 0.105), "between"),
        ("price: deep pools by execution, 0.08 to 0.10, WETH_Q",
         "price_definition:price_definition.WETH_Q.execution", (0.075, 0.105), "between"),
        ("price: deep pools by execution, 0.08 to 0.10, WBTC_Q",
         "price_definition:price_definition.WBTC_Q.execution", (0.075, 0.105), "between"),
        ("price: deep pools from reserves, 0.15, WETH_S",
         "price_definition:price_definition.WETH_S.pool", 0.15, "abs", 5e-3),
        ("price: deep pools from reserves, 0.15, WETH_Q",
         "price_definition:price_definition.WETH_Q.pool", 0.15, "abs", 5e-3),
        ("price: deep pools from reserves, 0.15, WBTC_Q",
         "price_definition:price_definition.WBTC_Q.pool", 0.15, "abs", 5e-3),
        ("price: aggregated Bitcoin 0.13", "price_definition:price_definition.aggregated.BTC", 0.13, "abs", 5e-3),
        ("price: aggregated Ether 0.15", "price_definition:price_definition.aggregated.ETH", 0.15, "abs", 5e-3),
        ("price: carbon pools differ by at most 0.04",
         "price_definition:price_definition.carbon.abs_difference_max", 0.045, "lt"),
    ]),

    ("3.5 Scaling diagnostics -- Table 4 (tab:scaling)", [
        ("poor fits at q = 5: deep, a tenth", "scaling:scaling.hourly.deep.q5.poor_share", 0.10, "abs", 5e-3),
        ("poor fits at q = 5: thin, about half", "scaling:scaling.hourly.thin.q5.poor_share", 0.46, "abs", 5e-3),
        ("poor fits at q = 5: active, about half", "scaling:scaling.hourly.active.q5.poor_share", 0.47, "abs", 5e-3),
        ("poor fits at q = 5: carbon, two thirds", "scaling:scaling.hourly.carbon.q5.poor_share", 0.68, "abs", 5e-3),
        ("poor fits at q = 5 exceed half, BCT", "scaling:scaling.transaction.BCT.q5.poor_share", 0.5, "gt"),
        ("poor fits at q = 5 exceed half, NCT", "scaling:scaling.transaction.NCT.q5.poor_share", 0.5, "gt"),
        ("poor fits at q = 5 exceed half, MCO2", "scaling:scaling.transaction.MCO2.q5.poor_share", 0.5, "gt"),
        ("width against fit at q = 5, Spearman", "scaling:scaling.hourly.width_vs_r2_at_q5.spearman", -0.59, "abs", 5e-3),
        ("poor fits at q = 3: deep, zero", "scaling:scaling.hourly.deep.q3.poor_share", 0.0, "abs", 1e-9),
        ("poor fits at q = 3: thin, about an eighth", "scaling:scaling.hourly.thin.q3.poor_share", 0.13, "abs", 5e-3),
        ("poor fits at q = 3: carbon, a sixth", "scaling:scaling.hourly.carbon.q3.poor_share", 0.16, "abs", 5e-3),
        ("poor fits at q = 3: active", "scaling:scaling.hourly.active.q3.poor_share", 0.07, "abs", 5e-3),
        ("poor fits at q = 2: carbon", "scaling:scaling.hourly.carbon.q2.poor_share", 0.03, "abs", 5e-3),
        ("poor fits at q = 2: MCO2", "scaling:scaling.transaction.MCO2.q2.poor_share", 0.21, "abs", 5e-3),
        ("poor fits at q = 3: MCO2", "scaling:scaling.transaction.MCO2.q3.poor_share", 0.64, "abs", 5e-3),
        ("poor fits at q = 3: NCT", "scaling:scaling.transaction.NCT.q3.poor_share", 0.22, "abs", 5e-3),
        ("poor fits at q = 3: BCT", "scaling:scaling.transaction.BCT.q3.poor_share", 0.12, "abs", 5e-3),
        ("r2 median at q = 3: carbon", "scaling:scaling.hourly.carbon.q3.r2_median", 0.979, "abs"),
        ("r2 median at q = 3: thin", "scaling:scaling.hourly.thin.q3.r2_median", 0.986, "abs"),
        ("r2 median at q = 3: deep", "scaling:scaling.hourly.deep.q3.r2_median", 0.996, "abs"),
        ("r2 median at q = 3: MCO2", "scaling:scaling.transaction.MCO2.q3.r2_median", 0.921, "abs"),
        ("crossover at q = 0.5: carbon", "scaling:scaling.hourly.carbon.q0.5.crossover_median", 0.21, "abs", 5e-3),
        ("crossover at q = 0.5: thin", "scaling:scaling.hourly.thin.q0.5.crossover_median", 0.15, "abs", 5e-3),
        ("crossover at q = 0.5: deep", "scaling:scaling.hourly.deep.q0.5.crossover_median", 0.04, "abs", 5e-3),
        ("crossover at q = 0.5: BCT", "scaling:scaling.transaction.BCT.q0.5.crossover_median", 0.23, "abs", 5e-3),
        ("crossover at q = 0.5: NCT", "scaling:scaling.transaction.NCT.q0.5.crossover_median", 0.24, "abs", 5e-3),
        ("windows: thin", "scaling:scaling.hourly.thin.q3.windows", 355, "int"),
        ("windows: carbon", "scaling:scaling.hourly.carbon.q3.windows", 38, "int"),
    ]),

    ("4.1 Hourly sampling", [
        ("hourly full-range widths all exceed every daily one",
         "calendar:hourly.reconstructed.range.width.min", 1.19, "gt"),
        ("hourly h(-5) above one for every token", "calendar:hourly.reconstructed.range.h_neg5.min",
         1.0, "gt"),
        ("unstaled MRW h(-5) is 0.81", "staleness:bias_table_p000_h_neg5", 0.81, "abs", 5e-3),
        ("MRW h(-5) still below one at half the intervals empty",
         "staleness:bias_table_p050_h_neg5", 1.0, "lt"),
        ("MRW h(-5) above one once most intervals are empty",
         "staleness:bias_table_p070_h_neg5", 1.0, "gt"),
        ("hourly full-range width above the token's daily value, BCT",
         "calendar:hourly.reconstructed.BCT.width", 1.712, "gt"),
        ("hourly full-range width above the token's daily value, NCT",
         "calendar:hourly.reconstructed.NCT.width", 2.073, "gt"),
        ("hourly full-range width above the token's daily value, MCO2",
         "calendar:hourly.reconstructed.MCO2.width", 0.808, "gt"),
        ("hourly full-range width above the token's daily value, KLIMA",
         "calendar:hourly.reconstructed.KLIMA.width", 0.885, "gt"),
        ("spread of the hourly full-range widths about 0.9",
         "calendar:hourly.reconstructed.range.width.max", (2.35, 2.55), "between"),
        ("spread of the hourly full-range widths about 0.9, smallest",
         "calendar:hourly.reconstructed.range.width.min", (1.50, 1.65), "between"),
        ("restricted hourly width, smallest",
         "calendar:hourly.reconstructed.range.width_pos.min", 0.236, "abs"),
        ("restricted hourly width, largest",
         "calendar:hourly.reconstructed.range.width_pos.max", 0.466, "abs"),
    ]),

    ("4.1 Positive-q estimates -- Table 5 (tab:hourly)", [
        ("BCT daily h(2)", "calendar:daily.reconstructed.BCT.h2_pos", 0.522, "abs"),
        ("BCT daily width+", "calendar:daily.reconstructed.BCT.width_pos", 0.345, "abs"),
        ("BCT hourly N", "calendar:hourly.reconstructed.BCT.n", 40_348, "int"),
        ("BCT hourly no-trade %", "calendar:hourly.reconstructed.BCT.no_trade_pct",
         46.1, "abs", 5e-2),
        ("BCT hourly h(2)", "calendar:hourly.reconstructed.BCT.h2_pos", 0.507, "abs"),
        ("BCT hourly width+", "calendar:hourly.reconstructed.BCT.width_pos", 0.426, "abs"),
        ("NCT daily h(2)", "calendar:daily.reconstructed.NCT.h2_pos", 0.505, "abs"),
        ("NCT daily width+", "calendar:daily.reconstructed.NCT.width_pos", 0.427, "abs"),
        ("NCT hourly N", "calendar:hourly.reconstructed.NCT.n", 37_590, "int"),
        ("NCT hourly no-trade %", "calendar:hourly.reconstructed.NCT.no_trade_pct",
         64.0, "abs", 5e-2),
        ("NCT hourly h(2)", "calendar:hourly.reconstructed.NCT.h2_pos", 0.514, "abs"),
        ("NCT hourly width+", "calendar:hourly.reconstructed.NCT.width_pos", 0.466, "abs"),
        ("MCO2 daily h(2)", "calendar:daily.reconstructed.MCO2.h2_pos", 0.389, "abs"),
        ("MCO2 daily width+", "calendar:daily.reconstructed.MCO2.width_pos", 0.383, "abs"),
        ("MCO2 hourly N", "calendar:hourly.reconstructed.MCO2.n", 38_452, "int"),
        ("MCO2 hourly no-trade %", "calendar:hourly.reconstructed.MCO2.no_trade_pct",
         73.1, "abs", 5e-2),
        ("MCO2 hourly h(2)", "calendar:hourly.reconstructed.MCO2.h2_pos", 0.437, "abs"),
        ("MCO2 hourly width+", "calendar:hourly.reconstructed.MCO2.width_pos", 0.295, "abs"),
        ("KLIMA daily h(2)", "calendar:daily.reconstructed.KLIMA.h2_pos", 0.585, "abs"),
        ("KLIMA daily width+", "calendar:daily.reconstructed.KLIMA.width_pos", 0.213, "abs"),
        ("KLIMA hourly N", "calendar:hourly.reconstructed.KLIMA.n", 40_346, "int"),
        ("KLIMA hourly no-trade %", "calendar:hourly.reconstructed.KLIMA.no_trade_pct",
         22.6, "abs", 5e-2),
        ("KLIMA hourly h(2)", "calendar:hourly.reconstructed.KLIMA.h2_pos", 0.523, "abs"),
        ("KLIMA hourly width+", "calendar:hourly.reconstructed.KLIMA.width_pos", 0.236, "abs"),
    ]),

    ("4.1 Common support on the calendar clock", [
        ("BCT rolling windows", "support:support.BCT.windows", 10, "int"),
        ("BCT windows below the threshold", "support:support.BCT.below_threshold", 2, "int"),
        ("NCT rolling windows", "support:support.NCT.windows", 9, "int"),
        ("NCT windows below the threshold", "support:support.NCT.below_threshold", 0, "int"),
        ("MCO2 rolling windows", "support:support.MCO2.windows", 9, "int"),
        ("MCO2 windows below the threshold", "support:support.MCO2.below_threshold", 0, "int"),
        ("support threshold", "support:support.threshold", 0.25, "abs", 5e-3),
    ]),

    ("3.4 Staleness bias -- Table 2 (tab:bias)", [
        ("caption: hourly carbon series sit at p = 0.23 at the least",
         "calendar:hourly.reconstructed.range.no_trade_pct.min", 23, "abs", 0.5),
        ("caption: hourly carbon series sit at p = 0.73 at the most",
         "calendar:hourly.reconstructed.range.no_trade_pct.max", 73, "abs", 0.5),
        ("p = 0.0 full-range width", "staleness:bias_table_p000_width", 0.706, "abs"),
        ("p = 0.0 inflation", "staleness:bias_table_p000_inflation", 1.00, "abs", 5e-3),
        ("p = 0.0 width+", "staleness:bias_table_p000_width_pos", 0.174, "abs"),
        ("p = 0.0 h(-5)", "staleness:bias_table_p000_h_neg5", 0.814, "abs"),
        ("p = 0.2 full-range width", "staleness:bias_table_p020_width", 0.710, "abs"),
        ("p = 0.2 inflation", "staleness:bias_table_p020_inflation", 1.01, "abs", 5e-3),
        ("p = 0.2 width+", "staleness:bias_table_p020_width_pos", 0.183, "abs"),
        ("p = 0.2 h(-5)", "staleness:bias_table_p020_h_neg5", 0.816, "abs"),
        ("p = 0.3 full-range width", "staleness:bias_table_p030_width", 0.747, "abs"),
        ("p = 0.3 inflation", "staleness:bias_table_p030_inflation", 1.06, "abs", 5e-3),
        ("p = 0.3 width+", "staleness:bias_table_p030_width_pos", 0.185, "abs"),
        ("p = 0.3 h(-5)", "staleness:bias_table_p030_h_neg5", 0.834, "abs"),
        ("p = 0.4 full-range width", "staleness:bias_table_p040_width", 0.811, "abs"),
        ("p = 0.4 inflation", "staleness:bias_table_p040_inflation", 1.15, "abs", 5e-3),
        ("p = 0.4 width+", "staleness:bias_table_p040_width_pos", 0.188, "abs"),
        ("p = 0.4 h(-5)", "staleness:bias_table_p040_h_neg5", 0.871, "abs"),
        ("p = 0.5 full-range width", "staleness:bias_table_p050_width", 0.902, "abs"),
        ("p = 0.5 inflation", "staleness:bias_table_p050_inflation", 1.28, "abs", 5e-3),
        ("p = 0.5 width+", "staleness:bias_table_p050_width_pos", 0.185, "abs"),
        ("p = 0.5 h(-5)", "staleness:bias_table_p050_h_neg5", 0.954, "abs"),
        ("p = 0.7 full-range width", "staleness:bias_table_p070_width", 1.284, "abs"),
        ("p = 0.7 inflation", "staleness:bias_table_p070_inflation", 1.82, "abs", 5e-3),
        ("p = 0.7 width+", "staleness:bias_table_p070_width_pos", 0.209, "abs"),
        ("p = 0.7 h(-5)", "staleness:bias_table_p070_h_neg5", 1.248, "abs"),
        ("p = 0.9 full-range width", "staleness:bias_table_p090_width", 1.766, "abs"),
        ("p = 0.9 inflation", "staleness:bias_table_p090_inflation", 2.50, "abs", 5e-3),
        ("p = 0.9 width+", "staleness:bias_table_p090_width_pos", 0.237, "abs"),
        ("p = 0.9 h(-5)", "staleness:bias_table_p090_h_neg5", 1.669, "abs"),
        ("replications per cell", "staleness:design_reps_per_cell", 100, "int"),
        ("series length of the base cell", "staleness:design_n_base", 20_000, "int"),
        ("intermittency of the base cell", "staleness:design_lambda2_base", 0.10, "abs"),
    ]),

    ("3.4 Staleness bias -- the threshold", [
        ("length grid", "staleness:design_n_grid", [5_000, 20_000, 50_000], "set"),
        ("intermittency grid", "staleness:design_lambda2_grid", [0.05, 0.10, 0.15], "set"),
        ("knee at N = 5,000", "staleness:knee_N5000_lambda2_0.10_iid_knee110",
         0.30, "abs", 5e-3),
        ("knee at N = 50,000", "staleness:knee_N50000_lambda2_0.10_iid_knee110",
         0.40, "abs", 5e-3),
        ("inflation at p = 0.30 stays below 1.25x, independent deletion",
         "staleness:worst_iid_max_inflation_to_p030", 1.25, "lt"),
        ("inflation at p = 0.30 reaches 3x under Markov runs",
         "staleness:worst_markov20_max_inflation_to_p030", 3.0, "abs", 0.1),
        ("clustered (markov5): width 10% inflated by p = 0.05, N = 5000, lambda2 = 0.05",
         "staleness:knee_N5000_lambda2_0.05_markov5_knee110", 0.05, "le"),
        ("clustered (markov20): width 10% inflated by p = 0.05, N = 5000, lambda2 = 0.05",
         "staleness:knee_N5000_lambda2_0.05_markov20_knee110", 0.05, "le"),
        ("clustered (markov5): width 10% inflated by p = 0.05, N = 5000, lambda2 = 0.10",
         "staleness:knee_N5000_lambda2_0.10_markov5_knee110", 0.05, "le"),
        ("clustered (markov20): width 10% inflated by p = 0.05, N = 5000, lambda2 = 0.10",
         "staleness:knee_N5000_lambda2_0.10_markov20_knee110", 0.05, "le"),
        ("clustered (markov5): width 10% inflated by p = 0.05, N = 5000, lambda2 = 0.15",
         "staleness:knee_N5000_lambda2_0.15_markov5_knee110", 0.05, "le"),
        ("clustered (markov20): width 10% inflated by p = 0.05, N = 5000, lambda2 = 0.15",
         "staleness:knee_N5000_lambda2_0.15_markov20_knee110", 0.05, "le"),
        ("clustered (markov5): width 10% inflated by p = 0.05, N = 20000, lambda2 = 0.05",
         "staleness:knee_N20000_lambda2_0.05_markov5_knee110", 0.05, "le"),
        ("clustered (markov20): width 10% inflated by p = 0.05, N = 20000, lambda2 = 0.05",
         "staleness:knee_N20000_lambda2_0.05_markov20_knee110", 0.05, "le"),
        ("clustered (markov5): width 10% inflated by p = 0.05, N = 20000, lambda2 = 0.10",
         "staleness:knee_N20000_lambda2_0.10_markov5_knee110", 0.05, "le"),
        ("clustered (markov20): width 10% inflated by p = 0.05, N = 20000, lambda2 = 0.10",
         "staleness:knee_N20000_lambda2_0.10_markov20_knee110", 0.05, "le"),
        ("clustered (markov5): width 10% inflated by p = 0.05, N = 20000, lambda2 = 0.15",
         "staleness:knee_N20000_lambda2_0.15_markov5_knee110", 0.05, "le"),
        ("clustered (markov20): width 10% inflated by p = 0.05, N = 20000, lambda2 = 0.15",
         "staleness:knee_N20000_lambda2_0.15_markov20_knee110", 0.05, "le"),
        ("clustered (markov5): width 10% inflated by p = 0.05, N = 50000, lambda2 = 0.05",
         "staleness:knee_N50000_lambda2_0.05_markov5_knee110", 0.05, "le"),
        ("clustered (markov20): width 10% inflated by p = 0.05, N = 50000, lambda2 = 0.05",
         "staleness:knee_N50000_lambda2_0.05_markov20_knee110", 0.05, "le"),
        ("clustered (markov5): width 10% inflated by p = 0.05, N = 50000, lambda2 = 0.10",
         "staleness:knee_N50000_lambda2_0.10_markov5_knee110", 0.05, "le"),
        ("clustered (markov20): width 10% inflated by p = 0.05, N = 50000, lambda2 = 0.10",
         "staleness:knee_N50000_lambda2_0.10_markov20_knee110", 0.05, "le"),
        ("clustered (markov5): width 10% inflated by p = 0.05, N = 50000, lambda2 = 0.15",
         "staleness:knee_N50000_lambda2_0.15_markov5_knee110", 0.05, "le"),
        ("clustered (markov20): width 10% inflated by p = 0.05, N = 50000, lambda2 = 0.15",
         "staleness:knee_N50000_lambda2_0.15_markov20_knee110", 0.05, "le"),
        # The manuscript quotes 1.22x as a separate 100-replication robustness
        # run, on top of a main design of 40. The design now runs 100 per cell,
        # so there is one number, not two, and the claim to check is that the
        # design the manuscript describes is the design that ran.
        ("replications per cell", "staleness:design_reps_per_cell", 100, "int"),
        ("p = 0.30 is recommended as safe", "staleness:worst_iid_max_inflation_to_p030",
         1.35, "lt"),
        ("hourly series, shortest", "calendar:hourly.reconstructed.range.n.min",
         37_590, "int"),
        ("hourly series, longest", "calendar:hourly.reconstructed.range.n.max",
         40_348, "int"),
        ("width+ across the p grid, smallest", "staleness:bias_table_width_pos_min",
         0.174, "abs"),
        ("width+ across the p grid, largest", "staleness:bias_table_width_pos_max",
         0.237, "abs"),
        ("width+ near zero for a monofractal",
         "staleness:informativeness_N20000_lambda2_0.000_width_pos", 0.05, "lt"),
        ("width+ correlates with lambda2",
         "staleness:informativeness_N20000_corr_medians", 0.99, "gt"),
        ("the positive-q width is monotone in lambda2",
         "staleness:informativeness_N20000_monotone", True, "bool"),
        ("clustered inactivity (runs of twenty) moves width+ by up to 55%",
         "staleness:base_markov20_width_pos_inflation_max_abs_dev", 0.55, "abs", 5e-3),
        ("clustered inactivity (runs of five) moves width+ by up to 26%",
         "staleness:base_markov5_width_pos_inflation_max_abs_dev", 0.26, "abs", 5e-3),
        ("scattered inactivity: width+ up about 6% at p = 0.3",
         "staleness:bias_table_p030_width_pos", 0.174 * 1.06, "abs", 0.004),
        ("scattered inactivity: width+ up about 6% at p = 0.5",
         "staleness:bias_table_p050_width_pos", 0.174 * 1.06, "abs", 0.004),
        ("scattered inactivity: width+ up 20% at p = 0.7",
         "staleness:bias_table_p070_width_pos", 0.174 * 1.20, "abs", 0.004),
        ("Table 3: bias at n = 20,000, lambda2 = 0.10, about an eighth below",
         "staleness:informativeness_N20000_lambda2_0.100_width_pos_bias", (-0.20 * 0.15, -0.20 * 0.10), "between"),
        ("Table 3: bias at n = 20,000, lambda2 = 0.15, about an eighth below",
         "staleness:informativeness_N20000_lambda2_0.150_width_pos_bias", (-0.30 * 0.15, -0.30 * 0.10), "between"),
        ("Table 3: RMSE at n = 20,000, lambda2 = 0.10",
         "staleness:informativeness_N20000_lambda2_0.100_width_pos_rmse", 0.054, "abs"),
        ("Table 3: RMSE at n = 5,000, lambda2 = 0.15",
         "staleness:informativeness_N5000_lambda2_0.150_width_pos_rmse", 0.103, "abs"),
        ("Table 3: RMSE at n = 50,000, lambda2 = 0.05",
         "staleness:informativeness_N50000_lambda2_0.050_width_pos_rmse", 0.022, "abs"),
    ]),

    ("3.4 Size of the overlapping-window inference", [
        ("overlapping-window HAC rejection rate, 15.6%",
         "power:power.size.NCT_vs_BCT.hac.rejection", 0.16, "abs", 6e-3),
        ("HAC size, MCO2 design, 12%", "power:power.size.MCO2_vs_BCT.hac.rejection", 0.12, "abs", 5e-3),
        ("HAC lags", "power:power.hac_lags", 16, "int"),
        ("power: base intermittency calibrated to 0.35",
         "power:power.calibration.lambda2_base", 0.35, "abs", 5e-3),
        ("power: alternatives stepped downward", "power:power.ladder.direction", "down", "prefix"),
        ("non-overlapping windows restore correct size",
         "power:power.size.NCT_vs_BCT.nonoverlap_perm.rejection", (0.03, 0.07), "between"),
        ("window length of the simulated design", "power:power.window", 4_000, "int"),
        ("step of the overlapping design", "power:power.step_overlap", 250, "int"),
    ]),

    ("4.2 Within the quality ladder", [
        ("zero returns are essentially absent", "tiers:tiers.BCT.zero_return_share",
         0.005, "lt"),
        ("NCT - BCT gap", "tiers:tiers.main.NCT_vs_BCT.gap", 0.024, "abs"),
        ("NCT - BCT p", "tiers:tiers.main.NCT_vs_BCT.p", 0.64, "p", 5e-3),
        ("NCT windows", "tiers:tiers.main.NCT_vs_BCT.n_other", 9, "int"),
        ("NCT - BCT matched p", "tiers:tiers.NCT_vs_BCT.matched.p", 1.0, "p", 5e-3),
        ("BCT windows", "tiers:tiers.main.NCT_vs_BCT.n_base", 99, "int"),
        ("MCO2 - BCT gap", "tiers:tiers.main.MCO2_vs_BCT.gap", -0.163, "abs"),
        ("MCO2 - BCT p below 0.001", "tiers:tiers.main.MCO2_vs_BCT.p", 0.001, "lt"),
        ("MCO2 windows", "tiers:tiers.main.MCO2_vs_BCT.n_other", 14, "int"),
        ("MCO2 - BCT gap by execution price", "execution_q3/tiers:tiers.main.MCO2_vs_BCT.gap", 0.018, "abs"),
        ("MCO2 - BCT p by execution price", "execution_q3/tiers:tiers.main.MCO2_vs_BCT.p", 0.72, "p", 5e-3),
        ("NCT - BCT gap by execution price, zero", "execution_q3/tiers:tiers.main.NCT_vs_BCT.gap", 0.005, "lt"),
        ("BCT window span, days", "tiers:tiers.BCT.span_days_median", 1.3, "abs", 5e-2),
        ("NCT window span, days", "tiers:tiers.NCT.span_days_median", 111.4, "abs", 0.5),
        ("most BCT windows start in 2021", "tiers:tiers.BCT.windows_start_2021", 49.5, "gt"),
        ("BCT window volume, about seventy times NCT's", "tiers:tiers.BCT.usd_per_window_median",
         (65 * 0.311e6, 75 * 0.311e6), "between"),
        ("NCT window volume, USD", "tiers:tiers.NCT.usd_per_window_median",
         0.31e6, "abs", 5e3),
        ("BCT windows inside NCT's intensity range",
         "tiers:tiers.NCT_vs_BCT.intensity_shared_base_windows", 8, "int"),
        ("matched NCT - BCT gap, zero", "tiers:tiers.NCT_vs_BCT.matched.gap", 0.000, "abs"),
        ("matched NCT windows", "tiers:tiers.NCT_vs_BCT.matched.n_other", 9, "int"),
        ("matched BCT windows against NCT", "tiers:tiers.NCT_vs_BCT.matched.n_base", 24, "int"),
        ("matched MCO2 - BCT gap", "tiers:tiers.MCO2_vs_BCT.matched.gap", -0.217, "abs"),
        ("matched MCO2 - BCT p", "tiers:tiers.MCO2_vs_BCT.matched.p", 0.001, "p"),
        ("matched MCO2 windows", "tiers:tiers.MCO2_vs_BCT.matched.n_other", 14, "int"),
        ("matched BCT windows against MCO2", "tiers:tiers.MCO2_vs_BCT.matched.n_base", 31, "int"),
    ]),

    ("4.2 Robustness -- Appendix B, Table A2 (tab:robust)", [
        ("full data window, NCT gap", "tiers:tiers.robust.full_window.NCT_vs_BCT.gap",
         0.041, "abs"),
        ("full data window, NCT p", "tiers:tiers.robust.full_window.NCT_vs_BCT.p",
         0.448, "p"),
        ("full data window, NCT windows",
         "tiers:tiers.robust.full_window.NCT_vs_BCT.n_other", 10, "int"),
        ("full data window, MCO2 gap", "tiers:tiers.robust.full_window.MCO2_vs_BCT.gap",
         -0.151, "abs"),
        ("full data window, MCO2 p", "tiers:tiers.robust.full_window.MCO2_vs_BCT.p",
         0.002, "p"),
        ("full data window, MCO2 windows",
         "tiers:tiers.robust.full_window.MCO2_vs_BCT.n_other", 15, "int"),
        *[(f"to {end}, {c} {what}", f"tiers:tiers.robust.to_{end.replace('-', '_')}.{c}_vs_BCT.{field}",
           value, kind)
          for end in ("2026-01-01", "2026-04-01")
          for c, rows in (("NCT", ((0.024, "gap", "abs"), (0.642, "p", "p"), (9, "n_other", "int"))),
                          ("MCO2", ((-0.163, "gap", "abs"), (0.001, "p", "lt"), (14, "n_other", "int"))))
          for value, field, kind in rows
          for what in [{"gap": "gap", "p": "p", "n_other": "windows"}[field]]],
        ("isolated trades, NCT gap", "tiers:tiers.robust.isolated.NCT_vs_BCT.gap",
         -0.114, "abs"),
        ("isolated trades, NCT p", "tiers:tiers.robust.isolated.NCT_vs_BCT.p", 0.133, "p"),
        ("isolated trades, NCT windows",
         "tiers:tiers.robust.isolated.NCT_vs_BCT.n_other", 8, "int"),
        ("isolated trades, BCT windows",
         "tiers:tiers.robust.isolated.NCT_vs_BCT.n_base", 68, "int"),
        ("isolated trades, MCO2 gap", "tiers:tiers.robust.isolated.MCO2_vs_BCT.gap",
         -0.126, "abs"),
        ("isolated trades, MCO2 p", "tiers:tiers.robust.isolated.MCO2_vs_BCT.p",
         0.10, "p", 5e-3),
        ("isolated trades, MCO2 windows",
         "tiers:tiers.robust.isolated.MCO2_vs_BCT.n_other", 7, "int"),
        ("fill first return, NCT gap", "tiers:tiers.robust.fill_first.NCT_vs_BCT.gap",
         0.011, "abs"),
        ("fill first return, NCT p", "tiers:tiers.robust.fill_first.NCT_vs_BCT.p",
         0.712, "p"),
        ("fill first return, MCO2 gap", "tiers:tiers.robust.fill_first.MCO2_vs_BCT.gap",
         -0.173, "abs"),
        ("fill first return, MCO2 p below 0.001", "tiers:tiers.robust.fill_first.MCO2_vs_BCT.p",
         0.001, "lt"),
        ("NCT gap across specifications, smallest (isolated trades)",
         "tiers:tiers.robust.NCT_vs_BCT.gap_range", (-0.114, 0.041), "pair", 5e-4),
        ("NCT cleaning grid: gap within 0.07 of zero, smallest",
         "tiers:tiers.cleaning.NCT_vs_BCT.gap_range", (-0.07, 0.07), "pair", 0.07),
        ("NCT significant in none of the 32 cleaning settings",
         "tiers:tiers.cleaning.NCT_vs_BCT.significant_5pct", 0, "int"),
        ("MCO2 significant in all 32 cleaning settings",
         "tiers:tiers.cleaning.MCO2_vs_BCT.significant_5pct", 32, "int"),
        ("isolated filter removes, MCO2",
         "tiers:tiers.robust.isolated.removed_share.MCO2", 0.531, "abs"),
        ("isolated filter removes, BCT",
         "tiers:tiers.robust.isolated.removed_share.BCT", 0.317, "abs"),
        ("isolated filter removes, NCT",
         "tiers:tiers.robust.isolated.removed_share.NCT", 0.163, "abs"),
        ("BCT median shifts by up to 0.05 across the cleaning settings",
         "tiers:tiers.cleaning.BCT_width_median_range", (0.339, 0.392), "pair", 5e-3),
        ("every fit good: MCO2 gap", "scaling:scaling.ladder.MCO2_vs_BCT.every_fit_good.gap",
         -0.204, "abs"),
        ("every fit good: MCO2 p", "scaling:scaling.ladder.MCO2_vs_BCT.every_fit_good.p",
         0.002, "p"),
        ("every fit good: MCO2 windows", "scaling:scaling.ladder.MCO2_vs_BCT.every_fit_good.windows",
         [87, 5], "set"),
    ]),

    ("4.2 Power of the ladder comparison", [
        ("size of the design, NCT", "power:power.curve.NCT_vs_BCT.size", (0.02, 0.08), "between"),
        ("size of the design, MCO2", "power:power.curve.MCO2_vs_BCT.size", (0.02, 0.08), "between"),
        ("80% power beyond about 0.13, NCT design",
         "power:power.curve.NCT_vs_BCT.gap_at_power_0.8", 0.13, "abs", 5e-3),
        ("80% power beyond about 0.11, MCO2 design",
         "power:power.curve.MCO2_vs_BCT.gap_at_power_0.8", 0.11, "abs", 5e-3),
        ("replications per rung", "power:power.curve.NCT_vs_BCT.reps", 500, "int"),
    ]),

    ("4.3 Between asset classes -- Table 6 (tab:contrasts)", [
        ("estimate, all: gap", "asset_class:asset_class.contrast.A.all.gaps_filled.gap", 0.196, "abs"),
        ("estimate, all: p", "asset_class:asset_class.contrast.A.all.gaps_filled.p", 0.001, "lt"),
        ("estimate, p <= 0.30: gap", "asset_class:asset_class.contrast.A.no_trade_le_0.30.gaps_filled.gap", 0.196, "abs"),
        ("estimate, p <= 0.30: p", "asset_class:asset_class.contrast.A.no_trade_le_0.30.gaps_filled.p", 0.001, "lt"),
        ("estimate, p <= 0.05: gap", "asset_class:asset_class.contrast.A.no_trade_le_0.05.gaps_filled.gap", 0.134, "abs"),
        ("estimate, p <= 0.05: p", "asset_class:asset_class.contrast.A.no_trade_le_0.05.gaps_filled.p", 0.002, "p"),
        ("control, all: gap", "asset_class:asset_class.contrast.B.all.gaps_filled.gap", -0.008, "abs"),
        ("control, all: p", "asset_class:asset_class.contrast.B.all.gaps_filled.p", 0.575, "p"),
        ("control, p <= 0.30: gap", "asset_class:asset_class.contrast.B.no_trade_le_0.30.gaps_filled.gap", -0.009, "abs"),
        ("control, p <= 0.30: p", "asset_class:asset_class.contrast.B.no_trade_le_0.30.gaps_filled.p", 0.553, "p"),
        ("control, p <= 0.05: gap", "asset_class:asset_class.contrast.B.no_trade_le_0.05.gaps_filled.gap", -0.009, "abs"),
        ("control, p <= 0.05: p", "asset_class:asset_class.contrast.B.no_trade_le_0.05.gaps_filled.p", 0.691, "p"),
        ("reference, all: gap", "asset_class:asset_class.contrast.C.all.gaps_filled.gap", 0.204, "abs"),
        ("reference, all: p below 0.001", "asset_class:asset_class.contrast.C.all.gaps_filled.p", 0.001, "lt"),
        ("reference, p <= 0.30: gap", "asset_class:asset_class.contrast.C.no_trade_le_0.30.gaps_filled.gap", 0.206, "abs"),
        ("reference, p <= 0.30: p below 0.001", "asset_class:asset_class.contrast.C.no_trade_le_0.30.gaps_filled.p", 0.001, "lt"),
        ("reference, p <= 0.05: gap", "asset_class:asset_class.contrast.C.no_trade_le_0.05.gaps_filled.gap", 0.143, "abs"),
        ("reference, p <= 0.05: p below 0.001", "asset_class:asset_class.contrast.C.no_trade_le_0.05.gaps_filled.p", 0.001, "lt"),
        ("estimate, all: conventional windows", "asset_class:asset_class.contrast.A.all.gaps_filled.n_a", 30, "int"),
        ("estimate, all: carbon windows", "asset_class:asset_class.contrast.A.all.gaps_filled.n_b", 38, "int"),
        ("estimate, p <= 0.30: conventional windows", "asset_class:asset_class.contrast.A.no_trade_le_0.30.gaps_filled.n_a", 29, "int"),
        ("estimate, p <= 0.30: carbon windows", "asset_class:asset_class.contrast.A.no_trade_le_0.30.gaps_filled.n_b", 11, "int"),
        ("estimate, p <= 0.05: conventional windows", "asset_class:asset_class.contrast.A.no_trade_le_0.05.gaps_filled.n_a", 16, "int"),
        ("estimate, p <= 0.05: carbon windows", "asset_class:asset_class.contrast.A.no_trade_le_0.05.gaps_filled.n_b", 4, "int"),
        ("control, all: aggregated windows", "asset_class:asset_class.contrast.B.all.gaps_filled.n_b", 20, "int"),
        ("conventional median width", "asset_class:asset_class.contrast.A.all.gaps_filled.median_a", 0.145, "abs"),
        ("carbon median width", "asset_class:asset_class.contrast.A.all.gaps_filled.median_b", 0.341, "abs"),
        ("control under 0.01 under every restriction, all", "asset_class:asset_class.contrast.B.all.gaps_filled.gap", (-0.01, 0.01), "between"),
        ("control under 0.01 under every restriction, no_trade_le_0.30", "asset_class:asset_class.contrast.B.no_trade_le_0.30.gaps_filled.gap", (-0.01, 0.01), "between"),
        ("control under 0.01 under every restriction, no_trade_le_0.05", "asset_class:asset_class.contrast.B.no_trade_le_0.05.gaps_filled.gap", (-0.01, 0.01), "between"),
        ("estimate, all: token p", "asset_class:asset_class.token_test.A.all.p_two_sided", 0.057, "abs"),
        ("estimate, p <= 0.30: token p", "asset_class:asset_class.token_test.A.no_trade_le_0.30.p_two_sided", 0.1, "abs", 0.005),
        ("estimate, p <= 0.05: token p", "asset_class:asset_class.token_test.A.no_trade_le_0.05.p_two_sided", 0.1, "abs", 0.005),
        ("control, all: token p", "asset_class:asset_class.token_test.B.all.p_two_sided", 0.4, "abs", 0.005),
        ("control, p <= 0.30: token p", "asset_class:asset_class.token_test.B.no_trade_le_0.30.p_two_sided", 0.4, "abs", 0.005),
        ("control, p <= 0.05: token p", "asset_class:asset_class.token_test.B.no_trade_le_0.05.p_two_sided", 0.8, "abs", 0.005),
        ("reference, all: token p", "asset_class:asset_class.token_test.C.all.p_two_sided", 0.133, "abs", 0.005),
        ("reference, p <= 0.30: token p", "asset_class:asset_class.token_test.C.no_trade_le_0.30.p_two_sided", 0.3, "abs", 0.005),
        ("reference, p <= 0.05: token p", "asset_class:asset_class.token_test.C.no_trade_le_0.05.p_two_sided", 0.33, "abs", 0.005),
        ("estimate, all: floor of the token test", "asset_class:asset_class.token_test.A.all.min_attainable_p_two_sided", 0.029, "abs"),
        ("estimate, all: assignments", "asset_class:asset_class.token_test.A.all.assignments", 35, "int"),
        ("estimate, p <= 0.30: assignments", "asset_class:asset_class.token_test.A.no_trade_le_0.30.assignments", 20, "int"),
        ("estimate, p <= 0.05: assignments", "asset_class:asset_class.token_test.A.no_trade_le_0.05.assignments", 10, "int"),
        ("estimate, p <= 0.30: floor of the token test", "asset_class:asset_class.token_test.A.no_trade_le_0.30.min_attainable_p_two_sided", 0.1, "abs", 0.005),
        ("estimate, p <= 0.05: floor of the token test", "asset_class:asset_class.token_test.A.no_trade_le_0.05.min_attainable_p_two_sided", 0.1, "abs", 0.005),
    ]),

    ("4.3 Between asset classes -- token level, power and staleness", [
        ("BCT windows", "asset_class:asset_class.token.BCT.windows", 10, "int"),
        ("NCT windows", "asset_class:asset_class.token.NCT.windows", 9, "int"),
        ("MCO2 windows", "asset_class:asset_class.token.MCO2.windows", 9, "int"),
        ("KLIMA windows", "asset_class:asset_class.token.KLIMA.windows", 10, "int"),
        ("WETH_S windows", "asset_class:asset_class.token.WETH_S.windows", 10, "int"),
        ("WETH_Q windows", "asset_class:asset_class.token.WETH_Q.windows", 10, "int"),
        ("WBTC_Q windows", "asset_class:asset_class.token.WBTC_Q.windows", 10, "int"),
        ("BTC windows", "asset_class:asset_class.token.BTC.windows", 10, "int"),
        ("ETH windows", "asset_class:asset_class.token.ETH.windows", 10, "int"),
        ("leave-one-out gap, smallest", "asset_class:asset_class.leave_one_out.A.all.min", 0.175, "abs"),
        ("leave-one-out gap, largest", "asset_class:asset_class.leave_one_out.A.all.max", 0.219, "abs"),
        ("separation is complete", "asset_class:asset_class.separation.complete", True, "bool"),
        ("observed split ranks first by signed gap", "asset_class:asset_class.token_test.A.all.rank_upper", 1, "int"),
        ("token-level two-sided p", "asset_class:asset_class.token_test.A.all.p_two_sided", 0.057, "abs"),
        ("token-level one-sided p", "asset_class:asset_class.token_test.A.all.p_upper", 0.029, "abs"),
        ("carbon tokens left at p <= 0.30", "asset_class:asset_class.token_test.A.no_trade_le_0.30.tokens_b", 3, "int"),
        ("carbon tokens left at p <= 0.05", "asset_class:asset_class.token_test.A.no_trade_le_0.05.tokens_b", 2, "int"),
        ("rank test, carbon vs deep: p", "thin_benchmark:thin_benchmark.both.rank_p_two_sided", 0.057, "abs"),
        ("rank test, carbon vs deep: shift 0.20", "thin_benchmark:thin_benchmark.both.rank_shift", 0.2, "abs", 0.005),
        ("rank test, carbon vs deep: interval low", "thin_benchmark:thin_benchmark.both.rank_ci_low", 0.11, "abs", 0.005),
        ("rank test, carbon vs deep: interval high", "thin_benchmark:thin_benchmark.both.rank_ci_high", 0.27, "abs", 0.005),
        ("size of the window test, all", "asset_class:asset_class.power.C.all.clean.size.reject_rate", (0.02, 0.08), "between"),
        ("rejects a true gap of 0.15 in at least 95%, all", "asset_class:asset_class.power.C.all.clean.gap_0.15.reject_rate", 0.94, "ge"),
        ("size of the window test, no_trade_le_0.30", "asset_class:asset_class.power.C.no_trade_le_0.30.clean.size.reject_rate", (0.02, 0.08), "between"),
        ("rejects a true gap of 0.15 in at least 95%, no_trade_le_0.30", "asset_class:asset_class.power.C.no_trade_le_0.30.clean.gap_0.15.reject_rate", 0.94, "ge"),
        ("size of the window test, no_trade_le_0.05", "asset_class:asset_class.power.C.no_trade_le_0.05.clean.size.reject_rate", (0.02, 0.08), "between"),
        ("rejects a true gap of 0.15 in at least 95%, no_trade_le_0.05", "asset_class:asset_class.power.C.no_trade_le_0.05.clean.gap_0.15.reject_rate", 0.94, "ge"),
        ("BCT median staleness is roughly half to four fifths", "asset_class:asset_class.token.BCT.no_trade_median", (0.44, 0.8), "between"),
        ("NCT median staleness is roughly half to four fifths", "asset_class:asset_class.token.NCT.no_trade_median", (0.44, 0.8), "between"),
        ("MCO2 median staleness is roughly half to four fifths", "asset_class:asset_class.token.MCO2.no_trade_median", (0.44, 0.8), "between"),
        ("KLIMA: fewer than one hour in ten empty", "asset_class:asset_class.token.KLIMA.no_trade_median", 0.1, "lt"),
        ("WETH_S staleness about a quarter at most", "asset_class:asset_class.token.WETH_S.no_trade_median", 0.26, "lt"),
        ("WETH_Q staleness about a quarter at most", "asset_class:asset_class.token.WETH_Q.no_trade_median", 0.26, "lt"),
        ("WBTC_Q staleness about a quarter at most", "asset_class:asset_class.token.WBTC_Q.no_trade_median", 0.26, "lt"),
    ]),

    ("4.3 The matched benchmark -- Table 7 (tab:benchmark_tests)", [
        ("asset: gap", "thin_benchmark:thin_benchmark.asset.gap", 0.089, "abs"),
        ("asset: reassignment p", "thin_benchmark:thin_benchmark.asset.token_p", 0.123, "abs"),
        ("asset: rank p", "thin_benchmark:thin_benchmark.asset.rank_p_two_sided", 0.207, "abs"),
        ("asset: interval low", "thin_benchmark:thin_benchmark.asset.rank_ci_low", -0.03, "abs", 0.005),
        ("asset: interval high", "thin_benchmark:thin_benchmark.asset.rank_ci_high", 0.17, "abs", 0.005),
        ("thinness: gap", "thin_benchmark:thin_benchmark.thinness.gap", 0.107, "abs"),
        ("thinness: reassignment p", "thin_benchmark:thin_benchmark.thinness.token_p", 0.102, "abs"),
        ("thinness: rank p", "thin_benchmark:thin_benchmark.thinness.rank_p_two_sided", 0.002, "abs"),
        ("thinness: interval low", "thin_benchmark:thin_benchmark.thinness.rank_ci_low", 0.02, "abs", 0.005),
        ("thinness: interval high", "thin_benchmark:thin_benchmark.thinness.rank_ci_high", 0.24, "abs", 0.005),
        ("both: gap", "thin_benchmark:thin_benchmark.both.gap", 0.196, "abs"),
        ("both: reassignment p", "thin_benchmark:thin_benchmark.both.token_p", 0.057, "abs"),
        ("both: rank p", "thin_benchmark:thin_benchmark.both.rank_p_two_sided", 0.057, "abs"),
        ("both: interval low", "thin_benchmark:thin_benchmark.both.rank_ci_low", 0.11, "abs", 0.005),
        ("both: interval high", "thin_benchmark:thin_benchmark.both.rank_ci_high", 0.27, "abs", 0.005),
        ("activity: gap", "thin_benchmark:thin_benchmark.activity.gap", 0.034, "abs"),
        ("activity: reassignment p", "thin_benchmark:thin_benchmark.activity.token_p", 0.485, "abs"),
        ("activity: rank p", "thin_benchmark:thin_benchmark.activity.rank_p_two_sided", 0.195, "abs"),
        ("activity: interval low", "thin_benchmark:thin_benchmark.activity.rank_ci_low", -0.03, "abs", 0.005),
        ("activity: interval high", "thin_benchmark:thin_benchmark.activity.rank_ci_high", 0.15, "abs", 0.005),
        ("busy: gap", "thin_benchmark:thin_benchmark.busy.gap", 0.091, "abs"),
        ("busy: reassignment p", "thin_benchmark:thin_benchmark.busy.token_p", 0.028, "abs"),
        ("busy: rank p", "thin_benchmark:thin_benchmark.busy.rank_p_two_sided", 0.011, "abs"),
        ("busy: interval low", "thin_benchmark:thin_benchmark.busy.rank_ci_low", 0.02, "abs", 0.005),
        ("busy: interval high", "thin_benchmark:thin_benchmark.busy.rank_ci_high", 0.16, "abs", 0.005),
        ("majors: gap", "thin_benchmark:thin_benchmark.majors.gap", -0.074, "abs"),
        ("majors: reassignment p", "thin_benchmark:thin_benchmark.majors.token_p", 0.119, "abs"),
        ("majors: rank p", "thin_benchmark:thin_benchmark.majors.rank_p_two_sided", 0.167, "abs"),
        ("majors: interval low", "thin_benchmark:thin_benchmark.majors.rank_ci_low", -0.22, "abs", 0.005),
        ("majors: interval high", "thin_benchmark:thin_benchmark.majors.rank_ci_high", 0.02, "abs", 0.005),
        ("thin tokens", "thin_benchmark:thin_benchmark.thin.tokens", 36, "int"),
        ("active tokens", "thin_benchmark:thin_benchmark.active.tokens", 6, "int"),
        ("35 of 36 thin medians above the deep median", "thin_benchmark:thin_benchmark.thinness.tokens_of_second_above_first_median", 35, "int"),
        ("asset: positive in all 60 variants", "thin_robustness:thin_robustness.asset.gap_positive", 60, "int"),
        ("asset: significant in 14 variants", "thin_robustness:thin_robustness.asset.significant_variants", 14, "int"),
        ("asset: thirteen of them dollar floors of 2 and above",
         "thin_robustness:thin_robustness.asset.significant_at_floor_ge_2", 13, "int"),
        ("dose: idle share and width across 45 conventional tokens", "thin_benchmark:thin_benchmark.dose.conventional.rho", 0.46, "abs", 0.005),
        ("floor 3: thin vs deep gap", "thin_robustness:thin_robustness.floor.min_usd=3.thinness.gap", 0.071, "abs"),
        ("floor 3: thin vs deep rank p", "thin_robustness:thin_robustness.floor.min_usd=3.thinness.rank_p", 0.04, "abs", 0.005),
        ("floor 5: thin vs deep gap", "thin_robustness:thin_robustness.floor.min_usd=5.thinness.gap", 0.038, "abs"),
        ("floor 20: thin vs deep gap removed", "thin_robustness:thin_robustness.floor.min_usd=20.thinness.gap", 0.01, "lt"),
    ]),

    ("4.3 Size of the asset", [
        ("45 conventional tokens", "size:size.tokens", 45, "int"),
        ("width vs idleness", "size:size.main.rho_width_idle", 0.46, "abs", 0.005),
        ("width vs smallness", "size:size.main.rho_width_small", 0.55, "abs", 0.005),
        ("idleness vs smallness", "size:size.main.rho_idle_small", 0.38, "abs", 0.005),
        ("partial, size given idleness", "size:size.main.partial_width_small", 0.45, "abs", 0.005),
        ("partial, size given idleness: p", "size:size.main.p_partial_width_small", 0.002, "p"),
        ("partial, idleness given size", "size:size.main.partial_width_idle", 0.32, "abs", 0.005),
        ("partial, idleness given size: p", "size:size.main.p_partial_width_idle", 0.03, "abs", 0.005),
        ("size partial between 0.44 and 0.71, volume", "size:size.volume.partial_width_small", (0.43, 0.71), "between"),
        ("size partial p at most 0.003, volume", "size:size.volume.p_partial_width_small", 0.0031, "lt"),
        ("idleness partial between 0.16 and 0.31, volume", "size:size.volume.partial_width_idle", (0.16, 0.31), "between"),
        ("size partial between 0.44 and 0.71, check", "size:size.check.partial_width_small", (0.43, 0.71), "between"),
        ("size partial p at most 0.003, check", "size:size.check.p_partial_width_small", 0.0031, "lt"),
        ("idleness partial between 0.16 and 0.31, check", "size:size.check.partial_width_idle", (0.16, 0.31), "between"),
        ("size partial between 0.44 and 0.71, late", "size:size.late.partial_width_small", (0.43, 0.71), "between"),
        ("size partial p at most 0.003, late", "size:size.late.p_partial_width_small", 0.0031, "lt"),
        ("idleness partial between 0.16 and 0.31, late", "size:size.late.partial_width_idle", (0.16, 0.31), "between"),
        ("size partial between 0.44 and 0.71, listed", "size:size.listed.partial_width_small", (0.43, 0.71), "between"),
        ("size partial p at most 0.003, listed", "size:size.listed.p_partial_width_small", 0.0031, "lt"),
        ("idleness partial between 0.16 and 0.31, listed", "size:size.listed.partial_width_idle", (0.16, 0.31), "between"),
        ("size partial between 0.44 and 0.71, weth_once", "size:size.weth_once.partial_width_small", (0.43, 0.71), "between"),
        ("size partial p at most 0.003, weth_once", "size:size.weth_once.p_partial_width_small", 0.0031, "lt"),
        ("idleness partial between 0.16 and 0.31, weth_once", "size:size.weth_once.partial_width_idle", (0.16, 0.31), "between"),
        ("size partial between 0.44 and 0.71, mismatch_recoded", "size:size.mismatch_recoded.partial_width_small", (0.43, 0.71), "between"),
        ("size partial p at most 0.003, mismatch_recoded", "size:size.mismatch_recoded.p_partial_width_small", 0.0031, "lt"),
        ("idleness partial between 0.16 and 0.31, mismatch_recoded", "size:size.mismatch_recoded.partial_width_idle", (0.16, 0.31), "between"),
        ("idleness partial on listed tokens, 0.17", "size:size.listed.partial_width_idle", 0.17, "abs", 0.005),
        ("listed tokens, 28", "size:size.listed.tokens", 28, "int"),
        ("17 unlisted", "size:size.unlisted", 17, "int"),
        ("cell large_busy: width", "size:size.cell.large_busy.width", 0.145, "abs"),
        ("cell large_thin: width", "size:size.cell.large_thin.width", 0.161, "abs"),
        ("cell small_busy: width", "size:size.cell.small_busy.width", 0.202, "abs"),
        ("cell small_thin: width", "size:size.cell.small_thin.width", 0.286, "abs"),
    ]),

    ("4.4 Implications for measurement practice", [
        ("execution price narrows a deep pool's spectrum by a third, WETH_S",
         "price_definition:price_definition.WETH_S.execution", (0.146 * 0.60, 0.146 * 0.72), "between"),
        ("execution price narrows a deep pool's spectrum by a third, WETH_Q",
         "price_definition:price_definition.WETH_Q.execution", (0.146 * 0.60, 0.146 * 0.72), "between"),
    ]),

    ("Abstract, Introduction and Conclusions", [
        ("abstract: wider than the deep pools, token-level p above 0.05",
         "asset_class:asset_class.token_test.A.all.p_two_sided", 0.05, "gt"),
        ("abstract: no wider than thin tokens, rank p above 0.05",
         "thin_benchmark:thin_benchmark.asset.rank_p_two_sided", 0.05, "gt"),
        ("abstract: no wider than thin tokens, reassignment p above 0.05",
         "thin_benchmark:thin_benchmark.asset.token_p", 0.05, "gt"),
        ("abstract: width goes with capitalisation, partial p",
         "size:size.main.p_partial_width_small", 0.01, "lt"),
        ("introduction: 45 conventional tokens", "size:size.tokens", 45, "int"),
        ("introduction: 42 benchmark pools beyond the deepest", "census:census.thin", 36, "int"),
        ("introduction: 42 benchmark pools beyond the deepest, active", "census:census.active", 6, "int"),
        ("introduction: four carbon tokens", "thin_benchmark:thin_benchmark.carbon.tokens", 4, "int"),
        ("conclusions: more than twice as wide as the deep pools (carbon median above twice 0.145)",
         "asset_class:asset_class.contrast.A.all.gaps_filled.median_b", 0.29, "gt"),
        ("conclusions: deep pools median 0.145",
         "asset_class:asset_class.contrast.A.all.gaps_filled.median_a", 0.145, "abs"),
        ("conclusions: every carbon token above every deep pool",
         "asset_class:asset_class.separation.complete", True, "bool"),
        ("conclusions: a third narrower by execution, WETH_S",
         "price_definition:price_definition.WETH_S.execution", (0.146 * 0.60, 0.146 * 0.72), "between"),
        ("conclusions: restrictive tier no wider, matched",
         "tiers:tiers.NCT_vs_BCT.matched.gap", 0.01, "lt"),
    ]),
]


@dataclass(frozen=True)
class Claim:
    section: str
    what: str
    source: str
    paper: object
    kind: str
    tol: float

    @property
    def module(self) -> str:
        return self.source.split(":", 1)[0]

    @property
    def key(self) -> str:
        return self.source.split(":", 1)[1]


def claims() -> list[Claim]:
    out = []
    for section, rows in CLAIMS:
        for row in rows:
            what, source, paper, kind = row[:4]
            tol = row[4] if len(row) > 4 else DEFAULT_TOL.get(kind, 0.0)
            out.append(Claim(section, what, source, paper, kind, tol))
    return out


# --- resolution -----------------------------------------------------------

ABSENT = object()
_FILES: dict[str, dict | None] = {}


def load(module: str) -> dict | None:
    """A module is read from the live results folder; "<config>/<module>"
    reads it from another configuration's folder, data/results_<config>/,
    for the few claims that compare the paper's setting with another."""
    if module not in _FILES:
        if "/" in module:
            config, name = module.split("/", 1)
            path = RESULTS_DIR.parent / f"results_{config}" / f"{name}.json"
        else:
            path = RESULTS_DIR / f"{module}.json"
        _FILES[module] = json.loads(path.read_text()) if path.exists() else None
    return _FILES[module]


def dotted(blob: dict, path: str):
    """Resolve a dotted path, preferring a literal key to a descent.

    Modules write mostly flat dictionaries whose keys contain dots, but a few
    nest, so at every level the longest remaining path is tried as a literal
    key before taking one segment and going deeper.
    """
    node = blob
    parts = path.split(".")
    for i, part in enumerate(parts):
        if not isinstance(node, dict):
            return ABSENT
        rest = ".".join(parts[i:])
        if rest in node:
            return node[rest]
        if part not in node:
            return ABSENT
        node = node[part]
    return node


def number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


# --- comparison -----------------------------------------------------------

def compare(claim: Claim, value) -> tuple[str, str]:
    kind, want, tol = claim.kind, claim.paper, claim.tol
    if kind == "set":
        if not isinstance(value, (list, tuple)):
            return "DISAGREE", "not a list"
        got = sorted(f"{v:g}" if isinstance(v, float) else str(v) for v in value)
        expect = sorted(f"{v:g}" if isinstance(v, float) else str(v) for v in want)
        return ("AGREE", "") if got == expect else ("DISAGREE", "")
    if kind == "pair":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return "DISAGREE", "not a [lo, hi] pair"
        lo, hi = (number(v) for v in value)
        if lo is None or hi is None:
            return "MISSING", "pair holds no numbers"
        ok = abs(lo - want[0]) <= tol and abs(hi - want[1]) <= tol
        return ("AGREE", "") if ok else ("DISAGREE", "")
    if kind == "bool":
        return ("AGREE", "") if bool(value) is bool(want) else ("DISAGREE", "")
    if kind in ("str", "prefix"):
        got = str(value).strip()
        ok = got == want if kind == "str" else got.startswith(want)
        return ("AGREE", "") if ok else ("DISAGREE", "")

    x = number(value)
    if x is None:
        return "MISSING", "value is not a number"
    if kind == "int":
        ok = x == float(want)
    elif kind == "abs":
        ok = abs(x - want) <= tol
    elif kind == "rel":
        ok = abs(x - want) <= tol * abs(want)
    elif kind == "p":
        if (x < ALPHA) != (want < ALPHA):
            return "DISAGREE", "crosses p = 0.05"
        ok = abs(x - want) <= tol
    elif kind == "between":
        ok = want[0] - tol <= x <= want[1] + tol
    elif kind in DIRECTIONS:
        ok = DIRECTIONS[kind](x, float(want))
    else:
        return "MISSING", f"unknown comparison kind {kind!r}"
    return ("AGREE", "") if ok else ("DISAGREE", "")


def verify(claim: Claim) -> dict:
    blob = load(claim.module)
    if blob is None:
        verdict, note, value = "MISSING", f"no data/results/{claim.module}.json", None
    else:
        value = dotted(blob, claim.key)
        if value is ABSENT:
            verdict, note, value = "MISSING", f"no key in {claim.module}.json", None
        elif value is None:
            verdict, note = "MISSING", "key holds null"
        else:
            verdict, note = compare(claim, value)
    return {"section": claim.section, "claim": claim.what, "source": claim.source,
            "kind": claim.kind, "tolerance": claim.tol, "paper": claim.paper,
            "computed": value, "verdict": verdict, "note": note}


# --- reporting ------------------------------------------------------------

def show(value, kind: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            return f"{value[0]:.3f}..{value[1]:.3f}"
        return ("[" + ", ".join(str(v) for v in value) + "]")[:20]
    if isinstance(value, float):
        text = f"{value:,.0f}" if abs(value) >= 1e4 else f"{value:.4g}"
    elif isinstance(value, int):
        text = f"{value:,}"
    else:
        text = str(value)
    if kind in DIRECTIONS:
        text = {"gt": "> ", "ge": ">= ", "lt": "< ", "le": "<= "}[kind] + text
    return text[:20]


def report(rows: list[dict]) -> None:
    section = None
    for row in rows:
        if row["section"] != section:
            section = row["section"]
            print(f"\n{section}\n{'-' * len(section)}")
        note = f"  {row['note']}" if row["note"] else ""
        print(f"  {row['verdict']:<8} {row['claim']:<44} "
              f"paper {show(row['paper'], row['kind']):>20}   "
              f"computed {show(row['computed']):>20}{note}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--section", help="only claims whose location contains this text")
    ap.add_argument("--json", action="store_true",
                    help="also write data/results/verify_paper.json")
    args = ap.parse_args()

    selected = claims()
    if args.section:
        needle = args.section.lower()
        selected = [c for c in selected if needle in c.section.lower()]
        if not selected:
            raise SystemExit(f"no claims in a section matching {args.section!r}")

    rows = [verify(c) for c in selected]
    report(rows)

    counts = {v: sum(r["verdict"] == v for r in rows)
              for v in ("AGREE", "DISAGREE", "MISSING")}
    print(f"\n{len(rows)} claims: {counts['AGREE']} agree, "
          f"{counts['DISAGREE']} disagree, {counts['MISSING']} missing")

    for verdict in ("DISAGREE", "MISSING"):
        bad = [r for r in rows if r["verdict"] == verdict]
        if bad:
            print(f"\n{verdict} ({len(bad)})")
            for r in bad:
                note = f"  [{r['note']}]" if r["note"] else ""
                print(f"  {r['section']}  --  {r['claim']}: paper "
                      f"{show(r['paper'], r['kind'])}, computed {show(r['computed'])}"
                      f"  <{r['source']}>{note}")

    if args.json:
        print("\nwrote", numbers("verify_paper", {"counts": counts, "claims": rows}))
    sys.exit(1 if counts["DISAGREE"] or counts["MISSING"] else 0)


if __name__ == "__main__":
    main()
