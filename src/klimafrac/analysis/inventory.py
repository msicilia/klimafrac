"""
Pool inventory, pool value and the October 2025 BCT redemption, read from
the chain (paper Section 4.2.2 "Pool inventory and redemption", Figure
fig7_depth; the sentences on the event in the Introduction and Conclusion).

Nothing is fetched or typed in here. The module reads what the fetchers wrote:
  data/inventory.csv, data/inventory_daily.csv, data/inventory_summary.json
      BCT, NCT, UBO and NBO totalSupply() in tonnes (fetch/inventory.py),
      reconciled to the unit against the mint and burn event stream
  data/burn.json
      the October 2025 burn located by bisection on totalSupply(), the
      burning account, its Safe configuration, the treasury manager() check,
      the pools and swap events in the burn transaction, and where the
      redeemed credits went (fetch/burn.py)
  data/swaps_polygon_BCT.csv.gz, data/swaps_polygon_NCT.csv.gz
      through trades.trades(key, full_window=True)
  data/raw/defillama/protocol_toucan-protocol/   optional, comparison only

Inventory. A value dated D in inventory_daily.csv is the supply immediately
before 00:00 UTC on D. Here tonnes on day D means the supply at the END of D,
i.e. the row dated D + 1, so that a day's tonnage and that day's trades
describe the same interval. The exact all-time peak is the block-level peak
from the event path; the monthly sample, which the paper previously quoted,
is reported next to it because it lands on the first of the following month.

Pool value. value(D) = tonnes at the end of D x the median price of the
pool's cleaned swaps executed on D (USDC per token). Days without a cleaned
swap carry the last median forward and are flagged `price_carried`; days
before the pool's first cleaned swap have no value. The median rather than
the close is used because a single late swap against thin liquidity would
otherwise set a whole day's valuation. Both pools are USDC-quoted, so no
KLIMA conversion enters. This replaces the vendor value panel: DefiLlama's
BCT series ends on 2024-03-29 and its implied price runs 0.35 -> 1.27 in the
final days, so it cannot reach the burn. The vendor series is kept as the
column `vendor_usd_defillama` (fetch/depth.load_depth with its documented
masks) for comparison only; it is never plotted and no number depends on it.

The fall in value is reported from two reference maxima, both taken over
days strictly before the reference day:
  launch   the largest daily value since the pool's first cleaned swap
  2022     the largest daily value in calendar year 2022
to the reference day, the last complete UTC day before the burn block (the
burn is at 00:33 UTC, so the day of the burn would already include it). The
fall is decomposed into its price and tonnage ratios, since value = price x
tonnes; a fall with a tonnage ratio above one is a repricing only. Values
after trades.SAMPLE_END are in the table and the figure but enter no cited
number.

Redemption. The burned BCT released the underlying TCO2 credits from the
pool. fetch/burn.py records where those credits went in the burn
transaction, how much the burning Safe still held at END_BLOCK, and every
later transfer of them out of the Safe, flagged when the recipient is the
zero address. A transfer to the zero address is the only on-chain retirement
the record can show, so the numbers here say whether any occurred, not
whether an off-chain registry later retired the credits.

Outputs (via klimafrac.results):
  data/results/inventory_monthly.csv      date, block, token, status, tonnes,
                                          change_tonnes (from the previous
                                          month), for BCT, NCT, UBO, NBO
  data/results/inventory_value_daily.csv  date, token, tonnes, trades,
                                          price_median, price, price_carried,
                                          value_usd, in_sample,
                                          vendor_usd_defillama
  data/results/inventory.json             every number the section cites
  figures/fig7_depth.png

If a swap file is missing the inventory and redemption numbers and the
monthly table are still written; the run then stops with a message naming
the file, or exits cleanly with --no-prices.

Usage:
    uv run python -m klimafrac.analysis.inventory
    uv run python -m klimafrac.analysis.inventory --no-prices
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from klimafrac.fetch.depth import load_depth
from klimafrac.fetch.inventory import load_events
from klimafrac.paths import data, figure
from klimafrac.plotstyle import INK, INK2, MUTED, SERIES, plt, save, style
from klimafrac.results import numbers, table
from klimafrac.trades import SAMPLE_END, trades

TOKENS = ("BCT", "NCT", "UBO", "NBO")
VALUED = {"BCT": "polygon_BCT", "NCT": "polygon_NCT"}
ZERO = "0x" + "0" * 40
TOL_TONNES = 1e-6


def iso_day(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


def complete_months(start: pd.Timestamp, end: pd.Timestamp) -> int:
    months = (end.year - start.year) * 12 + end.month - start.month
    return months - (end.day < start.day or
                     (end.day == start.day and end.time() < start.time()))


def load_json(name: str) -> dict:
    path = Path(data(name))
    if not path.exists():
        raise FileNotFoundError(f"{path} missing (run the fetcher that writes it)")
    return json.loads(path.read_text())


# ---------------------------------------------------------------- inventory

def monthly_table() -> pd.DataFrame:
    inv = pd.read_csv(data("inventory.csv"))
    rows = []
    for tok in TOKENS:
        part = inv[["date", "block", tok, f"{tok}_status"]].rename(
            columns={tok: "tonnes", f"{tok}_status": "status"})
        part.insert(2, "token", tok)
        part["change_tonnes"] = part.tonnes.diff()
        rows.append(part[["date", "block", "token", "status", "tonnes", "change_tonnes"]])
    return pd.concat(rows, ignore_index=True)


def end_of_day_tonnes() -> pd.DataFrame:
    """Supply at the end of each UTC day, one column per token."""
    daily = pd.read_csv(data("inventory_daily.csv"), parse_dates=["date"])
    wide = daily.pivot(index="date", columns="token", values="tonnes").sort_index()
    wide.index = wide.index - pd.Timedelta(days=1)
    return wide


def peak_numbers(summary: dict, monthly: pd.DataFrame) -> dict:
    out = {}
    for tok in ("BCT", "NCT"):
        s = summary["tokens"][tok]
        p = s["peak"]
        deployed = pd.Timestamp(s["deploy_timestamp"])
        reached = pd.Timestamp(p["first_timestamp"])
        m = monthly[monthly.token == tok]
        top = m.loc[m.tonnes.idxmax()]
        out[tok] = {
            "deploy_timestamp": s["deploy_timestamp"],
            "peak_tonnes": p["tonnes"],
            "peak_million_tonnes": round(p["tonnes"] / 1e6, 1),
            "peak_block": p["first_block"],
            "peak_timestamp": p["first_timestamp"],
            "peak_date": iso_day(reached),
            "peak_ended_timestamp": p["ended_timestamp"],
            "months_launch_to_peak": complete_months(deployed, reached),
            "years_launch_to_peak": round((reached - deployed).days / 365.25, 2),
            # A monthly grid reads the supply at 00:00 on the 1st, so a
            # late-May peak is sampled as "June".
            "monthly_sample_peak_date": top.date,
            "monthly_sample_peak_tonnes": float(top.tonnes),
            "end_block_tonnes": s["end_block_tonnes"],
        }
    return out


# --------------------------------------------------------------- redemption

def burn_numbers(burn: dict, monthly: pd.DataFrame) -> dict:
    b, month, safe, tr = burn["bisection"], burn["month"], burn["burner"], burn["treasury"]
    start = monthly[(monthly.token == "BCT") & (monthly.date == month["start"]["instant"][:10])]
    if len(start) != 1 or abs(float(start.tonnes.iloc[0]) - month["start"]["tonnes"]) > TOL_TONNES:
        raise ValueError("burn.json month start disagrees with inventory.csv")
    if burn["peak"]["supply_raw"] != load_json("inventory_summary.json")["tokens"]["BCT"]["peak"]["supply_raw"]:
        raise ValueError("burn.json peak disagrees with inventory_summary.json")

    txs = burn["transactions"]
    credits = [t["credits_leaving_pool"] for t in txs]
    side_transfers = [x for t in txs for x in t["bct_transfers"]
                      if x["to"] != ZERO and x["from"] == safe["address"]]
    held = burn["redeemed_credits_held_by_burner"]
    outflows = held["outflows_after_block_by_destination"]
    decreases = burn["event_cross_check"]["largest_net_decreases"]

    # The Safe's later outflow is compared with the burn plus the month's next
    # largest decrease: equality says the second redemption went to the Safe too.
    second = decreases[1]
    ev = load_events("BCT")
    second_ts = pd.Timestamp(int(ev.loc[ev.block == second["block"], "timestamp"].iloc[0]),
                             unit="s", tz="UTC")
    out_tonnes = sum(v["tonnes"] for v in outflows.values())

    return {
        "tonnes": b["decrease_tonnes"],
        "tonnes_rounded": round(b["decrease_tonnes"]),
        "block": b["block"],
        "timestamp": burn["block"]["timestamp"],
        "date": burn["block"]["timestamp"][:10],
        "tx_hashes": [t["hash"] for t in txs],
        "bisection_steps": b["steps"],
        "supply_before_tonnes": int(b["supply_before_raw"]) / 1e18,
        "supply_after_tonnes": int(b["supply_after_raw"]) / 1e18,
        "share_of_supply_before_block_pct": burn["shares"]["of_supply_before_block_pct"],
        "share_of_supply_at_month_start_pct": burn["shares"]["of_supply_at_month_start_pct"],
        "month_start_tonnes": month["start"]["tonnes"],
        "month_end_tonnes": month["end"]["tonnes"],
        "month_decline_tonnes": month["decline_tonnes"],
        "share_of_month_decline_pct": burn["shares"]["of_month_decline_pct"],
        "month_blocks_with_supply_change": burn["event_cross_check"]["blocks_with_supply_change_in_month"],
        "second_largest_decrease": {"block": second["block"], "tonnes": -second["net_tonnes"],
                                    "timestamp": second_ts.isoformat()},
        "burner": {
            "address": safe["address"], "is_safe": safe["is_safe"],
            "safe_version": safe.get("version"),
            "threshold": safe.get("threshold"), "owners": len(safe.get("owners", [])),
            "multisig": f"{safe['threshold']}-of-{len(safe['owners'])}" if safe["is_safe"] else None,
            "bct_received_from_treasury_tonnes": sum(
                v["tonnes"] for v in burn["burner_bct_inflows_since"]["by_sender"].values()
                if v["is_treasury"]),
        },
        "treasury": {
            "address": tr["address"],
            "manager_before_block": tr["manager_before_block"],
            "manager_at_block": tr["manager_at_block"],
            "burner_is_manager": (tr["manager_before_block"] == safe["address"]
                                  == tr["manager_at_block"]),
            "burner_roles": sorted(k.split("(")[0] for k, v in
                                   tr["roles_for_burner_at_block"].items() if v is True),
            "bct_returned_to_treasury_in_tx_tonnes": sum(x["tonnes"] for x in side_transfers
                                                         if x["to"] == tr["address"]),
        },
        "markets": {
            "registry_pools_touched": burn["markets"]["registry_pools_touched"],
            "swap_events": len(burn["markets"]["swap_events"]),
            "swap_signatures_checked": len(burn["markets"]["swap_signatures_checked"]),
            "no_market": not burn["markets"]["registry_pools_touched"]
                         and not burn["markets"]["swap_events"],
        },
        "credits": {
            "tco2_tokens": sum(c["distinct_tco2_tokens"] for c in credits),
            "tonnes_leaving_pool": sum(c["tonnes"] for c in credits),
            "tonnes_to_burner": sum(c["tonnes_by_destination"].get(safe["address"], 0.0)
                                    for c in credits),
            "tonnes_retired_in_tx": sum(c["tonnes_burned_in_same_tx"] for c in credits),
            "burner_held_at_block_tonnes": held["burner_tonnes_at_block"],
            "burner_held_at_end_block_tonnes": held["burner_tonnes_at_end_block"],
            "outflows": [{"to": k, **v} for k, v in outflows.items()],
            # Where the credits sat at the end of the window, and whether any
            # were retired there: a TCO2 retirement burns the token, so only a
            # transfer to the zero address records one on-chain.
            "destinations_at_end_block": held.get("destinations_at_end_block", {}),
            "tonnes_retired_by_end_block": sum(
                v["retired_tonnes_by_end_block"]
                for v in held.get("destinations_at_end_block", {}).values()),
            "outflow_tonnes": out_tonnes,
            "outflow_to_zero_address_tonnes": sum(v["tonnes"] for v in outflows.values()
                                                  if v["is_zero_address"]),
            "outflow_minus_burn_and_second_decrease_tonnes":
                out_tonnes - b["decrease_tonnes"] + second["net_tonnes"],
            "end_block": burn["end_block"],
        },
    }


# --------------------------------------------------------------- pool value

def swap_file(key: str) -> Path:
    return Path(data(f"swaps_{key}.csv.gz"))


def daily_value(tok: str, tonnes: pd.Series, vendor: pd.DataFrame | None) -> pd.DataFrame:
    d = trades(VALUED[tok], full_window=True).set_index("timestamp").price
    g = d.resample("1D")
    med, count = g.median(), g.count()
    days = pd.date_range(med.index[0], tonnes.index[-1].tz_localize("UTC"), freq="D")
    med, count = med.reindex(days), count.reindex(days, fill_value=0)
    out = pd.DataFrame({"date": days.tz_localize(None), "token": tok})
    out["tonnes"] = tonnes.reindex(out.date).to_numpy()
    out["trades"] = count.astype(int).to_numpy()
    out["price_median"] = med.to_numpy()
    out["price"] = med.ffill().to_numpy()
    out["price_carried"] = out.trades.eq(0).to_numpy()
    out["value_usd"] = out.tonnes * out.price
    out["in_sample"] = days < SAMPLE_END
    out["vendor_usd_defillama"] = (vendor[tok].reindex(out.date).to_numpy()
                                   if vendor is not None and tok in vendor else float("nan"))
    return out


def point(row: pd.Series) -> dict:
    return {"date": iso_day(row.date), "value_usd": row.value_usd, "price": row.price,
            "tonnes": row.tonnes, "price_carried": bool(row.price_carried)}


def fall(frm: pd.Series, to: pd.Series) -> dict:
    return {"from": point(frm), "to": point(to),
            "fall_pct": 100 * (1 - to.value_usd / frm.value_usd),
            "price_ratio": to.price / frm.price,
            "tonnes_ratio": to.tonnes / frm.tonnes}


def value_numbers(v: pd.DataFrame, burn_ts: pd.Timestamp) -> dict:
    ref_day = burn_ts.tz_localize(None).floor("D") - pd.Timedelta(days=1)
    out = {"reference_day": iso_day(ref_day), "definition":
           "value = end-of-day totalSupply tonnes x median cleaned-swap price that day "
           "(last median carried forward on days without a swap); maxima over days "
           "before the reference day, the last complete UTC day before the burn block"}
    for tok, g in v.groupby("token"):
        g = g.dropna(subset=["value_usd"]).set_index("date", drop=False)
        ref = g.loc[ref_day]
        before = g[g.date < ref_day]
        y2022 = g[g.date.dt.year == 2022]
        out[tok] = {
            "first_valued_day": iso_day(g.date.iloc[0]),
            "days_valued": len(g),
            "days_price_carried": int(g.price_carried.sum()),
            "days_price_carried_before_reference": int(before.price_carried.sum()),
            "from_launch_max": fall(before.loc[before.value_usd.idxmax()], ref),
            "from_2022_max": fall(y2022.loc[y2022.value_usd.idxmax()], ref),
            "last_in_sample": point(g[g.in_sample].iloc[-1]),
            "last": point(g.iloc[-1]),
        }
        both = g.dropna(subset=["vendor_usd_defillama"])
        both = both[both.vendor_usd_defillama > 0]
        out[tok]["vendor_comparison"] = {
            "common_days": len(both),
            "last_vendor_day": iso_day(both.date.iloc[-1]) if len(both) else None,
            "median_chain_over_vendor": float((both.value_usd / both.vendor_usd_defillama).median())
                                        if len(both) else None,
        }
    return out


# ------------------------------------------------------------------- figure

def plot(v: pd.DataFrame, tonnes: pd.DataFrame, peaks: dict, burn: dict, val: dict) -> str:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    burn_day = pd.Timestamp(burn["timestamp"]).tz_localize(None)

    for tok in VALUED:
        g = v[v.token == tok].dropna(subset=["value_usd"])
        left.plot(g.date, g.value_usd / 1e6, color=SERIES[tok], lw=1.4, label=tok)
        t = tonnes[tok]
        t = t[t.index >= g.date.iloc[0] - pd.Timedelta(days=30)]
        right.plot(t.index, t / 1e6, color=SERIES[tok], lw=1.8, label=tok)

    b = val["BCT"]["from_2022_max"]
    for p, offset, ha in ((b["from"], (40, 0), "left"), (b["to"], (-6, 40), "right")):
        x = pd.Timestamp(p["date"])
        left.plot([x], [p["value_usd"] / 1e6], "o", ms=4, color=INK)
        left.annotate(f"${p['value_usd'] / 1e6:,.1f}M\n{x:%d %b %Y}", (x, p["value_usd"] / 1e6),
                      xytext=offset, textcoords="offset points",
                      fontsize=8, color=INK2, ha=ha, va="center")
    left.set_yscale("log")
    style(left, "Pool value", f"tonnes × daily median swap price; BCT fall "
                              f"{b['fall_pct']:.1f}% from the 2022 maximum", "USD, millions (log)")
    left.legend(frameon=False, loc="lower left", fontsize=9)

    pk = peaks["BCT"]
    x = pd.Timestamp(pk["peak_timestamp"]).tz_localize(None)
    right.plot([x], [pk["peak_tonnes"] / 1e6], "o", ms=4, color=INK)
    right.annotate(f"peak {pk['peak_tonnes'] / 1e6:.2f} Mt\n{x:%d %b %Y}",
                   (x, pk["peak_tonnes"] / 1e6), xytext=(-14, -10), textcoords="offset points",
                   fontsize=8, color=INK2, ha="right", va="top")
    right.axvline(burn_day, color=MUTED, lw=0.9, ls=":")
    right.annotate(f"burn {burn['tonnes'] / 1e6:.2f} Mt\n{burn_day:%d %b %Y}\nno swap",
                   (burn_day, burn["supply_after_tonnes"] / 2e6), xytext=(-6, 0),
                   textcoords="offset points", fontsize=8, color=INK2, ha="right", va="center")
    right.set_ylim(bottom=0)
    style(right, "Pool inventory", "totalSupply(), tonnes of credits in the pool",
          "tonnes, millions")
    left.axvline(burn_day, color=MUTED, lw=0.9, ls=":")
    for ax in (left, right):
        ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    path = figure("fig7_depth.png")
    save(fig, path)
    return path


# --------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--no-prices", action="store_true",
                    help="write only what does not need swap prices")
    args = ap.parse_args()

    summary, burn_raw = load_json("inventory_summary.json"), load_json("burn.json")
    monthly = monthly_table()
    tonnes = end_of_day_tonnes()
    peaks = peak_numbers(summary, monthly)
    burn = burn_numbers(burn_raw, monthly)
    print(f"  wrote {table('inventory_monthly', monthly)}")

    out = {"peak": peaks, "burn": burn}
    missing = [str(swap_file(k)) for k in VALUED.values() if not swap_file(k).exists()]
    if args.no_prices or missing:
        reason = ("--no-prices" if args.no_prices else "missing " + ", ".join(missing))
        out["value"] = {"status": f"not computed: {reason}"}
        print(f"  wrote {numbers('inventory', out)}")
        if args.no_prices:
            return
        sys.exit(f"pool value and fig7_depth not computed: {reason} "
                 f"(run klimafrac.fetch.swaps, or pass --no-prices)")

    try:
        vendor = load_depth(units="usd", clean=True)
    except FileNotFoundError:
        vendor = None
    v = pd.concat([daily_value(tok, tonnes[tok], vendor) for tok in VALUED], ignore_index=True)
    out["value"] = value_numbers(v, pd.Timestamp(burn["timestamp"]))
    print(f"  wrote {table('inventory_value_daily', v.assign(date=v.date.dt.strftime('%Y-%m-%d')))}")
    print(f"  wrote {numbers('inventory', out)}")
    print(f"  wrote {plot(v, tonnes, peaks, burn, out['value'])}")


if __name__ == "__main__":
    main()
