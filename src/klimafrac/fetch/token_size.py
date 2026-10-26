"""
How large each token's asset is, from outside the chain.

A pool's idleness is read from the chain. Whether its token is a large asset
traded mostly elsewhere or a small one is not on the chain at all, and
analysis/size.py needs it. Two public sources supply it.

Snapshot (the measure). CoinMarketCap's historical ranking of SNAPSHOT, the
last weekly snapshot before the first hourly series begins (2021-10-08), so
that size is measured before the trading it is set against
(https://coinmarketcap.com/historical/20211003/), read from the listing feed
behind that page. The ranking of CHECK_SNAPSHOT, the first weekly snapshot
after the date by which the candidate pools had to exist, four months into
the sample, is kept as a check (`check_rank`). A token is tied to a ranked
asset by its contract and never by its name: the candidates are the ranked assets carrying the token's
on-chain symbol, and the one kept is the one whose record lists the token's
Polygon address, or else the Ethereum address that Polygon's own bridge names
as the token's origin (RootChainManager.childToRootToken, read on Ethereum).
The second route covers bridged copies of ranked assets whose records list
only the Ethereum contract. A token with no such asset is unlisted. Three tokens wrap a coin that has no
contract of its own -- WETH, WBTC and WPOL -- and take the rank of the coin
they wrap (UNDERLYING).

Late check. CoinGecko's public interface, which resolves a Polygon contract
address directly, at LATE_DATE, the last day of the estimation sample. It
serves only the past year without a paid plan, so it cannot supply the
snapshot; and a size measured after the trading it is set against is partly a
result of that trading. It is here to check the measure, not to be it.

Every response is stored as received under data/raw/token_size/, and a stored
response is never requested again, so the table can be rebuilt offline.

Output data/token_size.csv, one row per pool token:
  token, pool, base, role
  cmc_id, cmc_name, candidates     the ranked asset matched, and how many shared the symbol
  matched_by                       "address", "bridge", "underlying" or "" (unlisted)
  bridged_from                     the Ethereum token the bridge maps it to, if any
  snapshot_rank, snapshot_volume,  rank, worldwide 24-hour volume in dollars,
  snapshot_market_cap,             market value where the ranking states one,
  snapshot_markets                 and number of markets, on SNAPSHOT
  snapshot_rank_filled             the rank, or one past the last ranked asset if unlisted
  check_rank, check_volume,        the same on CHECK_SNAPSHOT
  check_market_cap, check_rank_filled
  sensitivity_id, sensitivity_rank the asset in KNOWN_MISMATCH and its rank on SNAPSHOT
  late_id, late_market_cap,        the second source's asset and its figures
  late_volume                      on LATE_DATE

Usage:
    uv run python -m klimafrac.fetch.token_size
"""

import json
import time
import urllib.error
import urllib.request

import pandas as pd

from klimafrac.fetch.chain import calldata, say, write_csv, write_json
from klimafrac.fetch.pools import POOLS, USDC_E_POLYGON
from klimafrac.paths import data, raw

MODULE = "token_size"
SNAPSHOT = "2021-10-03"          # the last weekly snapshot before the first hourly series (2021-10-08)
CHECK_SNAPSHOT = "2022-02-13"    # the first after the pools' existence date; a check
LATE_DATE = "26-05-2026"
# Contracts the ranking's own records misplace. The record of Telcoin (2394)
# lists the same address for Ethereum, Polygon and Base, so the rule above
# cannot tie the Polygon pool token to it, although the second source
# resolves the contract to Telcoin. The rule is kept; the asset is recorded
# as `sensitivity_id` so that analysis/size.py can report the test with the
# token coded as listed.
KNOWN_MISMATCH = {"0xdf7837de1f2fa4631d716cf2502f8b230f1dcc32": 2394}
ROLES = ("carbon", "control", "thin", "active")
CMC = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency"
ETHEREUM = "https://ethereum-rpc.publicnode.com"          # a public node, no key
ROOT_CHAIN_MANAGER = "0xa0c68c638235ee32657e8f720a23cec1bfc77c77"   # Polygon PoS bridge, on Ethereum
GECKO = "https://api.coingecko.com/api/v3"
# base token address: (symbol of the coin it wraps in the ranking, that coin's
# name there, its id in the second source)
UNDERLYING = {
    "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619": ("ETH", "Ethereum", "ethereum"),
    "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6": ("BTC", "Bitcoin", "bitcoin"),
    "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270": ("MATIC", "Polygon", "polygon-ecosystem-token"),
}


def fetch(url: str, path, pause: float, attempts: int = 8):
    """The stored response for `url`, requesting it only if it is not stored."""
    if path.exists():
        return json.loads(path.read_text())
    wait = 30.0
    for _ in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                       "accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as r:
                body = json.load(r)
            write_json(body, path)
            time.sleep(pause)
            return body
        except urllib.error.HTTPError as e:
            if e.code == 404:
                write_json({"not_found": True}, path)
                time.sleep(pause)
                return {"not_found": True}
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(wait)
        wait = min(2 * wait, 300.0)
    raise RuntimeError(f"no answer from {url.split('?')[0]} after {attempts} attempts")


def ranking(date: str = SNAPSHOT) -> pd.DataFrame:
    rows, start = [], 1
    while True:
        page = fetch(f"{CMC}/listings/historical?date={date}&limit=1000&start={start}"
                     "&convertId=2781", raw(MODULE, f"ranking_{date}_{start}.json"), 1.5)
        part = page.get("data") or []
        rows += part
        if len(part) < 1000:
            break
        start += 1000
    out = pd.DataFrame([{
        "cmc_id": r["id"], "symbol": r["symbol"], "cmc_name": r["name"], "rank": r["cmcRank"],
        "volume": r["quotes"][0].get("volume24h") if r.get("quotes") else None,
        "market_cap": r["quotes"][0].get("marketCap") if r.get("quotes") else None,
        "markets": r.get("numMarketPairs")} for r in rows])
    if out.cmc_id.duplicated().any() or out["rank"].isna().any():
        raise ValueError("the ranking repeats an asset or leaves one unranked")
    return out


def addresses(cmc_id: int, platform: str) -> set[str]:
    """The contract addresses a ranked asset's record lists on one platform."""
    detail = fetch(f"{CMC}/detail?id={cmc_id}", raw(MODULE, "assets", f"{cmc_id}.json"), 1.5)
    platforms = (detail.get("data") or {}).get("platforms") or []
    return {p["contractAddress"].lower() for p in platforms
            if p.get("contractPlatform") == platform and p.get("contractAddress")}


def bridged_from(base: str) -> str:
    """The Ethereum token Polygon's bridge maps `base` to, or "" if it maps to none."""
    path = raw(MODULE, "bridge", f"{base}.json")
    if path.exists():
        answer = json.loads(path.read_text())
    else:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_call", "params": [
            {"to": ROOT_CHAIN_MANAGER, "data": calldata("childToRootToken(address)", base)},
            "latest"]}).encode()
        request = urllib.request.Request(ETHEREUM, data=body,
                                         headers={"content-type": "application/json",
                                                  "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as r:
            answer = json.load(r)
        if "result" not in answer or len(answer["result"]) != 66:
            raise RuntimeError(f"bridge lookup for {base}: {answer}")
        write_json(answer, path)
        time.sleep(0.5)
    root = "0x" + answer["result"][-40:].lower()
    return root if int(root, 16) else ""


def late(base: str) -> dict:
    if base in UNDERLYING:
        coin = UNDERLYING[base][2]
    else:
        found = fetch(f"{GECKO}/coins/polygon-pos/contract/{base}",
                      raw(MODULE, "late", f"contract_{base}.json"), 7.0)
        if found.get("not_found") or "id" not in found:
            return {"late_id": "", "late_market_cap": None, "late_volume": None}
        coin = found["id"]
    day = fetch(f"{GECKO}/coins/{coin}/history?date={LATE_DATE}&localization=false",
                raw(MODULE, "late", f"history_{coin}_{LATE_DATE}.json"), 7.0)
    market = day.get("market_data") or {}
    return {"late_id": coin, "late_market_cap": (market.get("market_cap") or {}).get("usd"),
            "late_volume": (market.get("total_volume") or {}).get("usd")}


def main() -> None:
    listed = ranking(SNAPSHOT)
    check = ranking(CHECK_SNAPSHOT)
    say(f"ranking of {SNAPSHOT}: {len(listed):,} assets; of {CHECK_SNAPSHOT}: {len(check):,}")
    # An asset is identified once, by contract, among the assets either
    # ranking carries under the token's symbol; its rank is then read from
    # each ranking separately, and is missing where the asset is not in it.
    both = pd.concat([listed, check]).drop_duplicates("cmc_id")
    by_id = {"snapshot": listed.set_index("cmc_id"), "check": check.set_index("cmc_id")}
    pools = [p for p in POOLS if p.chain == "polygon" and p.role in ROLES
             and p.quote == USDC_E_POLYGON]
    rows = []
    for p in pools:
        row = {"token": p.symbol, "pool": p.key, "base": p.base, "role": p.role,
               "cmc_id": None, "cmc_name": "", "candidates": 0, "matched_by": "",
               "bridged_from": "", "sensitivity_id": KNOWN_MISMATCH.get(p.base)}
        if p.base in UNDERLYING:
            symbol, name, _ = UNDERLYING[p.base]
            hit = both[(both.symbol == symbol) & (both.cmc_name == name)]
            if len(hit) != 1:
                raise ValueError(f"{p.symbol}: {len(hit)} ranked assets named {name} ({symbol})")
            match, row["matched_by"], row["candidates"] = hit.iloc[0], "underlying", 1
        else:
            candidates = both[both.symbol.str.upper() == p.symbol.upper()]
            row["candidates"] = len(candidates)
            match = None
            row["bridged_from"] = root = bridged_from(p.base)
            for route, platform, wanted in (("address", "Polygon", p.base),
                                            ("bridge", "Ethereum", root)):
                for c in candidates.sort_values("rank").itertuples():
                    if wanted and wanted in addresses(c.cmc_id, platform):
                        match = both[both.cmc_id == c.cmc_id].iloc[0]
                        row["matched_by"] = route
                        break
                if match is not None:
                    break
        if match is not None:
            row.update({"cmc_id": int(match.cmc_id), "cmc_name": match.cmc_name})
            for which, frame in by_id.items():
                if int(match.cmc_id) in frame.index:
                    r = frame.loc[int(match.cmc_id)]
                    row.update({f"{which}_rank": int(r["rank"]), f"{which}_volume": r.volume,
                                f"{which}_market_cap": r.market_cap})
            row["snapshot_markets"] = match.markets
        if row["sensitivity_id"] in by_id["snapshot"].index:
            row["sensitivity_rank"] = int(by_id["snapshot"].loc[row["sensitivity_id"]]["rank"])
        row.update(late(p.base))
        rows.append(row)
        say(f"  {p.symbol:9s} {row['matched_by'] or 'unlisted':10s} "
            f"({row['candidates']} with the symbol)  late source: {row['late_id'] or '-'}")
    out = pd.DataFrame(rows)
    for which, frame in by_id.items():
        out[f"{which}_rank_filled"] = out[f"{which}_rank"].fillna(int(frame["rank"].max()) + 1)
    write_csv(out, data("token_size.csv"))
    say(f"{len(out)} tokens: {int((out.matched_by == 'address').sum())} matched by address, "
        f"{int((out.matched_by == 'bridge').sum())} through the bridge, "
        f"{int((out.matched_by == 'underlying').sum())} by the coin they wrap, "
        f"{int((out.matched_by == '').sum())} unlisted; "
        f"{int(out.snapshot_rank.notna().sum())} ranked on {SNAPSHOT}, "
        f"{int(out.check_rank.notna().sum())} on {CHECK_SNAPSHOT}; "
        f"{int(out.late_market_cap.fillna(0).gt(0).sum())} with a market value in the late source")


if __name__ == "__main__":
    main()
