"""
DefiLlama's protocol records for Toucan and KlimaDAO, stored verbatim.

Sources https://api.llama.fi/protocol/toucan-protocol and
https://api.llama.fi/protocol/klimadao. Only the Toucan USD value of carbon
locked is used, as a comparison column; tonnes come from the chain
(fetch/inventory.py) and pool value from reserves (fetch/reserves.py).
Window: vendor dates before 2026-07-26.

Outputs: data/depth.csv (date, chain, token, usd, tonnes, as reported).
Raw under data/raw/defillama/protocol_<name>/<retrieved>/. `load_depth()`
reads a stored snapshot; `clean=True` applies the masks derived from it.

Usage:
    uv run python -m klimafrac.fetch.depth [--refresh | --snapshot YYYY-MM-DD]
"""

import datetime as dt
import hashlib
import json
import math

import pandas as pd

from klimafrac.fetch.chain import WINDOW_END, say, write_csv, write_json
from klimafrac.fetch.defillama import (VendorError, cli, get, load_manifest, now,
                                       snapshot_dir, write_bytes)
from klimafrac.paths import data

DATASETS = {
    "protocol_toucan-protocol": "https://api.llama.fi/protocol/toucan-protocol",
    "protocol_klimadao": "https://api.llama.fi/protocol/klimadao",
}
TOUCAN = "protocol_toucan-protocol"

GLITCH_DROP = -0.5
GLITCH_RECOVERY = 0.5
# A vendor USD series detaches from its own quantity before it stops: the
# implied price (usd / tonnes) moves day after day while the tonnage does not.
# The mask is the final run of such days, and it is taken only if the implied
# price more than doubles or halves across the run -- ordinary price movement
# does not qualify, and a series that ends cleanly is not masked at all.
DAY_JUMP = 0.15          # |log change| in the implied price within one day
FLAT_QUANTITY = 0.01     # |log change| in tonnes below which quantity is flat
RUN_RATIO = 2.0          # the run's cumulative implied-price factor


def snapshot(dataset: str, snapshot_date: str | None = None, refresh: bool = False):
    path = snapshot_dir(dataset, snapshot_date, refresh)
    response = path / "response.json"
    url = DATASETS[dataset]
    if not response.exists():
        if snapshot_date:
            raise FileNotFoundError(f"{response} missing")
        write_bytes(get(url), response)
        write_json({"dataset": dataset, "url": url, "retrieved": path.name,
                    "fetched_at": now(),
                    "sha256": hashlib.sha256(response.read_bytes()).hexdigest()},
                   path / "manifest.json")
    manifest = load_manifest(path)
    if manifest is None:
        raise FileNotFoundError(f"{path} has no manifest")
    if manifest.get("url") != url or manifest.get("dataset") != dataset:
        raise VendorError(f"{path}: manifest records {manifest.get('url')!r} for "
                          f"{manifest.get('dataset')!r}, the code asks for {url!r} "
                          f"for {dataset!r}; fetch a new snapshot with --refresh")
    return path


def latest(dataset: str = TOUCAN, snapshot_date: str | None = None):
    path = snapshot_dir(dataset, snapshot_date, refresh=False)
    if not (path / "response.json").exists():
        raise FileNotFoundError(f"no {dataset} snapshot under {path.parent}; "
                                f"run klimafrac.fetch.depth")
    return path


def payload(dataset: str = TOUCAN, snapshot_date: str | None = None) -> dict:
    return json.loads((latest(dataset, snapshot_date) / "response.json").read_bytes())


def long_frame(payload: dict) -> pd.DataFrame:
    sources = [("all", payload)] + list(payload.get("chainTvls", {}).items())
    rows = []
    for chain, block in sources:
        for key, col in (("tokensInUsd", "usd"), ("tokens", "tonnes")):
            for point in block.get(key) or []:
                date = dt.datetime.fromtimestamp(point["date"], dt.timezone.utc).date()
                for token, value in (point.get("tokens") or {}).items():
                    rows.append((point["date"], str(date), chain, token, col, value))
    df = pd.DataFrame(rows, columns=["ts", "date", "chain", "token", "field", "value"])
    df = df[pd.to_datetime(df.date) < WINDOW_END.tz_localize(None)]
    df = df.sort_values("ts").groupby(["date", "chain", "token", "field"])["value"].last()
    return df.unstack("field").reset_index()[["date", "chain", "token", "usd", "tonnes"]]


def _drop_glitches(series: pd.Series) -> pd.Series:
    """Mask a one-day collapse that fully reverses the following day."""
    change = series.pct_change(fill_method=None)
    glitch = (change < GLITCH_DROP) & (change.shift(-1) > GLITCH_RECOVERY)
    return series.mask(glitch)


def usd_mask(usd: pd.DataFrame, tonnes: pd.DataFrame) -> dict:
    """Per token: from which date the USD series stopped tracking its quantity.

    Returns {token: {valid_to, points_removed, ...}}, with valid_to None where
    the series ends without detaching, so nothing is masked.
    """
    out = {}
    for token in usd.columns:
        pair = pd.DataFrame({"usd": usd[token],
                             "tonnes": tonnes[token] if token in tonnes else float("nan")})
        pair = pair[(pair.usd > 0) & (pair.tonnes > 0)].dropna().sort_index()
        record = {"valid_to": None, "points_removed": 0, "days_in_run": 0,
                  "implied_price_ratio_over_run": None,
                  "last_day_with_a_value": None}
        if len(pair) < 3:
            out[token] = record
            continue
        record["last_day_with_a_value"] = pair.index[-1].strftime("%Y-%m-%d")
        implied = pair.usd / pair.tonnes
        price_step = (implied / implied.shift()).apply(
            lambda r: abs(math.log(r)) if r > 0 else float("inf"))
        quantity_step = (pair.tonnes / pair.tonnes.shift()).apply(
            lambda r: abs(math.log(r)) if r > 0 else float("inf"))
        detached = (price_step > DAY_JUMP) & (quantity_step < FLAT_QUANTITY)
        k = len(pair)
        while k > 1 and bool(detached.iloc[k - 1]):
            k -= 1
        if k < len(pair):                       # a run reaching the last day
            ratio = implied.iloc[-1] / implied.iloc[k - 1]
            if ratio >= RUN_RATIO or ratio <= 1 / RUN_RATIO:
                cutoff = pair.index[k - 1]
                record.update(
                    valid_to=cutoff.strftime("%Y-%m-%d"),
                    days_in_run=len(pair) - k,
                    implied_price_ratio_over_run=float(ratio),
                    points_removed=int((usd.index > cutoff).sum()
                                       - usd[token][usd.index > cutoff].isna().sum()))
        out[token] = record
    return out


def load_depth(units: str = "usd", clean: bool = True, chain: str = "all",
               snapshot_date: str | None = None) -> pd.DataFrame:
    """Daily value (units="usd") or vendor tonnes (units="tonnes") per token.

    Reads the stored snapshot only. clean=True masks (a) single-day collapses
    of more than 50% that recover by more than 50% the next day, and (b) the
    tail of a USD series that detached from its own quantity, the cutoff being
    derived from the snapshot by `usd_mask` rather than fixed in advance.
    """
    df = long_frame(payload(TOUCAN, snapshot_date))
    df = df[df.chain == chain]
    wide = {field: df.pivot(index="date", columns="token", values=field).sort_index()
            for field in ("usd", "tonnes")}
    for field, frame in wide.items():
        frame.index = pd.to_datetime(frame.index)
    out = wide[{"usd": "usd", "tonnes": "tonnes"}[units]].copy()
    if clean:
        out = out.apply(_drop_glitches)
        if units == "usd":
            for token, record in usd_mask(wide["usd"], wide["tonnes"]).items():
                if record["valid_to"] and token in out.columns:
                    out.loc[out.index > pd.Timestamp(record["valid_to"]), token] = float("nan")
    return out


def main() -> None:
    args = cli(__doc__.strip().split("\n")[0])
    paths = {name: snapshot(name, args.snapshot, args.refresh) for name in DATASETS}
    for name, path in paths.items():
        size = (path / "response.json").stat().st_size
        say(f"  {name}: {path} ({size / 1e6:.2f} MB)")

    df = long_frame(json.loads((paths[TOUCAN] / "response.json").read_bytes()))
    write_csv(df, data("depth.csv"))
    say(f"  wrote depth.csv ({len(df):,} rows)")
    for (chain, token), g in df.groupby(["chain", "token"]):
        usd = g.dropna(subset=["usd"])
        say(f"  {chain:8s} {token:5s} {usd.date.min()} .. {usd.date.max()}  "
            f"{len(usd):,} days with a USD value")

    all_chains = df[df.chain == "all"]
    wide = {field: all_chains.pivot(index="date", columns="token", values=field).sort_index()
            for field in ("usd", "tonnes")}
    for frame in wide.values():
        frame.index = pd.to_datetime(frame.index)
    mask = usd_mask(wide["usd"], wide["tonnes"])
    write_json({"dataset": TOUCAN, "snapshot": paths[TOUCAN].name,
                "rule": (f"the final run of days on which the implied price "
                         f"(usd/tonnes) moved by more than {DAY_JUMP} in log while "
                         f"tonnes moved by less than {FLAT_QUANTITY}, taken only if "
                         f"the run changes the implied price by a factor of "
                         f"{RUN_RATIO} or more"),
                "derived_at": now(), "tokens": mask},
               paths[TOUCAN] / "usd_mask.json")
    for token, record in mask.items():
        if record["valid_to"]:
            say(f"  {token:5s} USD value detaches from tonnage after "
                f"{record['valid_to']}: {record['days_in_run']} days, implied price "
                f"x{record['implied_price_ratio_over_run']:.2f}, "
                f"{record['points_removed']} points masked by load_depth(clean=True)")
        else:
            say(f"  {token:5s} USD value tracks its tonnage to the end "
                f"({record['last_day_with_a_value']}): nothing masked")


if __name__ == "__main__":
    main()
