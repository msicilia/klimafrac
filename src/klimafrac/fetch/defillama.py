"""
DefiLlama requests, stored verbatim before anything is derived from them.

DefiLlama recomputes its history, so every response is written byte for
byte under data/raw/defillama/<dataset>/<retrieved YYYY-MM-DD>/ with a
manifest of the requests, and CSVs are derived only from a stored snapshot.

Chart series (https://coins.llama.fi/chart/<coin>) are requested on a
regular grid of instants up to 2026-07-26 00:00 UTC. Each returned point is
assigned to the instant it was requested for, the run failing if the
assignment is ambiguous; an instant with no point is a missing bar, counted
in data/vendor_manifest.json; an empty page is requested once more before
being believed. By default the newest snapshot of a dataset is used or
resumed, --refresh starts one dated today, --snapshot rebuilds from a given
one. A snapshot taken under other request parameters is refused.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from klimafrac.fetch.chain import (WINDOW_END, iso, say, utc_ts, write_csv,
                                   write_json)
from klimafrac.paths import RAW_DIR, data

CHART = "https://coins.llama.fi/chart/{coin}"
END_TS = utc_ts(WINDOW_END)
STEP = {"1h": 3600, "1d": 86400}
PAGE = {"1h": 400, "1d": 200}
WIDTH = {"1h": "30m", "1d": "6h"}
ATTEMPTS = 8
PAUSE = 0.3


class VendorError(RuntimeError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get(url: str, params: dict | None = None) -> bytes:
    delay, last = 2.0, ""
    for _ in range(ATTEMPTS):
        try:
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                json.loads(r.content)
                return r.content
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (400, 404):
                break
        except (requests.RequestException, ValueError) as e:
            last = repr(e)
        time.sleep(delay)
        delay = min(2 * delay, 60.0)
    raise VendorError(f"{url} {params}: {last}")


def write_bytes(content: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def snapshot_dir(dataset: str, snapshot: str | None, refresh: bool) -> Path:
    base = RAW_DIR / "defillama" / dataset
    if snapshot:
        path = base / snapshot
        if not path.exists():
            raise FileNotFoundError(f"no snapshot {path}")
        return path
    existing = sorted(p for p in base.glob("????-??-??") if p.is_dir()) if base.exists() else []
    if refresh or not existing:
        path = base / dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        path.mkdir(parents=True, exist_ok=True)
        return path
    return existing[-1]


def load_manifest(path: Path) -> dict | None:
    f = path / "manifest.json"
    return json.loads(f.read_text()) if f.exists() else None


def update_vendor_manifest(output: str, record: dict) -> None:
    path = Path(data("vendor_manifest.json"))
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[output] = record
    write_json(manifest, path)


# --- chart series ----------------------------------------------------------

def grid_pages(start: int, period: str) -> list[tuple[int, int]]:
    step, page = STEP[period], PAGE[period]
    n = math.ceil((END_TS - start) / step)
    return [(start + k * step, min(page, n - k)) for k in range(0, n, page)]


def spec_of(manifest: dict) -> dict:
    """The part of a stored series specification that defines the request."""
    return {k: {f: s[f] for f in ("coin", "symbol", "start")}
            for k, s in manifest["series"].items()}


def check_manifest(manifest: dict, dataset: str, period: str, spec: dict,
                   path: Path) -> None:
    """Nothing stored is reused until it was fetched under the same definition."""
    want = {"dataset": dataset, "period": period, "search_width": WIDTH[period],
            "window_end_exclusive": iso(END_TS)}
    for key, value in want.items():
        if manifest.get(key) != value:
            raise VendorError(f"{path}: manifest {key} is {manifest.get(key)!r}, "
                              f"the code asks for {value!r}; fetch a new snapshot "
                              f"with --refresh")
    stored = spec_of(manifest)
    current = spec_of({"series": spec})
    if stored != current:
        differ = sorted(set(stored) ^ set(current)) or \
            sorted(k for k in current if stored.get(k) != current[k])
        raise VendorError(f"{path}: the stored series differ from the code for "
                          f"{differ}; fetch a new snapshot with --refresh")


def is_empty_page(content: bytes, path: Path) -> bool:
    """A 200 OK carrying no coin at all -- not the same as a priced gap."""
    coins = json.loads(content).get("coins")
    if coins is None:
        raise VendorError(f"{path}: no 'coins' in the response")
    return not coins


def chart_snapshot(dataset: str, period: str, series: Callable[[], dict],
                   snapshot: str | None = None, refresh: bool = False) -> Path:
    """Fetch (or resume) a snapshot. `series()` -> {label: {coin, symbol, start}}."""
    path = snapshot_dir(dataset, snapshot, refresh)
    manifest = load_manifest(path)
    if snapshot and not manifest:
        raise FileNotFoundError(f"{path} has no manifest")
    spec = series()
    if manifest is None:
        manifest = {"dataset": dataset, "source": CHART, "period": period,
                    "search_width": WIDTH[period], "retrieved": path.name,
                    "window_end_exclusive": iso(END_TS), "started_at": now(),
                    "complete": False,
                    "series": {k: {**v, "start_instant": iso(v["start"])}
                               for k, v in spec.items()},
                    "pages": {}}
        write_json(manifest, path / "manifest.json")
    check_manifest(manifest, dataset, period, spec, path)
    if manifest.get("complete") and not manifest.get("empty_pages_pending", True):
        return path

    for label, s in manifest["series"].items():
        url = CHART.format(coin=s["coin"])
        pages = grid_pages(s["start"], period)
        for i, (start, span) in enumerate(pages):
            name = f"{label}/{start}_{span}.json"
            f = path / name
            params = {"start": start, "span": span, "period": period,
                      "searchWidth": WIDTH[period]}
            entry = manifest["pages"].get(name, {"fetched_at": "unrecorded"})
            if not f.exists():
                write_bytes(get(url, params), f)
                entry = {"fetched_at": now()}
                time.sleep(PAUSE)
            content = f.read_bytes()
            empty = is_empty_page(content, f)
            if empty and not entry.get("empty_retried"):
                again = get(url, params)
                time.sleep(PAUSE)
                entry["empty_retried"] = True
                empty = is_empty_page(again, f)
                if not empty:
                    write_bytes(again, f)
                    content = again
                    entry["empty_on_first_request_only"] = True
            entry.update(url=url, params=params, empty=empty,
                         sha256=hashlib.sha256(content).hexdigest())
            manifest["pages"][name] = entry
            if i % 20 == 19 or i == len(pages) - 1:
                write_json(manifest, path / "manifest.json")
                say(f"  {dataset}/{path.name} {label}: {i + 1}/{len(pages)} pages")
    empty = sorted(n for n, e in manifest["pages"].items() if e.get("empty"))
    manifest.update(complete=True, completed_at=now(), empty_pages_pending=False,
                    empty_pages={"count": len(empty), "pages": empty,
                                 "requested_twice": sorted(
                                     n for n, e in manifest["pages"].items()
                                     if e.get("empty_retried"))})
    write_json(manifest, path / "manifest.json")
    if empty:
        say(f"  {dataset}/{path.name}: {len(empty)} page(s) empty after a second "
            f"request -- those instants carry no vendor price at all")
    return path


def missing_runs(grid: list[int], present: set[int], step: int) -> list[dict]:
    runs, start, n = [], None, 0
    for t in grid + [None]:
        if t is not None and t not in present:
            start, n = (t, 1) if start is None else (start, n + 1)
        elif start is not None:
            runs.append({"first": iso(start), "bars": n})
            start, n = None, 0
    return sorted(runs, key=lambda r: -r["bars"])


def chart_frame(path: Path) -> tuple[pd.DataFrame, dict]:
    manifest = load_manifest(path)
    if not manifest or not manifest.get("complete"):
        raise FileNotFoundError(f"snapshot {path} is incomplete")
    period = manifest["period"]
    step = STEP[period]
    columns, info = [], {}
    for label, s in manifest["series"].items():
        values, max_offset, tie_points, empty_pages = {}, 0, 0, 0
        for start, span in grid_pages(s["start"], period):
            f = path / label / f"{start}_{span}.json"
            body = json.loads(f.read_bytes())
            coins = body.get("coins")
            if coins is None:
                raise ValueError(f"{f}: no 'coins' in response")
            keys = [k for k in coins if k.lower() == s["coin"].lower()]
            if len(coins) != len(keys):
                raise ValueError(f"{f}: unexpected coins {list(coins)}")
            if not keys:
                # An empty page, already requested a second time when the
                # snapshot was taken: the vendor has nothing for these instants.
                empty_pages += 1
                continue
            entry = coins[keys[0]]
            if str(entry.get("symbol", "")).upper() != s["symbol"].upper():
                raise ValueError(f"{f}: symbol {entry.get('symbol')!r}, expected {s['symbol']}")
            ties = []
            for point in entry["prices"]:
                q, r = divmod(point["timestamp"] - start, step)
                if 2 * r == step:          # exactly between two requested instants
                    ties.append((q, point))
                    continue
                k = q if 2 * r < step else q + 1
                instant = start + k * step
                if not 0 <= k < span:
                    raise ValueError(f"{f}: point {point} outside the requested page")
                if instant in values:
                    raise ValueError(f"{f}: two points for {iso(instant)}")
                values[instant] = float(point["price"])
                max_offset = max(max_offset, abs(point["timestamp"] - instant))
            # The vendor returns at most one point per requested instant, searching
            # both sides of it; a point exactly half a step away belongs to the one
            # neighbouring instant that has no other point.
            for q, point in ties:
                free = [start + k * step for k in (q, q + 1)
                        if 0 <= k < span and start + k * step not in values]
                if len(free) != 1:
                    raise ValueError(f"{f}: point {point} is ambiguous between two instants")
                values[free[0]] = float(point["price"])
                max_offset = max(max_offset, step // 2)
                tie_points += 1
        grid = [s["start"] + k * step for k in range(math.ceil((END_TS - s["start"]) / step))]
        runs = missing_runs(grid, set(values), step)
        priced = sorted(values)
        info[label] = {"coin": s["coin"], "first_instant": iso(grid[0]),
                       "last_instant": iso(grid[-1]), "bars": len(grid),
                       "priced": len(values), "missing": len(grid) - len(values),
                       "first_priced": iso(priced[0]) if priced else None,
                       "last_priced": iso(priced[-1]) if priced else None,
                       "max_offset_seconds": max_offset,
                       "points_exactly_half_a_step_away": tie_points,
                       "empty_pages": empty_pages,
                       "longest_missing_runs": runs[:10]}
        columns.append(pd.Series([values.get(t, float("nan")) for t in grid],
                                 index=pd.to_datetime(grid, unit="s"), name=label))
    frame = pd.concat(columns, axis=1).sort_index()
    frame.index.name = "date"
    return frame, info


def build_chart_csv(dataset: str, period: str, series: Callable[[], dict],
                    output: str, snapshot: str | None, refresh: bool) -> pd.DataFrame:
    path = chart_snapshot(dataset, period, series, snapshot, refresh)
    frame, info = chart_frame(path)
    write_csv(frame, data(output), index=True)
    update_vendor_manifest(output, {
        "snapshot": str(path.relative_to(RAW_DIR.parent)), "retrieved": path.name,
        "source": CHART, "period": period, "search_width": WIDTH[period],
        "window_end_exclusive": iso(END_TS),
        "convention": "value at instant t = vendor price point returned for t",
        "series": info, "built_at": now()})
    for label, i in info.items():
        say(f"  {output} {label:6s} {i['priced']:,}/{i['bars']:,} bars priced, "
            f"{i['missing']:,} missing ({i['empty_pages']} empty pages), "
            f"first priced {i['first_priced']}")
    return frame


# --- hourly against daily --------------------------------------------------

def hourly_daily_check(daily_output: str, hourly_output: str) -> dict | None:
    """Compare the two snapshots at the instants they share.

    Both series report the vendor's answer for 00:00 UTC, one asked with a 6h
    search width and one with 30m, so they need not agree to the cent. What
    the check establishes is that they describe the same asset over the same
    grid: a bar priced in one and missing in the other, or a large relative
    difference, is a property of the vendor worth recording rather than
    discovering later as a discrepancy between two tables. Free: both
    snapshots are already on disk.
    """
    day, hour = Path(data(daily_output)), Path(data(hourly_output))
    if not (day.exists() and hour.exists()):
        say(f"  hourly/daily check skipped: {day.name} or {hour.name} not built yet")
        return None
    d = pd.read_csv(day, parse_dates=["date"]).set_index("date").sort_index()
    h = pd.read_csv(hour, parse_dates=["date"]).set_index("date").sort_index()
    h = h[h.index.hour.isin([0]) & (h.index.minute == 0) & (h.index.second == 0)]
    shared = sorted(set(d.columns) & set(h.columns))
    instants = d.index.intersection(h.index)
    if not shared or not len(instants):
        raise VendorError(f"{day.name} and {hour.name} share no series or no "
                          f"instant: they cannot describe the same window")
    out = {"daily": daily_output, "hourly": hourly_output,
           "shared_instants": int(len(instants)), "checked_at": now(), "series": {}}
    for label in shared:
        a, b = d.loc[instants, label], h.loc[instants, label]
        both = a.notna() & b.notna()
        rel = ((a[both] - b[both]).abs() / a[both].abs().where(a[both] != 0)).dropna()
        out["series"][label] = {
            "both_priced": int(both.sum()),
            "daily_only": int((a.notna() & b.isna()).sum()),
            "hourly_only": int((a.isna() & b.notna()).sum()),
            "neither": int((a.isna() & b.isna()).sum()),
            "median_relative_difference": float(rel.median()) if len(rel) else None,
            "max_relative_difference": float(rel.max()) if len(rel) else None,
            "instants_differing_by_more_than_1pct": int((rel > 0.01).sum()),
        }
        s = out["series"][label]
        say(f"  {label:6s} 00:00 UTC: {s['both_priced']:,} instants priced in both, "
            f"{s['daily_only']:,} daily only, {s['hourly_only']:,} hourly only, "
            f"{s['instants_differing_by_more_than_1pct']:,} differing by over 1%")
    update_vendor_manifest(f"check_hourly_vs_daily:{daily_output}", out)
    return out


def cli(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--snapshot", help="rebuild from data/raw/defillama/.../YYYY-MM-DD")
    ap.add_argument("--refresh", action="store_true",
                    help="start a new snapshot dated today")
    return ap.parse_args()
