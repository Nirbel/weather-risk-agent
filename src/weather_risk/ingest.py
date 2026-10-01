"""Fetch → parse → store, for live APIs and for the recorded seed snapshot.

Each source fails independently. Failures are recorded in `source_status`, so answers
can say which data is missing instead of silently using less.

CLI:
    uv run python -m weather_risk.ingest                 # incremental refresh, all hubs
    uv run python -m weather_risk.ingest --hub denver    # one hub
    uv run python -m weather_risk.ingest --seed          # load data/seed/ into the DB
"""

import argparse
import asyncio
import gzip
import json
import logging
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from weather_risk.db import Store
from weather_risk.hubs import Hub, load_hubs, load_scoring_config
from weather_risk.scoring.snapshots import refresh_snapshots
from weather_risk.settings import get_settings
from weather_risk.sources import nri, nws, open_meteo, openfema
from weather_risk.sources.http import SourceError, make_client

log = logging.getLogger("weather_risk.ingest")

TTL = {
    "fema_nri": timedelta(days=30),
    "openfema": timedelta(hours=24),
    "open_meteo_forecast": timedelta(hours=1),
    "nws_alerts": timedelta(minutes=5),
}
# Open-Meteo free tier: 600 calls/min. Stay well under it when pulling years of history.
OPEN_METEO_CALLS_PER_MIN = 400


# -- apply: parsed payload → store (shared by live ingest and seed loading) ---------

def apply_archive(store: Store, hub: Hub, payload: dict) -> int:
    rows = open_meteo.parse_daily(payload)
    store.upsert_daily(hub.id, rows)
    store.cache_put(f"grid:{hub.id}", "open_meteo_archive", open_meteo.grid_cell(payload))
    return len(rows)


def apply_forecast(store: Store, hub: Hub, payload: dict, fetched_at: str | None = None) -> None:
    rows = [asdict(r) for r in open_meteo.parse_daily(payload)]
    store.cache_put(f"forecast:{hub.id}", "open_meteo_forecast", {"rows": rows}, fetched_at)


def apply_nws(store: Store, hub: Hub, payload: dict, fetched_at: str | None = None) -> None:
    alerts = [asdict(a) for a in nws.parse_alerts(payload)]
    store.cache_put(f"nws:{hub.id}", "nws_alerts", alerts, fetched_at)


def apply_declarations(store: Store, hub: Hub, payload: dict, fetched_at: str | None = None) -> None:
    decls = [asdict(d) for d in openfema.parse_declarations(payload)]
    store.cache_put(f"openfema:{hub.id}", "openfema", decls, fetched_at)


def apply_nri(store: Store, hubs: list[Hub], pages: list[dict], fetched_at: str | None = None) -> None:
    version, n, by_fips = nri.county_percentiles(pages, [h.county_fips for h in hubs])
    payload = {
        "version": version,
        "n_counties": n,
        "by_fips": {f: {hz: asdict(v) for hz, v in hz_map.items()} for f, hz_map in by_fips.items()},
    }
    store.cache_put("nri", "fema_nri", payload, fetched_at)


# -- live refresh ------------------------------------------------------------

def _fresh(store: Store, key: str, ttl: timedelta) -> bool:
    cached = store.cache_get(key)
    if not cached:
        return False
    return datetime.now(UTC) - datetime.fromisoformat(cached[0]) < ttl


async def _guarded(store: Store, source: str, errors: list[str], coro) -> None:
    try:
        await coro
        store.source_ok(source)
    except (SourceError, ValueError, KeyError) as exc:
        message = str(exc)
        store.source_error(source, message)
        errors.append(message)
        log.warning("%s failed: %s", source, message)


async def refresh_nri(store: Store, client: httpx.AsyncClient, hubs: list[Hub], force: bool) -> None:
    if not force and _fresh(store, "nri", TTL["fema_nri"]):
        return
    apply_nri(store, hubs, await nri.fetch_county_pages(client))


async def refresh_declarations(store: Store, client: httpx.AsyncClient, hub: Hub, since: date, force: bool) -> None:
    if not force and _fresh(store, f"openfema:{hub.id}", TTL["openfema"]):
        return
    apply_declarations(store, hub, await openfema.fetch_declarations(client, hub.county_fips, since))


async def refresh_archive(store: Store, client: httpx.AsyncClient, hub: Hub, start: date, end: date, pace: bool) -> int:
    """Fetch only dates not yet stored. History never changes, so it is never refetched."""
    have = store.daily_range(hub.id)
    ranges = []
    if have is None:
        ranges.append((start, end))
    else:
        if have[0] > start:
            ranges.append((start, have[0] - timedelta(days=1)))
        if have[1] < end:
            ranges.append((have[1] + timedelta(days=1), end))
    fetched = 0
    for a, b in ranges:
        fetched += apply_archive(store, hub, await open_meteo.fetch_archive(client, hub.lat, hub.lon, a, b))
        weight = open_meteo.call_weight((b - a).days + 1)
        if pace and weight > 20:
            await asyncio.sleep(weight * 60 / OPEN_METEO_CALLS_PER_MIN)
    return fetched


async def refresh_near_term(store: Store, client: httpx.AsyncClient, hub: Hub, force: bool, errors: list[str]) -> None:
    if force or not _fresh(store, f"forecast:{hub.id}", TTL["open_meteo_forecast"]):
        async def forecast():
            apply_forecast(store, hub, await open_meteo.fetch_forecast(client, hub.lat, hub.lon))
        await _guarded(store, "open_meteo_forecast", errors, forecast())
    if force or not _fresh(store, f"nws:{hub.id}", TTL["nws_alerts"]):
        async def alerts():
            apply_nws(store, hub, await nws.fetch_active_alerts(client, hub.lat, hub.lon))
        await _guarded(store, "nws_alerts", errors, alerts())


async def run_ingest(
    store: Store,
    client: httpx.AsyncClient,
    *,
    hubs: list[Hub] | None = None,
    today: date | None = None,
    history: bool = True,
    near_term: bool = True,
    force: bool = False,
    pace: bool = True,
) -> list[str]:
    """Refresh data and recompute snapshots. Returns error messages (empty = all sources OK)."""
    hubs = list(hubs or load_hubs())
    today = today or date.today()
    cfg = load_scoring_config()
    errors: list[str] = []
    if history:
        await _guarded(store, "fema_nri", errors, refresh_nri(store, client, list(load_hubs()), force))
        since = cfg["declarations_since"]
        for hub in hubs:
            await _guarded(store, "openfema", errors, refresh_declarations(store, client, hub, since, force))
        start, end = cfg["climatology"]["start"], open_meteo.archive_end(today)
        for hub in hubs:
            log.info("archive %s", hub.id)
            await _guarded(store, "open_meteo_archive", errors, refresh_archive(store, client, hub, start, end, pace))
    if near_term:
        for hub in hubs:
            await refresh_near_term(store, client, hub, force, errors)
    refresh_snapshots(store, today=today)
    return errors


# -- seed snapshot (recorded real responses, see scripts/record_seed.py) -------

def read_gz(path: Path) -> Any:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def write_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", compresslevel=9) as f:
        json.dump(payload, f, separators=(",", ":"))


def load_seed(store: Store, seed_dir: Path, today: date | None = None) -> dict:
    """Load the recorded seed into the store. Freshness is the recording time, not now."""
    manifest = json.loads((seed_dir / "manifest.json").read_text())
    at = manifest["recorded_at"]
    hubs = list(load_hubs())
    apply_nri(store, hubs, read_gz(seed_dir / "nri_pages.json.gz"), at)
    store.source_ok("fema_nri", at)
    for hub in hubs:
        apply_archive(store, hub, read_gz(seed_dir / "archive" / f"{hub.id}.json.gz"))
        apply_forecast(store, hub, read_gz(seed_dir / "forecast" / f"{hub.id}.json.gz"), at)
        apply_nws(store, hub, read_gz(seed_dir / "nws" / f"{hub.id}.json.gz"), at)
        apply_declarations(store, hub, read_gz(seed_dir / "openfema" / f"{hub.id}.json.gz"), at)
    for source in ("open_meteo_archive", "open_meteo_forecast", "nws_alerts", "openfema"):
        store.source_ok(source, at)
    refresh_snapshots(store, today=today or date.fromisoformat(manifest["today"]))
    return manifest


def ensure_data(store: Store, seed_dir: Path) -> bool:
    """Seed an empty database so a fresh install works without the paced history pull."""
    if store.has_weather():
        return False
    load_seed(store, seed_dir)
    return True


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Refresh weather and hazard data.")
    parser.add_argument("--hub", action="append", help="hub id (repeatable); default: all")
    parser.add_argument("--seed", action="store_true", help="load data/seed/ into the database and exit")
    parser.add_argument("--no-history", action="store_true", help="skip archive/NRI/FEMA; refresh near-term only")
    parser.add_argument("--force", action="store_true", help="ignore cache TTLs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = get_settings()
    store = Store(settings.db_path)
    if args.seed:
        manifest = load_seed(store, settings.seed_dir)
        print(f"Seed loaded (recorded {manifest['recorded_at']}).")
        return
    by_id = {h.id: h for h in load_hubs()}
    hubs = [by_id[h] for h in args.hub] if args.hub else list(by_id.values())
    async with make_client(settings.contact_email) as client:
        errors = await run_ingest(store, client, hubs=hubs, history=not args.no_history, force=args.force)
    for hub in hubs:
        rng = store.daily_range(hub.id)
        print(f"{hub.id:12} daily rows {rng[0]} → {rng[1]}" if rng else f"{hub.id:12} no daily rows")
    print("All sources OK." if not errors else f"{len(errors)} source error(s):\n  " + "\n  ".join(errors))


if __name__ == "__main__":
    asyncio.run(_main())
