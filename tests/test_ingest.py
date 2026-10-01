"""Ingest glue: live refresh (mocked HTTP) and seed loading."""

import json
from datetime import date
from pathlib import Path

import httpx
import respx

from weather_risk.db import Store
from weather_risk.hubs import hubs_by_id, load_hubs
from weather_risk.ingest import ensure_data, load_seed, refresh_archive, run_ingest, write_gz
from weather_risk.sources import nri, nws, open_meteo, openfema

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def mock_all_sources(nri_status: int = 200) -> None:
    respx.get(nri.URL).mock(return_value=httpx.Response(nri_status, json=load("nri_counties_sample.json")))
    respx.get(openfema.URL).mock(return_value=httpx.Response(200, json=load("openfema_harris.json")))
    respx.get(open_meteo.ARCHIVE_URL).mock(return_value=httpx.Response(200, json=load("open_meteo_archive_denver.json")))
    respx.get(open_meteo.FORECAST_URL).mock(return_value=httpx.Response(200, json=load("open_meteo_forecast_denver.json")))
    respx.get(nws.ALERTS_URL).mock(return_value=httpx.Response(200, json=load("nws_alerts_sample.json")))


@respx.mock
async def test_run_ingest_stores_every_source_and_snapshots():
    mock_all_sources()
    store = Store(":memory:")
    async with httpx.AsyncClient() as client:
        errors = await run_ingest(store, client, hubs=[hubs_by_id()["denver"]], today=date(2025, 1, 16), pace=False)
    assert errors == []
    assert store.daily_range("denver") == (date(2025, 1, 1), date(2025, 1, 10))
    assert store.cache_get("nri")[1]["version"] == "December 2025"
    assert len(store.cache_get("openfema:denver")[1]) == 16
    assert len(store.cache_get("forecast:denver")[1]["rows"]) == 7
    statuses = store.source_statuses()
    assert all(s["last_error"] is None for s in statuses.values())
    assert "denver" in store.latest_snapshots("exposure")
    assert "denver" in store.latest_snapshots("near_term")


@respx.mock
async def test_failed_source_is_recorded_and_others_continue():
    mock_all_sources(nri_status=400)
    store = Store(":memory:")
    async with httpx.AsyncClient() as client:
        errors = await run_ingest(store, client, hubs=[hubs_by_id()["denver"]], today=date(2025, 1, 16), pace=False)
    assert len(errors) == 1 and "fema_nri" in errors[0]
    assert store.source_statuses()["fema_nri"]["last_error"].startswith("fema_nri: HTTP 400")
    assert store.daily_range("denver") is not None  # archive still ingested
    gaps = store.latest_snapshots("exposure")["denver"]["breakdown"]["data_gaps"]
    assert any("National Risk Index unavailable" in g for g in gaps)


@respx.mock
async def test_archive_fetches_only_missing_dates():
    route = respx.get(open_meteo.ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=load("open_meteo_archive_denver.json"))
    )
    store = Store(":memory:")
    hub = hubs_by_id()["denver"]
    async with httpx.AsyncClient() as client:
        await refresh_archive(store, client, hub, date(2025, 1, 1), date(2025, 1, 10), pace=False)
        await refresh_archive(store, client, hub, date(2025, 1, 1), date(2025, 1, 10), pace=False)  # nothing missing
        await refresh_archive(store, client, hub, date(2025, 1, 1), date(2025, 1, 14), pace=False)
    assert route.call_count == 2
    params = route.calls[1].request.url.params
    assert (params["start_date"], params["end_date"]) == ("2025-01-11", "2025-01-14")


def make_seed(seed: Path) -> None:
    write_gz(seed / "nri_pages.json.gz", [load("nri_counties_sample.json")])
    for hub in load_hubs():
        write_gz(seed / "archive" / f"{hub.id}.json.gz", load("open_meteo_archive_denver.json"))
        write_gz(seed / "forecast" / f"{hub.id}.json.gz", load("open_meteo_forecast_denver.json"))
        write_gz(seed / "nws" / f"{hub.id}.json.gz", load("nws_alerts_sample.json"))
        write_gz(seed / "openfema" / f"{hub.id}.json.gz", load("openfema_harris.json"))
    (seed / "manifest.json").write_text(json.dumps({"recorded_at": "2026-10-01T12:00:00+00:00", "today": "2026-10-01"}))


def test_load_seed_fills_store_with_recording_time(tmp_path):
    make_seed(tmp_path)
    store = Store(":memory:")
    assert ensure_data(store, tmp_path) is True
    assert ensure_data(store, tmp_path) is False  # already has data
    assert all(store.daily_range(h.id) for h in load_hubs())
    assert store.source_statuses()["open_meteo_archive"]["last_success_at"] == "2026-10-01T12:00:00+00:00"
    assert store.cache_get("nws:denver")[0] == "2026-10-01T12:00:00+00:00"
    assert len(store.latest_snapshots("exposure")) == len(load_hubs())
    gaps = store.latest_snapshots("exposure")["denver"]["breakdown"]["data_gaps"]
    assert any("covers only" in g for g in gaps)  # 10 recorded days out of a 10-year window is disclosed


def test_load_seed_manifest_returned(tmp_path):
    make_seed(tmp_path)
    assert load_seed(Store(":memory:"), tmp_path)["today"] == "2026-10-01"
