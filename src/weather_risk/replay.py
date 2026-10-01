"""Offline replay of recorded real API responses (tests/fixtures/recorded/), for tests and eval.

Not used by the app: at runtime every request goes to the live APIs and is cached.
Requests that were not recorded fail loudly (HTTP 404), so a test can never silently
pass on made-up data.
"""

import gzip
import json
from datetime import date
from pathlib import Path

import httpx
import respx

from weather_risk.config import load_hubs
from weather_risk.sources import nri, open_meteo
from weather_risk.settings import ROOT

RECORDED = ROOT / "tests" / "fixtures" / "recorded"


def _read_gz(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def recorded_today() -> date:
    """The date the fixtures were recorded — replay must run 'as of' that day to match requests."""
    return date.fromisoformat(json.loads((RECORDED / "manifest.json").read_text())["today"])


def install(router: respx.MockRouter, *, nri_status: int = 200, open_meteo_status: int = 200) -> None:
    """Register replay routes on a respx router (e.g. the `respx_mock` fixture or respx.mock())."""
    by_coords = {(round(h.lat, 4), round(h.lon, 4)): h.id for h in load_hubs()}

    def archive(request: httpx.Request) -> httpx.Response:
        if open_meteo_status != 200:
            return httpx.Response(open_meteo_status, text="replay: simulated failure")
        p = request.url.params
        hub = by_coords.get((round(float(p["latitude"]), 4), round(float(p["longitude"]), 4)))
        path = RECORDED / "open_meteo" / f"{hub}_{p['start_date'][:4]}.json.gz"
        if hub is None or not path.exists():
            return httpx.Response(404, text=f"replay: no recording for {p}")
        payload = _read_gz(path)
        if payload["daily"]["time"][-1] != p["end_date"]:
            return httpx.Response(404, text=f"replay: recording ends {payload['daily']['time'][-1]}, not {p['end_date']}")
        return httpx.Response(200, json=payload)

    pages = _read_gz(RECORDED / "nri_pages.json.gz")

    def nri_page(request: httpx.Request) -> httpx.Response:
        if nri_status != 200:
            return httpx.Response(nri_status, text="replay: simulated failure")
        offset = int(request.url.params.get("resultOffset", 0))
        index = offset // nri.PAGE_SIZE
        return httpx.Response(200, json=pages[index] if index < len(pages) else {"features": []})

    router.get(open_meteo.ARCHIVE_URL).mock(side_effect=archive)
    router.get(nri.URL).mock(side_effect=nri_page)
