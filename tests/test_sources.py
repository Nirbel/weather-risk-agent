"""Source clients: parsing of real recorded responses + HTTP behaviour (all calls mocked)."""

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
import respx

from weather_risk.sources import nri, open_meteo
from weather_risk.sources.http import SourceError, get_json

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


# -- Open-Meteo ---------------------------------------------------------------

def test_parse_archive_real_response():
    rows = open_meteo.parse_daily(load("open_meteo_archive_denver.json"))
    assert len(rows) == 10
    assert rows[0].date == date(2025, 1, 1)
    assert rows[-1].date == date(2025, 1, 10)
    assert all(isinstance(r.tmin_c, float) for r in rows)


def test_parse_rejects_unexpected_units():
    payload = load("open_meteo_archive_denver.json")
    payload["daily_units"]["snowfall_sum"] = "inch"
    with pytest.raises(ValueError, match="snowfall_sum"):
        open_meteo.parse_daily(payload)


def test_grid_cell_reports_snapped_coordinates():
    cell = open_meteo.grid_cell(load("open_meteo_archive_denver.json"))
    assert set(cell) == {"latitude", "longitude", "elevation"}


def test_call_weight_matches_open_meteo_rule():
    # 2 weeks, 5 variables = 1 call; 10 years ≈ 261 calls.
    assert open_meteo.call_weight(14) == 1.0
    assert open_meteo.call_weight(7) == 1.0
    assert round(open_meteo.call_weight(3653)) == 261


def test_archive_end_lags_six_days():
    assert open_meteo.archive_end(date(2026, 10, 1)) == date(2026, 9, 25)


@respx.mock
async def test_fetch_archive_sends_expected_params():
    route = respx.get(open_meteo.ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=load("open_meteo_archive_denver.json"))
    )
    async with httpx.AsyncClient() as client:
        await open_meteo.fetch_archive(client, 39.8561, -104.6737, date(2025, 1, 1), date(2025, 1, 10))
    params = route.calls[0].request.url.params
    assert params["start_date"] == "2025-01-01"
    assert params["end_date"] == "2025-01-10"
    assert "snowfall_sum" in params["daily"]


# -- FEMA NRI -----------------------------------------------------------------

def _page(rows):
    return {"features": [{"attributes": {"STCOFIPS": f, "NRI_VER": "December 2025", **r}} for f, r in rows]}


def test_percentile_is_share_of_counties_strictly_lower():
    # HRCN building loss rates: A=None (not applicable), B=0.1, C=0.2, D=0.2
    page = _page([("A", {"HRCN_ALRB": None}), ("B", {"HRCN_ALRB": 0.1}),
                  ("C", {"HRCN_ALRB": 0.2}), ("D", {"HRCN_ALRB": 0.2})])
    version, n, result = nri.county_percentiles([page], ["A", "B", "C"])
    assert (version, n) == ("December 2025", 4)
    assert result["A"]["HRCN"] == nri.NriHazard(alrb=None, pct=0.0)
    assert result["B"]["HRCN"].pct == 25.0  # only A (treated as 0) is lower
    assert result["C"]["HRCN"].pct == 50.0  # A and B are lower; tie with D does not count


def test_percentiles_on_real_sample():
    version, n, result = nri.county_percentiles([load("nri_counties_sample.json")], ["12086", "08031"])
    assert version == "December 2025" and n == 6
    assert result["08031"]["HRCN"].pct == 0.0  # Denver: no hurricane exposure
    assert result["12086"]["HRCN"].pct > result["08031"]["HRCN"].pct


@respx.mock
async def test_fetch_county_pages_follows_paging():
    first = {"features": [{"attributes": {"STCOFIPS": str(i)}} for i in range(nri.PAGE_SIZE)]}
    second = {"features": [{"attributes": {"STCOFIPS": "x"}}]}
    route = respx.get(nri.URL).mock(side_effect=[httpx.Response(200, json=first), httpx.Response(200, json=second)])
    async with httpx.AsyncClient() as client:
        pages = await nri.fetch_county_pages(client)
    assert len(pages) == 2
    assert route.calls[1].request.url.params["resultOffset"] == str(nri.PAGE_SIZE)


@respx.mock
async def test_nri_error_body_raises_source_error():
    respx.get(nri.URL).mock(return_value=httpx.Response(200, json={"error": {"code": 400}}))
    async with httpx.AsyncClient() as client:
        with pytest.raises(SourceError):
            await nri.fetch_county_pages(client)


# -- shared HTTP helper -------------------------------------------------------

@respx.mock
async def test_get_json_retries_on_5xx_then_succeeds():
    route = respx.get("https://x.test/a").mock(
        side_effect=[httpx.Response(503), httpx.Response(429), httpx.Response(200, json={"ok": 1})]
    )
    async with httpx.AsyncClient() as client:
        assert await get_json(client, "https://x.test/a", source="t", backoff_s=0) == {"ok": 1}
    assert route.call_count == 3


@respx.mock
async def test_get_json_does_not_retry_client_errors():
    route = respx.get("https://x.test/a").mock(return_value=httpx.Response(404, text="nope"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(SourceError, match="HTTP 404"):
            await get_json(client, "https://x.test/a", source="t", backoff_s=0)
    assert route.call_count == 1


@respx.mock
async def test_get_json_gives_up_after_retries():
    respx.get("https://x.test/a").mock(side_effect=httpx.ConnectError("down"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(SourceError, match="failed after 3 attempts"):
            await get_json(client, "https://x.test/a", source="t", backoff_s=0)
