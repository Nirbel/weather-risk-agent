"""On-demand data services: rate limiting, per-year weather caching, NRI caching (HTTP mocked)."""

import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from weather_risk.config import hubs_by_id, load_hubs
from weather_risk.data import HazardService, RateLimiter, WeatherService
from weather_risk.db import Database
from weather_risk.sources import nri, open_meteo

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 10, 1)
DENVER = hubs_by_id()["denver"]


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


class FakeClock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.fixture
async def db():
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    yield database
    await database.dispose()


def no_wait_limiter() -> RateLimiter:
    clock = FakeClock()
    return RateLimiter(per_minute=10_000, clock=clock.now, sleep=clock.sleep)


# -- rate limiter --------------------------------------------------------------------

async def test_rate_limiter_waits_for_the_window_to_free_up():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=100, clock=clock.now, sleep=clock.sleep)
    await limiter.acquire(60)
    assert clock.sleeps == []
    await limiter.acquire(60)  # 120 > 100 → wait until the first 60 leaves the 60-s window
    assert clock.sleeps == [pytest.approx(60.0)]


async def test_rate_limiter_lets_an_oversized_request_through_alone():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=100, clock=clock.now, sleep=clock.sleep)
    await limiter.acquire(150)
    assert clock.sleeps == []


# -- weather on demand ---------------------------------------------------------------

@respx.mock
async def test_weather_fetches_each_missing_year_once(db):
    route = respx.get(open_meteo.ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=load("open_meteo_archive_denver.json")))
    async with httpx.AsyncClient() as client:
        weather = WeatherService(db, client, no_wait_limiter(), today_fn=lambda: TODAY)
        assert await weather.ensure(DENVER, date(2024, 1, 1), date(2025, 12, 31)) == []
        await weather.ensure(DENVER, date(2024, 1, 1), date(2025, 12, 31))
    assert route.call_count == 2  # 2024 and 2025, cached afterwards
    years = sorted(call.request.url.params["start_date"] for call in route.calls)
    assert years == ["2024-01-01", "2025-01-01"]
    assert (await weather.grid_cell(DENVER))["latitude"] == pytest.approx(39.86, abs=0.1)


@respx.mock
async def test_current_year_is_clipped_and_refreshed_only_when_stale(db):
    route = respx.get(open_meteo.ARCHIVE_URL).mock(
        return_value=httpx.Response(200, json=load("open_meteo_archive_denver.json")))
    now = [datetime(2026, 10, 1, 12, tzinfo=UTC)]
    async with httpx.AsyncClient() as client:
        weather = WeatherService(db, client, no_wait_limiter(), today_fn=lambda: TODAY, now_fn=lambda: now[0])
        await weather.ensure(DENVER, date(2026, 1, 1), date(2026, 12, 31))
        assert route.calls[0].request.url.params["end_date"] == "2026-09-25"  # archive lag
        now[0] += timedelta(hours=2)
        await weather.ensure(DENVER, date(2026, 1, 1), date(2026, 12, 31))
        assert route.call_count == 1  # fresh enough
        now[0] += timedelta(hours=13)
        await weather.ensure(DENVER, date(2026, 1, 1), date(2026, 12, 31))
    assert route.call_count == 2


@respx.mock
async def test_weather_failure_becomes_a_data_gap(db):
    respx.get(open_meteo.ARCHIVE_URL).mock(return_value=httpx.Response(400, text="bad request"))
    async with httpx.AsyncClient() as client:
        weather = WeatherService(db, client, no_wait_limiter(), today_fn=lambda: TODAY)
        rows, gaps = await weather.daily(DENVER, date(2025, 1, 1), date(2025, 12, 31))
    assert rows == []
    assert gaps and "Denver 2025" in gaps[0] and "Open-Meteo" in gaps[0]
    assert (await db.source_statuses())["open_meteo_archive"]["last_error"]


@respx.mock
async def test_concurrent_requests_share_one_fetch(db):
    async def slow(request):
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=load("open_meteo_archive_denver.json"))

    route = respx.get(open_meteo.ARCHIVE_URL).mock(side_effect=slow)
    async with httpx.AsyncClient() as client:
        weather = WeatherService(db, client, no_wait_limiter(), today_fn=lambda: TODAY)
        await asyncio.gather(*(weather.ensure(DENVER, date(2025, 1, 1), date(2025, 12, 31)) for _ in range(3)))
    assert route.call_count == 1


# -- FEMA NRI on demand --------------------------------------------------------------

@respx.mock
async def test_nri_is_fetched_once_and_cached(db):
    route = respx.get(nri.URL).mock(return_value=httpx.Response(200, json=load("nri_counties_sample.json")))
    async with httpx.AsyncClient() as client:
        hazards = HazardService(db, client, load_hubs())
        payload, gaps = await hazards.percentiles()
        await hazards.percentiles()
    assert route.call_count == 1
    assert payload["version"] == "December 2025"
    assert payload["by_fips"]["12086"]["HRCN"]["pct"] > payload["by_fips"]["08031"]["HRCN"]["pct"]
    assert gaps == []


@respx.mock
async def test_nri_failure_uses_stale_cache_and_says_so(db):
    respx.get(nri.URL).mock(side_effect=[httpx.Response(200, json=load("nri_counties_sample.json")),
                                         httpx.Response(400, text="down")])
    now = [datetime(2026, 10, 1, tzinfo=UTC)]
    async with httpx.AsyncClient() as client:
        hazards = HazardService(db, client, load_hubs(), now_fn=lambda: now[0])
        await hazards.percentiles()
        now[0] += timedelta(days=31)  # cache expired → refresh fails → stale data with a note
        payload, gaps = await hazards.percentiles()
    assert payload["version"] == "December 2025"
    assert gaps and "using data retrieved" in gaps[0]


@respx.mock
async def test_nri_unavailable_without_cache(db):
    respx.get(nri.URL).mock(return_value=httpx.Response(400, text="down"))
    async with httpx.AsyncClient() as client:
        payload, gaps = await HazardService(db, client, load_hubs()).percentiles()
    assert payload is None and "National Risk Index unavailable" in gaps[0]
