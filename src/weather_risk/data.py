"""On-demand data: Open-Meteo history and FEMA NRI, fetched when a question needs them and cached.

Nothing is preloaded. A completed calendar year of weather is fetched once and never again;
the current year is refreshed when older than 12 hours; NRI percentiles are cached 30 days.
Failures become data gaps that answers report (docs/DECISIONS.md D6, D14).
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import httpx

from weather_risk.config import Hub
from weather_risk.db import Database, utcnow
from weather_risk.sources import nri, open_meteo
from weather_risk.sources.http import SourceError
from weather_risk.sources.open_meteo import DailyRow
from weather_risk.timewindow import archive_end, year_span

# Open-Meteo free tier: 600 weighted calls/min. One year for one hub ≈ 26 calls.
OPEN_METEO_CALLS_PER_MIN = 400


class RateLimiter:
    """Sliding 60-second window of weighted calls, shared by all requests in the process."""

    def __init__(self, per_minute: float, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], object] = asyncio.sleep):
        self.per_minute, self.clock, self.sleep = per_minute, clock, sleep
        self.window: deque[tuple[float, float]] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self, weight: float) -> None:
        async with self.lock:
            while True:
                now = self.clock()
                while self.window and now - self.window[0][0] >= 60:
                    self.window.popleft()
                used = sum(w for _, w in self.window)
                if not self.window or used + weight <= self.per_minute:
                    self.window.append((now, weight))
                    return
                await self.sleep(60 - (now - self.window[0][0]))


class WeatherService:
    STALE_AFTER = timedelta(hours=12)

    def __init__(self, db: Database, client: httpx.AsyncClient, limiter: RateLimiter, *,
                 today_fn: Callable[[], date] = date.today, now_fn: Callable[[], datetime] = utcnow):
        self.db, self.client, self.limiter = db, client, limiter
        self.today_fn, self.now_fn = today_fn, now_fn
        self._locks: dict[tuple[str, int], asyncio.Lock] = {}

    def _needs_fetch(self, entry, complete: bool) -> bool:
        if entry is None:
            return True
        if entry.complete:
            return False
        fetched = entry.fetched_at if entry.fetched_at.tzinfo else entry.fetched_at.replace(tzinfo=UTC)
        return complete or self.now_fn() - fetched > self.STALE_AFTER

    async def _ensure_year(self, hub: Hub, year: int) -> str | None:
        today = self.today_fn()
        start, end, complete = year_span(year, today)
        if start > end:
            return None
        async with self._locks.setdefault((hub.id, year), asyncio.Lock()):
            entry = (await self.db.fetch_log(hub.id)).get(year)
            if not self._needs_fetch(entry, complete):
                return None
            await self.limiter.acquire(open_meteo.call_weight((end - start).days + 1))
            try:
                payload = await open_meteo.fetch_archive(self.client, hub.lat, hub.lon, start, end)
                rows = open_meteo.parse_daily(payload)
            except (SourceError, ValueError, KeyError) as exc:
                await self.db.source_error("open_meteo_archive", str(exc))
                cached = " — using the copy cached earlier" if entry else ""
                return f"{hub.name} {year}: Open-Meteo request failed ({exc}){cached}."
            await self.db.upsert_daily(hub.id, rows)
            await self.db.record_fetch(hub.id, year, complete=complete, grid=open_meteo.grid_cell(payload),
                                       fetched_at=self.now_fn())
            await self.db.source_ok("open_meteo_archive")
            return None

    async def ensure(self, hub: Hub, start: date, end: date) -> list[str]:
        """Make sure [start, end] is cached. Returns data-gap messages (empty = complete)."""
        end = min(end, archive_end(self.today_fn()))
        results = [await self._ensure_year(hub, y) for y in range(start.year, end.year + 1)]
        return [r for r in results if r]

    async def daily(self, hub: Hub, start: date, end: date) -> tuple[list[DailyRow], list[str]]:
        gaps = await self.ensure(hub, start, end)
        return await self.db.daily(hub.id, start, end), gaps

    async def grid_cell(self, hub: Hub) -> dict | None:
        log = await self.db.fetch_log(hub.id)
        entry = log[max(log)] if log else None
        if entry is None or entry.grid_lat is None:
            return None
        return {"latitude": entry.grid_lat, "longitude": entry.grid_lon, "elevation": entry.grid_elevation}


class HazardService:
    """FEMA NRI building-loss percentiles for the hub counties (computed over all US counties)."""

    TTL = timedelta(days=30)

    def __init__(self, db: Database, client: httpx.AsyncClient, hubs: tuple[Hub, ...] | list[Hub], *,
                 now_fn: Callable[[], datetime] = utcnow):
        self.db, self.client, self.hubs, self.now_fn = db, client, list(hubs), now_fn
        self._lock = asyncio.Lock()

    async def percentiles(self) -> tuple[dict | None, list[str]]:
        """Returns ({"version", "n_counties", "by_fips"}, gaps)."""
        async with self._lock:
            cached = await self.db.cache_get("nri")
            needed = {h.county_fips for h in self.hubs}
            if cached:
                age = self.now_fn() - datetime.fromisoformat(cached[0])
                if age < self.TTL and needed <= set(cached[1]["requested_fips"]):
                    return cached[1], []
            try:
                pages = await nri.fetch_county_pages(self.client)
                version, n, by_fips = nri.county_percentiles(pages, sorted(needed))
            except (SourceError, ValueError, KeyError) as exc:
                await self.db.source_error("fema_nri", str(exc))
                if cached:
                    return cached[1], [f"FEMA National Risk Index refresh failed ({exc}); "
                                       f"using data retrieved {cached[0]}."]
                return None, [f"FEMA National Risk Index unavailable ({exc}) — modeled-loss evidence is missing."]
            payload = {
                "version": version,
                "n_counties": n,
                "requested_fips": sorted(needed),  # a county absent from NRI must not trigger refetches
                "by_fips": {f: {hz: {"alrb": v.alrb, "pct": v.pct} for hz, v in m.items()} for f, m in by_fips.items()},
            }
            await self.db.cache_put("nri", "fema_nri", payload, fetched_at=self.now_fn())
            await self.db.source_ok("fema_nri")
            return payload, []
