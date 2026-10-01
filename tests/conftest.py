"""Shared fixtures: in-memory database, on-demand services replaying recorded real responses,
and a fake LiteLLM completion."""

from datetime import date, timedelta
from types import SimpleNamespace

import httpx
import pytest
import respx

from weather_risk import replay
from weather_risk.analysis import Analyzer, Exposure
from weather_risk.config import load_hubs, load_scoring_config
from weather_risk.data import HazardService, RateLimiter, WeatherService
from weather_risk.db import Database
from weather_risk.sources.open_meteo import DailyRow

RECORDED_TODAY = replay.recorded_today()


@pytest.fixture
async def db():
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    yield database
    await database.dispose()


def no_wait_limiter() -> RateLimiter:
    async def no_sleep(_):
        return None

    return RateLimiter(per_minute=1e9, sleep=no_sleep)


@pytest.fixture
async def services(db):
    """Analyzer over the recorded real responses, 'as of' the recording date. No network."""
    with respx.mock(assert_all_called=False) as router:
        replay.install(router)
        async with httpx.AsyncClient() as client:
            weather = WeatherService(db, client, no_wait_limiter(), today_fn=lambda: RECORDED_TODAY)
            hazards = HazardService(db, client, load_hubs())
            analyzer = Analyzer(weather, hazards, load_scoring_config(), load_hubs(), today_fn=lambda: RECORDED_TODAY)
            yield SimpleNamespace(db=db, weather=weather, hazards=hazards, analyzer=analyzer, router=router)


# -- fake LiteLLM completion ---------------------------------------------------------

def llm_response(content: str | None = None, tool_args: str | None = None):
    tool_calls = [SimpleNamespace(function=SimpleNamespace(arguments=tool_args))] if tool_args else None
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class FakeCompletion:
    """Replays scripted outcomes: a string (content), an Exception (raised), or a response object."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return llm_response(outcome) if isinstance(outcome, str) else outcome


# -- a fake Analyzer with hand-chosen scores (exact executor expectations) -----------

FAMILIES = list(load_scoring_config().families)
SYNTHETIC_SCORES = {  # every family scores 10 unless listed
    "minneapolis": {"winter": 90},
    "detroit": {"winter": 80},
    "chicago": {"winter": 78},
    "kansas_city": {"winter": 50},
    "miami": {"hurricane": 85, "flood": 60},
    "houston": {"hurricane": 80, "flood": 75},
    "dallas": {"severe_storm": 90, "heat": 60},
}


def synthetic_breakdown(hub_id: str, gaps: list[str] | None = None) -> dict:
    cfg = load_scoring_config()
    scores = {f: 10.0 for f in FAMILIES} | SYNTHETIC_SCORES.get(hub_id, {})
    families = {
        f: {"label": cfg.families[f].label, "score": s, "contribution": s / len(FAMILIES),
            "lenses": {"modeled": {"score": s, "weight": 0.5, "source": "FEMA National Risk Index",
                                   "indicators": [{"id": f"nri_{f}", "label": f"{f} indicator", "raw": s,
                                                   "unit": "percentile", "score": s}]}},
            "missing_lenses": [], "missing_weight_share": 0.0}
        for f, s in scores.items()
    }
    return {"hub_id": hub_id, "score": sum(scores.values()) / len(scores), "families": families,
            "family_weights": cfg.family_weights, "top_family": max(scores, key=scores.get),
            "data_gaps": gaps or [],
            "meta": {"observed_window": {"start": "2021-01-01", "end": "2025-12-31", "coverage_pct": 100.0},
                     "grid_cell": {"latitude": 39.8, "longitude": -104.7, "elevation": 1650},
                     "nri_version": "December 2025", "county": "Test County"}}


class FakeAnalyzer:
    """Same interface as weather_risk.analysis.Analyzer, no data access."""

    def __init__(self, gaps: dict[str, list[str]] | None = None):
        self.today = date(2026, 10, 1)
        self.cfg = load_scoring_config()
        self.hubs = {h.id: h for h in load_hubs()}
        self.gaps = gaps or {}
        self.weather = SimpleNamespace(grid_cell=self._grid_cell)

    async def _grid_cell(self, hub):
        return {"latitude": hub.lat + 0.03, "longitude": hub.lon, "elevation": 0}

    def window(self):
        return date(2021, 1, 1), date(2025, 12, 31)

    def data_end(self):
        return date(2026, 9, 25)

    async def exposure(self, hub_ids=None):
        start, end = self.window()
        ids = hub_ids or list(self.hubs)
        return Exposure(start, end, {h: synthetic_breakdown(h, self.gaps.get(h)) for h in ids}, "December 2025")

    async def daily(self, hub_id, start, end):
        """Denver 2025: 1.0 cm of snow on the first 30 days of the year, none otherwise (30 of 365 = 8.2 %)."""
        days = (end - start).days + 1
        rows = []
        for i in range(days):
            d = start + timedelta(days=i)
            snowy = hub_id == "denver" and d.year == 2025 and d <= date(2025, 1, 30)
            rows.append(DailyRow(d, 1.0 if snowy else 0.0, 0.0, 10.0, -5.0 if snowy else 0.0, 10.0))
        return rows, []
