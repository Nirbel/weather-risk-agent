"""Shared fixtures: a synthetic store with hand-chosen scores (for exact expectations)."""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from weather_risk.db import Store
from weather_risk.hubs import load_hubs, load_scoring_config
from weather_risk.sources.open_meteo import DailyRow

FAMILIES = list(load_scoring_config()["families"])
LABELS = {f: c["label"] for f, c in load_scoring_config()["families"].items()}

# Every family scores 10 unless listed here.
SYNTHETIC_SCORES = {
    "minneapolis": {"winter": 90},
    "detroit": {"winter": 80},
    "chicago": {"winter": 78},
    "columbus": {"winter": 50},
    "miami": {"hurricane": 85, "flood": 60},
    "houston": {"hurricane": 80, "flood": 75},
    "dallas": {"severe_storm": 90, "heat": 60},
}


def synthetic_breakdown(hub_id: str) -> dict:
    scores = {f: 10.0 for f in FAMILIES} | SYNTHETIC_SCORES.get(hub_id, {})
    families = {
        f: {
            "label": LABELS[f],
            "score": s,
            "contribution": s / len(FAMILIES),
            "lenses": {"modeled": {"score": s, "weight": 0.4, "source": "FEMA National Risk Index",
                                   "indicators": [{"id": f"nri_{f}", "label": f"{LABELS[f]} indicator",
                                                   "raw": s, "unit": "percentile", "score": s}]}},
            "missing_lenses": [],
            "missing_weight_share": 0.0,
        }
        for f, s in scores.items()
    }
    return {
        "hub_id": hub_id,
        "score": sum(scores.values()) / len(scores),
        "families": families,
        "family_weights": {f: 1 for f in FAMILIES},
        "top_family": max(scores, key=scores.get),
        "data_gaps": [],
        "meta": {"observed_window": {"start": "2016-01-01", "end": "2025-12-31", "coverage_pct": 100.0},
                 "nri_version": "December 2025", "declarations_since": "2000-01-01",
                 "grid_cell": {"latitude": 39.85, "longitude": -104.65, "elevation": 1650}},
    }


def synthetic_near_term(hub_id: str) -> tuple[float, str, dict]:
    if hub_id == "chicago":
        families = {f: {"score": 0.0, "drivers": []} for f in FAMILIES} | {
            "winter": {"score": 60.0, "drivers": ["2026-10-03: Daily snowfall (cm) 8.75 → 60 pts"]}}
        return 60.0, "Elevated", {"score": 60.0, "band": "Elevated", "top_family": "winter", "families": families,
                                  "unmapped_alerts": [], "data_gaps": [],
                                  "meta": {"forecast_start": "2026-10-01", "forecast_fetched_at": "2026-10-01T12:00:00+00:00"}}
    families = {f: {"score": 0.0, "drivers": []} for f in FAMILIES}
    return 0.0, "Low", {"score": 0.0, "band": "Low", "top_family": None, "families": families,
                        "unmapped_alerts": [], "data_gaps": [],
                        "meta": {"forecast_start": "2026-10-01", "forecast_fetched_at": "2026-10-01T12:00:00+00:00"}}


@pytest.fixture
def synthetic_store() -> Store:
    store = Store(":memory:")
    for hub in load_hubs():
        bd = synthetic_breakdown(hub.id)
        store.save_snapshot(hub.id, "exposure", bd["score"], None, bd)
        score, band, nt = synthetic_near_term(hub.id)
        store.save_snapshot(hub.id, "near_term", score, band, nt)
        store.cache_put(f"grid:{hub.id}", "open_meteo_archive", {"latitude": hub.lat, "longitude": hub.lon, "elevation": 0})
    # Denver 2025: 1.0 cm of snow on the first 30 days, none otherwise → 30 of 365 days (8.2 %).
    start = date(2016, 1, 1)
    rows = [DailyRow(start + timedelta(days=i), 0.0, 0.0, 10.0, 0.0, 10.0) for i in range((date(2026, 9, 25) - start).days + 1)]
    rows = [r if not (date(2025, 1, 1) <= r.date <= date(2025, 1, 30)) else DailyRow(r.date, 1.0, 0.0, 0.0, -5.0, 10.0) for r in rows]
    store.upsert_daily("denver", rows)
    for source in ("open_meteo_archive", "open_meteo_forecast", "nws_alerts", "fema_nri", "openfema"):
        store.source_ok(source, "2026-10-01T12:00:00+00:00")
    return store


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
