"""Cached data → deterministic results. Shared by the agent (chat) and the Analytics API,
so both always show the same numbers."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from weather_risk.config import Hub, ScoringConfig
from weather_risk.data import HazardService, WeatherService
from weather_risk.scoring.coverage import weather_coverage
from weather_risk.scoring.exposure import HubInputs, compute_exposure
from weather_risk.scoring.indicators import exceeds
from weather_risk.sources.open_meteo import DailyRow
from weather_risk.timewindow import archive_end, climatology_window


@dataclass
class Exposure:
    start: date
    end: date
    breakdowns: dict[str, dict]  # hub_id → breakdown (scores, lenses, indicators, meta, data_gaps)
    nri_version: str | None


class Analyzer:
    def __init__(self, weather: WeatherService, hazards: HazardService, cfg: ScoringConfig,
                 hubs: tuple[Hub, ...], *, today_fn: Callable[[], date] = date.today):
        self.weather, self.hazards, self.cfg = weather, hazards, cfg
        self.hubs = {h.id: h for h in hubs}
        self.today_fn = today_fn

    @property
    def today(self) -> date:
        return self.today_fn()

    def window(self) -> tuple[date, date]:
        return climatology_window(self.today, self.cfg.climatology.years)

    def data_end(self) -> date:
        return archive_end(self.today)

    def observed_variables(self) -> list[str]:
        """Daily variables the exposure score reads; a day counts as covered only if all are present."""
        return sorted({spec.variable for fam in self.cfg.families.values() for spec in fam.observed})

    def unrankable_note(self, hub: Hub, cov: dict, families: list[str] | None = None) -> str:
        labels = [self.cfg.families[f].label.split(" (")[0].lower() for f in families or self.cfg.families
                  if self.cfg.families[f].observed]
        return (f"{hub.name}: observed weather is incomplete for {cov['window']} — " + " and ".join(cov["problems"])
                + f". Its {', '.join(labels)} and overall scores are not computed, so it is not ranked on them.")

    async def exposure(self, hub_ids: list[str] | None = None) -> Exposure:
        """Exposure breakdowns for the given hubs (default: all), fetching missing data on demand."""
        start, end = self.window()
        nri, nri_gaps = await self.hazards.percentiles()

        async def one(hub: Hub) -> tuple[str, dict]:
            rows, weather_gaps = await self.weather.daily(hub, start, end)
            hub_nri = nri["by_fips"].get(hub.county_fips) if nri else None
            cov = weather_coverage(rows, start, end, self.observed_variables(), self.cfg.coverage)
            cov["window"] = f"{start.year}–{end.year}"
            breakdown = compute_exposure(hub.id, HubInputs(rows, hub_nri, weather_complete=cov["complete"]), self.cfg)
            gaps = [*nri_gaps, *weather_gaps, *breakdown["data_gaps"]]
            if nri and hub_nri is None:
                gaps.append(f"FEMA NRI has no record for {hub.county} — modeled-loss evidence is missing.")
            if not cov["complete"]:
                gaps.append(self.unrankable_note(hub, cov))
            breakdown["data_gaps"] = list(dict.fromkeys(gaps))
            breakdown["meta"] = {
                "observed_window": {"start": start.isoformat(), "end": end.isoformat(),
                                    "days_with_data": cov["days_with_data"], "expected_days": cov["expected_days"],
                                    "coverage_pct": cov["pct"], "longest_gap_days": cov["longest_gap_days"],
                                    "longest_gap": cov["longest_gap"], "complete": cov["complete"],
                                    "problems": cov["problems"]},
                "grid_cell": await self.weather.grid_cell(hub),
                "nri_version": nri["version"] if nri else None,
                "county": f"{hub.county}, {hub.state} (FIPS {hub.county_fips})",
            }
            return hub.id, breakdown

        ids = hub_ids or list(self.hubs)
        results = await asyncio.gather(*(one(self.hubs[h]) for h in ids))
        return Exposure(start, end, dict(results), nri["version"] if nri else None)

    async def daily(self, hub_id: str, start: date, end: date) -> tuple[list[DailyRow], list[str]]:
        return await self.weather.daily(self.hubs[hub_id], start, end)

    async def yearly_indicators(self, hub_ids: list[str] | None = None) -> dict[str, dict]:
        """Per-year counts of every observed indicator (Analytics: historical exposure metrics)."""
        start, end = self.window()
        specs = [s for fam in self.cfg.families.values() for s in fam.observed]
        out: dict[str, dict] = {}
        for hub_id in hub_ids or list(self.hubs):
            rows, gaps = await self.daily(hub_id, start, end)
            by_year: dict[int, dict[str, int]] = {}
            for spec in specs:
                for r in rows:
                    value = getattr(r, spec.variable)
                    if value is not None and exceeds(value, spec.op, spec.threshold):
                        by_year.setdefault(r.date.year, {}).setdefault(spec.id, 0)
                        by_year[r.date.year][spec.id] += 1
            out[hub_id] = {
                "years": {y: {s.id: by_year.get(y, {}).get(s.id, 0) for s in specs}
                          for y in range(start.year, end.year + 1)},
                "labels": {s.id: s.label for s in specs},
                "data_gaps": gaps,
            }
        return out


def build_analyzer(db, client, *, today_fn: Callable[[], date] = date.today, limiter=None) -> Analyzer:
    """Wire the on-demand services (shared by the API, the CLIs and eval).

    `limiter=None` → the real Open-Meteo rate limit; replay (no network) passes an unlimited one.
    """
    from weather_risk.config import load_hubs, load_scoring_config
    from weather_risk.data import OPEN_METEO_CALLS_PER_MIN, RateLimiter

    hubs = load_hubs()
    weather = WeatherService(db, client, limiter or RateLimiter(OPEN_METEO_CALLS_PER_MIN), today_fn=today_fn)
    return Analyzer(weather, HazardService(db, client, hubs), load_scoring_config(), hubs, today_fn=today_fn)
