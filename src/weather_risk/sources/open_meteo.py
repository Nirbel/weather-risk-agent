"""Open-Meteo historical weather API (ERA5 reanalysis) client.

Free tier is non-commercial; data is CC-BY 4.0. Requests longer than 2 weeks or with
more than 10 variables count as several API calls (see `call_weight`).
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

from weather_risk.sources.http import get_json
from weather_risk.timewindow import archive_end  # noqa: F401  (re-exported for callers)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0), ERA5 reanalysis"

# Open-Meteo daily variable → our column, with the unit we require.
DAILY_VARS = {
    "snowfall_sum": ("snowfall_cm", "cm"),
    "precipitation_sum": ("precip_mm", "mm"),
    "temperature_2m_max": ("tmax_c", "°C"),
    "temperature_2m_min": ("tmin_c", "°C"),
    "wind_gusts_10m_max": ("gust_kmh", "km/h"),
}


@dataclass(frozen=True)
class DailyRow:
    date: date
    snowfall_cm: float | None
    precip_mm: float | None
    tmax_c: float | None
    tmin_c: float | None
    gust_kmh: float | None


def call_weight(n_days: int, n_vars: int = len(DAILY_VARS)) -> float:
    """Fractional API-call cost of one request (Open-Meteo pricing rule)."""
    return max(1.0, n_days / 14) * max(1.0, n_vars / 10)


def _params(lat: float, lon: float) -> dict[str, Any]:
    return {"latitude": lat, "longitude": lon, "daily": ",".join(DAILY_VARS), "timezone": "auto"}


async def fetch_archive(client: httpx.AsyncClient, lat: float, lon: float, start: date, end: date) -> dict:
    params = _params(lat, lon) | {"start_date": start.isoformat(), "end_date": end.isoformat()}
    return await get_json(client, ARCHIVE_URL, source="open_meteo_archive", params=params)


def parse_daily(payload: dict) -> list[DailyRow]:
    """Parse a daily payload. Rejects unexpected units instead of silently converting."""
    units = payload.get("daily_units", {})
    for api_name, (_, unit) in DAILY_VARS.items():
        if units.get(api_name) != unit:
            raise ValueError(f"Open-Meteo returned {api_name} in {units.get(api_name)!r}, expected {unit!r}")
    daily = payload["daily"]
    columns = {col: daily[api_name] for api_name, (col, _) in DAILY_VARS.items()}
    return [
        DailyRow(date=date.fromisoformat(day), **{col: values[i] for col, values in columns.items()})
        for i, day in enumerate(daily["time"])
    ]


def grid_cell(payload: dict) -> dict[str, float]:
    """The reanalysis grid cell the coordinates snapped to."""
    return {k: payload[k] for k in ("latitude", "longitude", "elevation")}
