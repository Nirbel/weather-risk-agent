"""NWS active alerts (api.weather.gov). No key; a User-Agent is required.

Only structured fields are kept. Alert free text (description, instructions) is
dropped on purpose so it can never reach the LLM prompt.
"""

from dataclasses import dataclass

import httpx

from weather_risk.sources.http import get_json

ALERTS_URL = "https://api.weather.gov/alerts/active"


@dataclass(frozen=True)
class NwsAlert:
    event: str
    severity: str
    urgency: str
    certainty: str
    onset: str | None
    expires: str | None


async def fetch_active_alerts(client: httpx.AsyncClient, lat: float, lon: float) -> dict:
    return await get_json(client, ALERTS_URL, source="nws_alerts", params={"point": f"{lat:.4f},{lon:.4f}"})


def parse_alerts(payload: dict) -> list[NwsAlert]:
    alerts = []
    for feature in payload.get("features", []):
        p = feature.get("properties", {})
        if p.get("status") != "Actual" or p.get("messageType") == "Cancel":
            continue
        alerts.append(
            NwsAlert(
                event=p.get("event", "Unknown"),
                severity=p.get("severity") or "Unknown",
                urgency=p.get("urgency") or "Unknown",
                certainty=p.get("certainty") or "Unknown",
                onset=p.get("onset"),
                expires=p.get("expires"),
            )
        )
    return alerts
