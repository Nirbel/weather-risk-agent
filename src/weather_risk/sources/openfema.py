"""OpenFEMA Disaster Declarations Summaries (v2). No key.

One row per (disaster, designated area). We keep major disasters (DR) only and
de-duplicate by disaster number; incident-type filtering happens in scoring.
"""

from dataclasses import dataclass
from datetime import date

import httpx

from weather_risk.sources.http import get_json

URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
FIELDS = "disasterNumber,declarationType,declarationDate,incidentType,declarationTitle,fipsStateCode,fipsCountyCode"


@dataclass(frozen=True)
class Declaration:
    disaster_number: int
    incident_type: str
    declaration_date: date
    title: str


async def fetch_declarations(client: httpx.AsyncClient, county_fips: str, since: date) -> dict:
    flt = (
        f"fipsStateCode eq '{county_fips[:2]}' and fipsCountyCode eq '{county_fips[2:]}' "
        f"and declarationDate ge '{since.isoformat()}T00:00:00.000Z'"
    )
    params = {"$filter": flt, "$select": FIELDS, "$orderby": "declarationDate desc", "$top": 1000}
    return await get_json(client, URL, source="openfema", params=params)


def parse_declarations(payload: dict) -> list[Declaration]:
    """Major-disaster (DR) declarations only, one per disaster number."""
    seen: dict[int, Declaration] = {}
    for row in payload.get("DisasterDeclarationsSummaries", []):
        if row.get("declarationType") != "DR":
            continue
        number = int(row["disasterNumber"])
        seen.setdefault(
            number,
            Declaration(
                disaster_number=number,
                incident_type=row["incidentType"],
                declaration_date=date.fromisoformat(row["declarationDate"][:10]),
                title=row.get("declarationTitle", ""),
            ),
        )
    return sorted(seen.values(), key=lambda d: d.declaration_date, reverse=True)
