"""FEMA National Risk Index (county layer, ArcGIS REST). No key.

We use each hazard's building expected-annual-loss rate (`{HZ}_ALRB`) and compute
our own national percentile across all counties (see docs/DECISIONS.md D7):
  - the headline RISK_SCORE scales with county wealth (every metro scores 95-100);
  - the composite *_ALR_NPCTL includes agriculture losses.
"""

import bisect
from dataclasses import dataclass

import httpx

from weather_risk.sources.http import SourceError, get_json

URL = (
    "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/"
    "National_Risk_Index_Counties/FeatureServer/0/query"
)
HAZARDS = ("WNTW", "ISTM", "CWAV", "HRCN", "IFLD", "CFLD", "TRND", "HAIL", "SWND", "HWAV")
PAGE_SIZE = 2000


@dataclass(frozen=True)
class NriHazard:
    alrb: float | None  # building loss rate; None = hazard not applicable in this county
    pct: float  # share of US counties (0-100) with a strictly lower building loss rate


async def fetch_county_pages(client: httpx.AsyncClient) -> list[dict]:
    """All counties, ALRB fields only, paged (maxRecordCount = 2000)."""
    out_fields = ",".join(["STCOFIPS", "NRI_VER", *(f"{h}_ALRB" for h in HAZARDS)])
    pages, offset = [], 0
    while True:
        params = {
            "where": "1=1",
            "outFields": out_fields,
            "returnGeometry": "false",
            "orderByFields": "OBJECTID",
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }
        page = await get_json(client, URL, source="fema_nri", params=params)
        if "error" in page:
            raise SourceError("fema_nri", str(page["error"])[:200])
        pages.append(page)
        n = len(page.get("features", []))
        if n < PAGE_SIZE and not page.get("exceededTransferLimit"):
            return pages
        offset += n


def county_percentiles(pages: list[dict], fips: list[str]) -> tuple[str, int, dict[str, dict[str, NriHazard]]]:
    """Return (nri_version, n_counties, {fips: {hazard: NriHazard}}) for the requested counties."""
    rows = [f["attributes"] for page in pages for f in page.get("features", [])]
    if not rows:
        raise ValueError("NRI returned no counties")
    version = rows[0].get("NRI_VER") or "unknown"
    n = len(rows)
    by_fips = {r["STCOFIPS"]: r for r in rows}
    result: dict[str, dict[str, NriHazard]] = {}
    for hazard in HAZARDS:
        values = sorted((r.get(f"{hazard}_ALRB") or 0.0) for r in rows)
        for code in fips:
            row = by_fips.get(code)
            if row is None:
                continue
            value = row.get(f"{hazard}_ALRB")
            lower = bisect.bisect_left(values, value or 0.0)
            result.setdefault(code, {})[hazard] = NriHazard(alrb=value, pct=round(100.0 * lower / n, 1))
    return version, n, result

