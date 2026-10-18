"""FEMA National Risk Index (county layer, ArcGIS REST). No key.

We use each hazard's building expected-annual-loss rate (`{HZ}_ALRB`), not NRI's own scores
(see docs/DECISIONS.md D7):
  - the headline RISK_SCORE scales with county wealth (every metro scores 95-100);
  - the composite *_ALR_NPCTL includes agriculture losses.
The rate is scored on fixed log-scale anchors in scoring/exposure.py (D26).
"""

import httpx

from weather_risk.sources.http import SourceError, get_json

URL = (
    "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/"
    "National_Risk_Index_Counties/FeatureServer/0/query"
)
HAZARDS = ("WNTW", "ISTM", "CWAV", "HRCN", "IFLD", "CFLD", "TRND", "HAIL", "SWND", "HWAV")
PAGE_SIZE = 2000


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


def county_loss_rates(pages: list[dict], fips: list[str]) -> tuple[str, int, dict[str, dict[str, float | None]]]:
    """Return (nri_version, n_counties, {fips: {hazard: building loss rate}}) for the requested counties.
    A rate of None means the hazard is not applicable in that county."""
    rows = [f["attributes"] for page in pages for f in page.get("features", [])]
    if not rows:
        raise ValueError("NRI returned no counties")
    version = rows[0].get("NRI_VER") or "unknown"
    by_fips = {r["STCOFIPS"]: r for r in rows}
    result = {code: {hazard: by_fips[code].get(f"{hazard}_ALRB") for hazard in HAZARDS}
              for code in fips if code in by_fips}
    return version, len(rows), result
