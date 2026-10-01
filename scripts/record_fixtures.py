"""Record real API responses as deterministic test/eval fixtures (tests/fixtures/recorded/).

Fixtures are NOT a runtime dependency: the app always fetches on demand and caches.
Each file is the exact response to the request the runtime makes for one (hub, year).

    uv run python scripts/record_fixtures.py
"""

import asyncio
import gzip
import json
from datetime import UTC, date, datetime
from pathlib import Path

from weather_risk.config import load_hubs, load_scoring_config
from weather_risk.settings import get_settings
from weather_risk.sources import nri, open_meteo
from weather_risk.sources.http import make_client
from weather_risk.timewindow import climatology_window, year_span

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "recorded"
CALLS_PER_MIN = 400  # stay under Open-Meteo's 600/min free-tier limit


def write_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", compresslevel=9) as f:
        json.dump(payload, f, separators=(",", ":"))


async def main() -> None:
    today = date.today()
    first, last = climatology_window(today, load_scoring_config().climatology.years)
    years = list(range(first.year, today.year + 1))  # climatology + current year to date
    async with make_client(get_settings().contact_email) as client:
        write_gz(OUT / "nri_pages.json.gz", await nri.fetch_county_pages(client))
        print("NRI county pages recorded")
        for hub in load_hubs():
            for year in years:
                start, end, _ = year_span(year, today)
                path = OUT / "open_meteo" / f"{hub.id}_{year}.json.gz"
                if path.exists():
                    continue
                write_gz(path, await open_meteo.fetch_archive(client, hub.lat, hub.lon, start, end))
                await asyncio.sleep(open_meteo.call_weight((end - start).days + 1) * 60 / CALLS_PER_MIN)
            print(f"{hub.id}: {years[0]}–{years[-1]} recorded")
    manifest = {"recorded_at": datetime.now(UTC).isoformat(timespec="seconds"), "today": today.isoformat(),
                "years": years, "hubs": [h.id for h in load_hubs()], "attribution": open_meteo.ATTRIBUTION}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
