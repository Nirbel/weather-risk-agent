"""Record real API responses into data/seed/ (gzipped JSON).

The seed is used as test/eval data and to fill an empty database on first run, so a
fresh install works without the paced ~10-minute Open-Meteo history pull.

    uv run python scripts/record_seed.py           # resumes: skips archive files already recorded
    uv run python scripts/record_seed.py --force   # re-record everything
"""

import argparse
import asyncio
import json
from datetime import UTC, date, datetime

from weather_risk.hubs import load_hubs, load_scoring_config
from weather_risk.ingest import OPEN_METEO_CALLS_PER_MIN, write_gz
from weather_risk.settings import get_settings
from weather_risk.sources import nri, nws, open_meteo, openfema
from weather_risk.sources.http import make_client


async def main(force: bool) -> None:
    settings = get_settings()
    seed = settings.seed_dir
    cfg = load_scoring_config()
    today = date.today()
    start, end = cfg["climatology"]["start"], open_meteo.archive_end(today)
    since = cfg["declarations_since"]

    async with make_client(settings.contact_email) as client:
        pages = await nri.fetch_county_pages(client)
        write_gz(seed / "nri_pages.json.gz", pages)
        version = pages[0]["features"][0]["attributes"]["NRI_VER"]
        print(f"NRI: {sum(len(p['features']) for p in pages)} counties ({version})")

        for hub in load_hubs():
            write_gz(seed / "openfema" / f"{hub.id}.json.gz", await openfema.fetch_declarations(client, hub.county_fips, since))
            write_gz(seed / "forecast" / f"{hub.id}.json.gz", await open_meteo.fetch_forecast(client, hub.lat, hub.lon))
            write_gz(seed / "nws" / f"{hub.id}.json.gz", await nws.fetch_active_alerts(client, hub.lat, hub.lon))
            archive_path = seed / "archive" / f"{hub.id}.json.gz"
            if archive_path.exists() and not force:
                print(f"{hub.id}: archive already recorded, skipping")
                continue
            write_gz(archive_path, await open_meteo.fetch_archive(client, hub.lat, hub.lon, start, end))
            weight = open_meteo.call_weight((end - start).days + 1)
            print(f"{hub.id}: archive {start} → {end} recorded (~{weight:.0f} API calls), pacing…")
            await asyncio.sleep(weight * 60 / OPEN_METEO_CALLS_PER_MIN)

    manifest = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "today": today.isoformat(),
        "archive_start": start.isoformat(),
        "archive_end": end.isoformat(),
        "declarations_since": since.isoformat(),
        "nri_version": version,
        "hubs": [h.id for h in load_hubs()],
        "attribution": open_meteo.ATTRIBUTION,
    }
    (seed / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Seed recorded:", seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    asyncio.run(main(parser.parse_args().force))
