"""Optional: warm the cache before a demo and print the exposure table. Not needed at runtime —
the app fetches on demand — but a cold first ranking takes ~3 minutes on Open-Meteo's free tier.

    uv run python -m weather_risk.warm             # live APIs, rate-limited
    uv run python -m weather_risk.warm --replay    # recorded fixtures, no network (in-memory DB)
"""

import argparse
import asyncio

import httpx

from weather_risk import replay
from weather_risk.analysis import build_analyzer
from weather_risk.config import load_scoring_config, validate_config
from weather_risk.data import RateLimiter
from weather_risk.db import Database
from weather_risk.settings import get_settings
from weather_risk.sources.http import make_client


async def main(use_replay: bool) -> None:
    validate_config()
    settings = get_settings()
    db = Database("sqlite+aiosqlite:///:memory:" if use_replay else settings.database_url)
    await db.create_all()
    today = replay.recorded_today() if use_replay else None
    client = httpx.AsyncClient(transport=replay.transport()) if use_replay else make_client(settings.contact_email)
    try:
        async with client:
            analyzer = (build_analyzer(db, client, today_fn=lambda: today, limiter=RateLimiter(float("inf")))
                        if use_replay else build_analyzer(db, client))
            result = await analyzer.exposure()
    finally:
        await db.dispose()

    families = list(load_scoring_config().families)
    print(f"Exposure {result.start.year}–{result.end.year} · NRI {result.nri_version}")
    print(f"{'hub':13}{'overall':>8}" + "".join(f"{f[:9]:>11}" for f in families) + f"{'top':>14}")
    def cell(score: float | None, width: int) -> str:
        return f"{'—' if score is None else f'{score:.1f}':>{width}}"  # — = not scored (missing or incomplete data)

    for hub_id, bd in sorted(result.breakdowns.items(), key=lambda kv: -(kv[1]["score"] or -1)):
        fams = "".join(cell(bd["families"][f]["score"], 11) for f in families)
        print(f"{hub_id:13}{cell(bd['score'], 8)}{fams}{bd['top_family'] or '—':>14}")
        for gap in bd["data_gaps"]:
            print(f"   gap: {gap}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", action="store_true", help="use recorded fixtures instead of the live APIs")
    asyncio.run(main(parser.parse_args().replay))
