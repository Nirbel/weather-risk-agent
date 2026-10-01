"""Async SQLAlchemy store (SQLite via aiosqlite) — repository behaviour."""

from datetime import date

import pytest

from weather_risk.db import Database
from weather_risk.sources.open_meteo import DailyRow


@pytest.fixture
async def db():
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    yield database
    await database.dispose()


def row(day: int, snow: float | None = 0.0) -> DailyRow:
    return DailyRow(date(2025, 1, day), snow, 1.0, 5.0, -3.0, 20.0)


async def test_daily_rows_upsert_and_range_query(db):
    await db.upsert_daily("denver", [row(1, 1.0), row(2), row(3, None)])
    await db.upsert_daily("denver", [row(2, 7.5)])  # re-fetch replaces, never duplicates
    rows = await db.daily("denver", date(2025, 1, 2), date(2025, 1, 3))
    assert [(r.date.day, r.snowfall_cm) for r in rows] == [(2, 7.5), (3, None)]
    assert await db.daily("miami", date(2025, 1, 1), date(2025, 1, 3)) == []


async def test_fetch_log_tracks_completed_years(db):
    await db.record_fetch("denver", 2024, complete=True, grid={"latitude": 39.8, "longitude": -104.6, "elevation": 1650})
    await db.record_fetch("denver", 2026, complete=False, grid=None)
    log = await db.fetch_log("denver")
    assert log[2024].complete and not log[2026].complete
    assert log[2024].grid_lat == 39.8
    await db.record_fetch("denver", 2026, complete=True, grid=None)  # later fetch completes the year
    assert (await db.fetch_log("denver"))[2026].complete


async def test_cache_round_trip(db):
    await db.cache_put("nri", "fema_nri", {"version": "December 2025", "by_fips": {"08031": {}}})
    fetched_at, payload = await db.cache_get("nri")
    assert payload["version"] == "December 2025" and fetched_at
    assert await db.cache_get("missing") is None


async def test_source_status(db):
    await db.source_ok("open_meteo_archive")
    await db.source_error("fema_nri", "fema_nri: HTTP 503")
    statuses = await db.source_statuses()
    assert statuses["open_meteo_archive"]["last_success_at"]
    assert statuses["fema_nri"]["last_error"] == "fema_nri: HTTP 503"
    assert statuses["fema_nri"]["last_success_at"] is None


async def test_conversations_are_per_user_with_history(db):
    cid = await db.create_conversation("ana@example.com", "Midwest winter?")
    other = await db.create_conversation("bob@example.com", "Miami?")
    await db.add_turn(cid, "Midwest winter?", {"intent": "rank"}, {"answer": "Minneapolis"})
    await db.add_turn(cid, "Why?", {"intent": "explain"}, {"answer": "Cold days"})
    assert await db.conversation_owner(cid) == "ana@example.com"
    listed = await db.list_conversations("ana@example.com")
    assert [c["id"] for c in listed] == [cid] and listed[0]["title"] == "Midwest winter?"
    assert listed[0]["turns"] == 2
    recent = await db.recent_turns(cid, limit=1)
    assert recent[0]["user_text"] == "Why?" and recent[0]["plan"] == {"intent": "explain"}
    assert [t["user_text"] for t in await db.conversation_turns(cid)] == ["Midwest winter?", "Why?"]
    assert await db.conversation_owner(other) == "bob@example.com"
    assert await db.conversation_owner("nope") is None


# -- clean checkout: data/ does not exist yet -------------------------------------------

async def test_missing_database_directory_is_created(tmp_path):
    # Regression: a fresh clone has no data/ directory, and SQLite raised
    # "sqlite3.OperationalError: unable to open database file".
    path = tmp_path / "fresh-clone" / "data" / "app.db"
    database = Database(f"sqlite+aiosqlite:///{path}")
    await database.create_all()
    await database.upsert_daily("denver", [row(1)])
    assert [r.date for r in await database.daily("denver", date(2025, 1, 1), date(2025, 1, 31))] == [date(2025, 1, 1)]
    await database.dispose()
    assert path.is_file()
