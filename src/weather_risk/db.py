"""Persistence: async SQLAlchemy 2.0 over SQLite (aiosqlite). See docs/DECISIONS.md D14.

Tables are small and few: cached daily weather + a per-(hub, year) fetch log, a JSON cache
for the NRI percentiles, per-source status, and conversations for follow-up questions.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager, nullcontext
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, event, func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from weather_risk.sources.open_meteo import DailyRow

DAILY_COLS = ("snowfall_cm", "precip_mm", "tmax_c", "tmin_c", "gust_kmh")


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).isoformat(timespec="seconds")


class Base(DeclarativeBase):
    pass


class WeatherDay(Base):
    __tablename__ = "weather_daily"
    hub_id: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    snowfall_cm: Mapped[float | None] = mapped_column(Float)
    precip_mm: Mapped[float | None] = mapped_column(Float)
    tmax_c: Mapped[float | None] = mapped_column(Float)
    tmin_c: Mapped[float | None] = mapped_column(Float)
    gust_kmh: Mapped[float | None] = mapped_column(Float)


class WeatherFetch(Base):
    """One row per (hub, calendar year) fetched. A complete year is never fetched again."""

    __tablename__ = "weather_fetches"
    hub_id: Mapped[str] = mapped_column(String, primary_key=True)
    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    complete: Mapped[bool] = mapped_column(Boolean)
    grid_lat: Mapped[float | None] = mapped_column(Float)
    grid_lon: Mapped[float | None] = mapped_column(Float)
    grid_elevation: Mapped[float | None] = mapped_column(Float)


class CacheEntry(Base):
    __tablename__ = "http_cache"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[Any] = mapped_column(JSON)


class SourceStatus(Base):
    __tablename__ = "source_status"
    source: Mapped[str] = mapped_column(String, primary_key=True)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_email: Mapped[str | None] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Turn(Base):
    __tablename__ = "turns"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    user_text: Mapped[str] = mapped_column(Text)
    plan: Mapped[Any | None] = mapped_column(JSON)
    answer: Mapped[Any] = mapped_column(JSON)


class ScoreSnapshot(Base):
    """The alert baseline: each hub's overall score as analysts were last told it."""

    __tablename__ = "score_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    taken_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    scores: Mapped[Any] = mapped_column(JSON)  # hub_id → overall score


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    hub_id: Mapped[str] = mapped_column(String)
    old_score: Mapped[float] = mapped_column(Float)
    new_score: Mapped[float] = mapped_column(Float)
    delta: Mapped[float] = mapped_column(Float)
    delivery: Mapped[str] = mapped_column(String)


class AlertSettings(Base):
    """Single row (id 1), edited by an admin in the UI. Absent → env defaults."""

    __tablename__ = "alert_settings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    webhook_url: Mapped[str | None] = mapped_column(String)
    enabled: Mapped[bool] = mapped_column(Boolean)
    threshold: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_by: Mapped[str] = mapped_column(String)


class Database:
    def __init__(self, url: str):
        memory = ":memory:" in url
        self.engine = create_async_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool if memory else None,  # one shared connection keeps an in-memory DB alive
        )
        if not memory:
            @event.listens_for(self.engine.sync_engine, "connect")
            def _wal(dbapi_conn, _):  # concurrent readers while the API writes
                dbapi_conn.execute("PRAGMA journal_mode=WAL")

        self.session: async_sessionmaker[AsyncSession] = async_sessionmaker(self.engine, expire_on_commit=False)
        # SQLite has a single writer: serialize writes here instead of hitting "database is locked".
        # An in-memory DB shares one connection, so its reads are serialized too.
        self._write_lock = asyncio.Lock()
        self._memory = memory

    @asynccontextmanager
    async def _tx(self):
        async with self._write_lock, self.session.begin() as s:
            yield s

    @asynccontextmanager
    async def _read(self):
        async with self._write_lock if self._memory else nullcontext(), self.session() as s:
            yield s

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

    # -- weather -------------------------------------------------------------------
    async def upsert_daily(self, hub_id: str, rows: list[DailyRow]) -> None:
        if not rows:
            return
        values = [{"hub_id": hub_id, "date": r.date, **{c: getattr(r, c) for c in DAILY_COLS}} for r in rows]
        stmt = insert(WeatherDay).values(values)
        stmt = stmt.on_conflict_do_update(index_elements=["hub_id", "date"],
                                          set_={c: stmt.excluded[c] for c in DAILY_COLS})
        async with self._tx() as s:
            await s.execute(stmt)

    async def daily(self, hub_id: str, start: date, end: date) -> list[DailyRow]:
        async with self._read() as s:
            result = await s.scalars(
                select(WeatherDay).where(WeatherDay.hub_id == hub_id, WeatherDay.date.between(start, end))
                .order_by(WeatherDay.date)
            )
            return [DailyRow(r.date, *(getattr(r, c) for c in DAILY_COLS)) for r in result]

    async def record_fetch(self, hub_id: str, year: int, *, complete: bool, grid: dict | None,
                           fetched_at: datetime | None = None) -> None:
        async with self._tx() as s:
            await s.merge(WeatherFetch(
                hub_id=hub_id, year=year, fetched_at=fetched_at or utcnow(), complete=complete,
                grid_lat=grid["latitude"] if grid else None, grid_lon=grid["longitude"] if grid else None,
                grid_elevation=grid["elevation"] if grid else None,
            ))

    async def fetch_log(self, hub_id: str) -> dict[int, WeatherFetch]:
        async with self._read() as s:
            result = await s.scalars(select(WeatherFetch).where(WeatherFetch.hub_id == hub_id))
            return {f.year: f for f in result}

    # -- cache (payloads read whole, e.g. NRI percentiles) --------------------------
    async def cache_put(self, key: str, source: str, payload: Any, fetched_at: datetime | None = None) -> None:
        async with self._tx() as s:
            await s.merge(CacheEntry(key=key, source=source, fetched_at=fetched_at or utcnow(), payload=payload))

    async def cache_get(self, key: str) -> tuple[str, Any] | None:
        async with self._read() as s:
            entry = await s.get(CacheEntry, key)
            return (iso(entry.fetched_at), entry.payload) if entry else None

    # -- source status --------------------------------------------------------------
    async def source_ok(self, source: str) -> None:
        async with self._tx() as s:
            status = await s.get(SourceStatus, source) or SourceStatus(source=source)
            status.last_success_at = utcnow()
            s.add(status)

    async def source_error(self, source: str, error: str) -> None:
        async with self._tx() as s:
            status = await s.get(SourceStatus, source) or SourceStatus(source=source)
            status.last_error, status.last_error_at = error[:300], utcnow()
            s.add(status)

    async def source_statuses(self) -> dict[str, dict]:
        async with self._read() as s:
            return {
                st.source: {"last_success_at": iso(st.last_success_at), "last_error": st.last_error,
                            "last_error_at": iso(st.last_error_at)}
                for st in await s.scalars(select(SourceStatus))
            }

    # -- conversations ----------------------------------------------------------------
    async def create_conversation(self, user_email: str | None, title: str) -> str:
        cid = uuid.uuid4().hex
        now = utcnow()
        async with self._tx() as s:
            s.add(Conversation(id=cid, user_email=user_email, title=title[:80], created_at=now, updated_at=now))
        return cid

    async def conversation_owner(self, cid: str) -> str | None:
        async with self._read() as s:
            conv = await s.get(Conversation, cid)
            return conv.user_email if conv else None

    async def add_turn(self, cid: str, user_text: str, plan: dict | None, answer: dict) -> None:
        async with self._tx() as s:
            s.add(Turn(conversation_id=cid, created_at=utcnow(), user_text=user_text, plan=plan, answer=answer))
            conv = await s.get(Conversation, cid)
            conv.updated_at = utcnow()

    async def recent_turns(self, cid: str, limit: int = 3) -> list[dict]:
        async with self._read() as s:
            turns = list(await s.scalars(
                select(Turn).where(Turn.conversation_id == cid).order_by(Turn.id.desc()).limit(limit)))
        return [_turn(t) for t in reversed(turns)]

    async def conversation_turns(self, cid: str) -> list[dict]:
        async with self._read() as s:
            turns = await s.scalars(select(Turn).where(Turn.conversation_id == cid).order_by(Turn.id))
            return [_turn(t) for t in turns]

    async def list_conversations(self, user_email: str, limit: int = 50) -> list[dict]:
        async with self._read() as s:
            counts = select(Turn.conversation_id, func.count().label("n")).group_by(Turn.conversation_id).subquery()
            rows = await s.execute(
                select(Conversation, counts.c.n).outerjoin(counts, counts.c.conversation_id == Conversation.id)
                .where(Conversation.user_email == user_email)
                .order_by(Conversation.updated_at.desc()).limit(limit)
            )
            return [{"id": c.id, "title": c.title, "updated_at": iso(c.updated_at), "turns": n or 0} for c, n in rows]

    # -- alerts -----------------------------------------------------------------------
    async def latest_snapshot(self) -> dict | None:
        async with self._read() as s:
            snap = await s.scalar(select(ScoreSnapshot).order_by(ScoreSnapshot.id.desc()).limit(1))
            return {"taken_at": iso(snap.taken_at), "scores": snap.scores} if snap else None

    async def add_snapshot(self, scores: dict[str, float]) -> None:
        async with self._tx() as s:
            s.add(ScoreSnapshot(taken_at=utcnow(), scores=scores))

    async def add_alerts(self, rows: list[dict], delivery: str) -> None:
        now = utcnow()
        async with self._tx() as s:
            s.add_all(Alert(created_at=now, delivery=delivery, **row) for row in rows)

    async def recent_alerts(self, limit: int = 50) -> list[dict]:
        async with self._read() as s:
            alerts = await s.scalars(select(Alert).order_by(Alert.id.desc()).limit(limit))
            return [{"created_at": iso(a.created_at), "hub_id": a.hub_id, "old_score": a.old_score,
                     "new_score": a.new_score, "delta": a.delta, "delivery": a.delivery} for a in alerts]

    async def alert_settings(self) -> dict | None:
        async with self._read() as s:
            row = await s.get(AlertSettings, 1)
            return None if row is None else {
                "webhook_url": row.webhook_url, "enabled": row.enabled, "threshold": row.threshold,
                "updated_at": iso(row.updated_at), "updated_by": row.updated_by}

    async def save_alert_settings(self, *, webhook_url: str | None, enabled: bool, threshold: float,
                                  updated_by: str) -> None:
        async with self._tx() as s:
            await s.merge(AlertSettings(id=1, webhook_url=webhook_url, enabled=enabled, threshold=threshold,
                                        updated_at=utcnow(), updated_by=updated_by))


def _turn(t: Turn) -> dict:
    return {"user_text": t.user_text, "plan": t.plan, "answer": t.answer, "created_at": iso(t.created_at)}
