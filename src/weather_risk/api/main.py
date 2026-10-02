"""FastAPI app: the agent's HTTP interface. The Streamlit UI is just one client of it.

    uv run python scripts/run.py        (API + UI)   ·   API docs at http://localhost:8000/docs
"""

import asyncio
import contextlib
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from weather_risk.agent import llm
from weather_risk.agent.schemas import AgentAnswer, QueryPlan
from weather_risk.agent.service import Agent, ConversationAccessError
from weather_risk.alerts import AlertService, run_on_schedule
from weather_risk.analysis import build_analyzer
from weather_risk.api.auth import (MAX_PASSWORD, User, current_user, ensure_demo_admin, login, require_admin,
                                   require_admin_or_trigger_secret, signup)
from weather_risk.config import load_hubs, load_scoring_config, validate_config
from weather_risk.db import Database
from weather_risk.scoring.exposure import overall_score
from weather_risk.settings import DEFAULT_DEMO_PASSWORD, Settings, get_settings, llm_configured, voice_configured
from weather_risk.sources.http import make_client

MAX_AUDIO_BYTES = 10 * 1024 * 1024

log = logging.getLogger("weather_risk.api")


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD)


class SignupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # no "role" field: sign-up always creates an analyst
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(max_length=MAX_PASSWORD)


class LoginResponse(BaseModel):
    token: str
    email: str
    role: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


class AlertSettingsRequest(BaseModel):
    webhook_url: str | None = Field(default=None, max_length=2000)
    enabled: bool
    threshold: float = Field(gt=0, le=100, description="Points on the 0–100 overall Exposure Score")


def _explain_plan(hub_id: str) -> QueryPlan:
    return QueryPlan.model_validate({
        "intent": "explain", "hubs": [hub_id], "region": None, "hazards": None, "metric": None, "top_k": None,
        "time_preset": None, "year": None, "start_date": None, "end_date": None, "weight_overrides": None,
        "interpretation_notes": [], "clarification_question": None, "out_of_scope_reason": None,
    })


def create_app(settings: Settings | None = None, *, db: Database | None = None, analyzer=None,
               completion=None, transcription=None, resolve=None) -> FastAPI:
    """App factory. Tests inject a database, a (fake) analyzer, fake LLM completion/transcription
    and a fake DNS resolver for webhook destination checks."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        validate_config()  # fail fast on a bad hubs.yaml / scoring.yaml
        app.state.settings = settings or get_settings()
        if len((app.state.settings.jwt_secret or "").encode()) < 32:
            log.warning("JWT_SECRET is unset or shorter than 32 bytes — using a random one; "
                        "users log in again after a restart.")
            app.state.settings = app.state.settings.model_copy(update={"jwt_secret": secrets.token_urlsafe(32)})
        app.state.db = db or Database(app.state.settings.database_url)
        await app.state.db.create_all()
        await ensure_demo_admin(app.state.db, app.state.settings)
        if app.state.settings.demo_admin_password == DEFAULT_DEMO_PASSWORD:
            log.warning("The demo admin %s uses the default password from the task — set DEMO_ADMIN_PASSWORD "
                        "before exposing this server to others.", app.state.settings.demo_admin_email)
        client = make_client(app.state.settings.contact_email)
        app.state.analyzer = analyzer or build_analyzer(app.state.db, client)
        app.state.agent = Agent(app.state.db, app.state.analyzer, app.state.settings, completion=completion)
        app.state.alerts = AlertService(app.state.db, app.state.analyzer, app.state.settings, client,
                                        **({"resolve": resolve} if resolve else {}))
        scheduler = None
        if app.state.settings.scheduler_enabled:
            scheduler = asyncio.create_task(run_on_schedule(app.state.alerts, app.state.settings.alert_check_hours))
        yield
        if scheduler is not None:
            scheduler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await scheduler
        await client.aclose()
        await app.state.db.dispose()

    app = FastAPI(title="Weather Risk Intelligence Agent", version="0.2.0", lifespan=lifespan,
                  description="Ranks, compares and explains the weather-disruption exposure of US logistics hubs.")

    # -- auth --------------------------------------------------------------------------
    @app.post("/auth/login", response_model=LoginResponse, tags=["auth"])
    async def auth_login(body: LoginRequest, request: Request):
        user, token = await login(request.app.state.db, body.email, body.password, request.app.state.settings)
        return LoginResponse(token=token, email=user.email, role=user.role)

    @app.post("/auth/signup", response_model=LoginResponse, status_code=201, tags=["auth"])
    async def auth_signup(body: SignupRequest, request: Request):
        """Create an analyst account (@moveo.co.il only) and log in."""
        user, token = await signup(request.app.state.db, body.email, body.password, request.app.state.settings)
        return LoginResponse(token=token, email=user.email, role=user.role)

    # -- agent -------------------------------------------------------------------------
    @app.post("/chat", response_model=AgentAnswer, tags=["agent"])
    async def chat(body: ChatRequest, request: Request, user: User = Depends(current_user)):
        try:
            return await request.app.state.agent.ask(body.message, body.conversation_id, user.email)
        except ConversationAccessError as exc:
            raise HTTPException(404, "Conversation not found") from exc

    @app.post("/transcribe", tags=["agent"])
    async def transcribe(audio: UploadFile, request: Request, user: User = Depends(current_user)):
        """Voice input (bonus): speech → text via LiteLLM (Groq Whisper). The UI then sends the text to /chat."""
        st = request.app.state.settings
        if not voice_configured(st):
            raise HTTPException(503, f"Voice input needs an API key for {st.transcribe_model}.")
        data = await audio.read()
        if not data:
            raise HTTPException(400, "The recording is empty.")
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "The recording is too long (10 MB max).")
        try:
            text = await llm.transcribe(data, audio.filename or "question.wav", model=st.transcribe_model,
                                        transcription=transcription, timeout=st.llm_timeout_s)
        except Exception as exc:
            log.warning("transcription failed: %s", exc)
            raise HTTPException(502, f"Transcription failed ({exc.__class__.__name__}).") from exc
        return {"text": text}

    @app.get("/conversations", tags=["agent"])
    async def conversations(request: Request, user: User = Depends(current_user)):
        return await request.app.state.db.list_conversations(user.email)

    @app.get("/conversations/{conversation_id}", tags=["agent"])
    async def conversation(conversation_id: str, request: Request, user: User = Depends(current_user)):
        db: Database = request.app.state.db
        if await db.conversation_owner(conversation_id) != user.email:
            raise HTTPException(404, "Conversation not found")
        turns = await db.conversation_turns(conversation_id)
        return {"id": conversation_id, "turns": [{"user_text": t["user_text"], "answer": t["answer"],
                                                  "created_at": t["created_at"]} for t in turns]}

    # -- data & analytics (same deterministic code as chat) -----------------------------
    @app.get("/hubs", tags=["data"])
    def hubs(user: User = Depends(current_user)):
        return [h.model_dump() for h in load_hubs()]

    @app.get("/analytics/exposure", tags=["analytics"])
    async def analytics_exposure(request: Request, user: User = Depends(current_user)):
        """Exposure per hub and hazard family, with lens scores — the numbers chat cites."""
        cfg = load_scoring_config()
        result = await request.app.state.analyzer.exposure()
        names = {h.id: h for h in load_hubs()}
        rows = []
        for hub_id, bd in result.breakdowns.items():
            rows.append({
                "hub_id": hub_id, "hub": names[hub_id].name, "region": names[hub_id].region,
                "overall": overall_score(bd, cfg.family_weights),
                "top_family": bd["top_family"],
                "families": {f: {"label": fam["label"], "score": fam["score"],
                                 **{lens: data["score"] for lens, data in fam["lenses"].items()}}
                             for f, fam in bd["families"].items()},
                "data_gaps": bd["data_gaps"],
                "weather_complete": bd["meta"]["observed_window"].get("complete", True),
                "coverage_pct": bd["meta"]["observed_window"].get("coverage_pct"),
            })
        rows.sort(key=lambda r: -(r["overall"] or 0))
        return {"window": {"start": result.start.isoformat(), "end": result.end.isoformat()},
                "nri_version": result.nri_version, "hubs": rows}

    @app.get("/analytics/yearly", tags=["analytics"])
    async def analytics_yearly(request: Request, user: User = Depends(current_user)):
        """Per-year counts of every observed indicator (historical exposure metrics)."""
        return await request.app.state.analyzer.yearly_indicators()

    @app.get("/analytics/hubs/{hub_id}", response_model=AgentAnswer, tags=["analytics"])
    async def analytics_hub(hub_id: str, request: Request, user: User = Depends(current_user)):
        """Deterministic score breakdown for one hub — the same answer chat gives, without an LLM."""
        if hub_id not in {h.id for h in load_hubs()}:
            raise HTTPException(404, f"Unknown hub {hub_id!r}")
        return await request.app.state.agent.answer_plan(_explain_plan(hub_id), f"Explain {hub_id}", use_llm=False)

    # -- alerts (bonus): each user's own score-change webhook -----------------------------
    @app.get("/alerts", tags=["alerts"])
    async def alerts(request: Request, user: User = Depends(current_user)):
        """Your recent score-change alerts, your settings, and when the last check ran."""
        db: Database = request.app.state.db
        cfg = await request.app.state.alerts.config(user.email)
        names = {h.id: h.name for h in load_hubs()}
        snapshot = await db.latest_snapshot()
        st = request.app.state.settings
        return {"last_check": snapshot["taken_at"] if snapshot else None, "threshold": cfg["threshold"],
                "delivering": bool(cfg["enabled"] and cfg["webhook_url"]),
                "check_every_hours": st.alert_check_hours if st.scheduler_enabled else None,
                "alerts": [a | {"hub": names.get(a["hub_id"], a["hub_id"])}
                           for a in await db.recent_alerts(user.email)]}

    @app.get("/alerts/settings", tags=["alerts"])
    async def alert_settings(request: Request, user: User = Depends(current_user)):
        return await request.app.state.alerts.config(user.email)

    @app.put("/alerts/settings", tags=["alerts"])
    async def update_alert_settings(body: AlertSettingsRequest, request: Request, user: User = Depends(current_user)):
        """Save your webhook URL (any http(s) endpoint), on/off flag and threshold."""
        try:
            return await request.app.state.alerts.update_settings(user.email, **body.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/alerts/test", tags=["alerts"])
    async def test_alert(request: Request, user: User = Depends(current_user)):
        """Send a sample payload to your saved webhook URL."""
        return await request.app.state.alerts.send_test(user.email)

    @app.post("/alerts/run", tags=["alerts"])
    async def run_alerts(request: Request, caller: str = Depends(require_admin_or_trigger_secret)):
        """Check every subscriber now. Admin token, or header X-Alert-Secret for an external cron."""
        return await request.app.state.alerts.run()

    # -- ops ---------------------------------------------------------------------------
    @app.post("/admin/warm", status_code=202, tags=["admin"])
    async def admin_warm(request: Request, background: BackgroundTasks, user: User = Depends(require_admin)):
        """Fetch and cache all hubs' data in the background (optional, before a demo)."""
        async def job():
            result = await request.app.state.analyzer.exposure()
            log.info("cache warmed for %d hubs", len(result.breakdowns))

        background.add_task(job)
        return {"status": "started"}

    @app.get("/health", tags=["ops"])
    async def health(request: Request):
        st = request.app.state
        start, end = st.analyzer.window()
        return {
            "status": "ok",
            "llm_configured": llm_configured(st.settings),
            "voice": voice_configured(st.settings),
            "llm_models": [st.settings.llm_model, st.settings.llm_fallback_model],
            "exposure_window": {"start": start.isoformat(), "end": end.isoformat()},
            "hubs": len(load_hubs()),
            "sources": await st.db.source_statuses(),
        }

    return app


app = create_app()
