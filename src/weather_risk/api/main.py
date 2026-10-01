"""FastAPI app: the agent's HTTP interface. The Streamlit UI is just one client of it.

    uv run python scripts/run.py        (API + UI)   ·   API docs at http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from weather_risk.agent.schemas import AgentAnswer, QueryPlan
from weather_risk.agent.service import Agent, ConversationAccessError
from weather_risk.analysis import build_analyzer
from weather_risk.api.auth import User, current_user, login, require_admin
from weather_risk.config import load_hubs, load_scoring_config, validate_config
from weather_risk.db import Database
from weather_risk.scoring.exposure import overall_score
from weather_risk.settings import Settings, get_settings, llm_configured
from weather_risk.sources.http import make_client

log = logging.getLogger("weather_risk.api")


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class LoginResponse(BaseModel):
    token: str
    email: str
    role: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


def _explain_plan(hub_id: str) -> QueryPlan:
    return QueryPlan.model_validate({
        "intent": "explain", "hubs": [hub_id], "region": None, "hazards": None, "metric": None, "top_k": None,
        "time_preset": None, "year": None, "start_date": None, "end_date": None, "weight_overrides": None,
        "interpretation_notes": [], "clarification_question": None, "out_of_scope_reason": None,
    })


def create_app(settings: Settings | None = None, *, db: Database | None = None, analyzer=None,
               completion=None) -> FastAPI:
    """App factory. Tests inject a database, a (fake) analyzer and a fake LLM completion."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        validate_config()  # fail fast on a bad hubs.yaml / scoring.yaml
        app.state.settings = settings or get_settings()
        app.state.db = db or Database(app.state.settings.database_url)
        await app.state.db.create_all()
        client = None
        if analyzer is None:
            client = make_client(app.state.settings.contact_email)
            app.state.analyzer = build_analyzer(app.state.db, client)
        else:
            app.state.analyzer = analyzer
        app.state.agent = Agent(app.state.db, app.state.analyzer, app.state.settings, completion=completion)
        if app.state.settings.jwt_secret.startswith("dev-only"):
            log.warning("JWT_SECRET is the development default — set it in .env.")
        yield
        if client is not None:
            await client.aclose()
        if db is None:
            await app.state.db.dispose()

    app = FastAPI(title="Weather Risk Intelligence Agent", version="0.2.0", lifespan=lifespan,
                  description="Ranks, compares and explains the weather-disruption exposure of US logistics hubs.")

    # -- auth --------------------------------------------------------------------------
    @app.post("/auth/login", response_model=LoginResponse, tags=["auth"])
    def auth_login(body: LoginRequest, request: Request):
        user, token = login(body.email, request.app.state.settings)
        return LoginResponse(token=token, email=user.email, role=user.role)

    # -- agent -------------------------------------------------------------------------
    @app.post("/chat", response_model=AgentAnswer, tags=["agent"])
    async def chat(body: ChatRequest, request: Request, user: User = Depends(current_user)):
        try:
            return await request.app.state.agent.ask(body.message, body.conversation_id, user.email)
        except ConversationAccessError as exc:
            raise HTTPException(404, "Conversation not found") from exc

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
            "llm_models": [st.settings.llm_model, st.settings.llm_fallback_model],
            "exposure_window": {"start": start.isoformat(), "end": end.isoformat()},
            "hubs": len(load_hubs()),
            "sources": await st.db.source_statuses(),
        }

    return app


app = create_app()
