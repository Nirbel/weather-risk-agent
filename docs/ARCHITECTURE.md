# Architecture

A chat agent that ranks, compares and explains the weather-disruption exposure of 10 US logistics hubs.
The core rule: **the LLM never produces a number.** It turns a question into a schema-validated query plan
and explains the result. Deterministic Python computes every score, rank and percentage.

Why each choice was made: [DECISIONS.md](DECISIONS.md).

## 1. Components and how they talk

```
┌─────────────────────────┐  HTTP/JSON + bearer token  ┌──────────────────────────────────────────────┐
│ Streamlit UI  :8501     │ ─────────────────────────► │ FastAPI  :8000                               │
│ Home·Chat·Analytics·    │                            │                                              │
│ Alerts (no logic)       │                            │  Chat turn (agent/service.py)                │
└─────────────────────────┘                            │   1. Planner   ── LLM ──► QueryPlan (schema) │
                                                       │   2. Executor  (Python) ──► ResultBundle     │
  External cron ── POST /alerts/run ─────────────────► │   3. Explainer ── LLM ──► Explanation        │
                                                       │      + grounding check (numbers must match)  │
                                                       │   4. Assembler (code) ──► AgentAnswer        │
                                                       │                                              │
                                                       │  Analyzer (analysis.py): one scoring path    │
                                                       │  for chat, Analytics and alerts              │
                                                       │                                              │
                                                       │  Alerts (alerts.py): check ─► user webhooks  │
                                                       └───────┬──────────────────────┬───────────────┘
                                                               │                      │ LiteLLM
                               SQLite (WAL), async SQLAlchemy  │                      ▼
                                                               │              Groq gpt-oss-20b → Gemini fallback
                                                               │              Groq Whisper (voice input)
                                             ┌─────────────────┴──────────────────┐
                                             │ Data services (data.py)            │
                                             │ fetch on demand · cache · throttle │
                                             └────┬─────────────────────────┬─────┘
                                                  ▼                         ▼
                                          Open-Meteo ERA5 archive       FEMA National Risk Index
                                          (daily weather, 5 years)      (county loss rates, ArcGIS)
```

| Component | Code | Role | Talks to |
|---|---|---|---|
| UI | `ui/` | Log in / sign up, chat with history, Analytics charts, your webhook settings, voice input | API only, over HTTP (`API_URL`) |
| API | `api/main.py`, `api/auth.py` | Routes; email + password auth (scrypt hashes), signed tokens, roles | Agent, Analyzer, Alerts, DB |
| Agent | `agent/` | Plan → execute → explain → assemble, one turn at a time | LLM wrapper, Analyzer, DB (history) |
| LLM wrapper | `agent/llm.py` | The only door to a model: JSON schema, local Pydantic validation, one corrective retry, provider fallback | LiteLLM → Groq / Gemini |
| Analyzer | `analysis.py`, `scoring/` | Cached data → exposure breakdowns, per-year counts and "% of days" stats | Data services |
| Data services | `data.py`, `sources/` | Fetch missing data, cache it, rate-limit Open-Meteo, report gaps | Open-Meteo, FEMA NRI, DB |
| Alerts | `alerts.py` | Diff overall scores against each user's baseline and post changes to that user's webhook | Analyzer, DB, webhooks |
| Store | `db.py` | Async SQLAlchemy over one SQLite file | — |

### One chat turn

1. **Planner.** The question and the last 3 resolved plans go to the LLM. It returns a `QueryPlan`: intent, hubs or
   region, hazards, metric, time preset, weight overrides, and how it read the question. The plan is
   validated against the JSON schema and by extra rules (for example, *compare* needs at least 2 hubs). One
   retry with the errors; then a clear error answer.
2. **Executor.** Pure Python. It resolves dates, fetches any missing data, scores, ranks, runs the ±25 % weight
   sensitivity test, and returns rows, drivers, assumptions, sources and data gaps. It never ranks incomplete
   data as if it were complete (see *Data completeness* below).
3. **Explainer.** The LLM writes a summary and reasoning from the computed result only. A grounding check
   rejects any number not in the result or the question. One retry; then a deterministic template explanation.
4. **Assembler.** Code builds the `AgentAnswer`, so assumptions, uncertainty, sources and time window are
   present by construction. The turn is saved for follow-ups.

### Data completeness

- **Coverage rule** (`config/scoring.yaml → coverage`): a hub's weather record is complete only if at least
  99 % of the expected days have data and no run of missing days is longer than 3.
- **If it fails:** weather-based hazards (winter, flood, heat) and the overall score are not computed, so the
  hub is listed as *not ranked* / *not compared*, stat answers give no percentage, and the answer says why at
  high uncertainty.
- **A hazard with no evidence at all** (for example hurricane while FEMA NRI is unreachable) is handled the
  same way: no combined score that silently drops it.
- **Partial evidence inside a hazard** (NRI down, weather complete) still scores from the remaining lens,
  flagged as a data gap.

### Contracts (JSON schemas)

| Schema | Direction | File |
|---|---|---|
| `QueryPlan` | LLM → code | [schemas/query_plan.json](../schemas/query_plan.json) |
| `Explanation` | LLM → code | [schemas/explanation.json](../schemas/explanation.json) |
| `AgentAnswer` | API → UI | [schemas/agent_answer.json](../schemas/agent_answer.json) |

They are generated from the Pydantic models in `agent/schemas.py`; a test fails if they drift. `QueryPlan` is
flat with every field required and nullable, which Groq's strict mode and Gemini both accept.

### API

Interactive docs at `http://localhost:8000/docs`.

| Route | Who | Purpose |
|---|---|---|
| `POST /auth/login` | anyone | Email + password → signed bearer token with role |
| `POST /auth/signup` | `@moveo.co.il` only | Create an analyst account and log in |
| `POST /chat` | signed-in user | One agent turn; returns `AgentAnswer` |
| `GET /conversations`, `GET /conversations/{id}` | owner only | Conversation history |
| `POST /transcribe` | signed-in user | Voice: audio → text (the UI then calls `/chat`) |
| `GET /analytics/exposure`, `/analytics/yearly`, `/analytics/hubs/{id}` | signed-in user | The same numbers chat cites, without an LLM |
| `GET /alerts` | signed-in user | Your alert feed and when the last check ran |
| `GET/PUT /alerts/settings`, `POST /alerts/test` | signed-in user | Your webhook URL, on/off, threshold; send a test payload |
| `POST /alerts/run` | admin, or `X-Alert-Secret` | Run the check for every subscriber now |
| `POST /admin/warm` | admin | Fetch all hubs' data in the background |
| `GET /health` | anyone | Exposure window, source status, LLM and voice availability |

### Alerts

- **Trigger:** an in-process scheduler (daily by default, `ALERT_CHECK_HOURS`), or `POST /alerts/run` from
  an external cron.
- **Per user:** each user sets their own generic webhook (any http(s) endpoint: Make, n8n, Zapier,
  Slack/Teams workflows), on/off flag and threshold on the *Alerts* page.
- **Check:** score all hubs once, then compare each overall score with *each user's* baseline (the last
  value that user was told about). A change of at least their threshold is stored in their feed and posted
  to their webhook.
- **Safety:** http(s) only, 10 s timeout, no redirects. The destination is resolved and checked at save and
  at send: link-local / metadata addresses are always refused, private and loopback ones unless
  `ALLOW_PRIVATE_WEBHOOKS=true` (anyone with a `@moveo.co.il` address can sign up and choose a URL).
- **Data gaps:** a hub with a gap is skipped, so a failed fetch never looks like a score change.
- **Why alerts are rare:** exposure covers the previous 5 completed years. Scores move only when the window
  rolls over in January, FEMA publishes a new NRI release, the scoring config changes, or a data gap closes.

## 2. Repository structure

```
README.md                 How to run, demo questions
CLAUDE.md                 Design principles and development rules
config/
  hubs.yaml               10 hubs: location, county FIPS, Census region (validated at startup)
  scoring.yaml            Thresholds, anchors, weights, stat definitions (validated at startup)
docs/
  TASK.md  PLAN.md  DECISIONS.md  ARCHITECTURE.md  AI_SESSION.md
  session/                Exported AI session transcripts
schemas/                  JSON Schemas generated from the Pydantic models
scripts/
  run.py                  Starts the API and the UI together; Ctrl-C stops both
  record_fixtures.py      Records real API responses as test/eval fixtures
src/weather_risk/
  settings.py             Env / .env settings
  config.py               Typed models for hubs.yaml and scoring.yaml
  db.py                   Async SQLAlchemy models and queries
  data.py                 On-demand fetch, cache, rate limit, data gaps
  analysis.py             Cached data → exposure breakdowns and yearly counts
  timewindow.py           Date math ("last year", "last winter", archive lag)
  alerts.py               Score-change diff, alert run, webhook delivery, scheduler
  warm.py                 Optional cache warm-up CLI
  replay.py               Serves recorded responses to tests and eval (never at runtime)
  sources/                http.py · open_meteo.py · nri.py — one module per data source
  scoring/                indicators.py · exposure.py · robustness.py · stats.py
  agent/                  schemas · llm · planner · executor · explainer · grounding · answer · service · prompts/
  api/                    main.py (routes) · auth.py (email + password, sign-up rule, tokens)
ui/
  streamlit_app.py        Log in / sign up and top navigation
  api_client.py           Thin HTTP client
  app_pages/              home.py · chat.py · analytics.py · alerts.py
eval/                     cases.yaml (22 cases) · run.py (offline, --live) · online.py (--online, end to end)
tests/                    One test file per core module; fixtures/recorded/ holds real API responses
Dockerfile                One image for both services
docker-compose.yml        Two services: backend (FastAPI) and frontend (Streamlit)
```

No LangChain or agent framework, and no plugin registry: one module per data source, one wrapper per model call.

## 3. Data storage

**Choice: SQLite in one file (`data/app.db`), through async SQLAlchemy 2.0 with aiosqlite.**

**Why SQLite**
- The data is small: 10 hubs × 5 years ≈ 18k daily rows, plus one cached NRI payload.
- It needs date-range queries ("% of days in a window") and atomic writes for conversations and alerts.
- Zero ops, a single file, and reviewers can open it with `sqlite3`.

**Why async SQLAlchemy** (D14)
- Typed models that fit the async FastAPI and httpx stack.
- Moving to Postgres is a change of `DATABASE_URL`.

**Concurrency.** SQLite allows one writer, so the store serializes writes with an async lock. WAL mode lets
reads run while the API writes. The app runs one API process, which also hosts the alert scheduler.

| Table | Holds |
|---|---|
| `weather_daily` | One row per hub per day: snowfall, precipitation, Tmax, Tmin, max gust |
| `weather_fetches` | Which (hub, year) was fetched, when, whether complete, and the ERA5 grid cell |
| `http_cache` | Payloads read whole: NRI percentiles for the hub counties, with fetch time and version |
| `source_status` | Last success and last error per source (feeds answers' data notes and `/health`) |
| `users` | Email, role, salted scrypt password hash (never the password) |
| `conversations`, `turns` | Per-user conversations; each turn stores the question, resolved plan and answer |
| `score_snapshots` | One row per alert check: each hub's overall score |
| `webhook_settings` | Per user: webhook URL, on/off, threshold, and that user's baseline scores |
| `webhook_alerts` | Per user: score changes (hub, old → new, delta) and webhook delivery status |

**Caching and refresh**

| Data | Rule |
|---|---|
| Weather, completed year | Fetched once per (hub, year), never again |
| Weather, current year | Refetched when older than 12 h (used only by stat questions) |
| FEMA NRI | Cached 30 days; on failure, the cached copy is used and the answer says so |

**Rejected**
- Postgres: no concurrency need, and it adds a service. It is the migration path.
- Plain JSON files: no range queries or transactions.
- DuckDB: built for analytics; this workload is small and transactional.

## 4. Running

- **Default:** `uv run python scripts/run.py`. It starts the API (:8000), waits for `/health`, then starts
  Streamlit (:8501); if either exits, it stops the other.
- **Server:** `docker compose up -d --build` runs the same image as two services. `backend` runs uvicorn and
  owns `.env` and the data volume. `frontend` runs Streamlit and reaches the backend at `http://backend:8000`.

See the [README](../README.md).
