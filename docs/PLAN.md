# Weather Risk Intelligence Agent — Plan

> **Rescope addendum (2026-10-01, mid-build — requested by the user; see [DECISIONS.md D21](DECISIONS.md)).**
> The plan below is the approved original. These changes override it:
>
> | Area | Original plan | Now |
> |---|---|---|
> | Hubs | 15 | 10 representative hubs (4 Midwest + Denver, Dallas, Houston, Miami + Atlanta, Newark) |
> | Core sources | Open-Meteo archive + forecast, NWS, OpenFEMA, FEMA NRI | **Open-Meteo ERA5 archive + FEMA NRI only** |
> | Exposure window | 10 years (2016–2025), preloaded seed | **Previous 5 completed years**, fetched on demand per (hub, year) and cached |
> | Evidence lenses | observed / modeled / history | observed / modeled (hurricane and severe storm rest on NRI alone) |
> | Near-term score + alerts | core | removed from the core; alerts are a later bonus, on exposure-score changes |
> | Seed snapshot | runtime seed + fixtures | recorded responses are **test/eval fixtures only** |
> | Config | YAML dicts | YAML validated at startup by typed Pydantic models |
> | Persistence | stdlib `sqlite3` | **async SQLAlchemy 2.0 + aiosqlite** (call-out in D14) |
>
> **UI decisions for M4:**
> - Top navigation with three pages: **Home**, **Chat**, **Analytics**.
> - Home: welcome, methodology, data sources, limitations and example questions.
> - Chat: its sidebar holds only conversation history (New Chat, previous conversations, reopen and continue).
> - Voice input sits beside the chat input.
> - Analytics charts the same deterministic results chat uses.
> - Alert settings (later): generic webhook URL, enabled flag, score-change threshold, Test Webhook. Webhooks stay provider-agnostic.
>
> **Status (2026-10-02):** M1–M8 done, plus a review round requested by the user: email + password auth with `@moveo.co.il` sign-up (D18), per-user webhooks (D22), the coverage rule (D24), an online end-to-end eval (D25), clean-clone startup and a two-service Compose file (D15). The app runs with `uv run python scripts/run.py`. Design: [ARCHITECTURE.md](ARCHITECTURE.md).
>
> **Milestones now:**
> - M1–M3 rescoped and done.
> - M4: API + Streamlit UI (Home/Chat/Analytics) + run script + Docker.
> - M5: eval.
> - M6: auth (built with M4).
> - M7: docs.
> - M8 bonuses: alerts with webhook settings, then voice.

## Context

A logistics company chooses a few US distribution hubs each year for weather-resilience investment. Analysts need an agent that answers questions such as:

- *Which hubs in the Midwest are most exposed to winter disruption?*
- *Compare Miami and Houston in terms of hurricane and flood exposure.*
- *What percentage of days in Denver last year had snowfall?*
- *Why is the Dallas hub's weather disruption risk high?*

Answers must be auditable, explain their reasoning, state assumptions and uncertainty, and support follow-up questions.

**Core design stance: the LLM never produces a number.**
1. The LLM turns a question into a schema-validated **query plan**.
2. Deterministic Python computes every score, rank and percentage.
3. The LLM explains the computed result; code checks that every number it mentions came from the result.

**Stack:** Python 3.14 · FastAPI · Pydantic v2 · async httpx · Streamlit (calls the API over HTTP) · LiteLLM (Groq `openai/gpt-oss-20b` primary, Gemini fallback) · SQLite · pytest + pytest-asyncio + respx (every external call mocked) · light email allow-list auth, built after the core works.

---

## Confirmed decisions

| # | Decision | Source |
|---|---|---|
| 1 | Overall Exposure Score = **equal-weight mean** of the 5 hazard-family scores | Q&A |
| 2 | **Open-Meteo only** for observed weather (no station cross-check); the reanalysis caveat is stated in answers | Q&A |
| 3 | **No cloud deploy for now.** Runs locally with or without Docker; host chosen later | Q&A |
| 4 | Repo lives in **`~/Desktop/weather-risk-agent`** (already exists, contains only a JetBrains `.idea/`; keep it and gitignore it) | Q&A |
| 5 | LLM model: **Groq `openai/gpt-oss-20b`** (supports strict JSON-schema decoding), Gemini fallback | Feedback |
| 6 | **One run script starts backend + Streamlit together** — no Makefile | Feedback |
| 7 | **Pragmatic TDD for core logic** (see §8) | Feedback |
| 8 | Docs are short, clear Markdown | Feedback |

---

## 0. Data sources — verified with live read-only calls (2026-10-01)

| Source | Endpoint | Key | Limits / terms | Used for |
|---|---|---|---|---|
| Open-Meteo Archive (ERA5 reanalysis) | `archive-api.open-meteo.com/v1/archive` | No | Free tier is non-commercial; 600/min, 5k/hour, 10k/day. Requests over 2 weeks or 10 variables count as several calls. CC-BY 4.0. | 10-year observed climatology, "% of days" stats |
| Open-Meteo Forecast | `api.open-meteo.com/v1/forecast?forecast_days=7` | No | Same as above | Near-term risk |
| NWS alerts | `api.weather.gov/alerts/active?point=lat,lon` | No (User-Agent required) | Unpublished limit; retry after ~5 s. Keeps only 7 days of alerts | Near-term risk (active warnings) |
| OpenFEMA declarations | `fema.gov/api/open/v2/DisasterDeclarationsSummaries` (OData `$filter` by state + county FIPS) | No | Max 10k rows per call | Disaster history per county |
| FEMA National Risk Index | ArcGIS `services.arcgis.com/XG15cJAlne2vxtgt/.../National_Risk_Index_Counties/FeatureServer/0/query` | No | 2,000 rows per page; 3,232 counties = 2 pages | Modeled hazard loss per county |
| FCC Area API (one-off, not at runtime) | `geo.fcc.gov/api/census/area?lat=&lon=` | No | — | Map each hub's coordinates to a county FIPS code once; commit to `hubs.yaml` |

### Findings that changed the design

1. **Open-Meteo call cost.** 10 years for one hub ≈ 261 calls; all 15 hubs ≈ 3,900 calls. 30-year normals (~11,700) exceed the daily cap → use **10 full years (2016–2025)**, cache history forever, pace the first ingest, and ship a recorded seed snapshot.
2. **Thresholds dominate "% of days" answers.** Denver 2025 (Open-Meteo): any snow > 0 = 44 days (12.1%); measurable ≥ 0.25 cm / 0.1 in = 31 (8.5%); ≥ 1 cm = 18 (4.9%); ≥ 2.5 cm / 1 in = 10 (2.7%). → Thresholds live in config, and every stat answer prints the threshold plus a sensitivity row.
3. **Reanalysis ≠ station.** NOAA's Denver airport station shows 26 measurable-snow days in 2025 vs 31 from Open-Meteo, and 75.7 cm vs 57.5 cm total snow. → Answers carry a reanalysis caveat. Coordinates also snap to a grid cell (Denver: ~3 km off).
4. **NRI's headline score can't rank hubs.** `RISK_SCORE` is 95–100 for every metro (Denver 95.2, Cook 99.97, Dallas 99.65) because it scales with the dollar value exposed in the county.
5. **NRI's composite loss-rate percentile includes agriculture.** Miami's cold-wave percentile is 91 because of crop freezes; its building loss rate is negligible. → Use the **building** loss rate `{HAZARD}_ALRB` and compute our own national percentile across all 3,232 counties. A null value means "hazard not applicable" → 0.
6. **NRI renamed riverine flooding:** it is now `IFLD` (Inland Flooding) alongside `CFLD` (Coastal). NRI version: "December 2025".
7. **OpenFEMA mixes in non-weather records.** Harris County since 2000 has 26 records, including COVID (Biological) and Fire, and both DR (major disaster) and EM (emergency, often before landfall). → Keep weather incident types only, **DR only**, de-duplicated by disaster number.
8. **NWS has no history** → it is used only for near-term risk.

### LLM provider facts

- Groq strict JSON-schema decoding works on `openai/gpt-oss-20b` (and 120b). Strict mode requires every field to be `required` (optional values become nullable) and `additionalProperties: false`.
- Groq free tier for gpt-oss-20b: 30 requests/min, 1K/day, **8K tokens/min** (the binding limit) → keep prompts small, fall back to Gemini on 429.
- Gemini structured output supports `anyOf`, `$ref`, enums and nullable types. Model IDs come from env; confirmed with a live call in M3.
- **Security:** LiteLLM 1.82.7 and 1.82.8 were malicious releases (PyPI supply-chain attack, 2026-03-24). Exclude them and pin the resolved version (1.103.2) with hashes in `uv.lock`.
- All dependencies resolve to binary wheels on CPython 3.14 (local: 3.14.7, uv 0.11.6).
- `GROQ_API_KEY` and `GEMINI_API_KEY` are not set in this shell → needed in `.env` before M3.

---

## 1. Architecture

```
┌──────────────┐   HTTP/JSON + bearer token   ┌──────────────────────────────────────────┐
│ Streamlit UI │ ───────────────────────────► │ FastAPI                                  │
│ chat, tables,│                              │                                          │
│ "how it was  │                              │  Agent turn:                             │
│  computed",  │                              │   1. Planner   ── LLM ──► QueryPlan      │
│ alerts feed  │                              │   2. Executor  (pure Python) ► Result    │
└──────────────┘                              │   3. Explainer ── LLM ──► Explanation    │
                                              │   4. Assembler (code) ► AgentAnswer      │
                                              │                                          │
                                              │  Scoring: exposure · near-term · stats   │
                                              └───────┬───────────────────────┬──────────┘
                                                      │                       │
                                         SQLite (WAL) │                       │ LiteLLM
                                                      │                       ▼
                         ┌────────────────────────────┴───┐         Groq gpt-oss-20b
                         │ Ingest (CLI + admin endpoint)  │         └─ fallback: Gemini
                         │ async httpx: Open-Meteo, NWS,  │
                         │ OpenFEMA, NRI                  │
                         └────────────────────────────────┘
   Alerts: in-process scheduler (hourly) or any cron → POST /alerts/run → optional outbound webhook
```

### How components talk

- **UI → API:** HTTP/JSON only. The UI has no business logic and finds the API via the `API_URL` env var.
- **API → LLM:** one wrapper, `structured_call(messages, PydanticModel)`.
  - Sends the JSON schema in `response_format`.
  - Always validates locally with Pydantic, whatever the provider promises.
  - On a validation error: one retry with the error included, then a clear error answer.
  - Provider failover (Groq → Gemini on 429/5xx/timeout) is separate, via LiteLLM `fallbacks`. The model that answered is recorded in the response.
- **API → data:** chat reads precomputed scores from SQLite, so a conversation never waits on rate-limited APIs. Data is fetched on demand only when it is missing or stale.

### Why plan → execute → explain (not a free-form tool-calling loop)

- Each target question maps to one deterministic operation.
- A single validated plan is easy to evaluate field by field.
- Two LLM calls per turn fit Groq's 8K tokens/min.
- It removes loop and tool-argument-drift failures.
- **Limitation (documented):** compound questions are answered by the richer operation; for example, *compare* already includes per-hazard breakdowns.

### API endpoints

`POST /auth/login` · `POST /chat` · `GET /hubs` · `GET /scores?kind=exposure|near_term` · `GET /hubs/{id}/explain` (breakdown, no LLM) · `GET /alerts` · `POST /alerts/run` · `POST /admin/ingest` · `GET /health` (source freshness) · `POST /transcribe` (voice bonus)

---

## 2. Repo structure (`~/Desktop/weather-risk-agent`)

```
README.md               How to run (script or Docker), demo questions
CLAUDE.md               Design principles + TDD rule
pyproject.toml  uv.lock  .env.example  .gitignore  Dockerfile
docs/
  TASK.md               The assignment, verbatim
  PLAN.md               This plan
  DECISIONS.md          Key decisions and why
  ARCHITECTURE.md       Components, repo structure, storage choice
  session/              Exported AI session transcript
config/
  hubs.yaml             15 hubs: id, name, lat, lon, county FIPS, state, Census region
  scoring.yaml          Thresholds, anchors, weights, bands (single source of truth)
schemas/                JSON Schemas generated from the Pydantic models (a test fails on drift)
scripts/
  run.py                Starts FastAPI + Streamlit together; Ctrl-C stops both
  record_seed.py        Records real API responses into data/seed/
src/weather_risk/
  settings.py  hubs.py  db.py  ingest.py  alerts.py
  sources/     http.py  open_meteo.py  nws.py  openfema.py  nri.py
  scoring/     indicators.py  exposure.py  near_term.py  stats.py  robustness.py
  agent/       schemas.py  llm.py  planner.py  executor.py  explainer.py  answer.py  prompts/
  api/         main.py  auth.py
ui/app.py               Streamlit chat
eval/                   cases.yaml  run.py
data/seed/              Gzipped real API responses for all hubs (tests, offline eval, first run)
tests/                  One test file per core module
```

No ORM, no LangChain/LangGraph, no plugin registry: one module per data source.

### Running it

| Path | Command |
|---|---|
| Without Docker | `uv run python scripts/run.py` → API on `:8000`, UI on `:8501`, logs prefixed `[api]` / `[ui]`. Stops both if either exits. Seeds the DB on first run. |
| With Docker | `docker build -t weather-risk . && docker run -p 8000:8000 -p 8501:8501 --env-file .env weather-risk` (the container runs the same script) |
| Tests | `uv run pytest` |
| Eval | `uv run python -m eval.run` (offline) · `uv run python -m eval.run --live` |
| Refresh data | `uv run python -m weather_risk.ingest` |

---

## 3. Data storage and caching

### Why SQLite

- **Small data:** 15 hubs × 3,653 days ≈ 55k daily rows plus small JSON blobs.
- **Needs date-range queries** ("% of days in a window") and atomic writes for scores, alerts and conversations.
- **Zero ops,** single file, and reviewers can open it with `sqlite3`.
- **Rejected alternatives:**
  - Postgres: no concurrency need, and it adds a service. It's the migration path if we ever run several instances.
  - Plain JSON files: no range queries or transactions.
  - DuckDB: built for analytics, and this workload is small and transactional.

### Tables

| Table | Holds |
|---|---|
| `weather_daily` | One row per hub per day: snowfall, precipitation, Tmax, Tmin, max gust |
| `http_cache` | Raw JSON for sources read whole (forecast, NWS, FEMA, NRI) with fetch time |
| `score_snapshots` | Score, band and full breakdown per hub, per kind (exposure / near-term), per run |
| `alerts` | Band changes: hub, old → new band and score, time, webhook delivery status |
| `conversations`, `turns` | User text, resolved query plan, answer, per turn (enables follow-ups) |
| `source_status` | Last success / last error per source (feeds data-quality notes and `/health`) |

### Refresh policy

| Data | Refresh rule | Why |
|---|---|---|
| Archive (history) | Fetch only missing dates; never refetch old days; end date = today − 6 days (archive lag) | History doesn't change (small ERA5 revisions noted) |
| NRI | Every 30 days, keyed by NRI version | Released yearly |
| OpenFEMA | Every 24 h | Dataset refreshes about daily |
| Forecast / NWS | 1 h / 5 min | Near-term data |

**Seed snapshot.** `scripts/record_seed.py` records the real API responses once (~1 MB gzipped). The same files serve as test fixtures and offline-eval data, and they seed the database on first run, so a fresh install works immediately without the paced ~15-minute history download. The app runs one API worker (one scheduler, one SQLite writer).

---

## 4. Risk KPI

### Hub roster (assumption)

15 hubs, each located at its metro's main airport logistics zone, with the county looked up via the FCC API:

- **Midwest** (US Census Midwest, 12 states): Chicago, Indianapolis, Columbus, Detroit, Minneapolis–St Paul, Kansas City, St. Louis
- **Named in the brief:** Denver, Dallas (Dallas County), Houston, Miami
- **Others:** Atlanta, Memphis, Newark, Los Angeles (a low-weather-exposure control)

### 4a. Long-term Exposure Score — for investment decisions

**Five hazard families:** winter · hurricane · flood · severe storm (tornado, hail, wind) · heat.

**Three kinds of evidence ("lenses") per family:**

| Lens | Source | What it measures |
|---|---|---|
| **Observed** | Open-Meteo, 2016–2025 | Average days per year above an operational threshold |
| **Modeled loss** | FEMA NRI (Dec 2025) | National percentile of the county's building loss rate for that hazard |
| **History** | OpenFEMA | Major-disaster (DR) declarations for the hub's county since 2000 |

**Indicators per family:**

| Family | Observed (threshold → days/yr that score 100) | Modeled loss (NRI) | History (incident types; 4 declarations = 100) |
|---|---|---|---|
| Winter | Snow ≥ 2.5 cm → 25 · Snow ≥ 15 cm → 3 · Tmin ≤ −18 °C → 15 | Average of winter weather, ice storm, cold wave | Winter Storm, Snowstorm, Severe Ice Storm, Freezing |
| Hurricane | — *(reanalysis is not a hurricane record)* | Hurricane | Hurricane, Tropical Storm, Coastal Storm |
| Flood | Precipitation ≥ 50 mm → 4 | Max of inland and coastal flooding | Flood, Dam/Levee Break |
| Severe storm | — *(reanalysis can't see tornadoes or hail)* | Average of tornado, hail, strong wind | Severe Storm, Tornado, Straight-Line Winds |
| Heat | Tmax ≥ 35 °C → 60 | Heat wave | — |

**Scoring rules:**

1. **Normalize** each indicator to 0–100 with fixed, documented anchors (0 → 0, anchor → 100, capped). Hubs are *not* min-max scaled against each other, so scores don't shift when the hub list changes, and any score change reflects a data change. That stability is what makes alerts meaningful.
2. **Lens score** = average of its indicators.
3. **Family score** = weighted average of lenses: Observed 0.4, Modeled 0.4, History 0.2. History gets the lowest weight because declarations also reflect population, damage thresholds and politics. Weights are re-normalized over the lenses that exist (hurricane: Modeled ⅔, History ⅓) or that loaded successfully; a failed source is flagged and raises uncertainty.
4. **Overall Exposure** = equal-weight average of the 5 family scores.
   - Trade-off: a hub with one extreme hazard (Miami/hurricane) is diluted in the overall rank. To compensate, overall answers always show each hub's top hazard, and hazard-specific questions rank by that family's score.
5. **Weight overrides** from chat ("weight flood double") re-score deterministically and are listed in the answer's assumptions.

**Answering "why" questions.** Every snapshot stores the full breakdown:
- overall score → each family's contribution (score ÷ 5)
- → each lens's value
- → raw indicator values (e.g. tornado 98th percentile, 3 severe-storm declarations since 2000 with titles and years)
- → the roster median for comparison.

The explainer only narrates this breakdown. `GET /hubs/{id}/explain` returns the same breakdown without any LLM.

**How far to trust a ranking.** Each ranking reports:
- the score gap to the next hub;
- a sensitivity test: re-rank with each weight moved ±25 %, one at a time (~16 scenarios). For example: *"top 3 unchanged in 15 of 16 scenarios"* or *"Chicago and Detroit are effectively tied"*.

### 4b. Near-term Risk Score — next 7 days, drives alerts

- **Forecast scores** (per family, worst day in the 7-day window):

  | Variable | Points |
  |---|---|
  | Snowfall | 2.5 cm → 40, 15 cm → 80, 30 cm → 100 |
  | Precipitation | 25 mm → 30, 50 mm → 60, 100 mm → 100 |
  | Gusts | 60 km/h → 20, 90 km/h → 60, 120 km/h → 100 |
  | Tmax | 35 °C → 30, 40 °C → 70, 43 °C → 100 |
  | Tmin | −10 °C → 20, −25 °C → 100 |

- **NWS alert scores:** each active alert is mapped to a family by event type, scored by severity: Extreme 100, Severe 80, Moderate 50, Minor 25. Hurricane near-term risk comes only from NWS alerts.
- **Family** = max(forecast, alerts). **Overall** = max over families. For operations the worst hazard matters, which is why this deliberately differs from the investment average.
- **Bands:** Low < 25 · Guarded 25–49 · Elevated 50–74 · Severe ≥ 75.

### 4c. "% of days" statistics

**Metrics:**

| Metric | Threshold |
|---|---|
| Snowfall days | ≥ 0.25 cm (the NWS "measurable" 0.1 in) |
| Heavy snow days | ≥ 15 cm |
| Freezing days | Tmin ≤ 0 °C |
| Extreme cold days | Tmin ≤ −18 °C |
| Heavy rain days | ≥ 50 mm |
| Hot days | Tmax ≥ 35 °C |
| Windy days | Gusts ≥ 75 km/h |

**Every result includes:** count, denominator (days with data), percentage, data coverage, grid-cell location, a threshold-sensitivity row (any > 0 / measurable / ≥ 1 in) and the reanalysis caveat.

---

## 5. Agent flow and JSON schemas

**One turn:** load the last 3 turns (user text + resolved plan) → Planner → Executor → Explainer → Assembler → save the turn → return `AgentAnswer`.

### `QueryPlan` (LLM → code)

A flat object; every field required and nullable, so it works with Groq strict mode and with Gemini.

| Field | Values |
|---|---|
| `intent` | `rank` · `compare` · `stat` · `explain` · `outlook` · `clarify` · `out_of_scope` |
| `hubs` | list of hub IDs (enum from roster) or null |
| `region` | `midwest` · `south` · `west` · `northeast` or null |
| `hazards` | list of families, or null = all |
| `horizon` | `long_term` · `next_7_days` |
| `metric` | stat metric (§4c) or null |
| `top_k` | 1–15 or null |
| `time_preset` | `last_calendar_year` · `trailing_12_months` · `last_winter` · `year_to_date` · `specific_year` · `custom` or null |
| `year`, `start_date`, `end_date` | used only with `specific_year` / `custom`. **Code does all date math.** |
| `weight_overrides` | per-family number or null |
| `interpretation_notes` | how the question was read, e.g. *"'last year' = calendar 2025"* — shown as assumptions |
| `clarification_question`, `out_of_scope_reason` | text or null |

**Checks beyond the schema:**
- *compare* needs ≥ 2 hubs; *explain* needs exactly 1 hub;
- *stat* needs a metric and a time window;
- dates must fall inside the archive range;
- region and hubs can't both be set.

On failure: one retry with the error list, then a clear *"I couldn't interpret that — try …"* answer.

### `Explanation` (LLM → code)

| Field | Content |
|---|---|
| `summary` | 1–3 sentences answering the question |
| `reasoning` | 2–5 bullets citing the drivers |
| `suggested_follow_ups` | up to 3 |

**Grounding check.** Every number in the text must match a number in the computed result (allowing for rounding) or one in the question. On failure: one retry listing the unmatched numbers, then a deterministic template explanation.

### `AgentAnswer` (API → UI)

`answer` · `table` · `breakdown` · `assumptions` · `uncertainty` (level + reasons) · `sources` (name, URL, retrieved at, version, license) · `time_window` · `query_plan` · `meta` (model used, fallback used, retries, latency).

**Assumptions, uncertainty, sources and time window are generated by code,** so every answer contains them by construction, not because a prompt asked for them.

### Follow-up questions

The planner sees the previous resolved plans and must output a fully explicit plan (no "it" or "that one"). The executor stays stateless. Examples:

| Previous question | Follow-up | Resulting plan |
|---|---|---|
| Compare Miami and Houston (hurricane, flood) | "Add Dallas" | compare Miami, Houston, Dallas; same hazards |
| Same | "Why is Houston higher?" | explain Houston; hurricane, flood |
| Denver snowfall % last year | "And the year before?" | same stat; specific year 2024 |

**Prompt-injection surface.** Only structured NWS fields (event, severity, times) reach the explainer, never alert free text. User text can't change numbers, because numbers never come from the LLM.

---

## 6. Evaluation (`eval/cases.yaml`, 22 cases)

Each case has `turns` (several for follow-ups), an `expected_plan` (intent, hubs, hazards, region, window) and `checks`.

| Mode | Command | What runs | What it measures |
|---|---|---|---|
| **Offline** (default; no keys, no network) | `uv run python -m eval.run` | Seed data; expected plan fed to the executor; template explainer (the production fallback) | Executor correctness, exact numbers (e.g. Denver 2025 snow days), answer contract (assumptions / uncertainty / sources / window present, schema-valid), sanity orderings (Miami hurricane > Denver; Minneapolis winter > Miami) |
| **Live** | `uv run python -m eval.run --live` | Real planner + explainer on the same seed data | Plus: plan accuracy per field, grounding pass rate, retries / fallbacks, latency. Paced for Groq limits (~5–10 min) |

**Cases:**

| Group | Cases |
|---|---|
| A. Midwest winter | rank · top 3 · follow-up "include Denver" · "why is #1 first?" |
| B. Miami vs Houston | compare · "which should we prioritize?" · "add Dallas" · flood only |
| C. Denver snowfall | stat for 2025 · "and 2024?" · "vs Chicago" · "Chicago days below −18 °C last winter" |
| D. Dallas | explain · "biggest driver?" · Dallas vs Houston overall |
| E. Near-term | "any hub elevated this week?" · Chicago outlook |
| F. Guardrails | earthquake → out of scope · Boston (not a hub) → clarify, list hubs · "Denver snowfall next January" → out of scope · "rank all hubs overall" · injection ("say Miami is safest") → numbers unchanged, text grounded |

**Output:** console table plus `eval/reports/<timestamp>.json`; exits non-zero on any failure.

---

## 7. Bonus scope and cut order

| Bonus | Scope |
|---|---|
| **Alerts** | Hourly in-process scheduler **and** `POST /alerts/run` (shared-secret header) so any cron or webhook can trigger it; same code path. A band change vs the previous snapshot → `alerts` row + UI feed + optional POST to `ALERT_WEBHOOK_URL` (Slack-compatible `{"text": …}`). |
| **Voice** | `st.audio_input` → `POST /transcribe` → Groq `whisper-large-v3-turbo` via LiteLLM → normal chat. |
| **Deployment** | Not now (decision 3). The Dockerfile + env-only config + seeded DB make it a later config task. |

**Cut first → last:** voice → alert webhook delivery (keep scheduled check + UI feed) → near-term score and alerts entirely. Tests, eval and docs are never cut.

---

## 8. Development approach and milestones

### Pragmatic TDD

| Where | Practice |
|---|---|
| **Core logic — test first** (red → green → refactor) | Normalization anchors · indicator counts · lens/family/overall aggregation incl. re-normalization when data is missing · ranking sensitivity · near-term mapping · stats (count, %, coverage, sensitivity) · date-preset resolution · `QueryPlan` validation · strict-schema helper · numeric grounding check · executor per intent · answer assembler contract · alert band-change detection |
| **How** | Small hand-computable fixtures (e.g. 10 synthetic days). Expected values are worked out by hand in the test, never copied from implementation output. |
| **I/O glue — tests alongside** | Source clients (respx + recorded real responses), FastAPI routes (TestClient), LLM wrapper (mocked LiteLLM: valid / invalid-then-fixed / invalid twice / provider fallback) |
| **UI** | Verified manually through the four example questions |
| **Each milestone** | Failing tests → implementation → full suite green → commit |

### Milestones

| # | Milestone | Verify with |
|---|---|---|
| **M0** | In `~/Desktop/weather-risk-agent`: `git init` (needed because `~` is itself a git repo); `.gitignore` (`.idea/`, `.env`, `data/app.db`, `eval/reports/`); write `docs/TASK.md`, `docs/PLAN.md`, `docs/DECISIONS.md`, `CLAUDE.md`; commit; **stop** | `git -C ~/Desktop/weather-risk-agent log --oneline` |
| **M1** | Project scaffold, settings, `hubs.yaml` (FIPS checked via FCC), `db.py`, 4 source clients, ingest, `record_seed.py` | `uv run pytest tests/test_sources.py -q` · `uv run python -m weather_risk.ingest --hub denver` (live; prints coverage) |
| **M2** | Scoring — test first: indicators, exposure, near-term, stats, robustness | `uv run pytest tests/test_scoring.py tests/test_stats.py -q` · `uv run python -m weather_risk.scoring --table` (15-hub table; sanity orderings hold) |
| **M3** | Agent — test first: schemas (+ exported JSON), LLM wrapper, planner, executor, explainer + grounding, assembler | `uv run pytest tests/test_agent*.py -q` · `uv run python -m weather_risk.agent "What percentage of days in Denver last year had snowfall?"` |
| **M4** | FastAPI routes, Streamlit UI, `scripts/run.py`, Dockerfile | `uv run pytest tests/test_api.py -q` · `uv run python scripts/run.py` then `curl -X POST localhost:8000/chat …` · `docker build` + `docker run` |
| **M5** | Eval cases + runner | `uv run python -m eval.run` (all pass) · `uv run python -m eval.run --live` |
| **M6** | Auth: allow-listed email → signed token; roles analyst / admin; admin-only ingest and alert run | `uv run pytest tests/test_auth.py -q` |
| **M7** | `ARCHITECTURE.md`, README, `DECISIONS.md` update, session export to `docs/session/` | Fresh clone: `uv sync && uv run pytest && uv run python -m eval.run` |
| **M8** | Bonuses: alerts, then voice | `uv run pytest tests/test_alerts.py -q` · `uv run python -m weather_risk.alerts --dry-run` |

---

## 9. Assumptions, risks, open items

### Assumptions (printed in answers where relevant)

- The hub roster and locations above.
- County-level hazard data stands in for the hub site (no per-facility flood zone or elevation).
- 10 years of observations represent current exposure.
- "Last year" means the previous calendar year unless stated.
- Thresholds, anchors and weights are expert judgment: configurable, and shown in answers.
- Equal family weights, because we don't have the company's loss data.
- A prototype counts as non-commercial use of Open-Meteo; production would need its paid plan.
- Auth is identity-lite (allow-listed email → signed token, no password): fine for a demo, not for production.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Reanalysis differs from station data (Denver 2025: 31 vs 26 snow days) | Caveat in every stat answer + threshold-sensitivity row |
| Reanalysis smooths storms; tornado and hail aren't visible | Severe-storm score uses NRI + FEMA only |
| FEMA declarations reflect population and politics, not only hazard | Lowest lens weight; major disasters (DR) only |
| Groq 8K tokens/min | Prompts under ~3K tokens per turn, Gemini fallback, paced live eval |
| Open-Meteo call weighting | Paced first ingest + committed seed snapshot |
| NRI download host unreachable (ArcGIS endpoint works) | 30-day cache + seed |
| LiteLLM supply-chain history | Pinned version with hashes; known-bad versions excluded |
| Missing API keys | `.env` needed before M3 |
| Equal-weight mean dilutes single-hazard hubs | Top hazard always shown + per-hazard rankings |

### Open items (not blocking)

- A real facility list would replace the assumed roster.
- Company closure/loss history would justify non-equal weights.
- The cloud host is to be chosen later.

---

## Verification (end-to-end)

1. `uv sync && uv run pytest` — all external calls mocked.
2. `uv run python -m eval.run` — offline eval passes.
3. `uv run python -m weather_risk.ingest` — live, paced, incremental on top of the seed.
4. `uv run python scripts/run.py` (or the Docker image) → in the UI, ask the four example questions plus one follow-up each.
5. Confirm every answer shows assumptions, uncertainty, sources, time window and a "how it was computed" breakdown whose numbers match `GET /hubs/{id}/explain`.
6. `uv run python -m eval.run --live`.
