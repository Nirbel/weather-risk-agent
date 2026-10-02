# CLAUDE.md

Weather Risk Intelligence Agent: a chat agent that ranks, compares and explains the weather-disruption exposure of US logistics hubs.

- Task: [docs/TASK.md](docs/TASK.md)
- Plan: [docs/PLAN.md](docs/PLAN.md)
- Decisions: [docs/DECISIONS.md](docs/DECISIONS.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- AI session: [docs/AI_SESSION.md](docs/AI_SESSION.md)

## Design principles

- **Numbers come from code.** Scores, ranks and percentages come from deterministic Python. The LLM turns questions into a schema-validated query plan and explains the computed results. This keeps every number auditable, which is the point of the scoring requirement.
- **Validate all LLM output** against Pydantic/JSON schemas. On failure, retry once with the error included, then return a clear error.
- **Real data only.** Weather data comes only from real API responses. When a source fails or has gaps, say so in the answer, because analysts need to know how far to trust a ranking.
- **Every answer states** its assumptions, its uncertainty, its data sources and its time window.
- **Build only what the assignment asks**, with no extra abstractions. If time runs short, cut bonuses before tests or docs.

## Development rules

- **Pragmatic TDD for core logic** (scoring, stats, coverage rule, date resolution, plan validation, grounding check, executor, answer contract, alert diff, email/password rules):
  - Write a failing test first, then implement, then refactor.
  - Use small hand-computable fixtures. Work out expected values by hand; never copy them from the implementation's output.
- **I/O glue** (HTTP clients, API routes, LLM wrapper): write tests alongside the code. Mock every external call (respx, mocked LiteLLM).
- **Each milestone:** full suite green, then commit.
- **Config is the source of truth.** Thresholds, anchors and weights live in `config/scoring.yaml`; hubs live in `config/hubs.yaml`. Both are validated at startup by typed models (`weather_risk/config.py`). Don't hard-code them.
- **Data is fetched on demand and cached** (Open-Meteo ERA5 + FEMA NRI). Recorded responses in `tests/fixtures/recorded/` are for tests and eval only — never a runtime dependency.
- **Persistence:** async SQLAlchemy 2.0 + SQLite (aiosqlite).
- **Before changing an agreed technology choice,** call it out and explain why first.
- **LLM:** go through LiteLLM only. Primary is Groq `openai/gpt-oss-20b` (strict JSON schema); Gemini is the fallback. Keep prompts compact, because Groq's free tier allows 8K tokens/min.
- **Running:**
  - `uv run python scripts/run.py` starts the API and Streamlit together.
  - Docker is optional: `docker compose up -d --build` runs two services (backend, frontend) from one image.
  - There is no Makefile.
- **Docs:** short, clear Markdown.
- **Streamlit:** use the `developing-with-streamlit` skill.
