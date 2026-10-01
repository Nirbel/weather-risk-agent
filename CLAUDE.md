# CLAUDE.md

Weather Risk Intelligence Agent: a chat agent that ranks, compares and explains the weather-disruption exposure of US logistics hubs.

- Task: [docs/TASK.md](docs/TASK.md)
- Plan: [docs/PLAN.md](docs/PLAN.md)
- Decisions: [docs/DECISIONS.md](docs/DECISIONS.md)

## Design principles

- **Numbers come from code.** Scores, ranks and percentages come from deterministic Python. The LLM turns questions into a schema-validated query plan and explains the computed results. This keeps every number auditable, which is the point of the scoring requirement.
- **Validate all LLM output** against Pydantic/JSON schemas. On failure, retry once with the error included, then return a clear error.
- **Real data only.** Weather data comes only from real API responses. When a source fails or has gaps, say so in the answer, because analysts need to know how far to trust a ranking.
- **Every answer states** its assumptions, its uncertainty, its data sources and its time window.
- **Build only what the assignment asks**, with no extra abstractions. If time runs short, cut bonuses before tests or docs.

## Development rules

- **Pragmatic TDD for core logic** (scoring, stats, date resolution, plan validation, grounding check, executor, answer contract, alert diff):
  - Write a failing test first, then implement, then refactor.
  - Use small hand-computable fixtures. Work out expected values by hand; never copy them from the implementation's output.
- **I/O glue** (HTTP clients, API routes, LLM wrapper): write tests alongside the code. Mock every external call (respx, mocked LiteLLM).
- **Each milestone:** full suite green, then commit.
- **Config is the source of truth.** Thresholds, anchors, weights and bands live in `config/scoring.yaml`; hubs live in `config/hubs.yaml`. Don't hard-code them.
- **LLM:** go through LiteLLM only. Primary is Groq `openai/gpt-oss-20b` (strict JSON schema); Gemini is the fallback. Keep prompts compact, because Groq's free tier allows 8K tokens/min.
- **Running:**
  - `uv run python scripts/run.py` starts the API and Streamlit together.
  - Docker is optional; the image runs the same script.
  - There is no Makefile.
- **Docs:** short, clear Markdown.
