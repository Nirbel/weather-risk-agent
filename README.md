# Weather Risk Intelligence Agent

A chat agent that ranks, compares and explains how exposed US logistics hubs are to weather disruption, to help analysts choose which hubs to invest in.

- **Numbers come from code, not the LLM.** The LLM turns a question into a schema-validated query plan and explains the computed result. A grounding check rejects any number it invents.
- **Every answer states** its assumptions, uncertainty, data sources and time window.
- **Real public data, fetched on demand:** Open-Meteo ERA5 daily weather (previous 5 completed years) and the FEMA National Risk Index.
- **Bonuses:** score-change alerts to a webhook (scheduled or triggered), and voice questions.

Design (components, repo structure, storage): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Decisions: [docs/DECISIONS.md](docs/DECISIONS.md) · Plan: [docs/PLAN.md](docs/PLAN.md) · Task: [docs/TASK.md](docs/TASK.md)

## Run it

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env    # then set GROQ_API_KEY and/or GEMINI_API_KEY, AUTH_ALLOWLIST, JWT_SECRET
uv sync
uv run python scripts/run.py
```

| What | Where |
|---|---|
| UI (Home · Chat · Analytics) | http://localhost:8501 — log in with an email from `AUTH_ALLOWLIST` |
| API + interactive docs | http://localhost:8000/docs |

`scripts/run.py` is the way to run the app. It starts the API, waits until it is healthy, starts the UI, and stops both on Ctrl-C (or if either one exits).

**Before a demo (optional):** data is fetched when a question needs it. A cold first ranking of all 10 hubs takes about 3 minutes on Open-Meteo's free tier. The alert scheduler's first check, 30 seconds after startup, also fills the cache. To fill it by hand:

```bash
uv run python -m weather_risk.warm
```

**Docker (optional, not needed).** The image runs the same script:

```bash
docker build -t weather-risk .
docker run --env-file .env -p 8000:8000 -p 8501:8501 -v "$PWD/data:/app/data" weather-risk
```

## Try these

- Which hubs in the Midwest are most exposed to winter disruption? → *Why is Minneapolis first?*
- Compare Miami and Houston in terms of hurricane and flood exposure. → *Add Dallas to the comparison.*
- What percentage of days in Denver last year had snowfall? → *And the year before?*
- Why is the Dallas hub's weather disruption risk high?
- Rank all hubs, but weight hurricane risk twice as much.

## Alerts and voice (bonuses)

- **Score-change alerts.** A daily check in the API process compares each hub's overall Exposure Score with the last value analysts were told about. A change of at least the threshold (default 5 points) is stored and posted as JSON to a webhook. The payload has a `text` summary, so a Slack incoming webhook works as-is.
  - Admins set the webhook URL, on/off and threshold at the bottom of **Analytics**, with *Send test* and *Check now* buttons.
  - An external cron can trigger a check:

    ```bash
    curl -X POST localhost:8000/alerts/run -H "X-Alert-Secret: $ALERT_TRIGGER_SECRET"
    ```

  - Exposure covers 5 completed years, so scores move only when the window rolls over (January), FEMA publishes a new NRI release, or the config changes. Hubs with data gaps are skipped, so a failed fetch never looks like a change.
- **Voice.** With `GROQ_API_KEY` set, the chat box shows a microphone. The recording is transcribed by Groq Whisper (through LiteLLM) and asked like a typed question.

## Test and evaluate

```bash
uv run pytest                      # unit + integration tests; every external call mocked or replayed
uv run python -m eval.run          # 22 eval cases, offline (no keys, no network)
uv run python -m eval.run --live   # same cases with the real LLM planner + explainer
```

The offline eval replays recorded real API responses (`tests/fixtures/recorded/`) through the full pipeline. Reports go to `eval/reports/`.

## Ask from the terminal

```bash
uv run python -m weather_risk.agent "Why is the Dallas hub's weather disruption risk high?"
uv run python -m weather_risk.warm --replay      # exposure table from recorded data, no network
```

## Configuration

| File / variable | What it controls |
|---|---|
| `config/hubs.yaml` | The 10 hubs: location, county FIPS, Census region (validated at startup) |
| `config/scoring.yaml` | Thresholds, anchors, weights, stat definitions (validated at startup) |
| `LLM_MODEL` / `LLM_FALLBACK_MODEL` | Default `groq/openai/gpt-oss-20b` → `gemini/gemini-3.8-flash` |
| `AUTH_ALLOWLIST` | `email:role,…` with role `analyst` or `admin` |
| `JWT_SECRET` | Signs login tokens; use 32+ random bytes |
| `ALERT_WEBHOOK_URL` / `ALERT_THRESHOLD` | Alert defaults until an admin saves settings in the UI (threshold default 5 points) |
| `ALERT_CHECK_HOURS` / `SCHEDULER_ENABLED` | Scheduled check interval (default 24 h) / turn the scheduler off |
| `ALERT_TRIGGER_SECRET` | Lets an external cron call `POST /alerts/run` with header `X-Alert-Secret` |
| `TRANSCRIBE_MODEL` | Voice input, default `groq/whisper-large-v3-turbo` |
| `DATABASE_URL` | Default `sqlite+aiosqlite:///data/app.db` |

## Data and licenses

- Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), ERA5 reanalysis. The free tier is non-commercial; production use needs Open-Meteo's paid plan.
- [FEMA National Risk Index](https://hazards.fema.gov/nri/) (US public domain).
