# Weather Risk Intelligence Agent

A chat agent that ranks, compares and explains how exposed 10 US logistics hubs are to weather disruption,
so analysts can choose which hubs to invest in.

- **Numbers come from code, not the LLM.** The LLM turns a question into a schema-validated query plan and
  explains the computed result; a grounding check rejects any number it invents.
- **Every answer states** its assumptions, uncertainty, data sources and time window.
- **Real public data, fetched on demand:** Open-Meteo ERA5 daily weather (previous 5 completed years) and
  the FEMA National Risk Index. Incomplete data is never ranked as if it were complete.
- **Bonuses:** score-change alerts to your own webhook, and voice questions.


**Live demo:** [https://weather-risk.duckdns.org](https://weather-risk.duckdns.org)



**Documentation:** [Architecture](docs/ARCHITECTURE.md) (components, repo structure, storage) ·
[Decisions](docs/DECISIONS.md) · [Plan](docs/PLAN.md) · [Task](docs/TASK.md) ·
[AI session](docs/AI_SESSION.md) (how it was built with Claude Code; transcript in [docs/session/](docs/session/))

## Run it

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.14 for you)
and a free LLM key from [Groq](https://console.groq.com/keys) (primary) and/or
[Google AI Studio](https://aistudio.google.com/apikey) (Gemini, the fallback).

```bash
git clone https://github.com/Nirbel/weather-risk-agent.git
```

```bash
cd weather-risk-agent
```

```bash
uv sync
```

```bash
cp .env.example .env
```

Open `.env` and set `GROQ_API_KEY` and/or `GEMINI_API_KEY`. Everything else is optional. Then start the
API and the UI together (Ctrl-C stops both):

```bash
uv run python scripts/run.py
```

| What | Where |
|---|---|
| UI (Home · Chat · Analytics · Alerts) | http://localhost:8501 |
| API and interactive docs | http://localhost:8000/docs |

**Log in** with the demo admin **`admin@moveo.co.il`** / **`12345678`**, or use *Sign up* with any
`@moveo.co.il` address (new accounts are analysts).

**First answer:** data is fetched when a question needs it and then cached in `data/app.db` (created
automatically). A cold cache takes about 3 minutes on Open-Meteo's free tier. The alert scheduler starts
that download 30 seconds after startup, so it is usually done by the time you ask.

## Try these

- Which hubs in the Midwest are most exposed to winter disruption? → *Why is Minneapolis first?*
- Compare Miami and Houston in terms of hurricane and flood exposure. → *Add Dallas to the comparison.*
- What percentage of days in Denver last year had snowfall? → *And the year before?*
- Why is the Dallas hub's weather disruption risk high?
- Rank all hubs, but weight hurricane risk twice as much.

Each answer has *Assumptions, sources and time window* and *How this was computed* (the query plan and the
score breakdown). The Analytics page charts the same numbers.

## Accounts

| Who | How | Can |
|---|---|---|
| Demo admin | `admin@moveo.co.il` / `12345678` (created on first start) | Everything, plus *Check now* on the Alerts page |
| Analyst | *Sign up* with an `@moveo.co.il` email (password ≥ 8 characters) | Chat, Analytics, their own alerts |

- Passwords are stored as salted scrypt hashes.
- Each user sees only their own conversations.
- Sign-up cannot create an admin.
- **Before you expose the app to others,** set `DEMO_ADMIN_PASSWORD` in `.env`; changing it and restarting
  rotates the password.

## Alerts and voice (bonuses)

- **Score-change alerts.** On the **Alerts** page each user sets their own webhook: URL, on/off, threshold,
  *Save*, *Test webhook*.
  - The webhook is generic: any endpoint that accepts a JSON `POST`, such as a Make, Zapier or n8n
    scenario, a Slack or Teams workflow, or your own service.
  - A daily check compares each hub's overall exposure score with the last value you were told about, and
    posts changes of at least your threshold.
  - Scores cover 5 completed years, so they move only when the window rolls over (January), FEMA publishes
    a new NRI release, or the scoring config changes. Alerts are rare by design.
  - Internal addresses (localhost, private networks) are refused unless `ALLOW_PRIVATE_WEBHOOKS=true`, for
    example for a self-hosted n8n on your LAN.
  - An external cron can trigger a check when `ALERT_TRIGGER_SECRET` is set in `.env`:

    ```bash
    curl -X POST localhost:8000/alerts/run -H "X-Alert-Secret: $ALERT_TRIGGER_SECRET"
    ```

- **Voice.** With `GROQ_API_KEY` set, the chat box shows a microphone. The recording is transcribed by Groq
  Whisper and asked like a typed question.

## Test and evaluate

```bash
uv run pytest
```

Unit and integration tests; every external call is mocked or replayed (no keys, no network).

```bash
uv run python -m eval.run
```

22 eval cases, **offline**: recorded real API responses, expected plans, deterministic explanations.
Fast and reproducible; no keys or network.

```bash
uv run python -m eval.run --online
```

**Online, end to end:** real LLM → real Open-Meteo and FEMA APIs (fresh download, about 4 minutes) →
deterministic computation → real LLM explanation → validation.

- Live data changes, so no score is hard-coded. Each answer is checked against what the code computes in
  the same run:
  - plan validity, intent, hubs, hazards and time period;
  - numbers and ranking;
  - numeric grounding;
  - follow-up context;
  - assumptions, sources and uncertainty.
- It then injects failures (LLM down, all LLMs down, Open-Meteo down, FEMA NRI down) and checks that the
  answer degrades gracefully.
- Turns are paced for the Groq free tier, so a full run takes 15–20 minutes. If every LLM provider is
  rate-limited or out of quota, a case cools down and retries once, then counts as *inconclusive*
  (provider capacity, not the agent); re-run those cases later with `--only`.
- Add `--cache data/eval.db` to keep the downloaded data between runs, or `--only B1,X3` for a subset.
- **LLM budget:** a full online run uses about 200K LLM tokens. That is the Groq free tier's daily limit, so on
  free keys run it about once a day (Gemini covers the overflow while its daily quota lasts), or run subsets.

```bash
uv run python -m eval.run --live
```

The real LLM on the recorded data (plan accuracy with stable numbers).

Reports are written to `eval/reports/`.

## Run on a server with Docker Compose (optional)

`docker-compose.yml` runs two services from one image:

- `backend`: FastAPI on port 8000. It reads `.env` and keeps the cache in the `app-data` volume.
- `frontend`: Streamlit on port 8501. It talks to the backend over the compose network and starts once the
  backend is healthy.

```bash
cp .env.example .env
```

```bash
docker compose up -d --build
```

Then open `http://<server>:8501`. Change the published ports with `UI_PORT` / `API_PORT`, for example
`UI_PORT=80 docker compose up -d`.

## Ask from the terminal

```bash
uv run python -m weather_risk.agent "Why is the Dallas hub's weather disruption risk high?"
```

```bash
uv run python -m weather_risk.warm
```

`weather_risk.warm` fills the cache before a demo and prints the exposure table. Add `--replay` to use the
recorded data with no network.

## Configuration

| File / variable | What it controls |
|---|---|
| `config/hubs.yaml` | The 10 hubs: location, county FIPS, Census region (validated at startup) |
| `config/scoring.yaml` | Thresholds, anchors, weights, the coverage rule, stat definitions (validated at startup) |
| `GROQ_API_KEY` / `GEMINI_API_KEY` | LLM keys (at least one) |
| `LLM_MODEL` / `LLM_FALLBACK_MODEL` | Default `groq/openai/gpt-oss-20b` → `gemini/gemini-3.8-flash` |
| `DEMO_ADMIN_EMAIL` / `DEMO_ADMIN_PASSWORD` | Demo admin, default `admin@moveo.co.il` / `12345678` |
| `JWT_SECRET` | Signs login tokens; empty → a random one per start (users log in again after a restart) |
| `ALERT_THRESHOLD` / `ALERT_CHECK_HOURS` | Default alert threshold (5 points) and check interval (24 h) |
| `ALLOW_PRIVATE_WEBHOOKS` | `true` lets users target private or loopback hosts (default `false`) |
| `ALERT_TRIGGER_SECRET` | Lets an external cron call `POST /alerts/run` with header `X-Alert-Secret` |
| `DATABASE_URL` | Default `sqlite+aiosqlite:///data/app.db` (the folder is created automatically) |

## Data and licenses

- Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), ERA5 reanalysis. The free tier
  is non-commercial; production use needs Open-Meteo's paid plan.
- [FEMA National Risk Index](https://hazards.fema.gov/nri/) (US public domain).
