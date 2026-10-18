"""Runtime settings, read from environment variables and `.env`."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO_PASSWORD = "12345678"  # demo admin password; override with DEMO_ADMIN_PASSWORD
DEFAULT_SIGNUP_DOMAIN = "example.co.il"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Storage (async SQLAlchemy URL; SQLite file by default)
    database_url: str = f"sqlite+aiosqlite:///{ROOT / 'data' / 'app.db'}"

    # Data sources (NWS asks for a contact in the User-Agent)
    contact_email: str = "weather-risk-agent@example.com"

    # LLM (via LiteLLM, which reads provider keys from the environment)
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_model: str = "groq/openai/gpt-oss-20b"
    llm_fallback_model: str = "gemini/gemini-3.8-flash"
    llm_timeout_s: float = 30.0
    transcribe_model: str = "groq/whisper-large-v3-turbo"

    # Auth: email + password; sign-up only for @<signup_domain> (always role analyst).
    # The demo admin is created on first start if missing.
    signup_domain: str = DEFAULT_SIGNUP_DOMAIN  # e.g. your own domain; compared exactly, case-insensitive
    demo_admin_email: str = f"admin@{DEFAULT_SIGNUP_DOMAIN}"
    demo_admin_password: str = DEFAULT_DEMO_PASSWORD  # the startup log warns while this default is in use
    jwt_secret: str | None = None  # empty or < 32 bytes → a generated secret, saved in jwt_secret_file
    jwt_secret_file: Path | None = ROOT / "data" / ".jwt_secret"  # None → a new secret per start (log in again)
    token_ttl_hours: int = 12

    # Score-change alerts: each user sets their own webhook URL, on/off and threshold in the UI
    scheduler_enabled: bool = True
    alert_check_hours: float = 24.0
    alert_threshold: float = 5.0  # default threshold: points on the 0–100 overall Exposure Score
    alert_trigger_secret: str | None = None  # header X-Alert-Secret for POST /alerts/run (external cron)
    allow_private_webhooks: bool = False  # true: users may target private/loopback hosts (e.g. LAN n8n)

    # Open-Meteo free tier allows 600 calls/min per IP. The limiter is per process: when you run N API
    # processes (e.g. uvicorn --workers N), set this to about 400 / N.
    open_meteo_calls_per_minute: float = 400.0

    # UI → API
    api_url: str = "http://localhost:8000"
    # Public path prefix when a reverse proxy serves the API under a sub-path and strips it,
    # e.g. "/api" for https://example.com/api/docs. Empty when the API is served at the root.
    root_path: str = ""


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # .env values are read by pydantic-settings; LiteLLM looks for keys in os.environ.
    for name, value in (("GROQ_API_KEY", settings.groq_api_key), ("GEMINI_API_KEY", settings.gemini_api_key)):
        if value:
            os.environ.setdefault(name, value)
    return settings


def llm_configured(settings: Settings) -> bool:
    return bool(settings.groq_api_key or settings.gemini_api_key)


def voice_configured(settings: Settings) -> bool:
    """Voice input needs a key for the transcription model's provider (Groq Whisper by default)."""
    provider = settings.transcribe_model.split("/", 1)[0]
    return bool({"groq": settings.groq_api_key, "gemini": settings.gemini_api_key}.get(provider))
