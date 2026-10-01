"""Runtime settings, read from environment variables and `.env`."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


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

    # Auth: "email:role,email:role" with role in {analyst, admin}
    auth_allowlist: str = ""
    jwt_secret: str = "dev-only-change-me-and-use-at-least-32-bytes"
    token_ttl_hours: int = 12

    # Alerts
    scheduler_enabled: bool = True
    near_term_refresh_minutes: int = 60
    alert_webhook_url: str | None = None
    alert_trigger_secret: str | None = None  # header X-Alert-Secret for POST /alerts/run

    # UI → API
    api_url: str = "http://localhost:8000"


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
