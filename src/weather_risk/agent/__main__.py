"""Ask the agent from the command line (needs an LLM key in .env).

    uv run python -m weather_risk.agent "What percentage of days in Denver last year had snowfall?"
"""

import asyncio
import json
import sys

from weather_risk.agent.answer import render_markdown
from weather_risk.agent.service import Agent
from weather_risk.analysis import build_analyzer
from weather_risk.config import validate_config
from weather_risk.db import Database
from weather_risk.settings import get_settings, llm_configured
from weather_risk.sources.http import make_client


async def main(question: str) -> None:
    validate_config()
    settings = get_settings()
    if not llm_configured(settings):
        sys.exit("Set GROQ_API_KEY and/or GEMINI_API_KEY in .env first (see .env.example).")
    db = Database(settings.database_url)
    await db.create_all()
    try:
        async with make_client(settings.contact_email) as client:
            answer = await Agent(db, build_analyzer(db, client), settings).ask(question, user_email="cli")
    finally:
        await db.dispose()
    print(render_markdown(answer))
    print("\nquery plan:", json.dumps({k: v for k, v in (answer.query_plan or {}).items() if v not in (None, [])}))
    print("meta:", json.dumps(answer.meta))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    asyncio.run(main(" ".join(sys.argv[1:])))
