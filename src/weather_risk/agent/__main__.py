"""Ask the agent from the command line (needs an LLM key in .env).

    uv run python -m weather_risk.agent "What percentage of days in Denver last year had snowfall?"
"""

import asyncio
import json
import sys

from weather_risk.agent.answer import render_markdown
from weather_risk.agent.service import Agent
from weather_risk.db import Store
from weather_risk.ingest import ensure_data
from weather_risk.settings import get_settings, llm_configured


async def main(question: str) -> None:
    settings = get_settings()
    if not llm_configured(settings):
        sys.exit("Set GROQ_API_KEY and/or GEMINI_API_KEY in .env first (see .env.example).")
    store = Store(settings.db_path)
    ensure_data(store, settings.seed_dir)
    answer = await Agent(store, settings).ask(question)
    print(render_markdown(answer))
    print("\nquery plan:", json.dumps({k: v for k, v in (answer.query_plan or {}).items() if v not in (None, [])}))
    print("meta:", json.dumps(answer.meta))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    asyncio.run(main(" ".join(sys.argv[1:])))
