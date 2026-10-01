"""Question (+ recent turns) → validated QueryPlan."""

import json
from datetime import date
from pathlib import Path

from weather_risk.agent.llm import CallMeta, structured_call
from weather_risk.agent.schemas import QueryPlan
from weather_risk.config import load_hubs, load_scoring_config
from weather_risk.settings import Settings

PROMPT = (Path(__file__).parent / "prompts" / "planner.md").read_text()


def _compact(plan: dict) -> dict:
    return {k: v for k, v in plan.items() if v not in (None, [], {})}


def build_messages(question: str, history: list[dict], *, today: date, window: tuple[date, date],
                   data_end: date) -> list[dict]:
    system = PROMPT.format(
        today=today.isoformat(),
        last_year=today.year - 1,
        window=f"{window[0].year}–{window[1].year}",
        data_end=data_end.isoformat(),
        hubs="\n".join(f"- {h.id}: {h.name}, {h.state}, {h.region}" for h in load_hubs()),
        metrics="; ".join(f"{key} = {m.label}" for key, m in load_scoring_config().stats.items()),
    )
    messages = [{"role": "system", "content": system}]
    for turn in history:  # earlier turns: the question and the plan we resolved it to
        messages.append({"role": "user", "content": turn["user_text"]})
        if turn.get("plan"):
            messages.append({"role": "assistant", "content": json.dumps(_compact(turn["plan"]))})
    messages.append({"role": "user", "content": question})
    return messages


async def plan_question(question: str, history: list[dict], *, settings: Settings, today: date,
                        window: tuple[date, date], data_end: date, completion=None) -> tuple[QueryPlan, CallMeta]:
    messages = build_messages(question, history, today=today, window=window, data_end=data_end)
    return await structured_call(messages, QueryPlan, models=[settings.llm_model, settings.llm_fallback_model],
                                 completion=completion, temperature=0.0, timeout=settings.llm_timeout_s)
