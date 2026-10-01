"""ResultBundle → Explanation. The LLM narrates; the grounding check keeps its numbers honest.

If the LLM is unavailable or still ungrounded after one retry, a deterministic template
built from the bundle is used instead — the answer never depends on the LLM being up.
"""

import json
from pathlib import Path

from weather_risk.agent.executor import ResultBundle
from weather_risk.agent.grounding import allowed_numbers, ungrounded_numbers
from weather_risk.agent.llm import CallMeta, LLMError, structured_call
from weather_risk.agent.schemas import Explanation
from weather_risk.settings import Settings

PROMPT = (Path(__file__).parent / "prompts" / "explainer.md").read_text()


def build_messages(question: str, bundle: ResultBundle) -> list[dict]:
    result = json.dumps(bundle.for_llm(), ensure_ascii=False, default=str)
    return [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"QUESTION: {question}\n\nRESULT:\n{result}"},
    ]


def template_explanation(bundle: ResultBundle) -> Explanation:
    return Explanation(
        summary=bundle.headline,
        reasoning=(bundle.points or [bundle.headline])[:5],
        suggested_follow_ups=bundle.follow_ups[:3],
    )


async def explain(question: str, bundle: ResultBundle, settings: Settings,
                  completion=None) -> tuple[Explanation, CallMeta | None, str]:
    """Returns (explanation, call metadata, "llm" | "template")."""
    allowed = allowed_numbers(bundle.for_llm(), question)

    def check(explanation: Explanation) -> None:
        bad = ungrounded_numbers(" ".join([explanation.summary, *explanation.reasoning]), allowed)
        if bad:
            listed = ", ".join(f"{b:g}" for b in bad)
            raise ValueError(f"These numbers are not in RESULT: {listed}. Use only numbers that appear in RESULT.")

    try:
        explanation, meta = await structured_call(
            build_messages(question, bundle), Explanation,
            models=[settings.llm_model, settings.llm_fallback_model],
            check=check, completion=completion, temperature=0.2, timeout=settings.llm_timeout_s,
        )
        return explanation, meta, "llm"
    except LLMError as exc:
        return template_explanation(bundle), exc.meta, "template"
