"""ResultBundle → Explanation. The LLM narrates; the grounding check keeps its numbers honest.

If the LLM is unavailable or still ungrounded after one retry, a deterministic template
built from the bundle is used instead — the answer never depends on the LLM being up.
"""

import json
import re
from pathlib import Path

from weather_risk.agent.executor import ResultBundle
from weather_risk.agent.grounding import allowed_numbers, rank_claim_errors, ungrounded_numbers
from weather_risk.agent.i18n import HEBREW, detect_language
from weather_risk.agent.llm import CallMeta, LLMError, structured_call
from weather_risk.agent.schemas import Explanation
from weather_risk.settings import Settings

PROMPT = (Path(__file__).parent / "prompts" / "explainer.md").read_text()
# Follow-ups the planner would route to out_of_scope: suggesting them sends the user into a refusal.
UNANSWERABLE = re.compile(r"\b(forecast\w*|outlook|tomorrow|upcoming|next (?:week|month|\d+ days)|7-day|"
                          r"this (?:week|weekend)|will it)\b", re.IGNORECASE)


def build_messages(question: str, bundle: ResultBundle, previous_question: str | None = None) -> list[dict]:
    result = json.dumps(bundle.for_llm(), ensure_ascii=False, default=str)
    previous = f"PREVIOUS QUESTION: {previous_question}\n" if previous_question else ""
    language = "Hebrew" if detect_language(question) == "he" else "English"
    return [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"{previous}QUESTION: {question}\nANSWER LANGUAGE: {language}\n\nRESULT:\n{result}"},
    ]


def template_explanation(bundle: ResultBundle) -> Explanation:
    return Explanation(
        summary=bundle.headline,
        reasoning=(bundle.points or [bundle.headline])[:5],
        suggested_follow_ups=bundle.follow_ups[:3],
    )


async def explain(question: str, bundle: ResultBundle, settings: Settings, completion=None,
                  previous_question: str | None = None) -> tuple[Explanation, CallMeta | None, str]:
    """Returns (explanation, call metadata, "llm" | "template")."""
    allowed = allowed_numbers(bundle.for_llm(), question) | allowed_numbers(None, previous_question or "")
    hebrew = detect_language(question) == "he"

    def check(explanation: Explanation) -> None:
        text = " ".join([explanation.summary, *explanation.reasoning])
        problems = []
        if bad := ungrounded_numbers(text, allowed):
            listed = ", ".join(f"{b:g}" for b in bad)
            problems.append(f"These numbers are not in RESULT: {listed}. Use only numbers that appear in RESULT.")
        if hebrew and not HEBREW.search(explanation.summary):  # the prompt rule alone was ignored live
            problems.append("The QUESTION is in Hebrew: write summary, reasoning and suggested_follow_ups in Hebrew.")
        if claims := rank_claim_errors(text, bundle.orders):
            problems.append("Ranking claims contradict RESULT: " + " ".join(claims)
                            + " State positions exactly as RESULT gives them, even if the question assumes otherwise.")
        if problems:
            raise ValueError(" ".join(problems))

    try:
        explanation, meta = await structured_call(
            build_messages(question, bundle, previous_question), Explanation,
            models=[settings.llm_model, settings.llm_fallback_model],
            check=check, completion=completion, temperature=0.2, timeout=settings.llm_timeout_s,
        )
        explanation.suggested_follow_ups = [q for q in explanation.suggested_follow_ups if not UNANSWERABLE.search(q)]
        return explanation, meta, "llm"
    except LLMError as exc:
        return template_explanation(bundle), exc.meta, "template"
