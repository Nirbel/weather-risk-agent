"""Assemble the AgentAnswer. Assumptions, uncertainty, sources and time windows come from code,
so every answer carries them by construction (docs/DECISIONS.md D4)."""

from typing import Any

from weather_risk.agent.executor import ResultBundle
from weather_risk.agent.i18n import column_labels
from weather_risk.agent.schemas import AgentAnswer, Explanation, QueryPlan, Source, TimeWindow, Uncertainty
from weather_risk.config import load_scoring_config

SOURCES = {
    "open_meteo_archive": ("Open-Meteo Historical Weather API (ERA5 reanalysis)",
                           "https://open-meteo.com/en/docs/historical-weather-api", "CC BY 4.0"),
    "fema_nri": ("FEMA National Risk Index (county)", "https://hazards.fema.gov/nri/",
                 "Building expected-annual-loss rate, scored on fixed log-scale anchors (config/scoring.yaml)"),
}
SOURCE_DETAILS_HE = {
    "open_meteo_archive": "רישיון CC BY 4.0",
    "fema_nri": "שיעור הנזק השנתי הצפוי למבנים, מקבל ציון לפי עוגנים לוגריתמיים קבועים (config/scoring.yaml)",
}
SEVERITY = {"low": 0, "medium": 1, "high": 2}


def _tr(lang: str, en: str, he: str) -> str:
    return he if lang == "he" else en


def _source_reasons(key: str, status: dict, lang: str = "en") -> list[tuple[str, str]]:
    name = SOURCES[key][0]
    success, error_at = status.get("last_success_at"), status.get("last_error_at")
    if not success:
        return [("high", _tr(lang, f"{name}: no successful download yet, so this evidence is missing.",
                             f"{name}: עדיין אין הורדה מוצלחת, ולכן הראיות האלה חסרות."))]
    if error_at and error_at > success:
        return [("medium", _tr(lang, f"{name}: the latest download failed ({status.get('last_error')}); "
                                     f"the last successful one was {success}.",
                               f"{name}: ההורדה האחרונה נכשלה ({status.get('last_error')}); "
                               f"ההורדה המוצלחת האחרונה הייתה ב-{success}."))]
    return []


def assemble(plan: QueryPlan, bundle: ResultBundle, explanation: Explanation, statuses: dict[str, dict],
             meta: dict[str, Any], conversation_id: str | None = None, lang: str = "en") -> AgentAnswer:
    """`lang` must match the ExecContext the bundle was built with (agent/i18n.py)."""
    reasons = list(bundle.uncertainty) + [("medium", gap) for gap in bundle.data_gaps]
    sources = []
    for key in bundle.sources:
        status = statuses.get(key, {})
        reasons += _source_reasons(key, status, lang)
        name, url, detail = SOURCES[key]
        sources.append(Source(name=name, url=url, retrieved_at=status.get("last_success_at"),
                              detail=SOURCE_DETAILS_HE[key] if lang == "he" else detail))
    if not reasons:
        reasons = [("low", _tr(lang, "Nothing was computed for this question.", "לא חושב דבר עבור שאלה זו.")
                    if not bundle.sources else _tr(lang, "All sources loaded and no data gaps were found.",
                                                   "כל המקורות נטענו ולא נמצאו פערים בנתונים."))]
    reasons.sort(key=lambda r: -SEVERITY[r[0]])
    assumptions = list(dict.fromkeys([*bundle.assumptions, *plan.interpretation_notes]))
    return AgentAnswer(
        conversation_id=conversation_id,
        intent=bundle.intent,
        answer=explanation.summary,
        reasoning=explanation.reasoning,
        suggested_follow_ups=explanation.suggested_follow_ups,
        table=bundle.rows,
        breakdown=bundle.details or None,
        assumptions=assumptions,
        uncertainty=Uncertainty(level=reasons[0][0], reasons=list(dict.fromkeys(r for _, r in reasons))),
        sources=sources,
        time_windows=[TimeWindow(**w) for w in bundle.windows],
        query_plan=plan.model_dump(mode="json"),
        meta=meta,
        language=lang,
        column_labels=column_labels(bundle.rows, load_scoring_config(), lang) if lang != "en" else {},
    )


ERROR_HEADLINES_HE = {
    "llm_unavailable": "מודל השפה אינו זמין כרגע, ולכן אני לא יכול לקרוא את השאלה. נסה שוב בעוד דקה — עמוד "
                       "Analytics עובד גם בלעדיו.",
    "invalid_plan": "לא הצלחתי להפוך את השאלה לשאילתה שאפשר לחשב. נסח אותה מחדש — למשל, ציין את המרכזים, "
                    "את הסיכון ואת טווח הזמן.",
}
ERROR_HEADLINES = {
    "llm_unavailable": "The language model is unavailable right now, so I can't read the question. Please try again "
                       "in a minute — the Analytics page works without it.",
    "invalid_plan": "I couldn't turn that question into a query I can compute. Please rephrase it — for example, "
                    "name the hubs, the hazard and the time frame.",
}


def error_answer(reason: str, conversation_id: str | None, meta: dict[str, Any],
                 kind: str = "invalid_plan", lang: str = "en") -> AgentAnswer:
    """`kind`: "llm_unavailable" (every provider failed) or "invalid_plan" (still invalid after the retry)."""
    return AgentAnswer(
        conversation_id=conversation_id,
        intent="error",
        answer=(ERROR_HEADLINES_HE if lang == "he" else ERROR_HEADLINES)[kind],
        reasoning=[reason],
        suggested_follow_ups=[_tr(lang, "Which hubs in the Midwest are most exposed to winter disruption?",
                                  "אילו מרכזים במערב התיכון הכי חשופים לשיבושי חורף?"),
                              _tr(lang, "What percentage of days in Denver last year had snowfall?",
                                  "באיזה אחוז מהימים בשנה שעברה ירד שלג ב-Denver?")],
        language=lang,
        table=[], breakdown=None, assumptions=[],
        uncertainty=Uncertainty(level="high", reasons=[reason]),
        sources=[], time_windows=[], query_plan=None, meta=meta | {"error_kind": kind},
    )


def render_markdown(answer: AgentAnswer) -> str:
    """Plain-text rendering for the CLI and eval reports."""
    lines = [answer.answer, ""]
    lines += [f"- {r}" for r in answer.reasoning]
    if answer.table:
        cols = list(answer.table[0])
        lines += ["", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(row.get(c, "")) for c in cols) + " |" for row in answer.table]
    lines += ["", f"Uncertainty: {answer.uncertainty.level}"] + [f"  - {r}" for r in answer.uncertainty.reasons]
    lines += ["Assumptions:"] + [f"  - {a}" for a in answer.assumptions]
    lines += ["Time windows:"] + [f"  - {w.label}" + (f": {w.start or '…'} → {w.end or '…'}" if w.start or w.end else "")
                                  for w in answer.time_windows]
    lines += ["Sources:"] + [f"  - {s.name} (retrieved {s.retrieved_at})" for s in answer.sources]
    return "\n".join(lines)
