"""Assemble the AgentAnswer. Assumptions, uncertainty, sources and time windows come from code,
so every answer carries them by construction (docs/DECISIONS.md D4)."""

from typing import Any

from weather_risk.agent.executor import ResultBundle
from weather_risk.agent.schemas import AgentAnswer, Explanation, QueryPlan, Source, TimeWindow, Uncertainty

SOURCES = {
    "open_meteo_archive": ("Open-Meteo Historical Weather API (ERA5 reanalysis)",
                           "https://open-meteo.com/en/docs/historical-weather-api", "CC BY 4.0"),
    "fema_nri": ("FEMA National Risk Index (county)", "https://hazards.fema.gov/nri/",
                 "Building expected-annual-loss rate; national percentile computed over all US counties"),
}
SEVERITY = {"low": 0, "medium": 1, "high": 2}


def _source_reasons(key: str, status: dict) -> list[tuple[str, str]]:
    name = SOURCES[key][0]
    success, error_at = status.get("last_success_at"), status.get("last_error_at")
    if not success:
        return [("high", f"{name}: no successful download yet, so this evidence is missing.")]
    if error_at and error_at > success:
        return [("medium", f"{name}: the latest download failed ({status.get('last_error')}); "
                           f"the last successful one was {success}.")]
    return []


def assemble(plan: QueryPlan, bundle: ResultBundle, explanation: Explanation, statuses: dict[str, dict],
             meta: dict[str, Any], conversation_id: str | None = None) -> AgentAnswer:
    reasons = list(bundle.uncertainty) + [("medium", gap) for gap in bundle.data_gaps]
    sources = []
    for key in bundle.sources:
        status = statuses.get(key, {})
        reasons += _source_reasons(key, status)
        name, url, detail = SOURCES[key]
        sources.append(Source(name=name, url=url, retrieved_at=status.get("last_success_at"), detail=detail))
    if not reasons:
        reasons = [("low", "Nothing was computed for this question." if not bundle.sources
                    else "All sources loaded and no data gaps were found.")]
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
    )


def error_answer(reason: str, conversation_id: str | None, meta: dict[str, Any]) -> AgentAnswer:
    return AgentAnswer(
        conversation_id=conversation_id,
        intent="error",
        answer="I couldn't turn that question into a query I can compute. Please rephrase it — for example, "
               "name the hubs, the hazard and the time frame.",
        reasoning=[reason],
        suggested_follow_ups=["Which hubs in the Midwest are most exposed to winter disruption?",
                              "What percentage of days in Denver last year had snowfall?"],
        table=[], breakdown=None, assumptions=[],
        uncertainty=Uncertainty(level="high", reasons=[reason]),
        sources=[], time_windows=[], query_plan=None, meta=meta,
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
