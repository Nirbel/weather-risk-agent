"""JSON contracts between the LLM and the code (docs/DECISIONS.md D3).

QueryPlan    LLM → code: what to compute. Every field required (nullable when not used),
             so the same schema works with Groq strict mode and with Gemini.
Explanation  LLM → code: narrative over numbers the code computed.
AgentAnswer  API → UI: the full answer; assumptions/uncertainty/sources/window are built by code.
"""

import copy
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from weather_risk.hubs import load_hubs, load_scoring_config

HUB_IDS = tuple(h.id for h in load_hubs())
FAMILIES = tuple(load_scoring_config()["families"])
METRICS = tuple(load_scoring_config()["stats"])

HubId = Literal[HUB_IDS]  # type: ignore[valid-type]
Family = Literal[FAMILIES]  # type: ignore[valid-type]
StatMetric = Literal[METRICS]  # type: ignore[valid-type]
Intent = Literal["rank", "compare", "stat", "explain", "outlook", "clarify", "out_of_scope"]
Region = Literal["midwest", "south", "west", "northeast"]
Horizon = Literal["long_term", "next_7_days"]
TimePreset = Literal["last_calendar_year", "trailing_12_months", "last_winter", "year_to_date", "specific_year", "custom"]


class WeightOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")
    winter: float | None
    hurricane: float | None
    flood: float | None
    severe_storm: float | None
    heat: float | None


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent = Field(description="rank hubs, compare hubs, stat (% of days), explain one hub's score, "
                                       "outlook (next 7 days), clarify, or out_of_scope")
    hubs: list[HubId] | None = Field(description="Explicit hubs, or null to use region / all hubs")
    region: Region | None = Field(description="US Census region filter, or null")
    hazards: list[Family] | None = Field(description="Hazard families to focus on, or null for all")
    horizon: Horizon = Field(description="long_term exposure (investment) or next_7_days (near-term risk)")
    metric: StatMetric | None = Field(description="For intent=stat: which day count")
    top_k: int | None = Field(description="How many ranked hubs to show, or null for all")
    time_preset: TimePreset | None = Field(description="For intent=stat: which window")
    year: int | None = Field(description="Only with time_preset=specific_year")
    start_date: date | None = Field(description="Only with time_preset=custom, YYYY-MM-DD")
    end_date: date | None = Field(description="Only with time_preset=custom, YYYY-MM-DD")
    weight_overrides: WeightOverrides | None = Field(description="Relative hazard weights the user asked for, else null")
    interpretation_notes: list[str] = Field(description="How ambiguous wording was read, e.g. 'last year = calendar 2025'")
    clarification_question: str | None = Field(description="For intent=clarify")
    out_of_scope_reason: str | None = Field(description="For intent=out_of_scope")

    @model_validator(mode="after")
    def _semantic_rules(self) -> "QueryPlan":
        errors: list[str] = []
        if self.hubs:
            self.hubs = list(dict.fromkeys(self.hubs))
        n_hubs = len(self.hubs or [])
        if self.hubs and self.region:
            errors.append("set either hubs or region, not both")
        match self.intent:
            case "compare" if n_hubs < 2:
                errors.append("compare needs at least 2 hubs")
            case "explain" if n_hubs != 1:
                errors.append("explain needs exactly 1 hub")
            case "stat":
                if n_hubs < 1:
                    errors.append("stat needs at least 1 hub")
                if self.metric is None:
                    errors.append("stat needs a metric")
                if self.time_preset is None:
                    errors.append("stat needs a time_preset")
            case "outlook" if self.horizon != "next_7_days":
                errors.append("outlook needs horizon next_7_days")
            case "rank" | "compare" if self.horizon == "next_7_days":
                errors.append("for next-7-days questions use intent outlook")
            case "clarify" if not self.clarification_question:
                errors.append("clarify needs a clarification_question")
            case "out_of_scope" if not self.out_of_scope_reason:
                errors.append("out_of_scope needs an out_of_scope_reason")
        if self.time_preset == "specific_year" and self.year is None:
            errors.append("time_preset specific_year needs year")
        if self.time_preset == "custom" and (self.start_date is None or self.end_date is None):
            errors.append("time_preset custom needs start_date and end_date")
        if self.top_k is not None and not 1 <= self.top_k <= len(HUB_IDS):
            errors.append(f"top_k must be between 1 and {len(HUB_IDS)}")
        if self.weight_overrides:
            values = [v for v in self.weight_overrides.model_dump().values() if v is not None]
            if any(v < 0 for v in values) or (values and not any(v > 0 for v in values)):
                errors.append("weights must be non-negative and not all zero")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(description="1-3 sentences that directly answer the question")
    reasoning: list[str] = Field(description="2-5 short bullets citing the drivers, using only numbers from RESULT")
    suggested_follow_ups: list[str] = Field(description="Up to 3 natural follow-up questions")

    @model_validator(mode="after")
    def _limits(self) -> "Explanation":
        if not self.summary.strip():
            raise ValueError("summary must not be empty")
        if not 1 <= len(self.reasoning) <= 6:
            raise ValueError("reasoning needs 1-6 bullets")
        if len(self.suggested_follow_ups) > 3:
            raise ValueError("at most 3 suggested_follow_ups")
        return self


class Source(BaseModel):
    name: str
    url: str
    retrieved_at: str | None = None
    detail: str | None = None


class Uncertainty(BaseModel):
    level: Literal["low", "medium", "high"]
    reasons: list[str]


class TimeWindow(BaseModel):
    label: str
    start: str | None = None
    end: str | None = None


class AgentAnswer(BaseModel):
    conversation_id: str | None = None
    intent: str
    answer: str
    reasoning: list[str]
    suggested_follow_ups: list[str]
    table: list[dict[str, Any]]
    breakdown: dict[str, Any] | None
    assumptions: list[str]
    uncertainty: Uncertainty
    sources: list[Source]
    time_windows: list[TimeWindow]
    query_plan: dict[str, Any] | None
    meta: dict[str, Any]


# -- strict schema for the LLM ---------------------------------------------------

_KEEP = {"type", "properties", "required", "items", "enum", "anyOf", "additionalProperties", "description", "const"}


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema → flat strict schema: refs inlined, objects closed, all properties required.

    Constraint keywords (format, minimum, …) are dropped for provider portability; Pydantic
    still enforces them locally, and a violation triggers the one retry.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return resolve(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
        out = {}
        for key, value in node.items():
            if key == "properties":  # property names are data, not keywords
                out[key] = {name: resolve(sub) for name, sub in value.items()}
            elif key in _KEEP:
                out[key] = resolve(value)
        if "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    return resolve(schema)


def export_schemas(directory) -> dict[str, dict]:
    """Write the contracts as JSON files (python -m weather_risk.agent.schemas). A test guards drift."""
    import json
    from pathlib import Path

    schemas = {
        "query_plan.json": strict_json_schema(QueryPlan),
        "explanation.json": strict_json_schema(Explanation),
        "agent_answer.json": AgentAnswer.model_json_schema(),
    }
    for name, schema in schemas.items():
        (Path(directory) / name).write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
    return schemas


if __name__ == "__main__":
    from weather_risk.settings import ROOT

    export_schemas(ROOT / "schemas")
    print("Wrote schemas/*.json")
