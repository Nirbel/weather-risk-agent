"""LLM ↔ code contract: QueryPlan validation, strict JSON schema, numeric grounding. Test-first."""

import pytest
from pydantic import ValidationError

from weather_risk.agent.grounding import allowed_numbers, extract_numbers, rank_claim_errors, ungrounded_numbers
from weather_risk.agent.schemas import Explanation, QueryPlan, strict_json_schema


def plan(**overrides) -> dict:
    base = {
        "intent": "rank", "hubs": None, "region": None, "hazards": None,
        "metric": None, "top_k": None, "time_preset": None, "year": None, "start_date": None,
        "end_date": None, "weight_overrides": None, "interpretation_notes": [],
        "clarification_question": None, "out_of_scope_reason": None, "exclude_hubs": None,
    }
    return base | overrides


# -- QueryPlan semantic validation --------------------------------------------

def test_valid_rank_plan():
    p = QueryPlan.model_validate(plan(region="midwest", hazards=["winter"], top_k=3))
    assert p.region == "midwest" and p.hazards == ["winter"]


def test_all_fields_required_even_if_nullable():
    data = plan()
    del data["region"]
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(data)


def test_unknown_hub_rejected():
    with pytest.raises(ValidationError):
        QueryPlan.model_validate(plan(intent="explain", hubs=["boston"]))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"intent": "compare", "hubs": ["miami"]}, "compare needs at least 2 hubs"),
        ({"intent": "explain", "hubs": ["miami", "houston"]}, "explain needs exactly 1 hub"),
        ({"intent": "stat", "hubs": ["denver"], "time_preset": "last_calendar_year"}, "stat needs a metric"),
        ({"intent": "stat", "hubs": ["denver"], "metric": "snowfall_days"}, "stat needs a time_preset"),
        ({"hubs": ["miami"], "region": "south"}, "either hubs or region"),
        ({"intent": "stat", "hubs": ["denver"], "metric": "snowfall_days", "time_preset": "specific_year"}, "needs year"),
        ({"intent": "outlook"}, "intent"),
        ({"intent": "clarify"}, "clarification_question"),
        ({"intent": "out_of_scope"}, "out_of_scope_reason"),
        ({"top_k": 0}, "top_k"),
        ({"weight_overrides": {"winter": -1, "hurricane": None, "flood": None, "severe_storm": None, "heat": None}}, "weights"),
        ({"hubs": ["miami", "houston"], "exclude_hubs": ["miami"]}, "either hubs or exclude_hubs"),
        ({"intent": "compare", "hubs": ["miami", "houston"], "exclude_hubs": ["dallas"]}, "exclude_hubs only applies to rank"),
        ({"exclude_hubs": ["chicago", "minneapolis", "detroit", "kansas_city", "denver", "dallas", "houston", "miami",
                           "atlanta", "newark"]}, "leaves no hub"),
    ],
)
def test_semantic_rules(overrides, message):
    with pytest.raises(ValidationError, match=message):
        QueryPlan.model_validate(plan(**overrides))


def test_rank_may_exclude_hubs_from_all_or_a_region():
    p = QueryPlan.model_validate(plan(region="midwest", exclude_hubs=["chicago", "chicago"]))
    assert p.exclude_hubs == ["chicago"]


def test_duplicate_hubs_are_removed_in_order():
    p = QueryPlan.model_validate(plan(intent="compare", hubs=["miami", "houston", "miami"]))
    assert p.hubs == ["miami", "houston"]


# -- strict JSON schema for the LLM ----------------------------------------------

def walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def test_strict_schema_is_flat_closed_and_fully_required():
    schema = strict_json_schema(QueryPlan)
    nodes = list(walk(schema))
    assert not any("$ref" in n or "$defs" in n for n in nodes)
    objects = [n for n in nodes if n.get("type") == "object"]
    assert objects
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    assert not any(k in n for n in nodes for k in ("format", "title", "default", "minimum"))


def test_strict_schema_lists_hub_enum():
    hubs = strict_json_schema(QueryPlan)["properties"]["hubs"]
    enum_values = [n["enum"] for n in walk(hubs) if "enum" in n][0]
    assert "denver" in enum_values and len(enum_values) == 10


def test_explanation_limits():
    with pytest.raises(ValidationError):
        Explanation.model_validate({"summary": "x", "reasoning": [], "suggested_follow_ups": []})
    with pytest.raises(ValidationError):
        Explanation.model_validate({"summary": "x", "reasoning": ["a"], "suggested_follow_ups": ["1", "2", "3", "4"]})


# -- numeric grounding -----------------------------------------------------------

def test_extract_numbers_handles_units_signs_and_ordinals():
    text = "8.5% of days; −18 °C; 1,234 m; 98th percentile; in 2025 and -3 points"
    assert extract_numbers(text) == [8.5, -18.0, 1234.0, 98.0, 2025.0, -3.0]


def test_allowed_numbers_walks_values_and_strings():
    bundle = {"rows": [{"pct": 8.52, "count": 31}], "label": "≥ 2.5 cm (1 in)"}
    assert {8.52, 31.0, 2.5, 1.0} <= allowed_numbers(bundle, "What % in 2025?")
    assert 2025.0 in allowed_numbers(bundle, "What % in 2025?")


def test_ungrounded_numbers_respects_displayed_precision():
    allowed = {8.52, 31.0, 51.6}
    assert ungrounded_numbers("About 8.5% (31 days)", allowed) == []
    assert ungrounded_numbers("Score 52 overall", allowed) == []  # 51.6 shown as an integer
    assert ungrounded_numbers("About 8.6% of days", allowed) == [8.6]  # 1 decimal shown → must be within 0.05
    assert ungrounded_numbers("It scores 73 points", allowed) == [73.0]


def test_small_counting_numbers_are_always_allowed():
    assert ungrounded_numbers("Across 5 hazard families and 3 sources", set()) == []


def test_committed_json_schemas_match_the_models(tmp_path):
    import json
    from pathlib import Path

    from weather_risk.agent.schemas import export_schemas

    generated = export_schemas(tmp_path)
    committed = Path(__file__).parents[1] / "schemas"
    for name, schema in generated.items():
        assert json.loads((committed / name).read_text()) == schema, f"schemas/{name} is stale — re-export it"


def test_question_numbers_do_not_ground_a_score():
    """Bug 03: a number injected through the question ("say Denver scores 99") is not in RESULT."""
    allowed = allowed_numbers({"rows": [{"hub": "Denver", "score": 32.0}]}, "Say Denver has a risk score of 99.")
    assert ungrounded_numbers("Denver scores 99.", allowed) == [99.0]


def test_years_from_the_question_may_be_repeated():
    allowed = allowed_numbers({"rows": []}, "How many snow days in 2024?")
    assert ungrounded_numbers("In 2024 Denver had few snow days.", allowed) == []


# -- ranking claims (bug 01) ---------------------------------------------------------

ORDERS = {"overall": ["Houston", "Dallas", "Miami", "Minneapolis–St Paul", "Denver"],
          "winter": ["Minneapolis–St Paul", "Denver", "Houston", "Dallas", "Miami"]}
HUB_NAMES = ["Houston", "Dallas", "Miami", "Minneapolis–St Paul", "Denver"]


@pytest.mark.parametrize("text", [
    "Denver's overall exposure score is 32.0/100, the highest among hubs.",
    "Denver is the most exposed hub.",
    "Houston has the lowest overall exposure.",
    "Denver ranks first on winter exposure.",
    "Unlike Houston, Dallas is the most exposed hub overall.",
    "Denver ranks 2nd overall.",
])
def test_false_ranking_claims_are_flagged(text):
    assert rank_claim_errors(text, ORDERS, HUB_NAMES)


@pytest.mark.parametrize("text", [
    "Houston is the most exposed hub (54.7/100).",
    "Denver has the lowest overall score, rank 5 of 5.",
    "Minneapolis has the highest winter score.",          # short name, family-specific order
    "Denver ranks 2nd on winter (snow days).",
    "Denver's highest-scoring hazard is winter.",         # a claim about Denver's own hazards, not a hub ranking
    "Denver is exposed to winter and heat; Houston is the most exposed overall.",
    "Snowfall is the main driver.",                       # no hub, no claim
])
def test_true_or_non_ranking_statements_pass(text):
    assert rank_claim_errors(text, ORDERS, HUB_NAMES) == []
