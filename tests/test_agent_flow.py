"""Agent flow with a mocked LLM: planner messages, explainer grounding, answer contract, follow-ups."""

import json
from datetime import date

import pytest

from conftest import FakeCompletion
from weather_risk.agent.answer import assemble
from weather_risk.agent.executor import ExecContext, execute
from weather_risk.agent.explainer import explain, template_explanation
from weather_risk.agent.planner import build_messages
from weather_risk.agent.schemas import Explanation, QueryPlan
from weather_risk.agent.service import Agent
from weather_risk.settings import Settings

TODAY = date(2026, 10, 1)
SETTINGS = Settings(llm_model="groq/openai/gpt-oss-20b", llm_fallback_model="gemini/gemini-3.8-flash")


def plan_json(**overrides) -> str:
    base = {
        "intent": "rank", "hubs": None, "region": None, "hazards": None, "horizon": "long_term",
        "metric": None, "top_k": None, "time_preset": None, "year": None, "start_date": None,
        "end_date": None, "weight_overrides": None, "interpretation_notes": [],
        "clarification_question": None, "out_of_scope_reason": None,
    }
    return json.dumps(base | overrides)


def explanation_json(summary: str, reasoning: list[str] | None = None) -> str:
    return json.dumps({"summary": summary, "reasoning": reasoning or ["Winter scores drive the order."],
                       "suggested_follow_ups": ["Why is Minneapolis first?"]})


MIDWEST_WINTER = plan_json(region="midwest", hazards=["winter"], top_k=3,
                           interpretation_notes=["'winter disruption' = winter hazard family"])


# -- planner -------------------------------------------------------------------

def test_planner_messages_carry_date_roster_and_history():
    history = [{"user_text": "Compare Miami and Houston", "plan": {"intent": "compare", "hubs": ["miami", "houston"]},
                "answer": {"answer": "Houston is higher."}}]
    msgs = build_messages("Add Dallas", history, today=TODAY, data_start=date(2016, 1, 1), data_end=date(2026, 9, 25))
    system = msgs[0]["content"]
    assert "2026-10-01" in system and "calendar 2025" in system
    assert "dallas: Dallas, TX, south" in system
    assert msgs[1] == {"role": "user", "content": "Compare Miami and Houston"}
    assert json.loads(msgs[2]["content"]) == {"intent": "compare", "hubs": ["miami", "houston"]}
    assert msgs[-1] == {"role": "user", "content": "Add Dallas"}


# -- explainer -----------------------------------------------------------------

@pytest.fixture
def bundle(synthetic_store):
    plan = QueryPlan.model_validate_json(MIDWEST_WINTER)
    return execute(plan, ExecContext(store=synthetic_store, today=TODAY))


async def test_grounded_explanation_is_accepted(bundle):
    fake = FakeCompletion(explanation_json("Minneapolis–St Paul ranks first with a winter score of 90."))
    explanation, meta, source = await explain("Midwest winter?", bundle, SETTINGS, completion=fake)
    assert source == "llm" and "90" in explanation.summary


async def test_invented_number_is_retried_then_falls_back_to_template(bundle):
    fake = FakeCompletion(explanation_json("Minneapolis scores 97."), explanation_json("Minneapolis scores 96."))
    explanation, meta, source = await explain("Midwest winter?", bundle, SETTINGS, completion=fake)
    assert source == "template"
    assert explanation.summary == bundle.headline
    assert "97" in fake.calls[1]["messages"][-1]["content"]  # the retry names the ungrounded number


async def test_llm_down_falls_back_to_template(bundle):
    fake = FakeCompletion(RuntimeError("down"), RuntimeError("down"))
    _, _, source = await explain("q", bundle, SETTINGS, completion=fake)
    assert source == "template"


def test_template_explanation_uses_bundle_points(bundle):
    e = template_explanation(bundle)
    assert isinstance(e, Explanation) and e.summary == bundle.headline and e.reasoning


# -- answer contract -----------------------------------------------------------

def test_every_answer_states_assumptions_uncertainty_sources_window(bundle, synthetic_store):
    plan = QueryPlan.model_validate_json(MIDWEST_WINTER)
    answer = assemble(plan, bundle, template_explanation(bundle), synthetic_store.source_statuses(), meta={})
    assert answer.assumptions and answer.uncertainty.reasons and answer.sources and answer.time_windows
    assert "'winter disruption' = winter hazard family" in answer.assumptions
    assert answer.uncertainty.level == "medium"  # Detroit/Chicago near-tie
    names = {s.name for s in answer.sources}
    assert any("National Risk Index" in n for n in names)
    assert all(s.retrieved_at for s in answer.sources)


def test_failed_source_raises_uncertainty(bundle, synthetic_store):
    synthetic_store.source_error("fema_nri", "fema_nri: HTTP 503")
    plan = QueryPlan.model_validate_json(MIDWEST_WINTER)
    answer = assemble(plan, bundle, template_explanation(bundle), synthetic_store.source_statuses(), meta={})
    assert any("FEMA National Risk Index" in r and "failed" in r for r in answer.uncertainty.reasons)


# -- full agent turn -----------------------------------------------------------

async def test_agent_turn_and_follow_up(synthetic_store):
    fake = FakeCompletion(
        MIDWEST_WINTER,
        explanation_json("Minneapolis–St Paul ranks first with 90."),
        plan_json(intent="explain", hubs=["minneapolis"], hazards=["winter"]),
        explanation_json("Its winter score is 90."),
    )
    agent = Agent(synthetic_store, SETTINGS, completion=fake, today=TODAY)
    first = await agent.ask("Which Midwest hubs are most exposed to winter disruption?")
    assert first.intent == "rank" and first.table[0]["hub"] == "Minneapolis–St Paul"
    assert first.meta["explanation_source"] == "llm"

    second = await agent.ask("Why is it first?", conversation_id=first.conversation_id)
    assert second.conversation_id == first.conversation_id
    planner_msgs = fake.calls[2]["messages"]
    assert any(m["role"] == "assistant" and '"region": "midwest"' in m["content"] for m in planner_msgs)
    assert second.intent == "explain"


async def test_planner_failure_gives_clear_error_answer(synthetic_store):
    fake = FakeCompletion("not json", "still not json")
    answer = await Agent(synthetic_store, SETTINGS, completion=fake, today=TODAY).ask("???")
    assert answer.intent == "error"
    assert answer.uncertainty.level == "high"
    assert "couldn't" in answer.answer.lower()


async def test_offline_answer_from_plan_uses_no_llm(synthetic_store):
    agent = Agent(synthetic_store, SETTINGS, completion=FakeCompletion(), today=TODAY)
    answer = await agent.answer_plan(QueryPlan.model_validate_json(MIDWEST_WINTER), "q", use_llm=False)
    assert answer.meta["explanation_source"] == "template"
    assert answer.table[0]["hub"] == "Minneapolis–St Paul"
