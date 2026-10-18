"""Hebrew answers (bug 09): every code-built sentence a reader sees has a Hebrew version. Test-first.

Machine-readable parts (table keys, `details`, query plan) stay English on purpose: they are the API
contract and what the eval checks read. The UI shows translated column headers from `column_labels`.
"""

import re
from datetime import date

import pytest

from conftest import FakeAnalyzer
from weather_risk.agent.answer import assemble, error_answer
from weather_risk.agent.executor import ExecContext, execute
from weather_risk.agent.explainer import template_explanation
from weather_risk.agent.i18n import detect_language, family_label, indicator_label, nri_name, stat_label
from weather_risk.agent.schemas import QueryPlan
from weather_risk.config import load_scoring_config

HEBREW = re.compile(r"[֐-׿]")


def plan(**overrides) -> QueryPlan:
    base = {
        "intent": "rank", "hubs": None, "exclude_hubs": None, "region": None, "hazards": None, "metric": None,
        "top_k": None, "time_preset": None, "year": None, "start_date": None, "end_date": None,
        "weight_overrides": None, "interpretation_notes": [], "clarification_question": None,
        "out_of_scope_reason": None,
    }
    return QueryPlan.model_validate(base | overrides)


def test_detect_language():
    assert detect_language("אילו מרכזים במערב התיכון הכי חשופים לחורף?") == "he"
    assert detect_language("Which Midwest hubs are most exposed?") == "en"
    assert detect_language("Compare מיאמי and Houston") == "he"  # any Hebrew → answer in Hebrew


def test_every_configured_label_has_a_hebrew_version():
    cfg = load_scoring_config()
    for f in cfg.families:
        assert HEBREW.search(family_label(cfg, f, "he")) and HEBREW.search(family_label(cfg, f, "he", short=True))
        for spec in cfg.families[f].observed:
            assert HEBREW.search(indicator_label(spec, "he"))
    for code in cfg.nri_hazard_names:
        assert HEBREW.search(nri_name(cfg, code, "he"))
    for metric in cfg.stats.values():
        assert HEBREW.search(stat_label(metric, "he")) and HEBREW.search(stat_label(metric, "he", short=True))
        for sens in metric.sensitivity:
            assert HEBREW.search(stat_label(sens, "he", variable=metric.variable))
    assert family_label(cfg, "winter", "en", short=True) == "Winter"  # English stays as configured


CASES = {
    "rank all": (plan(), {}),
    "rank region except": (plan(region="midwest", hazards=["winter"], exclude_hubs=["detroit"]), {}),
    "rank weighted": (plan(weight_overrides={"winter": None, "hurricane": 2, "flood": None, "severe_storm": None,
                                             "heat": None}), {}),
    "rank with an incomplete hub": (plan(region="midwest"), {"incomplete": {"minneapolis"}}),
    "rank nobody": (plan(region="midwest"), {"incomplete": {"chicago", "minneapolis", "detroit", "kansas_city"}}),
    "rank empty scope": (plan(region="west", exclude_hubs=["denver"]), {}),
    "compare": (plan(intent="compare", hubs=["miami", "houston"], hazards=["hurricane", "flood"]), {}),
    "compare refused": (plan(intent="compare", hubs=["minneapolis", "chicago"]), {"incomplete": {"minneapolis"}}),
    "stat": (plan(intent="stat", hubs=["denver", "chicago"], metric="snowfall_days",
                  time_preset="last_calendar_year"), {}),
    "stat partial": (plan(intent="stat", hubs=["denver"], metric="snowfall_days", time_preset="last_calendar_year"),
                     {"missing_days": {"denver": {date(2025, 3, d) for d in range(1, 6)}}}),
    "stat window error": (plan(intent="stat", hubs=["denver"], metric="snowfall_days", time_preset="specific_year",
                               year=2030), {}),
    "explain": (plan(intent="explain", hubs=["dallas"]), {}),
    "explain one hazard": (plan(intent="explain", hubs=["minneapolis"], hazards=["winter"]), {}),
    "explain incomplete": (plan(intent="explain", hubs=["minneapolis"]), {"incomplete": {"minneapolis"}}),
    "out of scope": (plan(intent="out_of_scope", out_of_scope_reason="רעידות אדמה אינן במודל."), {}),
}


def prose(bundle) -> list[str]:
    """Every sentence a reader sees that code wrote (data gaps come from the data layer and stay English)."""
    texts = [bundle.headline, *bundle.points, *bundle.assumptions, *bundle.follow_ups,
             *(r for _, r in bundle.uncertainty), *(w["label"] for w in bundle.windows)]
    for row in bundle.rows:
        texts += [v for k, v in row.items() if isinstance(v, str) and k not in ("hub",)]
    return [t for t in texts if t and t not in bundle.data_gaps]


@pytest.mark.parametrize("name", list(CASES))
async def test_hebrew_answers_have_no_english_only_sentences(name):
    p, analyzer_args = CASES[name]
    b = await execute(p, ExecContext(FakeAnalyzer(**analyzer_args), lang="he"))
    english_only = [t for t in prose(b) if not HEBREW.search(t)]
    assert not english_only, english_only


async def test_numbers_and_keys_are_the_same_in_both_languages():
    en = await execute(plan(region="midwest", hazards=["winter"]), ExecContext(FakeAnalyzer()))
    he = await execute(plan(region="midwest", hazards=["winter"]), ExecContext(FakeAnalyzer(), lang="he"))
    assert [list(r) for r in he.rows] == [list(r) for r in en.rows]
    assert [r["score"] for r in he.rows] == [r["score"] for r in en.rows]
    assert he.orders == en.orders  # the ranking-claim check compares hub names, which stay as configured


async def test_assembled_hebrew_answer_labels_columns_and_sources():
    b = await execute(plan(intent="explain", hubs=["dallas"]), ExecContext(FakeAnalyzer(), lang="he"))
    statuses = {s: {"last_success_at": "2026-10-01T12:00:00+00:00", "last_error": None, "last_error_at": None}
                for s in ("open_meteo_archive", "fema_nri")}
    answer = assemble(plan(intent="explain", hubs=["dallas"]), b, template_explanation(b), statuses, meta={},
                      lang="he")
    assert answer.language == "he"
    assert set(answer.column_labels) >= set(answer.table[0]) and all(HEBREW.search(v)
                                                                     for v in answer.column_labels.values())
    assert all(HEBREW.search(s.detail) for s in answer.sources)
    assert all(HEBREW.search(r) for r in answer.uncertainty.reasons)


def test_hebrew_error_answers():
    answer = error_answer("x", None, {}, kind="llm_unavailable", lang="he")
    assert answer.language == "he" and HEBREW.search(answer.answer)
    assert all(HEBREW.search(q) for q in answer.suggested_follow_ups)


async def test_hebrew_question_gets_a_hebrew_explanation_or_the_hebrew_template():
    """Seen live: the model ignored the prompt's language rule. Language is now part of the check."""
    import json

    from conftest import FakeCompletion
    from weather_risk.agent.explainer import explain
    from weather_risk.settings import Settings

    question = "אילו מרכזים במערב התיכון הכי חשופים לחורף?"
    b = await execute(plan(region="midwest", hazards=["winter"]), ExecContext(FakeAnalyzer(), lang="he"))
    english = json.dumps({"summary": "Minneapolis–St Paul ranks first with 90.", "reasoning": ["Winter."],
                          "suggested_follow_ups": []})
    fake = FakeCompletion(english, english)
    explanation, _, source = await explain(question, b, Settings(), completion=fake)
    assert "ANSWER LANGUAGE: Hebrew" in fake.calls[0]["messages"][-1]["content"]
    assert "Hebrew" in fake.calls[1]["messages"][-1]["content"]  # the retry says what was wrong
    assert source == "template" and HEBREW.search(explanation.summary)

    hebrew = json.dumps({"summary": "Minneapolis–St Paul מדורג ראשון עם 90.", "reasoning": ["החורף מוביל."],
                         "suggested_follow_ups": []})
    explanation, _, source = await explain(question, b, Settings(), completion=FakeCompletion(hebrew))
    assert source == "llm" and HEBREW.search(explanation.summary)


def test_hebrew_forecast_follow_ups_are_recognised():
    from weather_risk.agent.explainer import UNANSWERABLE
    for q in ("מה התחזית ל-Denver?", "האם ירד שלג בשבוע הבא?", "מה צפוי מחר ב-Chicago?"):
        assert UNANSWERABLE.search(q), q
    assert not UNANSWERABLE.search("השווה בין Miami ל-Houston בחשיפה להוריקנים")
