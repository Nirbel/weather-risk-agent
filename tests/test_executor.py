"""Executor: QueryPlan → ResultBundle over a fake Analyzer with hand-chosen scores. Test-first."""

import json
from datetime import date

import pytest

from conftest import FakeAnalyzer
from weather_risk.agent.executor import ExecContext, execute
from weather_risk.agent.schemas import QueryPlan


def plan(**overrides) -> QueryPlan:
    base = {
        "intent": "rank", "hubs": None, "region": None, "hazards": None, "metric": None, "top_k": None,
        "time_preset": None, "year": None, "start_date": None, "end_date": None, "weight_overrides": None,
        "interpretation_notes": [], "clarification_question": None, "out_of_scope_reason": None,
    }
    return QueryPlan.model_validate(base | overrides)


@pytest.fixture
def ctx():
    return ExecContext(analyzer=FakeAnalyzer())


async def test_rank_midwest_winter_top3(ctx):
    b = await execute(plan(region="midwest", hazards=["winter"], top_k=3), ctx)
    assert [r["hub"] for r in b.rows] == ["Minneapolis–St Paul", "Detroit", "Chicago"]
    assert [r["score"] for r in b.rows] == [90.0, 80.0, 78.0]
    assert "Minneapolis–St Paul" in b.headline
    # Detroit (80) and Chicago (78) are 2 points apart < 3-point tie margin
    assert any("Detroit and Chicago" in reason for _, reason in b.uncertainty)
    assert set(b.sources) == {"open_meteo_archive", "fema_nri"}
    # assumptions are generated from config, so they cannot go stale after a calibration change
    assert any("winter 0.75/0.25" in a for a in b.assumptions)
    assert any("15 a year = 100" in a for a in b.assumptions)


async def test_rank_region_excludes_other_regions(ctx):
    b = await execute(plan(region="midwest"), ctx)
    assert {r["hub"] for r in b.rows} == {"Chicago", "Minneapolis–St Paul", "Detroit", "Kansas City"}


async def test_rank_all_overall_equal_weights(ctx):
    # houston (80+75+10+10+10)/5 = 37, dallas (90+60+30)/5 = 36, miami (85+60+30)/5 = 35
    b = await execute(plan(), ctx)
    assert [r["hub"] for r in b.rows[:3]] == ["Houston", "Dallas", "Miami"]
    assert b.rows[0]["score"] == 37.0
    assert len(b.rows) == 10


async def test_rank_with_weight_override(ctx):
    # winter ×3: Minneapolis (3·90 + 4·10) / 7 = 44.3 overtakes Houston (3·10 + 80 + 75 + 10 + 10) / 7 = 29.3
    overrides = {"winter": 3, "hurricane": None, "flood": None, "severe_storm": None, "heat": None}
    b = await execute(plan(weight_overrides=overrides), ctx)
    assert b.rows[0]["hub"] == "Minneapolis–St Paul" and b.rows[0]["score"] == 44.3
    assert any("winter ×3" in a for a in b.assumptions)


async def test_compare_two_hubs_on_two_hazards(ctx):
    # Miami (85+60)/2 = 72.5; Houston (80+75)/2 = 77.5 → Houston leads
    b = await execute(plan(intent="compare", hubs=["miami", "houston"], hazards=["hurricane", "flood"]), ctx)
    rows = {r["hub"]: r for r in b.rows}
    assert rows["Miami"]["combined"] == 72.5 and rows["Houston"]["combined"] == 77.5
    assert b.headline.startswith("Houston")
    assert "Hurricane / tropical storm: Miami higher by 5.0 points" in b.details["differences"]
    assert "Flood (inland, coastal, heavy rain): Houston higher by 15.0 points" in b.details["differences"]
    assert any("NRI alone" in a for a in b.assumptions)  # hurricane has no observed lens


async def test_stat_denver_last_year_snowfall(ctx):
    b = await execute(plan(intent="stat", hubs=["denver"], metric="snowfall_days", time_preset="last_calendar_year"), ctx)
    row = b.rows[0]
    assert (row["days meeting threshold"], row["days with data"], row["% of days"]) == (30, 365, 8.2)
    assert (b.windows[0]["start"], b.windows[0]["end"]) == ("2025-01-01", "2025-12-31")
    assert "8.2%" in b.headline and "Denver" in b.headline
    assert b.sources == ["open_meteo_archive"]
    assert b.details["sensitivity"]["Denver"]


async def test_stat_window_error_is_explained_not_raised(ctx):
    b = await execute(plan(intent="stat", hubs=["denver"], metric="snowfall_days", time_preset="specific_year",
                           year=2030), ctx)
    assert b.rows == [] and "no observed data" in b.headline


async def test_explain_dallas_orders_hazards_by_contribution(ctx):
    b = await execute(plan(intent="explain", hubs=["dallas"]), ctx)
    assert b.rows[0]["hazard"] == "Severe storm (tornado, hail, strong wind)"
    assert b.rows[0]["score"] == 90.0 and b.rows[0]["rank among hubs"] == "1 of 10"
    assert b.details["overall_rank"] == "2 of 10"  # 36 is second to Houston's 37
    assert "Dallas" in b.headline and "Severe storm" in b.headline


async def test_data_gaps_reach_the_bundle():
    ctx = ExecContext(analyzer=FakeAnalyzer(gaps={"miami": ["Miami 2023: Open-Meteo request failed."]}))
    b = await execute(plan(intent="compare", hubs=["miami", "houston"]), ctx)
    assert "Miami 2023: Open-Meteo request failed." in b.data_gaps


async def test_out_of_scope_returns_capabilities(ctx):
    b = await execute(plan(intent="out_of_scope", out_of_scope_reason="Earthquakes are not a modeled hazard."), ctx)
    assert b.rows == [] and b.headline == "Earthquakes are not a modeled hazard."
    assert b.details["what_i_can_answer"] and "Denver" in b.details["hubs"]


async def test_bundle_for_llm_is_compact_json(ctx):
    b = await execute(plan(intent="explain", hubs=["dallas"]), ctx)
    assert len(json.dumps(b.for_llm())) < 12_000  # keep the explainer prompt within Groq's token budget


# -- coverage rule: incomplete weather is never ranked as if complete -------------------

async def test_rank_leaves_out_a_hub_with_incomplete_weather():
    # Midwest winter: Minneapolis 90 would be first, but its weather is incomplete → Detroit 80, Chicago 78, KC 50
    ctx = ExecContext(analyzer=FakeAnalyzer(incomplete={"minneapolis"}))
    b = await execute(plan(region="midwest", hazards=["winter"]), ctx)
    ranked = [(r["rank"], r["hub"], r["score"]) for r in b.rows if r["rank"] is not None]
    assert ranked == [(1, "Detroit", 80.0), (2, "Chicago", 78.0), (3, "Kansas City", 50.0)]
    unranked = [r for r in b.rows if r["rank"] is None]
    assert [(r["hub"], r["score"]) for r in unranked] == [("Minneapolis–St Paul", None)]
    assert "incomplete" in unranked[0]["note"]
    assert b.headline.startswith("Detroit")
    assert any(level == "high" and "Minneapolis–St Paul" in reason and "not ranked" in reason
               for level, reason in b.uncertainty)


async def test_hazard_scored_from_nri_alone_still_ranks_that_hub():
    # Hurricane has no observed-weather lens, so a weather gap does not affect it.
    ctx = ExecContext(analyzer=FakeAnalyzer(incomplete={"minneapolis"}))
    b = await execute(plan(region="midwest", hazards=["hurricane"]), ctx)
    assert all(r["rank"] is not None for r in b.rows) and len(b.rows) == 4


async def test_no_rankable_hub_gives_no_ranking():
    ctx = ExecContext(analyzer=FakeAnalyzer(incomplete={"chicago", "minneapolis", "detroit", "kansas_city"}))
    b = await execute(plan(region="midwest", hazards=["winter"]), ctx)
    assert b.headline.startswith("No hub can be ranked")
    assert all(r["rank"] is None and r["score"] is None for r in b.rows)
    assert b.uncertainty[0][0] == "high"


async def test_compare_refuses_when_fewer_than_two_hubs_are_complete():
    ctx = ExecContext(analyzer=FakeAnalyzer(incomplete={"minneapolis"}))
    b = await execute(plan(intent="compare", hubs=["minneapolis", "detroit"], hazards=["winter"]), ctx)
    assert b.headline.startswith("I can't compare")
    assert {r["hub"]: r["Winter"] for r in b.rows} == {"Minneapolis–St Paul": None, "Detroit": 80.0}
    assert any(level == "high" for level, _ in b.uncertainty)


async def test_compare_ranks_only_the_complete_hubs():
    ctx = ExecContext(analyzer=FakeAnalyzer(incomplete={"minneapolis"}))
    b = await execute(plan(intent="compare", hubs=["minneapolis", "detroit", "chicago"], hazards=["winter"]), ctx)
    assert b.headline.startswith("Detroit has the higher winter exposure (80.0/100) vs Chicago (78.0)")
    assert any("Minneapolis–St Paul" in reason and "not compared" in reason for _, reason in b.uncertainty)


async def test_explain_an_incomplete_hub_says_so_instead_of_scoring_it():
    ctx = ExecContext(analyzer=FakeAnalyzer(incomplete={"minneapolis"}))
    b = await execute(plan(intent="explain", hubs=["minneapolis"]), ctx)
    assert "not scored" in b.headline and "6-day gap" in b.headline
    rows = {r["hazard"].split(" (")[0]: r for r in b.rows}
    assert rows["Winter"]["score"] is None and rows["Winter"]["rank among hubs"] == "not ranked"
    assert rows["Hurricane / tropical storm"]["score"] == 10.0  # NRI-only family is still scored
    assert b.details["overall_score"] is None and b.details["overall_rank"] == "not ranked"


async def test_explain_ranks_only_among_complete_hubs():
    # overall: Houston 37, Dallas 36, … ; Minneapolis is incomplete → Dallas is 2 of 9
    ctx = ExecContext(analyzer=FakeAnalyzer(incomplete={"minneapolis"}))
    b = await execute(plan(intent="explain", hubs=["dallas"]), ctx)
    assert b.details["overall_rank"] == "2 of 9"


async def test_stat_with_a_four_day_gap_is_partial_not_a_percentage():
    # Denver 2025 without Jan 10–13: 361/365 = 98.9 % and a 4-day gap → no percentage is given
    missing = {date(2025, 1, d) for d in range(10, 14)}
    ctx = ExecContext(analyzer=FakeAnalyzer(missing_days={"denver": missing}))
    b = await execute(plan(intent="stat", hubs=["denver"], metric="snowfall_days", time_preset="last_calendar_year"), ctx)
    row = b.rows[0]
    assert (row["% of days"], row["days meeting threshold"], row["days with data"], row["coverage %"]) == \
        (None, None, 361, 98.9)
    assert "incomplete" in b.headline and "%" not in b.headline.split("(")[0]
    assert b.uncertainty[0][0] == "high"


async def test_stat_with_a_two_day_gap_is_still_answered():
    # Jan 10–11 missing: 363/365 = 99.4 %, gap 2 → complete; snowy days Jan 1–30 minus 2 = 28 of 363 = 7.7 %
    missing = {date(2025, 1, 10), date(2025, 1, 11)}
    ctx = ExecContext(analyzer=FakeAnalyzer(missing_days={"denver": missing}))
    b = await execute(plan(intent="stat", hubs=["denver"], metric="snowfall_days", time_preset="last_calendar_year"), ctx)
    assert (b.rows[0]["days meeting threshold"], b.rows[0]["days with data"], b.rows[0]["% of days"]) == (28, 363, 7.7)


async def test_a_hazard_with_no_evidence_is_not_silently_dropped():
    # FEMA NRI unreachable: hurricane has no evidence at all. A "hurricane + flood" comparison must not
    # quietly become a flood-only comparison.
    ctx = ExecContext(analyzer=FakeAnalyzer(no_evidence={"hurricane"}))
    b = await execute(plan(intent="compare", hubs=["miami", "houston"], hazards=["hurricane", "flood"]), ctx)
    assert b.headline.startswith("I can't compare") and "FEMA National Risk Index" in b.headline
    assert {r["hub"]: r["Flood"] for r in b.rows} == {"Miami": 60.0, "Houston": 75.0}  # still shown
    assert all(r["combined"] is None for r in b.rows)
    assert b.uncertainty[0][0] == "high"


async def test_ranking_on_a_hazard_with_no_evidence_ranks_nobody():
    ctx = ExecContext(analyzer=FakeAnalyzer(no_evidence={"hurricane"}))
    b = await execute(plan(hazards=["hurricane"]), ctx)
    assert b.headline.startswith("No hub can be ranked") and all(r["rank"] is None for r in b.rows)
    flood = await execute(plan(hazards=["flood"]), ctx)  # other hazards still rank normally
    assert flood.rows[0]["rank"] == 1
