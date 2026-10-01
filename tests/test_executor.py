"""Executor: QueryPlan → ResultBundle over a fake Analyzer with hand-chosen scores. Test-first."""

import json

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
