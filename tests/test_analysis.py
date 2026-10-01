"""Analyzer over recorded real responses: on-demand fetch → cache → deterministic exposure."""

from datetime import date

import httpx



async def test_exposure_for_all_hubs_uses_previous_five_years(services):
    result = await services.analyzer.exposure()
    assert (result.start, result.end) == (date(2021, 1, 1), date(2025, 12, 31))
    assert len(result.breakdowns) == 10
    assert result.nri_version == "December 2025"
    denver = result.breakdowns["denver"]
    assert denver["meta"]["observed_window"]["coverage_pct"] == 100.0
    assert denver["meta"]["grid_cell"]["latitude"] is not None
    assert denver["data_gaps"] == []


async def test_domain_sanity_orderings_on_real_data(services):
    # Properties any analyst would expect — not numbers copied from the implementation.
    bd = (await services.analyzer.exposure()).breakdowns
    fam = lambda hub, family: bd[hub]["families"][family]["score"]
    assert fam("minneapolis", "winter") > fam("miami", "winter")
    assert fam("minneapolis", "winter") > fam("dallas", "winter")
    assert fam("miami", "hurricane") > fam("denver", "hurricane")
    assert fam("houston", "hurricane") > fam("chicago", "hurricane")
    assert fam("dallas", "heat") > fam("minneapolis", "heat")
    assert fam("denver", "hurricane") == 0.0  # NRI: hurricane not applicable inland


async def test_data_is_fetched_once_then_served_from_cache(services):
    await services.analyzer.exposure(["denver"])
    calls = services.router.routes[0].call_count
    assert calls == 5  # one request per year 2021–2025
    await services.analyzer.exposure(["denver"])
    assert services.router.routes[0].call_count == calls


async def test_weather_outage_is_reported_as_a_gap(services, monkeypatch):
    async def no_backoff(_):
        return None

    monkeypatch.setattr("weather_risk.sources.http.asyncio.sleep", no_backoff)  # retries happen, instantly
    services.router.routes[0].mock(return_value=httpx.Response(503))
    bd = (await services.analyzer.exposure(["miami"])).breakdowns["miami"]
    assert any("Open-Meteo request failed" in g for g in bd["data_gaps"])
    assert bd["families"]["winter"]["missing_lenses"] == ["observed"]
    assert bd["families"]["hurricane"]["score"] is not None  # NRI-only family still scored


async def test_yearly_indicators_for_analytics(services):
    yearly = (await services.analyzer.yearly_indicators(["minneapolis"]))["minneapolis"]
    assert sorted(yearly["years"]) == [2021, 2022, 2023, 2024, 2025]
    assert set(yearly["years"][2025]) == set(yearly["labels"])
    assert sum(y["extreme_cold_days"] for y in yearly["years"].values()) > 0
