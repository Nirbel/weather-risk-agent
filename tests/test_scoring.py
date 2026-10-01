"""Scoring core — written test-first. Every expected value is worked out by hand in the comment."""

from datetime import date, timedelta

import pytest

from weather_risk.scoring import exposure, near_term, robustness
from weather_risk.scoring.indicators import days_per_year, exceeds, piecewise
from weather_risk.sources.open_meteo import DailyRow


def rows(values: list[float | None], variable: str, start: date = date(2020, 1, 1)) -> list[DailyRow]:
    blank = dict(snowfall_cm=0.0, precip_mm=0.0, tmax_c=10.0, tmin_c=0.0, gust_kmh=10.0)
    return [DailyRow(date=start + timedelta(days=i), **(blank | {variable: v})) for i, v in enumerate(values)]


# -- normalization ------------------------------------------------------------

def test_piecewise_interpolates_and_is_flat_outside():
    pts = [[0, 0], [2.5, 40], [15, 80], [30, 100]]
    assert piecewise(0, pts) == 0
    assert piecewise(8.75, pts) == pytest.approx(60)  # halfway between 2.5→40 and 15→80
    assert piecewise(15, pts) == 80
    assert piecewise(45, pts) == 100  # flat above the last point
    assert piecewise(-3, pts) == 0  # flat below the first point


def test_piecewise_handles_decreasing_scores():
    pts = [[-25, 100], [-10, 20], [0, 0]]  # colder = worse
    assert piecewise(-17.5, pts) == pytest.approx(60)  # midpoint of 100 and 20
    assert piecewise(-40, pts) == 100
    assert piecewise(5, pts) == 0


def test_exceeds_operators():
    assert exceeds(2.5, "ge", 2.5) and not exceeds(2.5, "gt", 2.5)
    assert exceeds(-18, "le", -18) and not exceeds(-17.9, "le", -18)


def test_days_per_year_scales_by_valid_days():
    # 730.5 days = 2 years exactly; 6 qualifying days → 3 per year. A null day is not counted as data.
    values = [3.0] * 6 + [0.0] * 724 + [None]
    rate, n_valid = days_per_year(rows(values, "snowfall_cm"), "snowfall_cm", "ge", 2.5)
    assert n_valid == 730
    assert rate == pytest.approx(6 * 365.25 / 730)


def test_days_per_year_none_when_no_data():
    assert days_per_year(rows([None, None], "tmin_c"), "tmin_c", "le", -18) == (None, 0)


# -- exposure: lens and family aggregation -----------------------------------

CFG = {
    "lens_weights": {"observed": 0.4, "modeled": 0.4, "history": 0.2},
    "family_weights": {"winter": 1, "hurricane": 1},
    "declarations_since": date(2000, 1, 1),
    "nri_hazard_names": {"WNTW": "Winter weather", "HRCN": "Hurricane"},
    "families": {
        "winter": {
            "label": "Winter",
            # 5 snow days/yr scores 100 → 2.5 days/yr scores 50
            "observed": [{"id": "snow_days", "label": "Snow days", "variable": "snowfall_cm", "op": "ge",
                          "threshold": 2.5, "full_score_at": 5}],
            "modeled": {"nri": ["WNTW"], "combine": "mean"},
            "history": {"incident_types": ["Snowstorm"], "full_score_at": 4},
        },
        "hurricane": {
            "label": "Hurricane",
            "observed": [],
            "modeled": {"nri": ["HRCN"], "combine": "mean"},
            "history": {"incident_types": ["Hurricane"], "full_score_at": 4},
        },
    },
}


def four_years_with_snow_days(n: int) -> list[DailyRow]:
    # 1461 days = exactly 4 × 365.25, so n snow days → n/4 per year
    return rows([3.0] * n + [0.0] * (1461 - n), "snowfall_cm")


def decl(kind: str, year: int, title: str = "X") -> dict:
    return {"disaster_number": year, "incident_type": kind, "declaration_date": f"{year}-06-01", "title": title}


def test_family_score_weights_three_lenses():
    # observed: 10 snow days in 4 years → 2.5/yr → 50; modeled WNTW pct 30 → 30;
    # history: 4 snowstorm DRs → 100.  0.4·50 + 0.4·30 + 0.2·100 = 52
    inputs = exposure.HubInputs(
        daily=four_years_with_snow_days(10),
        nri={"WNTW": {"pct": 30.0, "alrb": 0.001}, "HRCN": {"pct": 60.0, "alrb": 0.01}},
        declarations=[decl("Snowstorm", y) for y in (2001, 2005, 2010, 2020)] + [decl("Hurricane", 2008)],
    )
    result = exposure.compute_exposure("x", inputs, CFG)
    winter = result["families"]["winter"]
    assert winter["lenses"]["observed"]["score"] == pytest.approx(50)
    assert winter["lenses"]["modeled"]["score"] == 30
    assert winter["lenses"]["history"]["score"] == 100
    assert winter["score"] == pytest.approx(52)


def test_family_lens_weights_can_be_overridden_per_family():
    # winter with observed 0.6 / modeled 0.2 / history 0.2: 0.6·50 + 0.2·30 + 0.2·100 = 56
    cfg = CFG | {"families": CFG["families"] | {"winter": CFG["families"]["winter"] | {
        "lens_weights": {"observed": 0.6, "modeled": 0.2, "history": 0.2}}}}
    inputs = exposure.HubInputs(
        daily=four_years_with_snow_days(10),
        nri={"WNTW": {"pct": 30.0, "alrb": 0.001}, "HRCN": {"pct": 60.0, "alrb": 0.01}},
        declarations=[decl("Snowstorm", y) for y in (2001, 2005, 2010, 2020)],
    )
    winter = exposure.compute_exposure("x", inputs, cfg)["families"]["winter"]
    assert winter["score"] == pytest.approx(56)
    assert winter["lenses"]["observed"]["weight"] == 0.6


def test_family_without_observed_lens_renormalizes():
    # hurricane has no observed lens by design: (0.4·60 + 0.2·25) / 0.6 = 29/0.6 = 48.33
    inputs = exposure.HubInputs(
        daily=four_years_with_snow_days(0),
        nri={"WNTW": {"pct": 0.0, "alrb": None}, "HRCN": {"pct": 60.0, "alrb": 0.01}},
        declarations=[decl("Hurricane", 2008)],  # 1 of 4 → 25
    )
    hurricane = exposure.compute_exposure("x", inputs, CFG)["families"]["hurricane"]
    assert hurricane["score"] == pytest.approx(48.33, abs=0.01)
    assert hurricane["missing_lenses"] == []  # absent by design is not a data gap


def test_failed_source_is_flagged_and_weights_renormalize():
    # NRI unavailable → winter = (0.4·50 + 0.2·100) / 0.6 = 66.67, modeled lens reported missing
    inputs = exposure.HubInputs(
        daily=four_years_with_snow_days(10),
        nri=None,
        declarations=[decl("Snowstorm", y) for y in (2001, 2005, 2010, 2020)],
    )
    result = exposure.compute_exposure("x", inputs, CFG)
    winter = result["families"]["winter"]
    assert winter["score"] == pytest.approx(66.67, abs=0.01)
    assert winter["missing_lenses"] == ["modeled"]
    assert winter["missing_weight_share"] == pytest.approx(0.4)
    assert any("FEMA National Risk Index" in gap for gap in result["data_gaps"])


def test_declarations_before_window_are_ignored():
    cfg = CFG | {"declarations_since": date(2010, 1, 1)}
    inputs = exposure.HubInputs(daily=four_years_with_snow_days(0), nri={"WNTW": {"pct": 0, "alrb": None},
                                "HRCN": {"pct": 0, "alrb": None}},
                                declarations=[decl("Hurricane", 2005), decl("Hurricane", 2017)])
    history = exposure.compute_exposure("x", inputs, cfg)["families"]["hurricane"]["lenses"]["history"]
    assert history["indicators"][0]["raw"] == 1  # only the 2017 one


def test_overall_is_equal_weight_mean_and_names_top_family():
    # winter 52, hurricane 48.33 → overall (52 + 48.33)/2 = 50.17; top family = winter
    inputs = exposure.HubInputs(
        daily=four_years_with_snow_days(10),
        nri={"WNTW": {"pct": 30.0, "alrb": 0.001}, "HRCN": {"pct": 60.0, "alrb": 0.01}},
        declarations=[decl("Snowstorm", y) for y in (2001, 2005, 2010, 2020)] + [decl("Hurricane", 2008)],
    )
    result = exposure.compute_exposure("x", inputs, CFG)
    assert result["score"] == pytest.approx(50.17, abs=0.01)
    assert result["top_family"] == "winter"
    # contribution of each family = score × weight / Σweights
    assert result["families"]["winter"]["contribution"] == pytest.approx(26.0)


def test_rescore_with_weight_override():
    # weight hurricane ×3: (52·1 + 48.33·3) / 4 = 49.25
    breakdown = {"families": {"winter": {"score": 52.0}, "hurricane": {"score": 48.33}}}
    assert exposure.overall_score(breakdown, {"winter": 1, "hurricane": 3}) == pytest.approx(49.25, abs=0.01)


def test_modeled_combine_max():
    cfg = {**CFG, "families": {"flood": {"label": "Flood", "observed": [],
                                         "modeled": {"nri": ["IFLD", "CFLD"], "combine": "max"}, "history": None}},
           "family_weights": {"flood": 1}, "nri_hazard_names": {"IFLD": "Inland", "CFLD": "Coastal"}}
    inputs = exposure.HubInputs(daily=[], nri={"IFLD": {"pct": 40, "alrb": 1}, "CFLD": {"pct": 70, "alrb": 1}},
                                declarations=[])
    assert exposure.compute_exposure("x", inputs, cfg)["families"]["flood"]["score"] == 70


# -- near-term ----------------------------------------------------------------

NT_CFG = {
    "forecast": {
        "winter": [{"variable": "snowfall_cm", "label": "Daily snowfall (cm)", "points": [[0, 0], [2.5, 40], [15, 80], [30, 100]]}],
        "heat": [{"variable": "tmax_c", "label": "Daily maximum temp (°C)", "points": [[30, 0], [35, 30], [40, 70], [43, 100]]}],
    },
    "nws_severity": {"Extreme": 100, "Severe": 80, "Moderate": 50, "Minor": 25, "Unknown": 25},
    "nws_events": {"hurricane": ["Hurricane"], "winter": ["Winter", "Snow"], "heat": ["Heat"]},
    "bands": [{"min": 0, "band": "Low"}, {"min": 25, "band": "Guarded"}, {"min": 50, "band": "Elevated"}, {"min": 75, "band": "Severe"}],
}


def forecast(snow: list[float], tmax: list[float]) -> list[dict]:
    return [{"date": f"2026-10-0{i + 1}", "snowfall_cm": s, "tmax_c": t} for i, (s, t) in enumerate(zip(snow, tmax))]


def test_near_term_takes_worst_forecast_day_per_family():
    # worst snow day 8.75 cm → 60; worst heat day 37.5 °C → 50 (midpoint of 30 and 70); overall = max = 60
    result = near_term.compute_near_term(forecast([0, 8.75, 1], [20, 37.5, 25]), [], NT_CFG)
    assert result["families"]["winter"]["score"] == pytest.approx(60)
    assert result["families"]["heat"]["score"] == pytest.approx(50)
    assert result["score"] == pytest.approx(60)
    assert result["band"] == "Elevated"
    assert result["top_family"] == "winter"
    assert "2026-10-02" in result["families"]["winter"]["drivers"][0]


def test_nws_alert_overrides_quiet_forecast():
    # Winter Storm Warning, Severe → 80 → band Severe; hurricane alert, Moderate → 50
    alerts = [
        {"event": "Winter Storm Warning", "severity": "Severe", "urgency": "Expected", "certainty": "Likely", "onset": None, "expires": "2026-10-03T00:00:00-05:00"},
        {"event": "Hurricane Watch", "severity": "Moderate", "urgency": "Future", "certainty": "Possible", "onset": None, "expires": None},
        {"event": "Air Quality Alert", "severity": "Minor", "urgency": "Expected", "certainty": "Likely", "onset": None, "expires": None},
    ]
    result = near_term.compute_near_term(forecast([0, 0], [20, 20]), alerts, NT_CFG)
    assert result["families"]["winter"]["score"] == 80
    assert result["families"]["hurricane"]["score"] == 50
    assert result["band"] == "Severe"
    assert any("Air Quality Alert" in note for note in result["unmapped_alerts"])


def test_nws_alert_scaled_by_certainty():
    # Flood Watch, Severe (80) but only "Possible" (×0.7) → 56 → Elevated, not Severe
    cfg = NT_CFG | {"nws_events": NT_CFG["nws_events"] | {"flood": ["Flood"]},
                    "nws_certainty": {"Observed": 1.0, "Likely": 1.0, "Possible": 0.7, "Unlikely": 0.4, "Unknown": 0.7}}
    alert = {"event": "Flood Watch", "severity": "Severe", "urgency": "Future", "certainty": "Possible",
             "onset": None, "expires": None}
    result = near_term.compute_near_term([], [alert], cfg)
    assert result["families"]["flood"]["score"] == pytest.approx(56)
    assert result["band"] == "Elevated"


def test_band_boundaries():
    bands = NT_CFG["bands"]
    assert near_term.band_for(24.9, bands) == "Low"
    assert near_term.band_for(25, bands) == "Guarded"
    assert near_term.band_for(75, bands) == "Severe"


# -- ranking robustness -------------------------------------------------------

def test_rank_stability_detects_fragile_first_place():
    # Two families with one lens each. A: x=80, y=40 → 60. B: x=40, y=78 → 59.
    # x weight ×1.25: A 62.2 / B 56.9 (A first)   x ×0.75: A 57.1 / B 61.7 (B first)
    # y weight ×1.25: A 57.8 / B 61.1 (B first)   y ×0.75: A 62.9 / B 56.3 (A first)
    # 3 lens-weight scenarios ×2 cannot change anything (one lens per family) → 10 scenarios, top-1 holds in 8.
    def bd(x, y):
        lens = lambda s: {"score": s, "lenses": {"modeled": {"score": s, "weight": 0.4}}, "missing_lenses": []}
        return {"families": {"x": lens(x), "y": lens(y)}}

    breakdowns = {"A": bd(80, 40), "B": bd(40, 78), "C": bd(10, 10)}
    result = robustness.rank_stability(
        breakdowns, families=["x", "y"], family_weights={"x": 1, "y": 1},
        lenses=("observed", "modeled", "history"), perturbation=0.25, top_k=2,
    )
    assert result["scenarios"] == 10
    assert result["top1_unchanged"] == 8
    assert result["topk_unchanged"] == 10  # {A, B} stays the top-2 set
    assert result["baseline"][:2] == ["A", "B"]


def test_ties_within_margin():
    ranked = [("A", 60.0), ("B", 58.5), ("C", 40.0)]
    assert robustness.near_ties(ranked, margin=3.0, top_k=3) == [("A", "B", 1.5)]
