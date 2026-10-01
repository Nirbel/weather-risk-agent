"""Scoring core — test-first. Every expected value is worked out by hand in the comment."""

from datetime import date, timedelta

import pytest

from weather_risk.config import ScoringConfig
from weather_risk.scoring import exposure, robustness
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
    assert piecewise(-17.5, pts) == pytest.approx(60)
    assert piecewise(-40, pts) == 100 and piecewise(5, pts) == 0


def test_exceeds_operators():
    assert exceeds(2.5, "ge", 2.5) and not exceeds(2.5, "gt", 2.5)
    assert exceeds(-18, "le", -18) and not exceeds(-17.9, "le", -18)


def test_days_per_year_scales_by_valid_days():
    # 6 qualifying days among 730 valid days → 6 × 365.25 / 730 per year; a null day is not data
    values = [3.0] * 6 + [0.0] * 724 + [None]
    rate, n_valid = days_per_year(rows(values, "snowfall_cm"), "snowfall_cm", "ge", 2.5)
    assert n_valid == 730
    assert rate == pytest.approx(6 * 365.25 / 730)


def test_days_per_year_none_when_no_data():
    assert days_per_year(rows([None, None], "tmin_c"), "tmin_c", "le", -18) == (None, 0)


# -- exposure: lens and family aggregation -----------------------------------

CFG = ScoringConfig.model_validate({
    "climatology": {"years": 4},
    "coverage": {"min_pct": 99, "max_gap_days": 3},
    "family_weights": {"winter": 1, "hurricane": 1},
    "lens_weights": {"observed": 0.5, "modeled": 0.5},
    "nri_hazard_names": {"WNTW": "Winter weather", "HRCN": "Hurricane"},
    "robustness": {"perturbation": 0.25, "tie_margin": 3.0},
    "stats": {},
    "families": {
        "winter": {
            "label": "Winter",
            # 5 snow days/yr scores 100 → 2.5 days/yr scores 50
            "observed": [{"id": "snow_days", "label": "Snow days", "variable": "snowfall_cm", "op": "ge",
                          "threshold": 2.5, "full_score_at": 5}],
            "modeled": {"nri": ["WNTW"], "combine": "mean"},
        },
        "hurricane": {"label": "Hurricane", "observed": [], "modeled": {"nri": ["HRCN"], "combine": "mean"}},
    },
})
NRI = {"WNTW": {"pct": 30.0, "alrb": 0.001}, "HRCN": {"pct": 60.0, "alrb": 0.01}}


def four_years_with_snow_days(n: int) -> list[DailyRow]:
    # 1461 days = exactly 4 × 365.25, so n snow days → n/4 per year
    return rows([3.0] * n + [0.0] * (1461 - n), "snowfall_cm")


def with_lens_weights(cfg: ScoringConfig, family: str, observed: float, modeled: float) -> ScoringConfig:
    data = cfg.model_dump()
    data["families"][family]["lens_weights"] = {"observed": observed, "modeled": modeled}
    return ScoringConfig.model_validate(data)


def test_family_score_weights_two_lenses():
    # observed: 10 snow days in 4 years → 2.5/yr → 50; modeled WNTW pct 30 → 0.5·50 + 0.5·30 = 40
    result = exposure.compute_exposure("x", exposure.HubInputs(four_years_with_snow_days(10), NRI), CFG)
    winter = result["families"]["winter"]
    assert winter["lenses"]["observed"]["score"] == pytest.approx(50)
    assert winter["lenses"]["modeled"]["score"] == 30
    assert winter["score"] == pytest.approx(40)


def test_family_lens_weights_can_be_overridden_per_family():
    # winter 0.75 / 0.25: 0.75·50 + 0.25·30 = 45
    cfg = with_lens_weights(CFG, "winter", 0.75, 0.25)
    winter = exposure.compute_exposure("x", exposure.HubInputs(four_years_with_snow_days(10), NRI), cfg)["families"]["winter"]
    assert winter["score"] == pytest.approx(45)
    assert winter["lenses"]["observed"]["weight"] == 0.75


def test_family_without_observed_lens_uses_modeled_only():
    hurricane = exposure.compute_exposure("x", exposure.HubInputs(four_years_with_snow_days(0), NRI), CFG)["families"]["hurricane"]
    assert hurricane["score"] == 60
    assert hurricane["missing_lenses"] == []  # absent by design is not a data gap


def test_missing_nri_is_flagged_and_weights_renormalize():
    # NRI unavailable → winter = observed only = 50; modeled lens reported missing; hurricane has no evidence
    result = exposure.compute_exposure("x", exposure.HubInputs(four_years_with_snow_days(10), None), CFG)
    winter = result["families"]["winter"]
    assert winter["score"] == pytest.approx(50)
    assert winter["missing_lenses"] == ["modeled"]
    assert winter["missing_weight_share"] == pytest.approx(0.5)
    assert result["families"]["hurricane"]["score"] is None
    assert any("FEMA National Risk Index" in gap for gap in result["data_gaps"])
    assert result["score"] == pytest.approx(50)  # overall over the families that have evidence


def test_missing_weather_makes_weather_families_unscorable():
    # No weather at all must not quietly become an NRI-only winter score (coverage rule).
    result = exposure.compute_exposure("x", exposure.HubInputs([], NRI), CFG)
    winter = result["families"]["winter"]
    assert winter["missing_lenses"] == ["observed"] and winter["incomplete"] is True
    assert winter["score"] is None and result["score"] is None


def test_incomplete_weather_blocks_weather_families_and_overall():
    # Weather present but failing the coverage rule: winter (observed 0.5 + NRI 0.5) is NOT renormalized
    # to NRI alone; hurricane (NRI only, by design) is unaffected: 60. Overall needs every family → None.
    result = exposure.compute_exposure(
        "x", exposure.HubInputs(four_years_with_snow_days(10), NRI, weather_complete=False), CFG)
    winter, hurricane = result["families"]["winter"], result["families"]["hurricane"]
    assert winter["score"] is None and winter["incomplete"] is True
    assert winter["lenses"]["observed"]["score"] is None
    assert hurricane["score"] == 60 and not hurricane.get("incomplete")
    assert result["score"] is None
    assert winter["contribution"] is None and hurricane["contribution"] is None
    assert not any("remaining evidence" in gap for gap in result["data_gaps"])


def test_overall_score_needs_every_requested_family_complete():
    breakdown = {"families": {"winter": {"score": None, "incomplete": True}, "hurricane": {"score": 60.0}}}
    assert exposure.overall_score(breakdown, {"winter": 1, "hurricane": 1}) is None
    assert exposure.overall_score(breakdown, {"winter": 1, "hurricane": 1}, ["hurricane"]) == 60


def test_overall_is_equal_weight_mean_and_names_top_family():
    # winter 40, hurricane 60 → overall 50; contribution of winter = 40 × 1/2 = 20; top family hurricane
    result = exposure.compute_exposure("x", exposure.HubInputs(four_years_with_snow_days(10), NRI), CFG)
    assert result["score"] == pytest.approx(50)
    assert result["families"]["winter"]["contribution"] == pytest.approx(20)
    assert result["top_family"] == "hurricane"


def test_rescore_with_weight_override():
    # hurricane ×3: (40·1 + 60·3) / 4 = 55
    breakdown = {"families": {"winter": {"score": 40.0}, "hurricane": {"score": 60.0}}}
    assert exposure.overall_score(breakdown, {"winter": 1, "hurricane": 3}) == pytest.approx(55)


def test_modeled_combine_max():
    cfg = ScoringConfig.model_validate(CFG.model_dump() | {
        "family_weights": {"flood": 1}, "nri_hazard_names": {"IFLD": "Inland", "CFLD": "Coastal"},
        "families": {"flood": {"label": "Flood", "observed": [], "modeled": {"nri": ["IFLD", "CFLD"], "combine": "max"}}},
    })
    nri = {"IFLD": {"pct": 40, "alrb": 1}, "CFLD": {"pct": 70, "alrb": 1}}
    assert exposure.compute_exposure("x", exposure.HubInputs([], nri), cfg)["families"]["flood"]["score"] == 70


# -- ranking robustness -------------------------------------------------------

def test_rank_stability_detects_fragile_first_place():
    # Two families with one lens each. A: x=80, y=40 → 60. B: x=40, y=78 → 59.
    # x weight ×1.25: A 62.2 / B 56.9 (A first)   x ×0.75: A 57.1 / B 61.7 (B first)
    # y weight ×1.25: A 57.8 / B 61.1 (B first)   y ×0.75: A 62.9 / B 56.3 (A first)
    # 2 lens-weight scenarios ×2 cannot change anything (one lens per family) → 8 scenarios, top-1 holds in 6.
    def bd(x, y):
        lens = lambda s: {"score": s, "lenses": {"modeled": {"score": s, "weight": 0.5}}, "missing_lenses": []}
        return {"families": {"x": lens(x), "y": lens(y)}}

    breakdowns = {"A": bd(80, 40), "B": bd(40, 78), "C": bd(10, 10)}
    result = robustness.rank_stability(
        breakdowns, families=["x", "y"], family_weights={"x": 1, "y": 1},
        lenses=("observed", "modeled"), perturbation=0.25, top_k=2,
    )
    assert result["scenarios"] == 8
    assert result["top1_unchanged"] == 6
    assert result["topk_unchanged"] == 8  # {A, B} stays the top-2 set
    assert result["baseline"][:2] == ["A", "B"]


def test_ties_within_margin():
    ranked = [("A", 60.0), ("B", 58.5), ("C", 40.0)]
    assert robustness.near_ties(ranked, margin=3.0, top_k=3) == [("A", "B", 1.5)]
