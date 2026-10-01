"""Long-term Exposure Score (for investment decisions) — docs/DECISIONS.md D10–D12, D20, D24.

family score = weighted mean of its evidence lenses (observed weather, modeled NRI loss),
               renormalized over the lenses that exist by design or loaded successfully;
overall      = weighted mean of family scores (equal weights by default).
Incomplete weather (config coverage rule, D24) is not renormalized away: a family that uses
observed weather gets no score, and neither does the overall, so the hub is not ranked on them.
Every intermediate value is kept in the returned breakdown so "why" answers can cite it.
"""

from dataclasses import dataclass
from statistics import mean

from weather_risk.config import Family, ScoringConfig
from weather_risk.scoring.indicators import anchor_score, days_per_year
from weather_risk.sources.open_meteo import DailyRow

LENS_SOURCES = {
    "observed": "Open-Meteo archive (ERA5 reanalysis)",
    "modeled": "FEMA National Risk Index",
}


@dataclass
class HubInputs:
    daily: list[DailyRow]  # empty = no weather data available
    nri: dict[str, dict] | None  # hazard code → {"alrb", "pct"}; None = source unavailable
    weather_complete: bool = True  # False = the record fails the coverage rule (analysis.py decides)


def _observed_lens(family: Family, daily: list[DailyRow], complete: bool) -> dict | None:
    if not family.observed:
        return None  # lens not used for this family (by design)
    if not complete or not daily:
        return {"score": None, "indicators": [], "incomplete": True, "source": LENS_SOURCES["observed"]}
    indicators = []
    for spec in family.observed:
        rate, _ = days_per_year(daily, spec.variable, spec.op, spec.threshold)
        if rate is None:
            return {"score": None, "indicators": [], "source": LENS_SOURCES["observed"]}
        indicators.append({"id": spec.id, "label": spec.label, "raw": rate, "unit": "days/yr",
                           "full_score_at": spec.full_score_at, "score": anchor_score(rate, spec.full_score_at)})
    return {"score": mean(i["score"] for i in indicators), "indicators": indicators, "source": LENS_SOURCES["observed"]}


def _modeled_lens(family: Family, nri: dict[str, dict] | None, names: dict[str, str]) -> dict | None:
    if family.modeled is None:
        return None
    indicators = [
        {"id": f"nri_{code}", "label": f"{names[code]}: building loss rate, national percentile",
         "raw": nri[code]["pct"], "unit": "percentile", "applicable": nri[code]["alrb"] is not None,
         "score": nri[code]["pct"]}
        for code in family.modeled.nri
        if nri and code in nri
    ]
    if not indicators:
        return {"score": None, "indicators": [], "source": LENS_SOURCES["modeled"]}
    combine = max if family.modeled.combine == "max" else mean
    return {"score": combine(i["score"] for i in indicators), "combine": family.modeled.combine,
            "indicators": indicators, "source": LENS_SOURCES["modeled"]}


def family_score_from_lenses(lenses: dict[str, dict], factors: dict[str, float] | None = None) -> float | None:
    """Weighted mean of available lens scores. `factors` scale lens weights (robustness checks)."""
    factors = factors or {}
    available = [(lens["weight"] * factors.get(name, 1.0), lens["score"])
                 for name, lens in lenses.items() if lens.get("score") is not None]
    if not available or sum(w for w, _ in available) <= 0:
        return None
    return sum(w * s for w, s in available) / sum(w for w, _ in available)


def score_family(name: str, inputs: HubInputs, cfg: ScoringConfig) -> dict:
    family = cfg.families[name]
    weights = cfg.lens_weights_for(name)
    built = {
        "observed": _observed_lens(family, inputs.daily, inputs.weather_complete),
        "modeled": _modeled_lens(family, inputs.nri, cfg.nri_hazard_names),
    }
    lenses = {lens: data | {"weight": getattr(weights, lens)} for lens, data in built.items() if data is not None}
    missing = [lens for lens, data in lenses.items() if data["score"] is None]
    total_w = sum(data["weight"] for data in lenses.values())
    incomplete = bool(lenses.get("observed", {}).get("incomplete"))
    return {
        "label": family.label,
        "score": None if incomplete else family_score_from_lenses(lenses),
        "incomplete": incomplete,
        "lenses": lenses,
        "missing_lenses": missing,
        "missing_weight_share": sum(lenses[m]["weight"] for m in missing) / total_w if total_w else 0.0,
    }


def overall_score(breakdown: dict, family_weights: dict[str, float], families: list[str] | None = None) -> float | None:
    """Weighted mean of family scores. None if any requested family has incomplete weather data;
    a family with no evidence at all (e.g. NRI down) is skipped and reported as a data gap."""
    families = families or list(family_weights)
    if any(breakdown["families"].get(f, {}).get("incomplete") for f in families):
        return None
    pairs = [(family_weights.get(f, 0.0), breakdown["families"][f]["score"])
             for f in families if breakdown["families"].get(f, {}).get("score") is not None]
    total = sum(w for w, _ in pairs)
    if not pairs or total <= 0:
        return None
    return sum(w * s for w, s in pairs) / total


def compute_exposure(hub_id: str, inputs: HubInputs, cfg: ScoringConfig,
                     family_weights: dict[str, float] | None = None) -> dict:
    weights = family_weights or cfg.family_weights
    families = {name: score_family(name, inputs, cfg) for name in cfg.families}
    breakdown = {"hub_id": hub_id, "families": families, "family_weights": weights}
    breakdown["score"] = overall_score(breakdown, weights)
    scored = {f: v for f, v in families.items() if v["score"] is not None}
    total_w = sum(weights.get(f, 0.0) for f in scored)
    for f, fam in families.items():
        fam["contribution"] = (fam["score"] * weights.get(f, 0.0) / total_w
                               if breakdown["score"] is not None and f in scored and total_w else None)
    breakdown["top_family"] = max(scored, key=lambda f: scored[f]["score"]) if scored else None
    gaps = [f"{LENS_SOURCES[lens]} unavailable — {fam['label']} score uses the remaining evidence."
            for fam in families.values() if not fam["incomplete"] for lens in fam["missing_lenses"]]
    breakdown["data_gaps"] = list(dict.fromkeys(gaps))
    return breakdown
