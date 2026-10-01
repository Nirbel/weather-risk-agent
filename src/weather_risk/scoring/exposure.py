"""Long-term Exposure Score (for investment decisions) — docs/DECISIONS.md D10–D12.

family score = weighted mean of its evidence lenses (observed / modeled / history),
               renormalized over lenses that exist by design or loaded successfully;
overall      = weighted mean of family scores (equal weights by default).
Every intermediate value is kept in the returned breakdown so "why" answers can cite it.
"""

from dataclasses import dataclass
from datetime import date
from statistics import mean

from weather_risk.scoring.indicators import anchor_score, days_per_year
from weather_risk.sources.open_meteo import DailyRow

LENS_SOURCES = {
    "observed": "Open-Meteo archive (ERA5 reanalysis)",
    "modeled": "FEMA National Risk Index",
    "history": "OpenFEMA disaster declarations",
}


@dataclass
class HubInputs:
    daily: list[DailyRow]
    nri: dict[str, dict] | None  # hazard code → {"alrb", "pct"}; None = source unavailable
    declarations: list[dict] | None  # DR declarations; None = source unavailable


def _observed_lens(specs: list[dict], daily: list[DailyRow]) -> dict | None:
    if not specs:
        return None  # lens not used for this family (by design)
    indicators = []
    for spec in specs:
        rate, _ = days_per_year(daily, spec["variable"], spec["op"], spec["threshold"])
        if rate is None:
            return {"score": None, "indicators": [], "source": LENS_SOURCES["observed"]}
        indicators.append({
            "id": spec["id"],
            "label": spec["label"],
            "raw": rate,
            "unit": "days/yr",
            "full_score_at": spec["full_score_at"],
            "score": anchor_score(rate, spec["full_score_at"]),
        })
    return {"score": mean(i["score"] for i in indicators), "indicators": indicators, "source": LENS_SOURCES["observed"]}


def _modeled_lens(spec: dict | None, nri: dict[str, dict] | None, names: dict[str, str]) -> dict | None:
    if not spec:
        return None
    if nri is None:
        return {"score": None, "indicators": [], "source": LENS_SOURCES["modeled"]}
    indicators = [
        {
            "id": f"nri_{hz}",
            "label": f"{names.get(hz, hz)}: building loss rate, national percentile",
            "raw": nri[hz]["pct"],
            "unit": "percentile",
            "applicable": nri[hz]["alrb"] is not None,
            "score": nri[hz]["pct"],
        }
        for hz in spec["nri"]
        if hz in nri
    ]
    if not indicators:
        return {"score": None, "indicators": [], "source": LENS_SOURCES["modeled"]}
    combine = max if spec.get("combine") == "max" else mean
    return {
        "score": combine(i["score"] for i in indicators),
        "combine": spec.get("combine", "mean"),
        "indicators": indicators,
        "source": LENS_SOURCES["modeled"],
    }


def _history_lens(spec: dict | None, declarations: list[dict] | None, since: date) -> dict | None:
    if not spec:
        return None
    if declarations is None:
        return {"score": None, "indicators": [], "source": LENS_SOURCES["history"]}
    types = spec["incident_types"]
    matching = [
        d for d in declarations
        if d["incident_type"] in types and date.fromisoformat(str(d["declaration_date"])[:10]) >= since
    ]
    count = len(matching)
    indicator = {
        "id": "fema_major_disasters",
        "label": f"FEMA major-disaster declarations since {since.year} ({', '.join(types)})",
        "raw": count,
        "unit": "declarations",
        "full_score_at": spec["full_score_at"],
        "score": anchor_score(count, spec["full_score_at"]),
        "examples": [f"{str(d['declaration_date'])[:4]} {d['title'].title()}" for d in matching[:3]],
    }
    return {"score": indicator["score"], "indicators": [indicator], "source": LENS_SOURCES["history"]}


def family_score_from_lenses(lenses: dict[str, dict], factors: dict[str, float] | None = None) -> float | None:
    """Weighted mean of available lens scores. `factors` scale lens weights (robustness checks)."""
    factors = factors or {}
    available = [(lens["weight"] * factors.get(name, 1.0), lens["score"])
                 for name, lens in lenses.items() if lens.get("score") is not None]
    if not available:
        return None
    return sum(w * s for w, s in available) / sum(w for w, _ in available)


def score_family(fam_cfg: dict, inputs: HubInputs, cfg: dict) -> dict:
    lens_weights = fam_cfg.get("lens_weights", cfg["lens_weights"])
    built = {
        "observed": _observed_lens(fam_cfg.get("observed") or [], inputs.daily),
        "modeled": _modeled_lens(fam_cfg.get("modeled"), inputs.nri, cfg.get("nri_hazard_names", {})),
        "history": _history_lens(fam_cfg.get("history"), inputs.declarations, cfg["declarations_since"]),
    }
    lenses = {name: lens | {"weight": lens_weights[name]} for name, lens in built.items() if lens is not None}
    missing = [name for name, lens in lenses.items() if lens["score"] is None]
    total_w = sum(lens["weight"] for lens in lenses.values())
    missing_w = sum(lenses[name]["weight"] for name in missing)
    return {
        "label": fam_cfg["label"],
        "score": family_score_from_lenses(lenses),
        "lenses": lenses,
        "missing_lenses": missing,
        "missing_weight_share": missing_w / total_w if total_w else 0.0,
    }


def overall_score(breakdown: dict, family_weights: dict[str, float], families: list[str] | None = None) -> float | None:
    """Weighted mean of family scores; families without a score are skipped."""
    families = families or list(family_weights)
    pairs = [
        (family_weights.get(f, 0.0), breakdown["families"][f]["score"])
        for f in families
        if breakdown["families"].get(f, {}).get("score") is not None
    ]
    total = sum(w for w, _ in pairs)
    if not pairs or total <= 0:
        return None
    return sum(w * s for w, s in pairs) / total


def compute_exposure(hub_id: str, inputs: HubInputs, cfg: dict, family_weights: dict[str, float] | None = None) -> dict:
    weights = family_weights or cfg["family_weights"]
    families = {name: score_family(fam_cfg, inputs, cfg) for name, fam_cfg in cfg["families"].items()}
    breakdown = {"hub_id": hub_id, "families": families, "family_weights": weights}
    breakdown["score"] = overall_score(breakdown, weights)
    scored = {f: v for f, v in families.items() if v["score"] is not None}
    total_w = sum(weights.get(f, 0.0) for f in scored)
    for f, fam in families.items():
        fam["contribution"] = fam["score"] * weights.get(f, 0.0) / total_w if f in scored and total_w else None
    breakdown["top_family"] = max(scored, key=lambda f: scored[f]["score"]) if scored else None
    gaps = []
    for fam in families.values():
        for lens in fam["missing_lenses"]:
            gaps.append(f"{LENS_SOURCES[lens]} unavailable — {fam['label']} score uses the remaining evidence.")
    breakdown["data_gaps"] = list(dict.fromkeys(gaps))
    return breakdown
