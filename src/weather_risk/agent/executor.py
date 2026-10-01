"""QueryPlan → ResultBundle. Deterministic Python over stored data; the LLM is not involved.

Every number an answer shows is produced here (docs/DECISIONS.md D1).
"""

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from statistics import median
from typing import Any

from weather_risk.agent.schemas import QueryPlan
from weather_risk.db import Store
from weather_risk.hubs import Hub, hubs_by_id, load_scoring_config
from weather_risk.scoring.exposure import overall_score
from weather_risk.scoring.robustness import near_ties, rank_stability
from weather_risk.scoring.stats import day_stat
from weather_risk.sources.open_meteo import archive_end
from weather_risk.timewindow import WindowError, resolve_window

CAPABILITIES = [
    "Rank hubs (all, by region, or a list) by long-term exposure — overall or per hazard",
    "Compare hubs on chosen hazards (winter, hurricane, flood, severe storm, heat)",
    "Explain why a hub's score is high or low, down to the evidence",
    "Count days meeting a weather condition (snowfall, freezing, heavy rain, heat, wind) in a past window",
    "Near-term outlook: forecast + active NWS alerts for the next 7 days",
]
REGION_NAMES = {"midwest": "Midwest", "south": "South", "west": "West", "northeast": "Northeast"}


@dataclass
class ResultBundle:
    intent: str
    headline: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    points: list[str] = field(default_factory=list)  # deterministic reasoning bullets (template fallback)
    windows: list[dict[str, str | None]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)
    uncertainty: list[tuple[str, str]] = field(default_factory=list)  # (low|medium|high, reason)
    assumptions: list[str] = field(default_factory=list)
    follow_ups: list[str] = field(default_factory=list)

    def for_llm(self) -> dict:
        """Compact view for the explainer prompt (Groq free tier: 8K tokens/min)."""
        return {
            "intent": self.intent,
            "headline": self.headline,
            "rows": self.rows[:10],
            "details": self.details,
            "uncertainty": [reason for _, reason in self.uncertainty],
            "data_gaps": self.data_gaps,
        }


@dataclass
class ExecContext:
    store: Store
    today: date
    cfg: dict = field(default_factory=load_scoring_config)
    hubs: dict[str, Hub] = field(default_factory=hubs_by_id)

    def exposure(self) -> dict[str, dict]:
        return {h: s["breakdown"] for h, s in self.store.latest_snapshots("exposure").items()}

    def near_term(self) -> dict[str, dict]:
        return {h: s["breakdown"] for h, s in self.store.latest_snapshots("near_term").items()}

    def name(self, hub_id: str) -> str:
        return self.hubs[hub_id].name

    def family_label(self, family: str) -> str:
        return self.cfg["families"][family]["label"]

    def short_label(self, family: str) -> str:
        return self.family_label(family).split(" (")[0]


def r1(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


def ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _select_hubs(plan: QueryPlan, ctx: ExecContext) -> tuple[list[str], str]:
    if plan.hubs:
        return list(plan.hubs), "the selected hubs"
    if plan.region:
        ids = [h for h, hub in ctx.hubs.items() if hub.region == plan.region]
        return ids, f"{REGION_NAMES[plan.region]} hubs (US Census region)"
    return list(ctx.hubs), f"all {len(ctx.hubs)} hubs"


def _families(plan: QueryPlan, ctx: ExecContext) -> list[str]:
    return list(plan.hazards) if plan.hazards else list(ctx.cfg["families"])


def _weights(plan: QueryPlan, ctx: ExecContext) -> tuple[dict[str, float], list[str]]:
    weights = dict(ctx.cfg["family_weights"])
    notes = []
    if plan.weight_overrides:
        for family, value in plan.weight_overrides.model_dump().items():
            if value is not None:
                weights[family] = value
                notes.append(f"{family} ×{value:g}")
    return weights, notes


def _hazard_phrase(families: list[str], ctx: ExecContext) -> str:
    if len(families) == len(ctx.cfg["families"]):
        return "overall weather exposure"
    return " + ".join(ctx.short_label(f).lower() for f in families) + " exposure"


def _format_indicator(ind: dict) -> str:
    if ind["unit"] == "percentile":
        value = f"{ordinal(ind['raw'])} percentile nationally"
        if ind.get("applicable") is False:
            value = "not applicable (0)"
    elif ind["unit"] == "declarations":
        examples = f" (e.g. {', '.join(ind['examples'][:2])})" if ind.get("examples") else ""
        value = f"{ind['raw']} declarations{examples}"
    else:
        value = f"{ind['raw']:.1f} {ind['unit']}"
    return f"{ind['label']}: {value} → {ind['score']:.0f} pts"


def _drivers(bd: dict, families: list[str], n: int = 3) -> list[str]:
    indicators = [
        ind
        for f in families
        for lens in bd["families"][f]["lenses"].values()
        for ind in lens.get("indicators", [])
    ]
    return [_format_indicator(i) for i in sorted(indicators, key=lambda i: -i["score"])[:n]]


def _family_detail(bd: dict, family: str) -> dict:
    fam = bd["families"][family]
    return {
        "score": r1(fam["score"]),
        "lenses": {
            name: {
                "score": r1(lens["score"]),
                "weight": lens.get("weight"),
                "evidence": [_format_indicator(i) for i in lens.get("indicators", [])],
            }
            for name, lens in fam["lenses"].items()
        },
        "missing_evidence": fam["missing_lenses"],
    }


def _evidence_note(families: list[str], ctx: ExecContext) -> str:
    parts = []
    for f in families:
        fam = ctx.cfg["families"][f]
        weights = fam.get("lens_weights", ctx.cfg["lens_weights"])
        present = {"observed": bool(fam.get("observed")), "modeled": bool(fam.get("modeled")),
                   "history": bool(fam.get("history"))}
        parts.append(ctx.short_label(f).lower() + " " + "/".join(
            f"{weights[lens]:g}" if present[lens] else "–" for lens in ("observed", "modeled", "history")))
    return ("Evidence weights per hazard (observed weather / FEMA NRI building-loss percentile / FEMA disaster "
            "declarations; renormalized over the evidence available): " + "; ".join(parts) + ".")


def _anchor_note(families: list[str], ctx: ExecContext) -> str:
    example = next((ind for f in families for ind in ctx.cfg["families"][f].get("observed") or []), None)
    text = "Indicators are scored on fixed anchors from config/scoring.yaml"
    if example:
        text += f" (e.g. {example['label'].lower()}: {example['full_score_at']:g} a year = 100)"
    return text + ", so a hub's score does not depend on which other hubs are listed."


def _exposure_context(ctx: ExecContext, exposure: dict[str, dict], families: list[str]) -> tuple[list, list, list, list]:
    """Windows, sources, assumptions and data gaps shared by every long-term answer."""
    any_bd = next(iter(exposure.values()))
    meta = any_bd.get("meta", {})
    obs = meta.get("observed_window", {})
    windows = [
        {"label": "Observed weather climatology (Open-Meteo ERA5)", "start": obs.get("start"), "end": obs.get("end")},
        {"label": f"FEMA National Risk Index ({meta.get('nri_version') or 'version unknown'} release)", "start": None, "end": None},
        {"label": "FEMA major-disaster declarations", "start": meta.get("declarations_since"), "end": ctx.today.isoformat()},
    ]
    uses_observed = any(ctx.cfg["families"][f].get("observed") for f in families)
    sources = (["open_meteo_archive"] if uses_observed else []) + ["fema_nri", "openfema"]
    assumptions = [
        "Hub locations are assumed: each hub sits at its metro's main airport logistics zone; "
        "county-level hazard data stands in for the site.",
        _evidence_note(families, ctx),
        _anchor_note(families, ctx),
    ]
    if len(families) == len(ctx.cfg["families"]):
        assumptions.append("Overall exposure is the equal-weight average of 5 hazard families "
                           "(winter, hurricane, flood, severe storm, heat).")
    gaps = sorted({g for bd in exposure.values() for g in bd.get("data_gaps", [])})
    return windows, sources, assumptions, gaps


def _rank(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    hub_ids, scope = _select_hubs(plan, ctx)
    families = _families(plan, ctx)
    weights, weight_notes = _weights(plan, ctx)
    exposure = {h: bd for h, bd in ctx.exposure().items() if h in hub_ids}
    if not exposure:
        return ResultBundle("rank", "No exposure scores are stored yet — run the ingest first.")
    scores = {h: overall_score(bd, weights, families) for h, bd in exposure.items()}
    scores = {h: s for h, s in scores.items() if s is not None}
    order = sorted(scores, key=lambda h: (-scores[h], h))
    k = min(plan.top_k or len(order), len(order))

    rows = []
    for i, h in enumerate(order[:k], 1):
        bd = exposure[h]
        row: dict[str, Any] = {"rank": i, "hub": ctx.name(h), "score": r1(scores[h])}
        if len(families) > 1:
            row |= {ctx.short_label(f): r1(bd["families"][f]["score"]) for f in families}
        row["top hazard"] = ctx.short_label(max(families, key=lambda f: bd["families"][f]["score"] or 0))
        rows.append(row)

    check_k = min(k, 3)
    stability = rank_stability(
        {h: exposure[h] for h in order}, families=families, family_weights=weights,
        lenses=tuple(ctx.cfg["lens_weights"]), perturbation=ctx.cfg["robustness"]["perturbation"], top_k=check_k,
    )
    stability["baseline"] = [ctx.name(h) for h in stability["baseline"][:k]]
    stability["flips"] = [f"{f['scenario']}: {ctx.name(f['first'])} ranks first" for f in stability["flips"]]
    ties = near_ties([(ctx.name(h), scores[h]) for h in order], ctx.cfg["robustness"]["tie_margin"], check_k)

    uncertainty = [("medium", f"{a} and {b} are only {gap} points apart — treat them as tied.") for a, b, gap in ties]
    n = stability["scenarios"]
    if stability["top1_unchanged"] < n:
        uncertainty.append(("medium", f"First place changes in {n - stability['top1_unchanged']} of {n} weight "
                                      f"scenarios (each weight ±25 %), e.g. {stability['flips'][0]}."))
    else:
        uncertainty.append(("low", f"First place holds in all {n} weight scenarios (each weight moved ±25 %)."))
    if check_k > 1:
        uncertainty.append(("low" if stability["topk_unchanged"] == n else "medium",
                            f"The top-{check_k} set is unchanged in {stability['topk_unchanged']} of {n} weight scenarios."))

    windows, sources, assumptions, gaps = _exposure_context(ctx, exposure, families)
    if weight_notes:
        assumptions.append(f"Hazard weights changed for this answer: {', '.join(weight_notes)} (others ×1).")
    phrase = _hazard_phrase(families, ctx)
    top = order[0]
    runners = ", ".join(f"{ctx.name(h)} ({r1(scores[h])})" for h in order[1:3])
    headline = f"{ctx.name(top)} has the highest {phrase} among {scope} (score {r1(scores[top])}/100)"
    headline += f", followed by {runners}." if runners else "."
    drivers = {ctx.name(h): _drivers(exposure[h], families) for h in order[:check_k]}
    points = [f"{name}: {'; '.join(d[:2])}" for name, d in drivers.items()] + [reason for _, reason in uncertainty]
    return ResultBundle(
        intent="rank", headline=headline, rows=rows,
        details={"scope": scope, "hazards": [ctx.family_label(f) for f in families],
                 "weights": weights if weight_notes else "equal", "drivers": drivers, "robustness": stability},
        points=points, windows=windows, sources=sources, data_gaps=gaps, uncertainty=uncertainty,
        assumptions=assumptions,
        follow_ups=[f"Why is {ctx.name(top)} ranked first?", f"Compare {ctx.name(top)} and {ctx.name(order[1])}"]
        if len(order) > 1 else [],
    )


def _compare(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    hub_ids = list(plan.hubs or [])
    families = _families(plan, ctx)
    weights, weight_notes = _weights(plan, ctx)
    all_exposure = ctx.exposure()
    exposure = {h: all_exposure[h] for h in hub_ids if h in all_exposure}
    combined = {h: overall_score(bd, weights, families) for h, bd in exposure.items()}
    rows = []
    for h in hub_ids:
        bd = exposure[h]
        row: dict[str, Any] = {"hub": ctx.name(h)}
        row |= {ctx.short_label(f): r1(bd["families"][f]["score"]) for f in families}
        if len(families) > 1:
            row["combined"] = r1(combined[h])
        rows.append(row)

    differences, family_ties = [], []
    for f in families:
        ranked = sorted(hub_ids, key=lambda h: -(exposure[h]["families"][f]["score"] or 0))
        lead, second = ranked[0], ranked[1]
        gap = (exposure[lead]["families"][f]["score"] or 0) - (exposure[second]["families"][f]["score"] or 0)
        differences.append(f"{ctx.family_label(f)}: {ctx.name(lead)} higher by {gap:.1f} points"
                           if gap > 0 else f"{ctx.family_label(f)}: {ctx.name(lead)} and {ctx.name(second)} are level")
        if len(families) > 1 and gap < ctx.cfg["robustness"]["tie_margin"]:
            family_ties.append(f"On {ctx.short_label(f).lower()} alone, {ctx.name(lead)} and {ctx.name(second)} are "
                               f"within {gap:.1f} points — effectively tied.")
    order = sorted(hub_ids, key=lambda h: -(combined[h] or 0))
    gap = (combined[order[0]] or 0) - (combined[order[1]] or 0)
    phrase = _hazard_phrase(families, ctx)
    others = " and ".join(f"{ctx.name(h)} ({r1(combined[h])})" for h in order[1:])
    headline = f"{ctx.name(order[0])} has the higher {phrase} ({r1(combined[order[0]])}/100) vs {others}."
    uncertainty = [("low", t) for t in family_ties]
    if gap < ctx.cfg["robustness"]["tie_margin"]:
        uncertainty.append(("medium", f"{ctx.name(order[0])} and {ctx.name(order[1])} are only {gap:.1f} points apart "
                                      "— treat them as similarly exposed."))
    windows, sources, assumptions, gaps = _exposure_context(ctx, exposure, families)
    if weight_notes:
        assumptions.append(f"Hazard weights changed for this answer: {', '.join(weight_notes)} (others ×1).")
    detail = {ctx.name(h): {ctx.short_label(f): _family_detail(exposure[h], f) for f in families} for h in hub_ids}
    return ResultBundle(
        intent="compare", headline=headline, rows=rows,
        details={"hazards": [ctx.family_label(f) for f in families], "differences": differences, "hubs": detail},
        points=differences + [f"{ctx.name(h)}: {'; '.join(_drivers(exposure[h], families, 2))}" for h in order],
        windows=windows, sources=sources, data_gaps=gaps, uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=[f"Why is {ctx.name(order[0])} higher?", f"Add Dallas to the comparison"
                    if "dallas" not in hub_ids else "Rank all hubs on these hazards"],
    )


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def _stat(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    metric = ctx.cfg["stats"][plan.metric]
    hub_ids = list(plan.hubs or [])
    ranges = [r for h in hub_ids if (r := ctx.store.daily_range(h))]
    if not ranges:
        return ResultBundle("stat", "No observed daily data is stored for these hubs yet — run the ingest first.")
    data_start = max(r[0] for r in ranges)
    data_end = min(min(r[1] for r in ranges), archive_end(ctx.today))
    try:
        window = resolve_window(plan.time_preset, today=ctx.today, data_end=data_end, data_start=data_start,
                                year=plan.year, start=plan.start_date, end=plan.end_date)
    except WindowError as exc:
        return ResultBundle("stat", f"I can't compute that window: {exc}.",
                            details={"available_data": f"{data_start} to {data_end}"},
                            uncertainty=[("high", str(exc))])
    rows, sensitivity, grid_notes, uncertainty = [], {}, [], []
    for h in hub_ids:
        s = day_stat(ctx.store.daily(h, window.start, window.end), window.start, window.end, metric)
        rows.append({"hub": ctx.name(h), "days meeting threshold": s["count"], "days with data": s["days_with_data"],
                     "% of days": s["pct"], "coverage %": s["coverage_pct"]})
        sensitivity[ctx.name(h)] = s["sensitivity"]
        if s["coverage_pct"] < 95:
            uncertainty.append(("medium", f"{ctx.name(h)}: only {s['coverage_pct']}% of days in the window have data."))
        pcts = [x["pct"] for x in s["sensitivity"] if x["pct"]]
        if len(pcts) >= 2 and max(pcts) >= 2 * min(pcts):
            lo = min(s["sensitivity"], key=lambda x: x["pct"] or 0)
            hi = max(s["sensitivity"], key=lambda x: x["pct"] or 0)
            uncertainty.append(("medium", f"{ctx.name(h)}: the threshold matters — {hi['label']} gives {hi['pct']}% "
                                          f"of days, {lo['label']} gives {lo['pct']}%."))
        grid = ctx.store.cache_get(f"grid:{h}")
        if grid:
            cell, hub = grid[1], ctx.hubs[h]
            distance = _km(hub.lat, hub.lon, cell["latitude"], cell["longitude"])
            grid_notes.append(f"{ctx.name(h)}: values are for the reanalysis grid cell at {cell['latitude']:.2f}, "
                              f"{cell['longitude']:.2f} ({distance:.0f} km from the hub point).")
    first = rows[0]
    if len(rows) == 1:
        headline = (f"{first['% of days']}% of days at {first['hub']} in {window.label} had {metric['short']} "
                    f"({first['days meeting threshold']} of {first['days with data']} days).")
    else:
        parts = ", ".join(f"{r['hub']} {r['% of days']}% ({r['days meeting threshold']} days)" for r in rows)
        headline = f"Share of days in {window.label} with {metric['short']}: {parts}."
    assumptions = [
        f"Definition: {metric['label']}.",
        "Open-Meteo ERA5 reanalysis is modeled, not station-measured. It smooths daily extremes and can differ "
        "from airport station records by several days a year, so treat counts as estimates.",
        *grid_notes,
        *window.notes,
    ]
    points = [f"{name}: " + "; ".join(f"{x['label']} → {x['pct']}%" for x in rows_) for name, rows_ in sensitivity.items()]
    return ResultBundle(
        intent="stat", headline=headline, rows=rows,
        details={"metric": metric["label"], "window": window.label, "sensitivity": sensitivity},
        points=points, windows=[{"label": window.label, "start": window.start.isoformat(), "end": window.end.isoformat()}],
        sources=["open_meteo_archive"], uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=["And the year before?", "How does that compare to Chicago?"],
    )


def _explain(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    hub = plan.hubs[0]
    if plan.horizon == "next_7_days":
        return _outlook(plan, ctx)
    exposure = ctx.exposure()
    if hub not in exposure:
        return ResultBundle("explain", f"No exposure score is stored for {ctx.name(hub)} yet.")
    bd = exposure[hub]
    families = _families(plan, ctx)
    all_families = list(ctx.cfg["families"])
    n = len(exposure)
    overall_order = sorted(exposure, key=lambda h: -(exposure[h]["score"] or 0))

    def family_rank(f: str) -> int:
        return sorted(exposure, key=lambda h: -(exposure[h]["families"][f]["score"] or 0)).index(hub) + 1

    shown = sorted(families, key=lambda f: -(bd["families"][f]["contribution"] or 0))
    rows = [
        {
            "hazard": ctx.family_label(f),
            "score": r1(bd["families"][f]["score"]),
            "contribution to overall": r1(bd["families"][f]["contribution"]),
            "rank among hubs": f"{family_rank(f)} of {n}",
            "median across hubs": r1(median(exposure[h]["families"][f]["score"] or 0 for h in exposure)),
        }
        for f in shown
    ]
    top = shown[0]
    focus = len(families) < len(all_families)
    if focus:
        combined = overall_score(bd, ctx.cfg["family_weights"], families)
        headline = (f"{ctx.name(hub)} scores {r1(combined)}/100 on {_hazard_phrase(families, ctx)}; "
                    f"the biggest driver is {ctx.short_label(top).lower()} ({r1(bd['families'][top]['score'])}/100, "
                    f"rank {family_rank(top)} of {n}).")
    else:
        headline = (f"{ctx.name(hub)} has an overall exposure score of {r1(bd['score'])}/100 "
                    f"(rank {overall_order.index(hub) + 1} of {n}). The largest contributor is "
                    f"{ctx.short_label(top)} at {r1(bd['families'][top]['score'])}/100 (rank {family_rank(top)} of {n}).")
    windows, sources, assumptions, _ = _exposure_context(ctx, exposure, families)
    gaps = bd.get("data_gaps", [])
    uncertainty = [("medium", g) for g in gaps]
    return ResultBundle(
        intent="explain", headline=headline, rows=rows,
        details={
            "hub": f"{ctx.name(hub)} ({bd.get('meta', {}).get('county', ctx.hubs[hub].county)})",
            "overall_score": r1(bd["score"]),
            "overall_rank": f"{overall_order.index(hub) + 1} of {n}",
            "hazards": {ctx.short_label(f): _family_detail(bd, f) for f in shown},
            "top_drivers": _drivers(bd, families, 4),
        },
        points=_drivers(bd, families, 4), windows=windows, sources=sources, data_gaps=gaps,
        uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=[f"Which evidence drives {ctx.name(hub)}'s {ctx.short_label(top).lower()} score?",
                    f"Compare {ctx.name(hub)} with {ctx.name(overall_order[0] if overall_order[0] != hub else overall_order[1])}"],
    )


def _outlook(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    hub_ids, scope = _select_hubs(plan, ctx)
    near = {h: nt for h, nt in ctx.near_term().items() if h in hub_ids}
    if not near:
        return ResultBundle("outlook", "No near-term scores are stored yet — run the ingest first.")
    order = sorted(near, key=lambda h: (-near[h]["score"], h))
    rows = []
    for h in order:
        nt = near[h]
        top = nt.get("top_family")
        driver = nt["families"][top]["drivers"][0] if top and nt["families"][top]["drivers"] else "—"
        rows.append({"hub": ctx.name(h), "score": r1(nt["score"]), "band": nt["band"],
                     "top hazard": ctx.short_label(top) if top else "—", "main driver": driver})
    elevated = [r for r in rows if r["band"] in ("Elevated", "Severe")]
    if elevated:
        headline = (f"{len(elevated)} of {len(rows)} hubs are Elevated or Severe over the next 7 days: "
                    + ", ".join(f"{r['hub']} ({r['band']}, {r['score']}, {r['top hazard'].lower()})" for r in elevated) + ".")
    else:
        headline = (f"No hub among {scope} is above Guarded over the next 7 days; the highest is "
                    f"{rows[0]['hub']} ({rows[0]['band']}, score {rows[0]['score']}).")
    meta = near[order[0]].get("meta", {})
    start = meta.get("forecast_start")
    end = (date.fromisoformat(start) + timedelta(days=6)).isoformat() if start else None
    windows = [{"label": "7-day forecast", "start": start, "end": end},
               {"label": "NWS alerts active when fetched", "start": meta.get("alerts_fetched_at"), "end": None}]
    uncertainty = [("medium", "Forecast skill drops after about 3 days; treat days 4–7 as indicative.")]
    unmapped = sorted({a for h in order for a in near[h].get("unmapped_alerts", [])})
    if unmapped:
        uncertainty.append(("low", f"Active NWS alerts not mapped to a hazard family: {', '.join(unmapped)}."))
    gaps = sorted({g for h in order for g in near[h].get("data_gaps", [])})
    details = {ctx.name(h): {ctx.short_label(f): v["drivers"] for f, v in near[h]["families"].items() if v["drivers"]}
               for h in order[:5]}
    return ResultBundle(
        intent="outlook", headline=headline, rows=rows, details={"scope": scope, "drivers": details},
        points=[f"{r['hub']}: {r['main driver']}" for r in rows[:3] if r["main driver"] != "—"]
        or ["No forecast day or active alert crosses a scoring threshold."],
        windows=windows, sources=["open_meteo_forecast", "nws_alerts"], data_gaps=gaps, uncertainty=uncertainty,
        assumptions=["Near-term score = the worst of the 7-day forecast and active NWS alerts, per hazard; "
                     "the overall score is the worst hazard. Hurricane risk appears only through NWS alerts.",
                     "Bands: Low < 25, Guarded 25–49, Elevated 50–74, Severe ≥ 75."],
        follow_ups=[f"Why is {rows[0]['hub']}'s near-term risk {rows[0]['band']}?", "Which hubs are most exposed long term?"],
    )


def _no_compute(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    text = plan.clarification_question if plan.intent == "clarify" else plan.out_of_scope_reason
    return ResultBundle(
        intent=plan.intent, headline=text or "",
        details={"what_i_can_answer": CAPABILITIES, "hubs": [h.name for h in ctx.hubs.values()]},
        points=["I can answer: " + "; ".join(c[0].lower() + c[1:] for c in CAPABILITIES[:3]) + "."],
        follow_ups=["Which hubs in the Midwest are most exposed to winter disruption?",
                    "Compare Miami and Houston on hurricane and flood exposure"],
    )


HANDLERS = {"rank": _rank, "compare": _compare, "stat": _stat, "explain": _explain, "outlook": _outlook,
            "clarify": _no_compute, "out_of_scope": _no_compute}


def execute(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    return HANDLERS[plan.intent](plan, ctx)
