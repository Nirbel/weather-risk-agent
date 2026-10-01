"""QueryPlan → ResultBundle. Deterministic Python over on-demand, cached data; no LLM.

Every number an answer shows is produced here (docs/DECISIONS.md D1). The Analytics API calls the
same functions, so charts and chat always agree.
"""

import math
from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Any

from weather_risk.agent.schemas import QueryPlan
from weather_risk.scoring.coverage import weather_coverage
from weather_risk.scoring.exposure import overall_score
from weather_risk.scoring.robustness import near_ties, rank_stability
from weather_risk.scoring.stats import day_stat
from weather_risk.timewindow import WindowError, resolve_window

ARCHIVE_START = date(1940, 1, 1)  # Open-Meteo ERA5 archive starts in 1940
CAPABILITIES = [
    "Rank hubs (all, by region, or a list) by long-term exposure — overall or per hazard",
    "Compare hubs on chosen hazards (winter, hurricane, flood, severe storm, heat)",
    "Explain why a hub's score is high or low, down to the evidence",
    "Count past days meeting a weather condition (snowfall, freezing, heavy rain, heat, wind)",
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
        return {"intent": self.intent, "headline": self.headline, "rows": self.rows[:10], "details": self.details,
                "uncertainty": [reason for _, reason in self.uncertainty], "data_gaps": self.data_gaps}


@dataclass
class ExecContext:
    analyzer: Any  # weather_risk.analysis.Analyzer (or a test double with the same interface)

    @property
    def cfg(self):
        return self.analyzer.cfg

    @property
    def hubs(self):
        return self.analyzer.hubs

    def name(self, hub_id: str) -> str:
        return self.hubs[hub_id].name

    def family_label(self, family: str) -> str:
        return self.cfg.families[family].label

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
        return [h for h, hub in ctx.hubs.items() if hub.region == plan.region], \
            f"{REGION_NAMES[plan.region]} hubs (US Census region)"
    return list(ctx.hubs), f"all {len(ctx.hubs)} hubs"


def _families(plan: QueryPlan, ctx: ExecContext) -> list[str]:
    return list(plan.hazards) if plan.hazards else list(ctx.cfg.families)


def _weights(plan: QueryPlan, ctx: ExecContext) -> tuple[dict[str, float], list[str]]:
    weights = dict(ctx.cfg.family_weights)
    notes = []
    if plan.weight_overrides:
        for family, value in plan.weight_overrides.model_dump().items():
            if value is not None:
                weights[family] = value
                notes.append(f"{family} ×{value:g}")
    return weights, notes


def _hazard_phrase(families: list[str], ctx: ExecContext) -> str:
    if len(families) == len(ctx.cfg.families):
        return "overall weather exposure"
    return " + ".join(ctx.short_label(f).lower() for f in families) + " exposure"


def _format_indicator(ind: dict) -> str:
    if ind["unit"] == "percentile":
        value = "not applicable (0)" if ind.get("applicable") is False else f"{ordinal(ind['raw'])} percentile nationally"
    else:
        value = f"{ind['raw']:.1f} {ind['unit']}"
    return f"{ind['label']}: {value} → {ind['score']:.0f} pts"


def _drivers(bd: dict, families: list[str], n: int = 3) -> list[str]:
    indicators = [ind for f in families if not bd["families"][f].get("incomplete")
                  for lens in bd["families"][f]["lenses"].values() for ind in lens.get("indicators", [])]
    return [_format_indicator(i) for i in sorted(indicators, key=lambda i: -i["score"])[:n]]


# -- coverage rule (config/scoring.yaml → coverage, D24) ---------------------------------

def _blocked(bd: dict, families: list[str]) -> bool:
    """A requested family can't be scored because the hub's weather fails the coverage rule."""
    return any(bd["families"][f].get("incomplete") for f in families)


def _problems(bd: dict) -> str:
    return " and ".join(bd.get("meta", {}).get("observed_window", {}).get("problems") or []) \
        or "the weather record is incomplete"


def _rule(ctx: ExecContext) -> str:
    rule = ctx.cfg.coverage
    return (f"A score needs at least {rule.min_pct:g}% of days with data and no gap longer than "
            f"{rule.max_gap_days} days.")


def _blocked_reason(ctx: ExecContext, hub_id: str, bd: dict, verb: str) -> tuple[str, str]:
    return ("high", f"{ctx.name(hub_id)} is {verb}: its weather data is incomplete ({_problems(bd)}). {_rule(ctx)}")


def _names(ctx: ExecContext, hub_ids: list[str]) -> str:
    return " and ".join(ctx.name(h) for h in hub_ids)


def _family_detail(bd: dict, family: str) -> dict:
    fam = bd["families"][family]
    return {
        "score": r1(fam["score"]),
        "lenses": {name: {"score": r1(lens["score"]), "weight": lens.get("weight"),
                          "evidence": [_format_indicator(i) for i in lens.get("indicators", [])]}
                   for name, lens in fam["lenses"].items()},
        "missing_evidence": fam["missing_lenses"],
    }


def _evidence_note(families: list[str], ctx: ExecContext) -> str:
    parts = []
    for f in families:
        fam, weights = ctx.cfg.families[f], ctx.cfg.lens_weights_for(f)
        observed = f"{weights.observed:g}" if fam.observed else "–"
        modeled = f"{weights.modeled:g}" if fam.modeled else "–"
        parts.append(f"{ctx.short_label(f).lower()} {observed}/{modeled}")
    return ("Evidence weights per hazard (observed weather days / FEMA NRI building-loss percentile; "
            "renormalized over the evidence available): " + "; ".join(parts) + ".")


def _anchor_note(families: list[str], ctx: ExecContext) -> str:
    example = next((ind for f in families for ind in ctx.cfg.families[f].observed), None)
    text = "Indicators are scored on fixed anchors from config/scoring.yaml"
    if example:
        text += f" (e.g. {example.label.lower()}: {example.full_score_at:g} a year = 100)"
    return text + ", so a hub's score does not depend on which other hubs are listed."


def _exposure_context(ctx: ExecContext, exposure, families: list[str]) -> tuple[list, list, list, list]:
    """Windows, sources, assumptions and data gaps shared by every long-term answer."""
    windows = [
        {"label": "Observed weather (Open-Meteo ERA5), previous completed calendar years",
         "start": exposure.start.isoformat(), "end": exposure.end.isoformat()},
        {"label": f"FEMA National Risk Index ({exposure.nri_version or 'unavailable'} release)", "start": None, "end": None},
    ]
    uses_observed = any(ctx.cfg.families[f].observed for f in families)
    sources = (["open_meteo_archive"] if uses_observed else []) + ["fema_nri"]
    assumptions = [
        "Hub locations are assumed: each hub sits at its metro's main airport logistics zone; "
        "county-level hazard data stands in for the site.",
        _evidence_note(families, ctx),
        _anchor_note(families, ctx),
    ]
    nri_only = [ctx.short_label(f).lower() for f in families if not ctx.cfg.families[f].observed]
    if nri_only:
        assumptions.append(f"{' and '.join(nri_only).capitalize()} scores rest on FEMA NRI alone — "
                           "reanalysis weather cannot resolve hurricanes, tornadoes or hail.")
    if len(families) == len(ctx.cfg.families):
        assumptions.append("Overall exposure is the equal-weight average of 5 hazard families "
                           "(winter, hurricane, flood, severe storm, heat).")
    gaps = sorted({g for bd in exposure.breakdowns.values() for g in bd.get("data_gaps", [])})
    return windows, sources, assumptions, gaps


async def _rank(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    hub_ids, scope = _select_hubs(plan, ctx)
    families = _families(plan, ctx)
    weights, weight_notes = _weights(plan, ctx)
    exposure = await ctx.analyzer.exposure(hub_ids)
    breakdowns = exposure.breakdowns
    blocked = [h for h in hub_ids if _blocked(breakdowns[h], families)]
    blocked_rows = [{"rank": None, "hub": ctx.name(h), "score": None,
                     "note": f"Not ranked: incomplete weather data ({_problems(breakdowns[h])})"} for h in blocked]
    blocked_reasons = [_blocked_reason(ctx, h, breakdowns[h], "not ranked") for h in blocked]
    scores = {h: s for h, bd in breakdowns.items()
              if h not in blocked and (s := overall_score(bd, weights, families)) is not None}
    if not scores:
        windows, sources, assumptions, gaps = _exposure_context(ctx, exposure, families)
        if blocked:
            return ResultBundle(
                "rank", f"No hub can be ranked on {_hazard_phrase(families, ctx)}: the weather data is incomplete "
                        f"for {_names(ctx, blocked)}.",
                rows=blocked_rows, details={"scope": scope, "not_ranked": [ctx.name(h) for h in blocked]},
                points=[r for _, r in blocked_reasons], windows=windows, sources=sources, data_gaps=gaps,
                uncertainty=blocked_reasons, assumptions=assumptions)
        return ResultBundle("rank", "No hub could be scored — every data source failed.",
                            data_gaps=sorted({g for bd in breakdowns.values() for g in bd["data_gaps"]}),
                            uncertainty=[("high", "No evidence was available for the requested hazards.")])
    order = sorted(scores, key=lambda h: (-scores[h], h))
    k = min(plan.top_k or len(order), len(order))

    rows = []
    for i, h in enumerate(order[:k], 1):
        bd = breakdowns[h]
        row: dict[str, Any] = {"rank": i, "hub": ctx.name(h), "score": r1(scores[h])}
        if len(families) > 1:
            row |= {ctx.short_label(f): r1(bd["families"][f]["score"]) for f in families}
        row["top hazard"] = ctx.short_label(max(families, key=lambda f: bd["families"][f]["score"] or 0))
        rows.append(row)
    rows += blocked_rows

    check_k = min(k, 3)
    stability = rank_stability({h: breakdowns[h] for h in order}, families=families, family_weights=weights,
                               lenses=("observed", "modeled"), perturbation=ctx.cfg.robustness.perturbation,
                               top_k=check_k)
    stability["baseline"] = [ctx.name(h) for h in stability["baseline"][:k]]
    stability["flips"] = [f"{f['scenario']}: {ctx.name(f['first'])} ranks first" for f in stability["flips"]]
    ties = near_ties([(ctx.name(h), scores[h]) for h in order], ctx.cfg.robustness.tie_margin, check_k)

    uncertainty = blocked_reasons + [("medium", f"{a} and {b} are only {gap} points apart — treat them as tied.")
                                     for a, b, gap in ties]
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
    top = order[0]
    runners = ", ".join(f"{ctx.name(h)} ({r1(scores[h])})" for h in order[1:3])
    headline = f"{ctx.name(top)} has the highest {_hazard_phrase(families, ctx)} among {scope} (score {r1(scores[top])}/100)"
    headline += f", followed by {runners}." if runners else "."
    if blocked:
        headline += f" Not ranked (incomplete weather data): {_names(ctx, blocked)}."
    drivers = {ctx.name(h): _drivers(breakdowns[h], families) for h in order[:check_k]}
    return ResultBundle(
        intent="rank", headline=headline, rows=rows,
        details={"scope": scope, "hazards": [ctx.family_label(f) for f in families],
                 "weights": weights if weight_notes else "equal", "drivers": drivers, "robustness": stability,
                 "not_ranked": [ctx.name(h) for h in blocked]},
        points=[f"{name}: {'; '.join(d[:2])}" for name, d in drivers.items()] + [r for _, r in uncertainty],
        windows=windows, sources=sources, data_gaps=gaps, uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=[f"Why is {ctx.name(top)} ranked first?", f"Compare {ctx.name(top)} and {ctx.name(order[1])}"]
        if len(order) > 1 else [],
    )


async def _compare(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    hub_ids = list(plan.hubs or [])
    families = _families(plan, ctx)
    weights, weight_notes = _weights(plan, ctx)
    exposure = await ctx.analyzer.exposure(hub_ids)
    bds = exposure.breakdowns
    blocked = [h for h in hub_ids if _blocked(bds[h], families)]
    complete = [h for h in hub_ids if h not in blocked]
    combined = {h: overall_score(bds[h], weights, families) or 0.0 for h in complete}
    rows = []
    for h in hub_ids:
        row: dict[str, Any] = {"hub": ctx.name(h)} | {ctx.short_label(f): r1(bds[h]["families"][f]["score"])
                                                       for f in families}
        if len(families) > 1:
            row["combined"] = r1(combined.get(h))
        if h in blocked:
            row["note"] = f"Not compared: incomplete weather data ({_problems(bds[h])})"
        rows.append(row)

    differences, family_ties = [], []
    for f in families:
        scored = [h for h in hub_ids if bds[h]["families"][f]["score"] is not None]
        if len(scored) < 2:
            missing = [h for h in hub_ids if h not in scored]
            differences.append(f"{ctx.family_label(f)}: not comparable — no score for {_names(ctx, missing)}")
            continue
        ranked = sorted(scored, key=lambda h: -(bds[h]["families"][f]["score"] or 0))
        lead, second = ranked[0], ranked[1]
        gap = (bds[lead]["families"][f]["score"] or 0) - (bds[second]["families"][f]["score"] or 0)
        differences.append(f"{ctx.family_label(f)}: {ctx.name(lead)} higher by {gap:.1f} points" if gap > 0
                           else f"{ctx.family_label(f)}: {ctx.name(lead)} and {ctx.name(second)} are level")
        if len(families) > 1 and gap < ctx.cfg.robustness.tie_margin:
            family_ties.append(f"On {ctx.short_label(f).lower()} alone, {ctx.name(lead)} and {ctx.name(second)} "
                               f"are within {gap:.1f} points — effectively tied.")
    windows, sources, assumptions, gaps = _exposure_context(ctx, exposure, families)
    blocked_reasons = [_blocked_reason(ctx, h, bds[h], "not compared") for h in blocked]
    if len(complete) < 2:
        why = "; ".join(f"{ctx.name(h)}'s weather data is incomplete ({_problems(bds[h])})" for h in blocked)
        return ResultBundle(
            intent="compare", rows=rows,
            headline=f"I can't compare {_names(ctx, hub_ids)} on {_hazard_phrase(families, ctx)}: {why}, "
                     "so its score is not computed.",
            details={"hazards": [ctx.family_label(f) for f in families], "differences": differences,
                     "not_compared": [ctx.name(h) for h in blocked]},
            points=differences, windows=windows, sources=sources, data_gaps=gaps, uncertainty=blocked_reasons,
            assumptions=assumptions)
    order = sorted(complete, key=lambda h: -combined[h])
    gap = combined[order[0]] - combined[order[1]]
    others = " and ".join(f"{ctx.name(h)} ({r1(combined[h])})" for h in order[1:])
    headline = f"{ctx.name(order[0])} has the higher {_hazard_phrase(families, ctx)} ({r1(combined[order[0]])}/100) vs {others}."
    if blocked:
        headline += f" Not compared (incomplete weather data): {_names(ctx, blocked)}."
    uncertainty = blocked_reasons + [("low", t) for t in family_ties]
    if gap < ctx.cfg.robustness.tie_margin:
        uncertainty.append(("medium", f"{ctx.name(order[0])} and {ctx.name(order[1])} are only {gap:.1f} points "
                                      "apart — treat them as similarly exposed."))
    if weight_notes:
        assumptions.append(f"Hazard weights changed for this answer: {', '.join(weight_notes)} (others ×1).")
    detail = {ctx.name(h): {ctx.short_label(f): _family_detail(bds[h], f) for f in families} for h in hub_ids}
    return ResultBundle(
        intent="compare", headline=headline, rows=rows,
        details={"hazards": [ctx.family_label(f) for f in families], "differences": differences, "hubs": detail},
        points=differences + [f"{ctx.name(h)}: {'; '.join(_drivers(bds[h], families, 2))}" for h in order],
        windows=windows, sources=sources, data_gaps=gaps, uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=[f"Why is {ctx.name(order[0])} higher?",
                    "Add Dallas to the comparison" if "dallas" not in hub_ids else "Rank all hubs on these hazards"],
    )


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


async def _stat(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    metric = ctx.cfg.stats[plan.metric]
    hub_ids = list(plan.hubs or [])
    try:
        window = resolve_window(plan.time_preset, today=ctx.analyzer.today, data_end=ctx.analyzer.data_end(),
                                data_start=ARCHIVE_START, year=plan.year, start=plan.start_date, end=plan.end_date)
    except WindowError as exc:
        return ResultBundle("stat", f"I can't compute that window: {exc}.",
                            details={"available_data": f"{ARCHIVE_START} to {ctx.analyzer.data_end()}"},
                            uncertainty=[("high", str(exc))])
    rows, sensitivity, grid_notes, uncertainty, gaps, partial = [], {}, [], [], [], {}
    for h in hub_ids:
        daily, hub_gaps = await ctx.analyzer.daily(h, window.start, window.end)
        gaps += hub_gaps
        cov = weather_coverage(daily, window.start, window.end, [metric.variable], ctx.cfg.coverage)
        if not cov["complete"]:  # partial record: no percentage, say why (coverage rule, D24)
            partial[h] = " and ".join(cov["problems"])
            rows.append({"hub": ctx.name(h), "days meeting threshold": None, "days with data": cov["days_with_data"],
                         "% of days": None, "coverage %": cov["pct"],
                         "note": "Partial: incomplete data, no percentage given"})
            uncertainty.append(("high", f"{ctx.name(h)}: the weather record for {window.label} is incomplete "
                                        f"({partial[h]}), so no share of days is given. {_rule(ctx)}"))
            continue
        s = day_stat(daily, window.start, window.end, metric)
        rows.append({"hub": ctx.name(h), "days meeting threshold": s["count"], "days with data": s["days_with_data"],
                     "% of days": s["pct"], "coverage %": cov["pct"]})
        sensitivity[ctx.name(h)] = s["sensitivity"]
        pcts = [x["pct"] for x in s["sensitivity"] if x["pct"]]
        if len(pcts) >= 2 and max(pcts) >= 2 * min(pcts):
            lo = min(s["sensitivity"], key=lambda x: x["pct"] or 0)
            hi = max(s["sensitivity"], key=lambda x: x["pct"] or 0)
            uncertainty.append(("medium", f"{ctx.name(h)}: the threshold matters — {hi['label']} gives {hi['pct']}% "
                                          f"of days, {lo['label']} gives {lo['pct']}%."))
        cell = await ctx.analyzer.weather.grid_cell(ctx.hubs[h])
        if cell:
            hub = ctx.hubs[h]
            grid_notes.append(f"{ctx.name(h)}: values are for the reanalysis grid cell at {cell['latitude']:.2f}, "
                              f"{cell['longitude']:.2f} ({_km(hub.lat, hub.lon, cell['latitude'], cell['longitude']):.0f} "
                              "km from the hub point).")
    answered = [r for r in rows if r["% of days"] is not None]
    if not answered:
        headline = (f"I can't give a reliable share of days with {metric.short} at {_names(ctx, hub_ids)} in "
                    f"{window.label}: the weather record is incomplete ({'; '.join(partial.values())}).")
    elif len(rows) == 1:
        first = rows[0]
        headline = (f"{first['% of days']}% of days at {first['hub']} in {window.label} had {metric.short} "
                    f"({first['days meeting threshold']} of {first['days with data']} days).")
    else:
        parts = ", ".join(f"{r['hub']} {r['% of days']}% ({r['days meeting threshold']} days)" for r in answered)
        headline = f"Share of days in {window.label} with {metric.short}: {parts}."
        if partial:
            headline += f" {_names(ctx, list(partial))}: incomplete data, no percentage given."
    assumptions = [
        f"Definition: {metric.label}.",
        "Open-Meteo ERA5 reanalysis is modeled, not station-measured. It smooths daily extremes and can differ "
        "from airport station records by several days a year, so treat counts as estimates.",
        *grid_notes, *window.notes,
    ]
    return ResultBundle(
        intent="stat", headline=headline, rows=rows,
        details={"metric": metric.label, "window": window.label, "sensitivity": sensitivity},
        points=[f"{name}: " + "; ".join(f"{x['label']} → {x['pct']}%" for x in rows_) for name, rows_ in sensitivity.items()],
        windows=[{"label": window.label, "start": window.start.isoformat(), "end": window.end.isoformat()}],
        sources=["open_meteo_archive"], data_gaps=gaps, uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=["And the year before?", "How does that compare to Chicago?"],
    )


async def _explain(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    hub = plan.hubs[0]
    exposure = await ctx.analyzer.exposure()  # all hubs: ranks and medians need the full roster
    bds = exposure.breakdowns
    bd = bds[hub]
    families = _families(plan, ctx)
    # Ranks and medians use only hubs with a score: an incomplete hub is not ranked (coverage rule).
    ranked_overall = sorted((h for h in bds if bds[h]["score"] is not None), key=lambda h: -bds[h]["score"])
    overall_rank = f"{ranked_overall.index(hub) + 1} of {len(ranked_overall)}" if hub in ranked_overall \
        else "not ranked"

    def family_rank(f: str) -> str:
        scored = sorted((h for h in bds if bds[h]["families"][f]["score"] is not None),
                        key=lambda h: -bds[h]["families"][f]["score"])
        return f"{scored.index(hub) + 1} of {len(scored)}" if hub in scored else "not ranked"

    def family_median(f: str) -> float | None:
        values = [bds[h]["families"][f]["score"] for h in bds if bds[h]["families"][f]["score"] is not None]
        return r1(median(values)) if values else None

    shown = sorted(families, key=lambda f: (-(bd["families"][f]["contribution"] or 0),
                                            -(bd["families"][f]["score"] if bd["families"][f]["score"] is not None else -1)))
    rows = [{"hazard": ctx.family_label(f), "score": r1(bd["families"][f]["score"]),
             "contribution to overall": r1(bd["families"][f]["contribution"]),
             "rank among hubs": family_rank(f), "median across hubs": family_median(f)}
            for f in shown]
    top = shown[0]
    windows, sources, assumptions, _ = _exposure_context(ctx, exposure, families)
    gaps = bd.get("data_gaps", [])
    uncertainty = [("medium", g) for g in gaps]
    if _blocked(bd, families):
        headline = (f"{ctx.name(hub)} is not scored on {_hazard_phrase(families, ctx)}: its weather data for "
                    f"{exposure.start.year}–{exposure.end.year} is incomplete ({_problems(bd)}), so it is not ranked.")
        nri_only = [f for f in shown if bd["families"][f]["score"] is not None]
        if nri_only:
            headline += " Hazards scored from FEMA NRI alone: " + ", ".join(
                f"{ctx.short_label(f).lower()} {r1(bd['families'][f]['score'])}/100" for f in nri_only) + "."
        uncertainty.insert(0, _blocked_reason(ctx, hub, bd, "not scored"))
    elif len(families) < len(ctx.cfg.families):
        combined = overall_score(bd, ctx.cfg.family_weights, families)
        headline = (f"{ctx.name(hub)} scores {r1(combined)}/100 on {_hazard_phrase(families, ctx)}; the biggest "
                    f"driver is {ctx.short_label(top).lower()} ({r1(bd['families'][top]['score'])}/100, "
                    f"rank {family_rank(top)}).")
    else:
        headline = (f"{ctx.name(hub)} has an overall exposure score of {r1(bd['score'])}/100 "
                    f"(rank {overall_rank}). The largest contributor is "
                    f"{ctx.short_label(top)} at {r1(bd['families'][top]['score'])}/100 (rank {family_rank(top)}).")
    other = next((h for h in ranked_overall if h != hub), None)
    return ResultBundle(
        intent="explain", headline=headline, rows=rows,
        details={"hub": f"{ctx.name(hub)} ({bd.get('meta', {}).get('county', ctx.hubs[hub].county)})",
                 "overall_score": r1(bd["score"]), "overall_rank": overall_rank,
                 "hazards": {ctx.short_label(f): _family_detail(bd, f) for f in shown},
                 "top_drivers": _drivers(bd, families, 4)},
        points=_drivers(bd, families, 4) or [r for _, r in uncertainty[:1]], windows=windows, sources=sources,
        data_gaps=gaps, uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=[f"Which evidence drives {ctx.name(hub)}'s {ctx.short_label(top).lower()} score?"]
        + ([f"Compare {ctx.name(hub)} with {ctx.name(other)}"] if other else []),
    )


async def _no_compute(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    text = plan.clarification_question if plan.intent == "clarify" else plan.out_of_scope_reason
    return ResultBundle(
        intent=plan.intent, headline=text or "",
        details={"what_i_can_answer": CAPABILITIES, "hubs": [h.name for h in ctx.hubs.values()]},
        points=["I can answer: " + "; ".join(c[0].lower() + c[1:] for c in CAPABILITIES[:3]) + "."],
        follow_ups=["Which hubs in the Midwest are most exposed to winter disruption?",
                    "Compare Miami and Houston on hurricane and flood exposure"],
    )


HANDLERS = {"rank": _rank, "compare": _compare, "stat": _stat, "explain": _explain,
            "clarify": _no_compute, "out_of_scope": _no_compute}


async def execute(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    return await HANDLERS[plan.intent](plan, ctx)
