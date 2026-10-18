"""QueryPlan → ResultBundle. Deterministic Python over on-demand, cached data; no LLM.

Every number an answer shows is produced here (docs/DECISIONS.md D1). The Analytics API calls the
same functions, so charts and chat always agree.
"""

import math
from dataclasses import dataclass, field
from datetime import date
from statistics import median
from typing import Any

from weather_risk.agent.i18n import family_label, indicator_label, nri_name, region_name, stat_label
from weather_risk.agent.schemas import QueryPlan
from weather_risk.scoring.coverage import weather_coverage
from weather_risk.scoring.exposure import overall_score
from weather_risk.scoring.robustness import near_ties, rank_stability
from weather_risk.scoring.stats import day_stat
from weather_risk.timewindow import ARCHIVE_LAG_DAYS, WindowError, resolve_window

ARCHIVE_START = date(1940, 1, 1)  # Open-Meteo ERA5 archive starts in 1940
CAPABILITIES = [
    "Rank hubs (all, by region, or a list) by long-term exposure — overall or per hazard",
    "Compare hubs on chosen hazards (winter, hurricane, flood, severe storm, heat)",
    "Explain why a hub's score is high or low, down to the evidence",
    "Count past days meeting a weather condition (snowfall, freezing, heavy rain, heat, wind)",
]
CAPABILITIES_HE = [
    "דירוג מרכזים (כולם, לפי אזור או רשימה) לפי חשיפה ארוכת טווח — כוללת או לפי סיכון",
    "השוואת מרכזים בסיכונים שנבחרו (חורף, הוריקן, הצפה, סערה קשה, חום)",
    "הסבר מדוע הציון של מרכז גבוה או נמוך, עד רמת הראיות",
    "ספירת ימים בעבר שעמדו בתנאי מזג אוויר (שלג, קיפאון, גשם כבד, חום, רוח)",
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
    # Computed orders for the ranking-claim check: "overall" (the hazards asked about) and each family,
    # hub names highest first. Not sent to the LLM; rows and ranks in `details` already carry them.
    orders: dict[str, list[str]] = field(default_factory=dict)

    def for_llm(self) -> dict:
        """Compact view for the explainer prompt (Groq free tier: 8K tokens/min)."""
        return {"intent": self.intent, "headline": self.headline, "rows": self.rows[:10], "details": self.details,
                "uncertainty": [reason for _, reason in self.uncertainty], "data_gaps": self.data_gaps}




@dataclass
class ExecContext:
    analyzer: Any  # weather_risk.analysis.Analyzer (or a test double with the same interface)
    lang: str = "en"  # "he": every sentence for a reader in Hebrew (agent/i18n.py); row keys and numbers unchanged

    @property
    def cfg(self):
        return self.analyzer.cfg

    @property
    def hubs(self):
        return self.analyzer.hubs

    def tr(self, en: str, he: str) -> str:
        return he if self.lang == "he" else en

    def name(self, hub_id: str) -> str:
        return self.hubs[hub_id].name

    def family_label(self, family: str) -> str:
        return family_label(self.cfg, family, self.lang)

    def short_label(self, family: str) -> str:
        """English short label: also the table's column key, so it never changes with the language."""
        return family_label(self.cfg, family, "en", short=True)

    def hazard(self, family: str) -> str:
        """A hazard's short name inside a sentence ("winter" / "חורף")."""
        return family_label(self.cfg, family, self.lang, short=True) if self.lang == "he" \
            else self.short_label(family).lower()

    def join(self, items: list[str]) -> str:
        return self.tr(" and ", " ו-").join(items)


def r1(x: float | None) -> float | None:
    return None if x is None else round(x, 1)


def _select_hubs(plan: QueryPlan, ctx: ExecContext) -> tuple[list[str], str]:
    if plan.hubs:
        return list(plan.hubs), ctx.tr("the selected hubs", "המרכזים שנבחרו")
    excluded = [h for h in plan.exclude_hubs or [] if h in ctx.hubs]
    if plan.region:
        hub_ids = [h for h, hub in ctx.hubs.items() if hub.region == plan.region and h not in excluded]
        region = region_name(plan.region, ctx.lang, REGION_NAMES[plan.region])
        scope = ctx.tr(f"{region} hubs (US Census region)", f"מרכזי {region} (אזור מפקד האוכלוסין של ארה״ב)")
    else:
        hub_ids = [h for h in ctx.hubs if h not in excluded]
        scope = ctx.tr(f"all {len(ctx.hubs)} hubs", f"כל {len(ctx.hubs)} המרכזים")
    if excluded:
        scope += ctx.tr(f" except {_names(ctx, excluded)}", f" מלבד {_names(ctx, excluded)}")
    return hub_ids, scope


def _families(plan: QueryPlan, ctx: ExecContext) -> list[str]:
    return list(plan.hazards) if plan.hazards else list(ctx.cfg.families)


def _weights(plan: QueryPlan, ctx: ExecContext) -> tuple[dict[str, float], list[str]]:
    weights = dict(ctx.cfg.family_weights)
    notes = []
    if plan.weight_overrides:
        for family, value in plan.weight_overrides.model_dump().items():
            if value is not None:
                weights[family] = value
                notes.append(f"{ctx.tr(family, ctx.hazard(family))} ×{value:g}")
    return weights, notes


def _hazard_phrase(families: list[str], ctx: ExecContext) -> str:
    """English: "overall weather exposure" / "winter exposure"; Hebrew: "בחשיפה הכוללת…" / "בחשיפה לחורף"."""
    if len(families) == len(ctx.cfg.families):
        return ctx.tr("overall weather exposure", "בחשיפה הכוללת למזג האוויר")
    joined = " + ".join(ctx.hazard(f) for f in families)
    return ctx.tr(f"{joined} exposure", f"בחשיפה ל{joined}")


def _per_million(rate: float, ctx: ExecContext) -> str:
    """Building loss rate → dollars lost per $1M of building value (D26)."""
    dollars = rate * 1e6
    if dollars == 0:
        return "$0"
    if dollars < 0.01:
        return ctx.tr("under $0.01", "פחות מ-$0.01")
    return f"${dollars:,.0f}" if dollars >= 10 else f"${dollars:.2g}"


def _indicator_label(ind: dict, ctx: ExecContext) -> str:
    if ctx.lang != "he":
        return ind["label"]
    if ind["id"].startswith("nri_") and ind["id"][4:] in ctx.cfg.nri_hazard_names:
        return f"{nri_name(ctx.cfg, ind['id'][4:], 'he')}: נזק למבנים"
    spec = next((s for fam in ctx.cfg.families.values() for s in fam.observed if s.id == ind["id"]), None)
    return indicator_label(spec, "he") if spec else ind["label"]


def _format_indicator(ind: dict, ctx: ExecContext | None = None) -> str:
    ctx = ctx or ExecContext(analyzer=None)  # English
    if ind["unit"] == "loss rate":
        rate = _per_million(ind["raw"] or 0.0, ctx)
        value = (ctx.tr("not applicable (0)", "לא רלוונטי (0)") if ind.get("applicable") is False
                 else ctx.tr(f"{rate} per $1M of building value a year", f"{rate} לכל מיליון $ של ערך מבנים בשנה"))
    else:
        value = ctx.tr(f"{ind['raw']:.1f} {ind['unit']}", f"{ind['raw']:.1f} ימים בשנה")
    return f"{_indicator_label(ind, ctx)}: {value} → {ind['score']:.0f} {ctx.tr('pts', 'נק׳')}"


def _drivers(bd: dict, families: list[str], ctx: ExecContext, n: int = 3) -> list[str]:
    indicators = [ind for f in families if not bd["families"][f].get("incomplete")
                  for lens in bd["families"][f]["lenses"].values() for ind in lens.get("indicators", [])]
    return [_format_indicator(i, ctx) for i in sorted(indicators, key=lambda i: -i["score"])[:n]]


# -- coverage rule (config/scoring.yaml → coverage, D24) ---------------------------------

# A requested hazard without a score (incomplete weather, or its only source down) blocks the hub:
# it is not ranked, compared or scored on that combination, and the answer says why.

def _blocked(bd: dict, families: list[str]) -> bool:
    return any(bd["families"][f]["score"] is None for f in families)


def _coverage_problems(bd: dict, ctx: ExecContext) -> str:
    return " and ".join(bd.get("meta", {}).get("observed_window", {}).get("problems") or []) \
        or ctx.tr("the weather record is incomplete", "רשומת מזג האוויר אינה מלאה")


def _why_unscored(ctx: ExecContext, bd: dict, families: list[str]) -> str:
    """'incomplete weather data (…)' or 'no evidence for hurricane: FEMA National Risk Index unavailable'."""
    missing = [f for f in families if bd["families"][f]["score"] is None]
    if any(bd["families"][f].get("incomplete") for f in missing):
        problems = _coverage_problems(bd, ctx)
        return ctx.tr(f"incomplete weather data ({problems})", f"נתוני מזג האוויר אינם מלאים ({problems})")
    sources = sorted({lens["source"] for f in missing for name, lens in bd["families"][f]["lenses"].items()
                      if name in bd["families"][f]["missing_lenses"]})
    hazards = ", ".join(ctx.hazard(f) for f in missing)
    return ctx.tr(f"no evidence for {hazards}: {' and '.join(sources) or 'its data source'} unavailable",
                  f"אין ראיות עבור {hazards}: {' ו-'.join(sources) or 'מקור הנתונים'} לא זמין")


def _rule(ctx: ExecContext) -> str:
    rule = ctx.cfg.coverage
    return ctx.tr(f" A score needs at least {rule.min_pct:g}% of days with data and no gap longer than "
                  f"{rule.max_gap_days} days.",
                  f" ציון דורש נתונים בלפחות {rule.min_pct:g}% מהימים וללא רצף חסר של יותר מ-"
                  f"{rule.max_gap_days} ימים.")


VERBS_HE = {"not ranked": "לא דורג", "not compared": "לא הושווה", "not scored": "לא קיבל ציון"}


def _blocked_reason(ctx: ExecContext, hub_id: str, bd: dict, families: list[str], verb: str) -> tuple[str, str]:
    incomplete = any(bd["families"][f].get("incomplete") for f in families)
    why = _why_unscored(ctx, bd, families)
    return ("high", ctx.tr(f"{ctx.name(hub_id)} is {verb}: {why}.", f"{ctx.name(hub_id)} {VERBS_HE[verb]}: {why}.")
            + (_rule(ctx) if incomplete else ""))


def _orders(ctx: ExecContext, bds: dict[str, dict], hub_ids: list[str], families: list[str],
            combined: dict[str, float]) -> dict[str, list[str]]:
    """Hub names highest first: on the requested hazards combined, and per family (scored hubs only)."""
    orders = {"overall": [ctx.name(h) for h in sorted(combined, key=lambda h: (-combined[h], h))]}
    for f in families:
        scored = [h for h in hub_ids if bds[h]["families"][f]["score"] is not None]
        orders[f] = [ctx.name(h) for h in sorted(scored, key=lambda h: (-bds[h]["families"][f]["score"], h))]
    return orders


def _names(ctx: ExecContext, hub_ids: list[str]) -> str:
    return ctx.join([ctx.name(h) for h in hub_ids])


def _family_detail(bd: dict, family: str, ctx: ExecContext) -> dict:
    fam = bd["families"][family]
    return {
        "score": r1(fam["score"]),
        "lenses": {name: {"score": r1(lens["score"]), "weight": lens.get("weight"),
                          "evidence": [_format_indicator(i, ctx) for i in lens.get("indicators", [])]}
                   for name, lens in fam["lenses"].items()},
        "missing_evidence": fam["missing_lenses"],
    }


def _evidence_note(families: list[str], ctx: ExecContext) -> str:
    parts = []
    for f in families:
        fam, weights = ctx.cfg.families[f], ctx.cfg.lens_weights_for(f)
        observed = f"{weights.observed:g}" if fam.observed else "–"
        modeled = f"{weights.modeled:g}" if fam.modeled else "–"
        parts.append(f"{ctx.hazard(f)} {observed}/{modeled}")
    return ctx.tr("Evidence weights per hazard (observed weather days / FEMA NRI building-loss rate; "
                  "renormalized over the evidence available): ",
                  "משקלי הראיות לכל סיכון (ימי מזג אוויר נצפים / שיעור נזק למבנים לפי FEMA NRI; "
                  "מנורמלים לפי הראיות הזמינות): ") + "; ".join(parts) + "."


def _anchor_note(families: list[str], ctx: ExecContext) -> str:
    example = next((ind for f in families for ind in ctx.cfg.families[f].observed), None)
    text = ctx.tr("Indicators are scored on fixed anchors from config/scoring.yaml",
                  "המדדים מקבלים ציון לפי עוגנים קבועים מ-config/scoring.yaml")
    if example:
        label = indicator_label(example, ctx.lang)
        text += ctx.tr(f" (e.g. {label.lower()}: {example.full_score_at:g} a year = 100)",
                       f" (למשל {label}: {example.full_score_at:g} בשנה = 100)")
    return text + ctx.tr(", so a hub's score does not depend on which other hubs are listed.",
                         ", ולכן הציון של מרכז אינו תלוי במרכזים האחרים ברשימה.")


def _exposure_context(ctx: ExecContext, exposure, families: list[str]) -> tuple[list, list, list, list]:
    """Windows, sources, assumptions and data gaps shared by every long-term answer."""
    release = exposure.nri_version or ctx.tr("unavailable", "לא זמינה")
    windows = [
        {"label": ctx.tr("Observed weather (Open-Meteo ERA5), previous completed calendar years",
                         "מזג אוויר נצפה (Open-Meteo ERA5), השנים הקלנדריות השלמות האחרונות"),
         "start": exposure.start.isoformat(), "end": exposure.end.isoformat()},
        {"label": ctx.tr(f"FEMA National Risk Index ({release} release)", f"מדד הסיכון הלאומי של FEMA (מהדורת {release})"),
         "start": None, "end": None},
    ]
    uses_observed = any(ctx.cfg.families[f].observed for f in families)
    sources = (["open_meteo_archive"] if uses_observed else []) + ["fema_nri"]
    assumptions = [
        ctx.tr("Hub locations are assumed: each hub sits at its metro's main airport logistics zone; "
               "county-level hazard data stands in for the site.",
               "מיקומי המרכזים הם הנחה: כל מרכז נמצא באזור הלוגיסטי של שדה התעופה הראשי במטרופולין, "
               "ונתוני הסיכון ברמת המחוז מייצגים את האתר."),
        _evidence_note(families, ctx),
        _anchor_note(families, ctx),
    ]
    nri_only = [ctx.hazard(f) for f in families if not ctx.cfg.families[f].observed]
    if nri_only:
        assumptions.append(ctx.tr(f"{' and '.join(nri_only).capitalize()} scores rest on FEMA NRI alone — "
                                  "reanalysis weather cannot resolve hurricanes, tornadoes or hail.",
                                  f"הציונים של {ctx.join(nri_only)} מבוססים על FEMA NRI בלבד — נתוני ניתוח-חוזר "
                                  "(reanalysis) אינם מזהים הוריקנים, טורנדו או ברד."))
    if len(families) == len(ctx.cfg.families):
        assumptions.append(ctx.tr("Overall exposure is the equal-weight average of 5 hazard families "
                                  "(winter, hurricane, flood, severe storm, heat).",
                                  "החשיפה הכוללת היא ממוצע במשקל שווה של 5 משפחות סיכון "
                                  "(חורף, הוריקן, הצפה, סערה קשה, חום)."))
    gaps = sorted({g for bd in exposure.breakdowns.values() for g in bd.get("data_gaps", [])})
    return windows, sources, assumptions, gaps


def _weights_note(weight_notes: list[str], ctx: ExecContext) -> str:
    notes = ", ".join(weight_notes)
    return ctx.tr(f"Hazard weights changed for this answer: {notes} (others ×1).",
                  f"משקלי הסיכונים שונו בתשובה זו: {notes} (האחרים ×1).")


def _scenario(flip: dict, ctx: ExecContext) -> str:
    """One weight-sensitivity scenario that changed first place, in the answer's language."""
    if ctx.lang != "he":
        return f"{flip['scenario']}: {ctx.name(flip['first'])} ranks first"
    what = (f"משקל {ctx.hazard(flip['name'])}" if flip["kind"] == "family"
            else {"observed": "משקל מזג האוויר הנצפה", "modeled": "משקל FEMA NRI"}.get(flip["name"], flip["name"]))
    return f"{what} {flip['change']}: {ctx.name(flip['first'])} במקום הראשון"


async def _rank(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    hub_ids, scope = _select_hubs(plan, ctx)
    if not hub_ids:
        return ResultBundle("rank", ctx.tr(f"There are no hubs left to rank: {scope} is empty.",
                                           f"לא נשארו מרכזים לדירוג: {scope} — הרשימה ריקה."),
                            details={"scope": scope},
                            uncertainty=[("high", ctx.tr(f"{scope.capitalize()} has no hubs.", f"{scope}: אין מרכזים."))],
                            follow_ups=[ctx.tr("Rank all hubs by overall weather exposure.",
                                               "דרג את כל המרכזים לפי החשיפה הכוללת למזג האוויר.")])
    families = _families(plan, ctx)
    weights, weight_notes = _weights(plan, ctx)
    exposure = await ctx.analyzer.exposure(hub_ids)
    breakdowns = exposure.breakdowns
    blocked = [h for h in hub_ids if _blocked(breakdowns[h], families)]
    blocked_rows = [{"rank": None, "hub": ctx.name(h), "score": None,
                     "note": ctx.tr("Not ranked: ", "לא דורג: ") + _why_unscored(ctx, breakdowns[h], families)}
                    for h in blocked]
    blocked_reasons = [_blocked_reason(ctx, h, breakdowns[h], families, "not ranked") for h in blocked]
    scores = {h: s for h, bd in breakdowns.items()
              if h not in blocked and (s := overall_score(bd, weights, families)) is not None}
    if not scores:
        windows, sources, assumptions, gaps = _exposure_context(ctx, exposure, families)
        if blocked:
            why = "; ".join(f"{ctx.name(h)} — {_why_unscored(ctx, breakdowns[h], families)}"
                            for h in blocked[:3]) + ("; …" if len(blocked) > 3 else "") + "."
            return ResultBundle(
                "rank", ctx.tr(f"No hub can be ranked on {_hazard_phrase(families, ctx)}: ",
                               f"אף מרכז לא ניתן לדירוג {_hazard_phrase(families, ctx)}: ") + why,
                rows=blocked_rows, details={"scope": scope, "not_ranked": [ctx.name(h) for h in blocked]},
                points=[r for _, r in blocked_reasons], windows=windows, sources=sources, data_gaps=gaps,
                uncertainty=blocked_reasons, assumptions=assumptions)
        return ResultBundle("rank", ctx.tr("No hub could be scored — every data source failed.",
                                           "אף מרכז לא קיבל ציון — כל מקורות הנתונים נכשלו."),
                            data_gaps=sorted({g for bd in breakdowns.values() for g in bd["data_gaps"]}),
                            uncertainty=[("high", ctx.tr("No evidence was available for the requested hazards.",
                                                         "לא היו ראיות זמינות עבור הסיכונים שהתבקשו."))])
    order = sorted(scores, key=lambda h: (-scores[h], h))
    k = min(plan.top_k or len(order), len(order))

    rows = []
    for i, h in enumerate(order[:k], 1):
        bd = breakdowns[h]
        row: dict[str, Any] = {"rank": i, "hub": ctx.name(h), "score": r1(scores[h])}
        if len(families) > 1:
            row |= {ctx.short_label(f): r1(bd["families"][f]["score"]) for f in families}
        top_family = max(families, key=lambda f: bd["families"][f]["score"] or 0)
        row["top hazard"] = ctx.tr(ctx.short_label(top_family), ctx.hazard(top_family))
        rows.append(row)
    rows += blocked_rows

    check_k = min(k, 3)
    stability = rank_stability({h: breakdowns[h] for h in order}, families=families, family_weights=weights,
                               lenses=("observed", "modeled"), perturbation=ctx.cfg.robustness.perturbation,
                               top_k=check_k)
    stability["baseline"] = [ctx.name(h) for h in stability["baseline"][:k]]
    stability["flips"] = [_scenario(f, ctx) for f in stability["flips"]]
    ties = near_ties([(ctx.name(h), scores[h]) for h in order], ctx.cfg.robustness.tie_margin, check_k)

    uncertainty = blocked_reasons + [("medium", ctx.tr(f"{a} and {b} are only {gap} points apart — treat them as tied.",
                                                       f"{a} ו-{b} רחוקים זה מזה רק {gap} נקודות — יש להתייחס "
                                                       "אליהם כשקולים."))
                                     for a, b, gap in ties]
    n = stability["scenarios"]
    changed = n - stability["top1_unchanged"]
    if stability["top1_unchanged"] < n:
        uncertainty.append(("medium", ctx.tr(f"First place changes in {changed} of {n} weight "
                                             f"scenarios (each weight ±25 %), e.g. {stability['flips'][0]}.",
                                             f"המקום הראשון משתנה ב-{changed} מתוך {n} תרחישי משקל "
                                             f"(כל משקל ±25%), למשל {stability['flips'][0]}.")))
    else:
        uncertainty.append(("low", ctx.tr(f"First place holds in all {n} weight scenarios (each weight moved ±25 %).",
                                          f"המקום הראשון נשמר בכל {n} תרחישי המשקל (כל משקל הוזז ב-±25%).")))
    if check_k > 1:
        kept = stability["topk_unchanged"]
        uncertainty.append(("low" if kept == n else "medium",
                            ctx.tr(f"The top-{check_k} set is unchanged in {kept} of {n} weight scenarios.",
                                   f"קבוצת {check_k} המובילים לא משתנה ב-{kept} מתוך {n} תרחישי משקל.")))

    windows, sources, assumptions, gaps = _exposure_context(ctx, exposure, families)
    if weight_notes:
        assumptions.append(_weights_note(weight_notes, ctx))
    top = order[0]
    runners = ", ".join(f"{ctx.name(h)} ({r1(scores[h])})" for h in order[1:3])
    phrase = _hazard_phrase(families, ctx)
    headline = ctx.tr(f"{ctx.name(top)} has the highest {phrase} among {scope} (score {r1(scores[top])}/100)",
                      f"{ctx.name(top)} מדורג ראשון {phrase} מבין {scope} (ציון {r1(scores[top])}/100)")
    headline += ctx.tr(f", followed by {runners}.", f", ואחריו {runners}.") if runners else "."
    if blocked:
        headline += ctx.tr(f" Not ranked (missing or incomplete data): {_names(ctx, blocked)}.",
                           f" לא דורגו (נתונים חסרים או חלקיים): {_names(ctx, blocked)}.")
    drivers = {ctx.name(h): _drivers(breakdowns[h], families, ctx) for h in order[:check_k]}
    return ResultBundle(
        intent="rank", headline=headline, rows=rows,
        details={"scope": scope, "hazards": [ctx.family_label(f) for f in families],
                 "weights": weights if weight_notes else "equal", "drivers": drivers, "robustness": stability,
                 "not_ranked": [ctx.name(h) for h in blocked]},
        points=[f"{name}: {'; '.join(d[:2])}" for name, d in drivers.items()] + [r for _, r in uncertainty],
        windows=windows, sources=sources, data_gaps=gaps, uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=[ctx.tr(f"Why is {ctx.name(top)} ranked first?", f"למה {ctx.name(top)} מדורג ראשון?"),
                    ctx.tr(f"Compare {ctx.name(top)} and {ctx.name(order[1])}",
                           f"השווה בין {ctx.name(top)} ל-{ctx.name(order[1])}")]
        if len(order) > 1 else [],
        orders=_orders(ctx, breakdowns, order, families, scores),
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
            row["note"] = ctx.tr("Not compared: ", "לא הושווה: ") + _why_unscored(ctx, bds[h], families)
        rows.append(row)

    differences, family_ties = [], []
    for f in families:
        label = ctx.family_label(f)
        scored = [h for h in hub_ids if bds[h]["families"][f]["score"] is not None]
        if len(scored) < 2:
            missing = _names(ctx, [h for h in hub_ids if h not in scored])
            differences.append(ctx.tr(f"{label}: not comparable — no score for {missing}",
                                      f"{label}: לא ניתן להשוואה — אין ציון ל-{missing}"))
            continue
        ranked = sorted(scored, key=lambda h: -(bds[h]["families"][f]["score"] or 0))
        lead, second = ctx.name(ranked[0]), ctx.name(ranked[1])
        gap = (bds[ranked[0]]["families"][f]["score"] or 0) - (bds[ranked[1]]["families"][f]["score"] or 0)
        differences.append(ctx.tr(f"{label}: {lead} higher by {gap:.1f} points",
                                  f"{label}: {lead} גבוה יותר ב-{gap:.1f} נקודות") if gap > 0
                           else ctx.tr(f"{label}: {lead} and {second} are level", f"{label}: {lead} ו-{second} שווים"))
        if len(families) > 1 and gap < ctx.cfg.robustness.tie_margin:
            family_ties.append(ctx.tr(f"On {ctx.hazard(f)} alone, {lead} and {second} "
                                      f"are within {gap:.1f} points — effectively tied.",
                                      f"ב{ctx.hazard(f)} בלבד, {lead} ו-{second} במרחק {gap:.1f} נקודות זה מזה "
                                      "— כמעט שקולים."))
    windows, sources, assumptions, gaps = _exposure_context(ctx, exposure, families)
    blocked_reasons = [_blocked_reason(ctx, h, bds[h], families, "not compared") for h in blocked]
    phrase = _hazard_phrase(families, ctx)
    if len(complete) < 2:
        why = "; ".join(f"{ctx.name(h)} — {_why_unscored(ctx, bds[h], families)}" for h in blocked)
        return ResultBundle(
            intent="compare", rows=rows,
            headline=ctx.tr(f"I can't compare {_names(ctx, hub_ids)} on {phrase}: {why}.",
                            f"אי אפשר להשוות את {_names(ctx, hub_ids)} {phrase}: {why}."),
            details={"hazards": [ctx.family_label(f) for f in families], "differences": differences,
                     "not_compared": [ctx.name(h) for h in blocked]},
            points=differences, windows=windows, sources=sources, data_gaps=gaps, uncertainty=blocked_reasons,
            assumptions=assumptions)
    order = sorted(complete, key=lambda h: -combined[h])
    gap = combined[order[0]] - combined[order[1]]
    others = ctx.join([f"{ctx.name(h)} ({r1(combined[h])})" for h in order[1:]])
    first = ctx.name(order[0])
    headline = ctx.tr(f"{first} has the higher {phrase} ({r1(combined[order[0]])}/100) vs {others}.",
                      f"{first} גבוה יותר {phrase} ({r1(combined[order[0]])}/100) לעומת {others}.")
    if blocked:
        headline += ctx.tr(f" Not compared (missing or incomplete data): {_names(ctx, blocked)}.",
                           f" לא הושוו (נתונים חסרים או חלקיים): {_names(ctx, blocked)}.")
    uncertainty = blocked_reasons + [("low", t) for t in family_ties]
    if gap < ctx.cfg.robustness.tie_margin:
        second = ctx.name(order[1])
        uncertainty.append(("medium", ctx.tr(f"{first} and {second} are only {gap:.1f} points "
                                             "apart — treat them as similarly exposed.",
                                             f"{first} ו-{second} רחוקים זה מזה רק {gap:.1f} נקודות — יש להתייחס "
                                             "אליהם כחשופים באופן דומה.")))
    if weight_notes:
        assumptions.append(_weights_note(weight_notes, ctx))
    detail = {ctx.name(h): {ctx.short_label(f): _family_detail(bds[h], f, ctx) for f in families} for h in hub_ids}
    add = ctx.tr("Add Dallas to the comparison", "הוסף את Dallas להשוואה") if "dallas" not in hub_ids \
        else ctx.tr("Rank all hubs on these hazards", "דרג את כל המרכזים לפי הסיכונים האלה")
    return ResultBundle(
        intent="compare", headline=headline, rows=rows,
        details={"hazards": [ctx.family_label(f) for f in families], "differences": differences, "hubs": detail},
        points=differences + [f"{ctx.name(h)}: {'; '.join(_drivers(bds[h], families, ctx, 2))}" for h in order],
        windows=windows, sources=sources, data_gaps=gaps, uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=[ctx.tr(f"Why is {first} higher?", f"למה {first} גבוה יותר?"), add],
        orders=_orders(ctx, bds, hub_ids, families, combined),
    )


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def _window_label(plan: QueryPlan, window, ctx: ExecContext) -> str:
    if ctx.lang != "he":
        return window.label
    match plan.time_preset:
        case "last_calendar_year" | "specific_year":
            return f"שנת {window.start.year}"
        case "year_to_date":
            return f"{window.start.year} מתחילת השנה"
        case "trailing_12_months":
            return "12 החודשים האחרונים"
        case "last_winter":
            return f"חורף {window.start.year}–{str(window.end.year)[2:]} (דצמבר–פברואר)"
    return f"{window.start} עד {window.end}"


async def _stat(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    metric = ctx.cfg.stats[plan.metric]
    hub_ids = list(plan.hubs or [])
    try:
        window = resolve_window(plan.time_preset, today=ctx.analyzer.today, data_end=ctx.analyzer.data_end(),
                                data_start=ARCHIVE_START, year=plan.year, start=plan.start_date, end=plan.end_date)
    except WindowError as exc:
        data_end = ctx.analyzer.data_end()
        requested = {"specific_year": (plan.year and f"{plan.year}-01-01", plan.year and f"{plan.year}-12-31"),
                     "custom": (plan.start_date and plan.start_date.isoformat(),
                                plan.end_date and plan.end_date.isoformat())}.get(plan.time_preset, (None, None))
        return ResultBundle(
            "stat", ctx.tr(f"I can't compute that window: {exc}.",
                           f"אי אפשר לחשב את חלון הזמן הזה: יש נתונים נצפים רק מ-{ARCHIVE_START} עד {data_end}."),
            details={"available_data": f"{ARCHIVE_START} to {data_end}"},
            windows=[{"label": ctx.tr("Requested window", "חלון הזמן המבוקש"), "start": requested[0], "end": requested[1]},
                     {"label": ctx.tr("Observed data available", "נתונים נצפים זמינים"),
                      "start": ARCHIVE_START.isoformat(), "end": data_end.isoformat()}],
            sources=["open_meteo_archive"],
            uncertainty=[("high", ctx.tr(str(exc), f"חלון הזמן המבוקש אינו בטווח הנתונים או אינו תקין ({exc})."))],
            assumptions=[ctx.tr(f"Only observed days can be counted: the Open-Meteo ERA5 archive runs from "
                                f"{ARCHIVE_START.year} and lags real time by about {ARCHIVE_LAG_DAYS} days. "
                                "Forecasts are not observations.",
                                f"אפשר לספור רק ימים שנצפו: ארכיון Open-Meteo ERA5 מתחיל ב-{ARCHIVE_START.year} "
                                f"ומפגר בכ-{ARCHIVE_LAG_DAYS} ימים אחרי הזמן האמיתי. תחזיות אינן תצפיות.")],
            follow_ups=[ctx.tr("What percentage of days in Denver last year had snowfall?",
                               "באיזה אחוז מהימים בשנה שעברה ירד שלג ב-Denver?")])
    label = _window_label(plan, window, ctx)
    short = stat_label(metric, ctx.lang, short=True)
    sens_labels = {t.label: stat_label(t, ctx.lang, variable=metric.variable) for t in metric.sensitivity}
    rows, sensitivity, grid_notes, uncertainty, gaps, partial = [], {}, [], [], [], {}
    for h in hub_ids:
        daily, hub_gaps = await ctx.analyzer.daily(h, window.start, window.end)
        gaps += hub_gaps
        cov = weather_coverage(daily, window.start, window.end, [metric.variable], ctx.cfg.coverage)
        if not cov["complete"]:  # partial record: no percentage, say why (coverage rule, D24)
            partial[h] = " and ".join(cov["problems"])
            rows.append({"hub": ctx.name(h), "days meeting threshold": None, "days with data": cov["days_with_data"],
                         "% of days": None, "coverage %": cov["pct"],
                         "note": ctx.tr("Partial: incomplete data, no percentage given",
                                        "חלקי: נתונים לא מלאים, לא ניתן אחוז")})
            uncertainty.append(("high", ctx.tr(f"{ctx.name(h)}: the weather record for {label} is incomplete "
                                               f"({partial[h]}), so no share of days is given. {_rule(ctx)}",
                                               f"{ctx.name(h)}: רשומת מזג האוויר עבור {label} אינה מלאה "
                                               f"({partial[h]}), ולכן לא ניתן אחוז ימים.{_rule(ctx)}")))
            continue
        s = day_stat(daily, window.start, window.end, metric)
        rows.append({"hub": ctx.name(h), "days meeting threshold": s["count"], "days with data": s["days_with_data"],
                     "% of days": s["pct"], "coverage %": cov["pct"]})
        sensitivity[ctx.name(h)] = [x | {"label": sens_labels.get(x["label"], x["label"])} for x in s["sensitivity"]]
        pcts = [x["pct"] for x in s["sensitivity"] if x["pct"]]
        if len(pcts) >= 2 and max(pcts) >= 2 * min(pcts):
            lo = min(sensitivity[ctx.name(h)], key=lambda x: x["pct"] or 0)
            hi = max(sensitivity[ctx.name(h)], key=lambda x: x["pct"] or 0)
            uncertainty.append(("medium", ctx.tr(f"{ctx.name(h)}: the threshold matters — {hi['label']} gives "
                                                 f"{hi['pct']}% of days, {lo['label']} gives {lo['pct']}%.",
                                                 f"{ctx.name(h)}: הסף משנה — {hi['label']} נותן {hi['pct']}% "
                                                 f"מהימים, {lo['label']} נותן {lo['pct']}%.")))
        cell = await ctx.analyzer.weather.grid_cell(ctx.hubs[h])
        if cell:
            hub = ctx.hubs[h]
            km = _km(hub.lat, hub.lon, cell["latitude"], cell["longitude"])
            grid_notes.append(ctx.tr(f"{ctx.name(h)}: values are for the reanalysis grid cell at {cell['latitude']:.2f}, "
                                     f"{cell['longitude']:.2f} ({km:.0f} km from the hub point).",
                                     f"{ctx.name(h)}: הערכים הם של תא הרשת בנתוני הניתוח-החוזר ב-{cell['latitude']:.2f}, "
                                     f"{cell['longitude']:.2f} ({km:.0f} ק״מ מנקודת המרכז)."))
    answered = [r for r in rows if r["% of days"] is not None]
    if not answered:
        problems = "; ".join(partial.values())
        headline = ctx.tr(f"I can't give a reliable share of days with {short} at {_names(ctx, hub_ids)} in "
                          f"{label}: the weather record is incomplete ({problems}).",
                          f"אי אפשר לתת אחוז ימים אמין עם {short} ב-{_names(ctx, hub_ids)} ({label}): "
                          f"רשומת מזג האוויר אינה מלאה ({problems}).")
    elif len(rows) == 1:
        first = rows[0]
        headline = ctx.tr(f"{first['% of days']}% of days at {first['hub']} in {label} had {short} "
                          f"({first['days meeting threshold']} of {first['days with data']} days).",
                          f"ב-{first['% of days']}% מהימים ב-{first['hub']} ({label}) היה {short} "
                          f"({first['days meeting threshold']} מתוך {first['days with data']} ימים).")
    else:
        parts = ", ".join(f"{r['hub']} {r['% of days']}% ({r['days meeting threshold']} {ctx.tr('days', 'ימים')})"
                          for r in answered)
        headline = ctx.tr(f"Share of days in {label} with {short}: {parts}.", f"אחוז הימים עם {short} ({label}): {parts}.")
        if partial:
            headline += ctx.tr(f" {_names(ctx, list(partial))}: incomplete data, no percentage given.",
                               f" {_names(ctx, list(partial))}: נתונים לא מלאים, לא ניתן אחוז.")
    notes = window.notes if ctx.lang != "he" else [
        f"החלון נחתך ל-{window.end} — ארכיון הניתוח-החוזר מפגר בכ-{ARCHIVE_LAG_DAYS} ימים אחרי הזמן האמיתי."
        for _ in window.notes]
    assumptions = [
        ctx.tr(f"Definition: {metric.label}.", f"הגדרה: {stat_label(metric, 'he')}."),
        ctx.tr("Open-Meteo ERA5 reanalysis is modeled, not station-measured. It smooths daily extremes and can differ "
               "from airport station records by several days a year, so treat counts as estimates.",
               "נתוני הניתוח-החוזר (reanalysis) של Open-Meteo ERA5 הם תוצר של מודל ולא מדידת תחנה. הם מחליקים "
               "קיצונים יומיים ויכולים להיות שונים מרשומות תחנות שדה התעופה בכמה ימים בשנה, ולכן יש להתייחס "
               "לספירות כהערכות."),
        *grid_notes, *notes,
    ]
    return ResultBundle(
        intent="stat", headline=headline, rows=rows,
        details={"metric": stat_label(metric, ctx.lang), "window": label, "sensitivity": sensitivity},
        points=[f"{name}: " + "; ".join(f"{x['label']} → {x['pct']}%" for x in rows_) for name, rows_ in sensitivity.items()],
        windows=[{"label": label, "start": window.start.isoformat(), "end": window.end.isoformat()}],
        sources=["open_meteo_archive"], data_gaps=gaps, uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=[ctx.tr("And the year before?", "ומה בשנה שלפני?"),
                    ctx.tr("How does that compare to Chicago?", "איך זה בהשוואה ל-Chicago?")],
    )


async def _explain(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    hub = plan.hubs[0]
    exposure = await ctx.analyzer.exposure()  # all hubs: ranks and medians need the full roster
    bds = exposure.breakdowns
    bd = bds[hub]
    families = _families(plan, ctx)
    not_ranked = ctx.tr("not ranked", "לא דורג")

    def position(i: int, n: int) -> str:
        return ctx.tr(f"{i} of {n}", f"{i} מתוך {n}")

    # Ranks and medians use only hubs with a score: an incomplete hub is not ranked (coverage rule).
    ranked_overall = sorted((h for h in bds if bds[h]["score"] is not None), key=lambda h: -bds[h]["score"])
    overall_rank = position(ranked_overall.index(hub) + 1, len(ranked_overall)) if hub in ranked_overall \
        else not_ranked

    def family_rank(f: str) -> str:
        scored = sorted((h for h in bds if bds[h]["families"][f]["score"] is not None),
                        key=lambda h: -bds[h]["families"][f]["score"])
        return position(scored.index(hub) + 1, len(scored)) if hub in scored else not_ranked

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
    name, phrase = ctx.name(hub), _hazard_phrase(families, ctx)
    top_score, top_rank = r1(bd["families"][top]["score"]), family_rank(top)
    if _blocked(bd, families):
        why = _why_unscored(ctx, bd, families)
        years = f"{exposure.start.year}–{exposure.end.year}"
        headline = ctx.tr(f"{name} is not scored on {phrase} for {years}: {why}, so it is not ranked.",
                          f"{name} לא קיבל ציון {phrase} לשנים {years}: {why}, ולכן הוא לא מדורג.")
        scored = [f for f in shown if bd["families"][f]["score"] is not None]
        if scored:
            headline += ctx.tr(" Hazards that could be scored: ", " סיכונים שכן קיבלו ציון: ") + ", ".join(
                f"{ctx.hazard(f)} {r1(bd['families'][f]['score'])}/100" for f in scored) + "."
        uncertainty.insert(0, _blocked_reason(ctx, hub, bd, families, "not scored"))
    elif len(families) < len(ctx.cfg.families):
        combined = r1(overall_score(bd, ctx.cfg.family_weights, families))
        headline = ctx.tr(f"{name} scores {combined}/100 on {phrase}; the biggest "
                          f"driver is {ctx.hazard(top)} ({top_score}/100, rank {top_rank}).",
                          f"{name} מקבל {combined}/100 {phrase}; הגורם המוביל הוא {ctx.hazard(top)} "
                          f"({top_score}/100, דירוג {top_rank}).")
    else:
        headline = ctx.tr(f"{name} has an overall exposure score of {r1(bd['score'])}/100 "
                          f"(rank {overall_rank}). The largest contributor is "
                          f"{ctx.short_label(top)} at {top_score}/100 (rank {top_rank}).",
                          f"ל-{name} ציון חשיפה כולל של {r1(bd['score'])}/100 (דירוג {overall_rank}). "
                          f"התורם הגדול ביותר הוא {ctx.hazard(top)} עם {top_score}/100 (דירוג {top_rank}).")
    other = next((h for h in ranked_overall if h != hub), None)
    return ResultBundle(
        intent="explain", headline=headline, rows=rows,
        details={"hub": f"{name} ({bd.get('meta', {}).get('county', ctx.hubs[hub].county)})",
                 "overall_score": r1(bd["score"]), "overall_rank": overall_rank,
                 "hazards": {ctx.short_label(f): _family_detail(bd, f, ctx) for f in shown},
                 "top_drivers": _drivers(bd, families, ctx, 4)},
        points=_drivers(bd, families, ctx, 4) or [r for _, r in uncertainty[:1]], windows=windows, sources=sources,
        data_gaps=gaps, uncertainty=uncertainty, assumptions=assumptions,
        follow_ups=[ctx.tr(f"Which evidence drives {name}'s {ctx.hazard(top)} score?",
                           f"אילו ראיות מניעות את ציון ה{ctx.hazard(top)} של {name}?")]
        + ([ctx.tr(f"Compare {name} with {ctx.name(other)}", f"השווה את {name} עם {ctx.name(other)}")]
           if other else []),
        orders=_orders(ctx, bds, list(bds), families,
                       {h: s for h in bds if (s := overall_score(bds[h], ctx.cfg.family_weights, families)) is not None}),
    )


async def _no_compute(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    text = plan.clarification_question if plan.intent == "clarify" else plan.out_of_scope_reason
    capabilities = CAPABILITIES_HE if ctx.lang == "he" else CAPABILITIES
    can = "; ".join(c[0].lower() + c[1:] for c in capabilities[:3]) if ctx.lang != "he" \
        else "; ".join(capabilities[:3])
    return ResultBundle(
        intent=plan.intent, headline=text or "",
        details={"what_i_can_answer": capabilities, "hubs": [h.name for h in ctx.hubs.values()]},
        points=[ctx.tr(f"I can answer: {can}.", f"אני יכול לענות על: {can}.")],
        follow_ups=[ctx.tr("Which hubs in the Midwest are most exposed to winter disruption?",
                           "אילו מרכזים במערב התיכון הכי חשופים לשיבושי חורף?"),
                    ctx.tr("Compare Miami and Houston on hurricane and flood exposure",
                           "השווה בין Miami ל-Houston בחשיפה להוריקנים ולהצפות")],
    )


HANDLERS = {"rank": _rank, "compare": _compare, "stat": _stat, "explain": _explain,
            "clarify": _no_compute, "out_of_scope": _no_compute}


async def execute(plan: QueryPlan, ctx: ExecContext) -> ResultBundle:
    return await HANDLERS[plan.intent](plan, ctx)
