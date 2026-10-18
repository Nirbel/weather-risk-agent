"""Answer language. The explainer LLM writes the narrative in the question's language;
everything code writes for a reader (headlines, uncertainty, assumptions, window labels, table headers)
comes from here, so the exact parts are never left to the LLM to translate.

Only English ("en") and Hebrew ("he") are supported. Labels for observed indicators and stat metrics are
generated from their config fields (variable, operator, threshold), so a new config entry is translated
without editing this file; hazard family and FEMA NRI names have a fixed Hebrew table.
"""

import re

from weather_risk.config import ScoringConfig

HEBREW = re.compile(r"[֐-׿]")


def detect_language(text: str) -> str:
    return "he" if HEBREW.search(text or "") else "en"


FAMILY_HE = {
    "winter": ("חורף", "חורף (שלג, קרח, קור קיצוני)"),
    "hurricane": ("הוריקן", "הוריקן / סערה טרופית"),
    "flood": ("הצפה", "הצפה (נחלית, חופית, גשם כבד)"),
    "severe_storm": ("סערה קשה", "סערה קשה (טורנדו, ברד, רוח חזקה)"),
    "heat": ("חום קיצוני", "חום קיצוני"),
}
NRI_HE = {
    "WNTW": "מזג אוויר חורפי", "ISTM": "סופת קרח", "CWAV": "גל קור", "HRCN": "הוריקן",
    "IFLD": "הצפה פנים-יבשתית", "CFLD": "הצפה חופית", "TRND": "טורנדו", "HAIL": "ברד",
    "SWND": "רוח חזקה", "HWAV": "גל חום",
}
REGION_HE = {"midwest": "המערב התיכון", "south": "הדרום", "west": "המערב", "northeast": "הצפון-מזרח"}
VARIABLE_HE = {  # variable → (what, unit)
    "snowfall_cm": ("שלג", "ס״מ"), "precip_mm": ("משקעים", "מ״מ"), "tmax_c": ("טמפרטורה מרבית", "°C"),
    "tmin_c": ("טמפרטורה מזערית", "°C"), "gust_kmh": ("משבי רוח", "קמ״ש"),
}
OPS = {"ge": "≥", "gt": ">", "le": "≤", "lt": "<"}

# Table keys stay English (API contract); these are the headers the UI shows for a Hebrew answer.
COLUMNS_HE = {
    "rank": "דירוג", "hub": "מרכז", "score": "ציון", "top hazard": "הסיכון המוביל", "combined": "ציון משולב",
    "note": "הערה", "hazard": "סיכון", "contribution to overall": "תרומה לציון הכולל",
    "rank among hubs": "דירוג בין המרכזים", "median across hubs": "חציון בין המרכזים",
    "days meeting threshold": "ימים שעברו את הסף", "days with data": "ימים עם נתונים", "% of days": "% מהימים",
    "coverage %": "% כיסוי",
}


def family_label(cfg: ScoringConfig, family: str, lang: str, short: bool = False) -> str:
    if lang == "he" and family in FAMILY_HE:
        return FAMILY_HE[family][0 if short else 1]
    label = cfg.families[family].label
    return label.split(" (")[0] if short else label


def nri_name(cfg: ScoringConfig, code: str, lang: str) -> str:
    return NRI_HE.get(code, cfg.nri_hazard_names[code]) if lang == "he" else cfg.nri_hazard_names[code]


def region_name(region: str, lang: str, english: str) -> str:
    return REGION_HE.get(region, english) if lang == "he" else english


def _condition(variable: str, op: str, threshold: float) -> str:
    what, unit = VARIABLE_HE[variable]
    return f"{what} {OPS[op]} {threshold:g} {unit}"


def indicator_label(spec, lang: str) -> str:
    """An observed indicator from config ("Days with snowfall ≥ 2.5 cm (1 in)")."""
    return f"ימים עם {_condition(spec.variable, spec.op, spec.threshold)}" if lang == "he" else spec.label


def stat_label(metric, lang: str, short: bool = False, variable: str | None = None) -> str:
    """A stat metric (label or `short` phrase) or one of its sensitivity thresholds (pass `variable`)."""
    if lang != "he":
        return metric.short if short else metric.label
    condition = _condition(variable or metric.variable, metric.op, metric.threshold)
    return condition if short or variable else f"ימים עם {condition}"


def column_labels(rows: list[dict], cfg: ScoringConfig, lang: str) -> dict[str, str]:
    """Display headers for a Hebrew table: fixed keys plus hazard short labels used as columns."""
    if lang != "he":
        return {}
    by_label = {family_label(cfg, f, "en", short=True): family_label(cfg, f, "he", short=True) for f in cfg.families}
    keys = list(dict.fromkeys(k for row in rows for k in row))
    return {k: COLUMNS_HE.get(k) or by_label.get(k) or k for k in keys}
