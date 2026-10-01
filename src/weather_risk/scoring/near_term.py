"""Near-term Risk Score (next 7 days, drives alerts) — docs/DECISIONS.md D10.

family = max(worst forecast day, worst active NWS alert × certainty); overall = max over families,
because operations are disrupted by the worst hazard this week.
"""

from weather_risk.scoring.indicators import piecewise


def band_for(score: float, bands: list[dict]) -> str:
    current = bands[0]["band"]
    for b in bands:
        if score >= b["min"]:
            current = b["band"]
    return current


def family_for_event(event: str, nws_events: dict[str, list[str]]) -> str | None:
    for family, keywords in nws_events.items():
        if any(k.lower() in event.lower() for k in keywords):
            return family
    return None


def compute_near_term(forecast_rows: list[dict], alerts: list[dict], cfg: dict) -> dict:
    names = list(dict.fromkeys([*cfg["forecast"], *cfg["nws_events"]]))
    families = {f: {"score": 0.0, "forecast_score": 0.0, "alert_score": 0.0, "drivers": []} for f in names}

    for family, rules in cfg["forecast"].items():
        best, driver = 0.0, None
        for rule in rules:
            for row in forecast_rows:
                value = row.get(rule["variable"])
                if value is None:
                    continue
                score = piecewise(value, rule["points"])
                if score > best:
                    best, driver = score, f"{row['date']}: {rule['label']} {value:g} → {score:.0f} pts"
        families[family]["forecast_score"] = best
        if driver:
            families[family]["drivers"].append(driver)

    unmapped = []
    for alert in alerts:
        family = family_for_event(alert["event"], cfg["nws_events"])
        if family is None:
            unmapped.append(f"{alert['event']} ({alert['severity']})")
            continue
        severity = cfg["nws_severity"].get(alert["severity"], cfg["nws_severity"]["Unknown"])
        certainty = cfg.get("nws_certainty", {}).get(alert.get("certainty", "Unknown"), 1.0)
        score = round(severity * certainty, 1)
        families[family]["alert_score"] = max(families[family]["alert_score"], score)
        until = f" until {alert['expires']}" if alert.get("expires") else ""
        families[family]["drivers"].append(
            f"NWS {alert['event']} ({alert['severity']}, {alert.get('certainty', 'Unknown')}){until} → {score:g} pts")

    for fam in families.values():
        fam["score"] = max(fam["forecast_score"], fam["alert_score"])
    top = max(families, key=lambda f: families[f]["score"])
    score = families[top]["score"]
    return {
        "score": score,
        "band": band_for(score, cfg["bands"]),
        "top_family": top if score > 0 else None,
        "families": families,
        "unmapped_alerts": unmapped,
        "horizon_days": len(forecast_rows),
    }
