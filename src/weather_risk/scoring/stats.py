"""Deterministic "% of days" statistics over a date window."""

from datetime import date

from weather_risk.scoring.indicators import exceeds
from weather_risk.sources.open_meteo import DailyRow


def _count(values: list[float], op: str, threshold: float) -> int:
    return sum(1 for v in values if exceeds(v, op, threshold))


def _pct(count: int, n: int) -> float | None:
    return round(100.0 * count / n, 1) if n else None


def day_stat(rows: list[DailyRow], start: date, end: date, metric: dict) -> dict:
    """Count days meeting the metric's threshold in [start, end], plus a threshold-sensitivity table."""
    in_window = [r for r in rows if start <= r.date <= end]
    values = [v for r in in_window if (v := getattr(r, metric["variable"])) is not None]
    days_in_window = (end - start).days + 1
    count = _count(values, metric["op"], metric["threshold"])
    return {
        "metric_label": metric["label"],
        "variable": metric["variable"],
        "op": metric["op"],
        "threshold": metric["threshold"],
        "count": count,
        "days_with_data": len(values),
        "days_in_window": days_in_window,
        "pct": _pct(count, len(values)),
        "coverage_pct": round(100.0 * len(values) / days_in_window, 1),
        "sensitivity": [
            {"label": s["label"], "count": (c := _count(values, s["op"], s["threshold"])), "pct": _pct(c, len(values))}
            for s in metric.get("sensitivity", [])
        ],
    }
