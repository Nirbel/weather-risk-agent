"""Small pure helpers shared by every score: normalization and threshold counting."""

import bisect
import operator
from collections.abc import Sequence

from weather_risk.sources.open_meteo import DailyRow

OPS = {"ge": operator.ge, "gt": operator.gt, "le": operator.le, "lt": operator.lt}


def piecewise(x: float, points: Sequence[Sequence[float]]) -> float:
    """Linear interpolation between (value, score) points sorted by value; flat outside the range."""
    xs = [p[0] for p in points]
    if x <= xs[0]:
        return float(points[0][1])
    if x >= xs[-1]:
        return float(points[-1][1])
    i = bisect.bisect_right(xs, x)
    (x0, y0), (x1, y1) = points[i - 1], points[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def anchor_score(value: float, full_score_at: float) -> float:
    """0 → 0, `full_score_at` → 100, capped (fixed anchors, not min-max — D11)."""
    return piecewise(value, [[0, 0], [full_score_at, 100]])


def exceeds(value: float, op: str, threshold: float) -> bool:
    return OPS[op](value, threshold)


def days_per_year(rows: list[DailyRow], variable: str, op: str, threshold: float) -> tuple[float | None, int]:
    """Mean qualifying days per year over days that have data. Returns (rate, n_valid_days)."""
    values = [getattr(r, variable) for r in rows]
    valid = [v for v in values if v is not None]
    if not valid:
        return None, 0
    count = sum(1 for v in valid if exceeds(v, op, threshold))
    return count * 365.25 / len(valid), len(valid)
