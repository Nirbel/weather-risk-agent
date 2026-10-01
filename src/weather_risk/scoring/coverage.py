"""Is a hub's weather record complete enough to score? (config/scoring.yaml → coverage)

A day has data only if every variable the answer needs is present. A window is complete if at
least `min_pct` of its days have data and no run of missing days is longer than `max_gap_days`.
"""

import math
from collections.abc import Iterable
from datetime import date, timedelta

from weather_risk.config import CoverageRule
from weather_risk.sources.open_meteo import DailyRow


def weather_coverage(rows: Iterable[DailyRow], start: date, end: date, variables: Iterable[str],
                     rule: CoverageRule) -> dict:
    variables = tuple(variables)
    have = {r.date for r in rows
            if start <= r.date <= end and all(getattr(r, v) is not None for v in variables)}
    expected = (end - start).days + 1
    longest, longest_start, run = 0, None, 0
    for i in range(expected):
        day = start + timedelta(days=i)
        run = 0 if day in have else run + 1
        if run > longest:
            longest, longest_start = run, day - timedelta(days=run - 1)
    with_data = len(have)
    pct = math.floor(1000.0 * with_data / expected) / 10  # floored: 98.96 % shows as 98.9, never "99.0"
    gap = None if longest_start is None else {
        "start": longest_start.isoformat(), "end": (longest_start + timedelta(days=longest - 1)).isoformat()}
    problems = []
    if 100 * with_data < rule.min_pct * expected:
        problems.append(f"{pct}% of days have data (at least {rule.min_pct:g}% needed)")
    if longest > rule.max_gap_days:
        problems.append(f"a {longest}-day gap with no data ({gap['start']} to {gap['end']}; "
                        f"at most {rule.max_gap_days} allowed)")
    return {"expected_days": expected, "days_with_data": with_data, "pct": pct, "longest_gap_days": longest,
            "longest_gap": gap, "complete": not problems, "problems": problems}
