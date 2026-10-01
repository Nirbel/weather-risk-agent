"""Weather-data coverage rule — written test-first, values worked out by hand.

A window is complete only if ≥ 99 % of its days have data AND no run of missing days is longer than 3.
"""

from datetime import date, timedelta

from weather_risk.config import CoverageRule
from weather_risk.scoring.coverage import weather_coverage
from weather_risk.sources.open_meteo import DailyRow

RULE = CoverageRule(min_pct=99, max_gap_days=3)
START = date(2024, 1, 1)
VARS = ("snowfall_cm", "tmin_c")


def days(n: int, *, missing: set[int] = frozenset(), null_gust: bool = False, null_snow: set[int] = frozenset()):
    """n consecutive days from START; `missing` day offsets have no row at all."""
    return [DailyRow(START + timedelta(days=i), None if i in null_snow else 0.0, 0.0, 10.0, -1.0,
                     None if null_gust else 5.0)
            for i in range(n) if i not in missing]


def end(n: int) -> date:
    return START + timedelta(days=n - 1)


def test_full_window_is_complete():
    c = weather_coverage(days(10), START, end(10), VARS, RULE)
    assert (c["expected_days"], c["days_with_data"], c["pct"], c["longest_gap_days"]) == (10, 10, 100.0, 0)
    assert c["complete"] and c["longest_gap"] is None and c["problems"] == []


def test_exactly_99_percent_with_short_gaps_is_complete():
    # 400 days, 4 isolated missing days → 396/400 = 99.0 %, longest gap 1
    c = weather_coverage(days(400, missing={10, 50, 90, 130}), START, end(400), VARS, RULE)
    assert (c["days_with_data"], c["pct"], c["longest_gap_days"]) == (396, 99.0, 1)
    assert c["complete"]


def test_below_99_percent_is_incomplete_and_the_percent_is_not_rounded_up():
    # 5 isolated missing days → 395/400 = 98.75 % → shown as 98.7 (floored, never "99.0")
    c = weather_coverage(days(400, missing={10, 50, 90, 130, 170}), START, end(400), VARS, RULE)
    assert c["pct"] == 98.7 and not c["complete"]
    assert c["problems"] == ["98.7% of days have data (at least 99% needed)"]


def test_gap_of_three_days_is_allowed_four_is_not():
    three = weather_coverage(days(400, missing={100, 101, 102}), START, end(400), VARS, RULE)
    assert three["complete"] and three["longest_gap_days"] == 3  # 397/400 = 99.25 %
    # 396/400 = 99.0 % passes the percentage, but the 4-day gap (Apr 10–13, 2024) fails
    four = weather_coverage(days(400, missing={100, 101, 102, 103}), START, end(400), VARS, RULE)
    assert four["pct"] == 99.0 and four["longest_gap_days"] == 4 and not four["complete"]
    assert four["longest_gap"] == {"start": "2024-04-10", "end": "2024-04-13"}
    assert four["problems"] == ["a 4-day gap with no data (2024-04-10 to 2024-04-13; at most 3 allowed)"]


def test_a_day_counts_only_if_every_required_variable_is_present():
    # snowfall missing on 2 days → those days count as missing; gust is not required here
    c = weather_coverage(days(10, null_gust=True, null_snow={3, 4}), START, end(10), VARS, RULE)
    assert (c["days_with_data"], c["longest_gap_days"]) == (8, 2)


def test_rows_outside_the_window_are_ignored_and_a_trailing_gap_counts():
    # window = first 10 days; rows exist for days 0–5 and 20–29 → days 6–9 missing at the end
    rows = [r for r in days(30) if not 6 <= (r.date - START).days <= 19]
    c = weather_coverage(rows, START, end(10), VARS, RULE)
    assert (c["days_with_data"], c["longest_gap_days"]) == (6, 4)
    assert c["longest_gap"] == {"start": "2024-01-07", "end": "2024-01-10"}


def test_no_data_at_all():
    c = weather_coverage([], START, end(10), VARS, RULE)
    assert (c["days_with_data"], c["pct"], c["longest_gap_days"], c["complete"]) == (0, 0.0, 10, False)
    assert len(c["problems"]) == 2
