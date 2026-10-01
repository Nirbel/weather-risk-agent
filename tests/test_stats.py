"""'% of days' statistics and time-window resolution — written test-first, values worked out by hand."""

from datetime import date, timedelta

import pytest

from weather_risk.scoring.stats import day_stat
from weather_risk.sources.open_meteo import DailyRow
from weather_risk.timewindow import WindowError, resolve_window

SNOW_METRIC = {
    "label": "Days with measurable snowfall",
    "variable": "snowfall_cm",
    "op": "ge",
    "threshold": 0.25,
    "sensitivity": [
        {"label": "Any snowfall (> 0 cm)", "op": "gt", "threshold": 0},
        {"label": "Measurable", "op": "ge", "threshold": 0.25},
        {"label": "≥ 2.5 cm", "op": "ge", "threshold": 2.5},
    ],
}


def snow_rows(values):
    start = date(2025, 1, 1)
    return [DailyRow(start + timedelta(days=i), v, 0.0, 0.0, 0.0, 0.0) for i, v in enumerate(values)]


def test_day_stat_counts_percent_and_coverage():
    # 10-day window, one day missing → 9 valid days.
    # measurable (≥0.25): 0.25, 0.3, 2.5, 3 → 4 of 9 = 44.4 %
    values = [0, 0.1, 0.25, 0.3, 2.5, 3, None, 0, 0, 0]
    s = day_stat(snow_rows(values), date(2025, 1, 1), date(2025, 1, 10), SNOW_METRIC)
    assert s["count"] == 4
    assert s["days_with_data"] == 9
    assert s["days_in_window"] == 10
    assert s["pct"] == pytest.approx(44.4, abs=0.05)
    assert s["coverage_pct"] == pytest.approx(90.0)


def test_day_stat_threshold_sensitivity():
    # any >0: 0.1, 0.25, 0.3, 2.5, 3 → 5/9 = 55.6 %;  ≥2.5: 2.5, 3 → 2/9 = 22.2 %
    values = [0, 0.1, 0.25, 0.3, 2.5, 3, None, 0, 0, 0]
    s = day_stat(snow_rows(values), date(2025, 1, 1), date(2025, 1, 10), SNOW_METRIC)
    by_label = {row["label"]: row for row in s["sensitivity"]}
    assert by_label["Any snowfall (> 0 cm)"]["count"] == 5
    assert by_label["Any snowfall (> 0 cm)"]["pct"] == pytest.approx(55.6, abs=0.05)
    assert by_label["≥ 2.5 cm"]["pct"] == pytest.approx(22.2, abs=0.05)


def test_day_stat_ignores_rows_outside_window():
    values = [5.0] * 10
    s = day_stat(snow_rows(values), date(2025, 1, 3), date(2025, 1, 4), SNOW_METRIC)
    assert (s["count"], s["days_in_window"]) == (2, 2)


def test_day_stat_with_no_data():
    s = day_stat([], date(2025, 1, 1), date(2025, 1, 10), SNOW_METRIC)
    assert s["pct"] is None and s["coverage_pct"] == 0


# -- time windows (today 2026-10-01, archive has data through 2026-09-25) -----------

TODAY, DATA_END = date(2026, 10, 1), date(2026, 9, 25)


def window(preset, **kw):
    return resolve_window(preset, today=TODAY, data_end=DATA_END, **kw)


def test_last_calendar_year():
    w = window("last_calendar_year")
    assert (w.start, w.end) == (date(2025, 1, 1), date(2025, 12, 31))
    assert "2025" in w.label


def test_last_winter_after_march():
    w = window("last_winter")
    assert (w.start, w.end) == (date(2025, 12, 1), date(2026, 2, 28))


def test_last_winter_in_february_means_previous_one():
    w = resolve_window("last_winter", today=date(2026, 2, 10), data_end=date(2026, 2, 4))
    assert (w.start, w.end) == (date(2024, 12, 1), date(2025, 2, 28))


def test_trailing_12_months_ends_at_data_end():
    w = window("trailing_12_months")
    assert (w.start, w.end) == (date(2025, 9, 26), DATA_END)


def test_year_to_date_and_current_specific_year_are_clipped():
    assert (window("year_to_date").start, window("year_to_date").end) == (date(2026, 1, 1), DATA_END)
    w = window("specific_year", year=2026)
    assert w.end == DATA_END and w.notes  # clipping is disclosed


def test_custom_range_validated():
    w = window("custom", start=date(2024, 3, 1), end=date(2024, 3, 31))
    assert (w.start, w.end) == (date(2024, 3, 1), date(2024, 3, 31))
    with pytest.raises(WindowError):
        window("custom", start=date(2024, 3, 31), end=date(2024, 3, 1))
    with pytest.raises(WindowError):
        window("custom", start=date(2027, 1, 1), end=date(2027, 2, 1))  # future
    with pytest.raises(WindowError):
        window("specific_year", year=None)


def test_window_before_stored_data_is_rejected():
    with pytest.raises(WindowError, match="2016-01-01"):
        resolve_window("specific_year", today=TODAY, data_end=DATA_END, data_start=date(2016, 1, 1), year=2010)
