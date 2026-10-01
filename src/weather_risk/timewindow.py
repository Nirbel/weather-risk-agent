"""Resolve a plan's time preset into concrete dates. All date math lives here, never in the LLM."""

from dataclasses import dataclass, field
from datetime import date, timedelta


class WindowError(ValueError):
    pass


@dataclass(frozen=True)
class Window:
    start: date
    end: date
    label: str
    notes: list[str] = field(default_factory=list)


ARCHIVE_LAG_DAYS = 6  # Open-Meteo ERA5 archive lags real time by about 6 days


def climatology_window(today: date, years: int) -> tuple[date, date]:
    """The previous `years` completed calendar years that the archive fully covers."""
    last = today - timedelta(days=ARCHIVE_LAG_DAYS)
    last_year = last.year if last == date(last.year, 12, 31) else last.year - 1
    return date(last_year - years + 1, 1, 1), date(last_year, 12, 31)


def archive_end(today: date) -> date:
    """Last day the reanalysis archive reliably covers."""
    return today - timedelta(days=ARCHIVE_LAG_DAYS)


def year_span(year: int, today: date) -> tuple[date, date, bool]:
    """Dates to request for one calendar year, and whether that year is complete in the archive."""
    end = min(date(year, 12, 31), archive_end(today))
    return date(year, 1, 1), end, end == date(year, 12, 31)


def _feb_end(year: int) -> date:
    return date(year, 3, 1) - timedelta(days=1)


def resolve_window(
    preset: str,
    *,
    today: date,
    data_end: date,
    data_start: date | None = None,
    year: int | None = None,
    start: date | None = None,
    end: date | None = None,
) -> Window:
    notes: list[str] = []
    match preset:
        case "last_calendar_year":
            y = today.year - 1
            w_start, w_end, label = date(y, 1, 1), date(y, 12, 31), f"calendar year {y}"
        case "specific_year":
            if year is None:
                raise WindowError("specific_year needs a year")
            w_start, w_end, label = date(year, 1, 1), date(year, 12, 31), f"calendar year {year}"
        case "year_to_date":
            w_start, w_end, label = date(today.year, 1, 1), data_end, f"{today.year} year to date"
        case "trailing_12_months":
            w_start, w_end, label = data_end - timedelta(days=364), data_end, "trailing 12 months"
        case "last_winter":
            # most recent fully completed Dec–Feb season
            end_year = today.year if today.month >= 3 else today.year - 1
            w_start, w_end = date(end_year - 1, 12, 1), _feb_end(end_year)
            label = f"winter {end_year - 1}–{str(end_year)[2:]} (Dec–Feb)"
        case "custom":
            if start is None or end is None:
                raise WindowError("custom window needs start_date and end_date")
            if start > end:
                raise WindowError(f"start {start} is after end {end}")
            w_start, w_end, label = start, end, f"{start} to {end}"
        case _:
            raise WindowError(f"unknown time preset {preset!r}")

    if w_start > data_end:
        raise WindowError(f"no observed data after {data_end} (the archive lags ~6 days; forecasts are not observations)")
    if data_start and w_start < data_start:
        raise WindowError(f"stored daily data starts on {data_start.isoformat()}")
    if w_end > data_end:
        notes.append(f"Window clipped to {data_end} — the reanalysis archive lags real time by about 6 days.")
        w_end = data_end
    return Window(w_start, w_end, label, notes)
