"""Typed validation of config/hubs.yaml and config/scoring.yaml — written test-first."""

from datetime import date

import pytest
import yaml

from weather_risk.config import ConfigError, load_hubs, load_scoring_config, parse_hubs, parse_scoring
from weather_risk.timewindow import climatology_window


def scoring_dict() -> dict:
    return yaml.safe_load(open("config/scoring.yaml"))


def hubs_dict() -> dict:
    return yaml.safe_load(open("config/hubs.yaml"))


# -- the real files ---------------------------------------------------------------

def test_real_config_files_validate():
    hubs = load_hubs()
    scoring = load_scoring_config()
    assert 8 <= len(hubs) <= 10
    assert {"denver", "dallas", "houston", "miami"} <= {h.id for h in hubs}
    assert sum(h.region == "midwest" for h in hubs) >= 3
    assert set(scoring.families) == {"winter", "hurricane", "flood", "severe_storm", "heat"}
    assert scoring.climatology.years == 5
    assert scoring.lens_weights_for("winter").observed == 0.75
    assert scoring.lens_weights_for("flood").observed == 0.5  # default


def test_hub_helpers():
    houston = {h.id: h for h in load_hubs()}["houston"]
    assert (houston.state_fips, houston.county_code) == ("48", "201")


# -- hubs.yaml rules --------------------------------------------------------------

@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["hubs"].append(dict(d["hubs"][0])), "duplicate hub id"),
        (lambda d: d["hubs"][0].update(county_fips="1703"), "county_fips"),
        (lambda d: d["hubs"][0].update(region="midwestern"), "region"),
        (lambda d: d["hubs"][0].update(lat=95), "lat"),
    ],
)
def test_invalid_hubs_are_rejected(mutate, message):
    data = hubs_dict()
    mutate(data)
    with pytest.raises(ConfigError, match=message):
        parse_hubs(data)


# -- scoring.yaml rules -----------------------------------------------------------

@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["family_weights"].pop("heat"), "family_weights"),
        (lambda d: d["families"]["hurricane"]["modeled"].update(nri=["QUAKE"]), "nri"),
        (lambda d: d["families"]["hurricane"].update(modeled=None), "at least one evidence lens"),
        (lambda d: d["families"]["winter"]["observed"][0].update(full_score_at=0), "full_score_at"),
        (lambda d: d["climatology"].update(years=0), "years"),
        (lambda d: d["stats"]["snowfall_days"].update(op="approx"), "op"),
        (lambda d: d.update(unknown_section=1), "unknown_section"),
    ],
)
def test_invalid_scoring_is_rejected(mutate, message):
    data = scoring_dict()
    mutate(data)
    with pytest.raises(ConfigError, match=message):
        parse_scoring(data)


# -- default exposure window: previous N completed calendar years ------------------

def test_climatology_window_is_previous_five_completed_years():
    assert climatology_window(date(2026, 10, 1), 5) == (date(2021, 1, 1), date(2025, 12, 31))


def test_climatology_window_waits_for_archive_lag_in_early_january():
    # On 2026-01-03 the archive (≈6-day lag) does not yet hold all of 2025 → 2020–2024.
    assert climatology_window(date(2026, 1, 3), 5) == (date(2020, 1, 1), date(2024, 12, 31))
    assert climatology_window(date(2026, 1, 7), 5) == (date(2021, 1, 1), date(2025, 12, 31))
