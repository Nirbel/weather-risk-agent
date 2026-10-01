"""Hub roster (config/hubs.yaml) and scoring config (config/scoring.yaml) loaders."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from weather_risk.settings import ROOT

CONFIG_DIR = ROOT / "config"
REGIONS = ("midwest", "south", "west", "northeast")


@dataclass(frozen=True)
class Hub:
    id: str
    name: str
    state: str
    region: str
    lat: float
    lon: float
    county_fips: str
    county: str

    @property
    def state_fips(self) -> str:
        return self.county_fips[:2]

    @property
    def county_code(self) -> str:
        return self.county_fips[2:]


@lru_cache
def load_hubs(path: Path = CONFIG_DIR / "hubs.yaml") -> tuple[Hub, ...]:
    data = yaml.safe_load(path.read_text())
    hubs = tuple(Hub(**{**h, "county_fips": str(h["county_fips"])}) for h in data["hubs"])
    for hub in hubs:
        if hub.region not in REGIONS:
            raise ValueError(f"{hub.id}: unknown region {hub.region!r}")
        if len(hub.county_fips) != 5 or not hub.county_fips.isdigit():
            raise ValueError(f"{hub.id}: county_fips must be 5 digits")
    return hubs


def hubs_by_id() -> dict[str, Hub]:
    return {h.id: h for h in load_hubs()}


@lru_cache
def load_scoring_config(path: Path = CONFIG_DIR / "scoring.yaml") -> dict[str, Any]:
    return yaml.safe_load(path.read_text())
