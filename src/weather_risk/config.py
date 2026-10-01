"""Typed models for config/hubs.yaml and config/scoring.yaml, validated at startup.

A bad config fails fast with a readable ConfigError instead of a KeyError mid-request.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from weather_risk.settings import ROOT

CONFIG_DIR = ROOT / "config"

Region = Literal["midwest", "south", "west", "northeast"]
Variable = Literal["snowfall_cm", "precip_mm", "tmax_c", "tmin_c", "gust_kmh"]
Op = Literal["ge", "gt", "le", "lt"]
NriCode = Literal["WNTW", "ISTM", "CWAV", "HRCN", "IFLD", "CFLD", "TRND", "HAIL", "SWND", "HWAV"]


class ConfigError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# -- hubs.yaml ---------------------------------------------------------------------

class Hub(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z_]*$")
    name: str
    state: str = Field(pattern=r"^[A-Z]{2}$")
    region: Region
    lat: float = Field(ge=18, le=72)
    lon: float = Field(ge=-180, le=-60)
    county_fips: str = Field(pattern=r"^\d{5}$")
    county: str

    @property
    def state_fips(self) -> str:
        return self.county_fips[:2]

    @property
    def county_code(self) -> str:
        return self.county_fips[2:]


class HubsFile(_Strict):
    hubs: list[Hub] = Field(min_length=1)

    @field_validator("hubs")
    @classmethod
    def _unique(cls, hubs: list[Hub]) -> list[Hub]:
        ids = [h.id for h in hubs]
        if dupes := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"duplicate hub id(s): {', '.join(dupes)}")
        return hubs


# -- scoring.yaml ------------------------------------------------------------------

class LensWeights(_Strict):
    observed: float = Field(ge=0)
    modeled: float = Field(ge=0)

    @model_validator(mode="after")
    def _positive(self) -> "LensWeights":
        if self.observed + self.modeled <= 0:
            raise ValueError("lens weights must not all be zero")
        return self


class ObservedIndicator(_Strict):
    id: str
    label: str
    variable: Variable
    op: Op
    threshold: float
    full_score_at: float = Field(gt=0)


class ModeledLens(_Strict):
    nri: list[NriCode] = Field(min_length=1)
    combine: Literal["mean", "max"]


class Family(_Strict):
    label: str
    lens_weights: LensWeights | None = None
    observed: list[ObservedIndicator] = []
    modeled: ModeledLens | None = None

    @model_validator(mode="after")
    def _has_evidence(self) -> "Family":
        if not self.observed and self.modeled is None:
            raise ValueError("a family needs at least one evidence lens (observed or modeled)")
        return self


class Threshold(_Strict):
    label: str
    op: Op
    threshold: float


class StatMetric(_Strict):
    short: str
    label: str
    variable: Variable
    op: Op
    threshold: float
    sensitivity: list[Threshold] = []


class Robustness(_Strict):
    perturbation: float = Field(gt=0, lt=1)
    tie_margin: float = Field(ge=0)


class Climatology(_Strict):
    years: int = Field(ge=1, le=10)


class ScoringConfig(_Strict):
    climatology: Climatology
    family_weights: dict[str, float]
    lens_weights: LensWeights
    families: dict[str, Family]
    nri_hazard_names: dict[NriCode, str]
    robustness: Robustness
    stats: dict[str, StatMetric]

    @model_validator(mode="after")
    def _consistent(self) -> "ScoringConfig":
        if set(self.family_weights) != set(self.families):
            raise ValueError("family_weights must list exactly the families "
                             f"({', '.join(self.families)}); got {', '.join(self.family_weights)}")
        if any(w < 0 for w in self.family_weights.values()) or sum(self.family_weights.values()) <= 0:
            raise ValueError("family_weights must be non-negative and not all zero")
        used = {code for fam in self.families.values() if fam.modeled for code in fam.modeled.nri}
        if missing := used - set(self.nri_hazard_names):
            raise ValueError(f"nri_hazard_names lacks {', '.join(sorted(missing))}")
        return self

    def lens_weights_for(self, family: str) -> LensWeights:
        return self.families[family].lens_weights or self.lens_weights


# -- loading -----------------------------------------------------------------------

def _parse(model: type[BaseModel], data: dict, name: str):
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
        raise ConfigError(f"{name} is invalid — {problems}") from exc


def parse_hubs(data: dict) -> tuple[Hub, ...]:
    return tuple(_parse(HubsFile, data, "config/hubs.yaml").hubs)


def parse_scoring(data: dict) -> ScoringConfig:
    return _parse(ScoringConfig, data, "config/scoring.yaml")


@lru_cache
def load_hubs(path: Path = CONFIG_DIR / "hubs.yaml") -> tuple[Hub, ...]:
    return parse_hubs(yaml.safe_load(path.read_text()))


@lru_cache
def load_scoring_config(path: Path = CONFIG_DIR / "scoring.yaml") -> ScoringConfig:
    return parse_scoring(yaml.safe_load(path.read_text()))


def hubs_by_id() -> dict[str, Hub]:
    return {h.id: h for h in load_hubs()}


def validate_config() -> tuple[tuple[Hub, ...], ScoringConfig]:
    """Called at startup (API lifespan, CLIs): fail fast on a bad config."""
    return load_hubs(), load_scoring_config()
