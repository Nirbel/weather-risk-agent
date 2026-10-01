"""Build scoring inputs from the store and save score snapshots.

Also a CLI to print the current table:
    uv run python -m weather_risk.scoring.snapshots
"""

from datetime import date

from weather_risk.db import Store
from weather_risk.hubs import Hub, load_hubs, load_scoring_config
from weather_risk.scoring.exposure import HubInputs, compute_exposure
from weather_risk.scoring.near_term import compute_near_term

COVERAGE_WARNING_PCT = 95.0


def hub_inputs(store: Store, hub: Hub, cfg: dict) -> HubInputs:
    clim = cfg["climatology"]
    nri = store.cache_get("nri")
    decls = store.cache_get(f"openfema:{hub.id}")
    return HubInputs(
        daily=store.daily(hub.id, clim["start"], clim["end"]),
        nri=nri[1]["by_fips"].get(hub.county_fips) if nri else None,
        declarations=decls[1] if decls else None,
    )


def exposure_for_hub(store: Store, hub: Hub, cfg: dict) -> dict:
    inputs = hub_inputs(store, hub, cfg)
    breakdown = compute_exposure(hub.id, inputs, cfg)
    clim = cfg["climatology"]
    expected = (clim["end"] - clim["start"]).days + 1
    with_data = sum(1 for r in inputs.daily if r.tmin_c is not None)
    coverage = round(100.0 * with_data / expected, 1)
    nri = store.cache_get("nri")
    grid = store.cache_get(f"grid:{hub.id}")
    breakdown["meta"] = {
        "observed_window": {"start": clim["start"].isoformat(), "end": clim["end"].isoformat(),
                            "days_with_data": with_data, "coverage_pct": coverage},
        "grid_cell": grid[1] if grid else None,
        "nri_version": nri[1]["version"] if nri else None,
        "declarations_since": cfg["declarations_since"].isoformat(),
        "county": f"{hub.county}, {hub.state} (FIPS {hub.county_fips})",
    }
    if coverage < COVERAGE_WARNING_PCT:
        breakdown["data_gaps"].append(
            f"Observed weather covers only {coverage}% of days in {clim['start'].year}–{clim['end'].year}."
        )
    return breakdown


def near_term_for_hub(store: Store, hub: Hub, cfg: dict) -> dict:
    forecast = store.cache_get(f"forecast:{hub.id}")
    alerts = store.cache_get(f"nws:{hub.id}")
    result = compute_near_term(forecast[1]["rows"] if forecast else [], alerts[1] if alerts else [], cfg["near_term"])
    result["meta"] = {
        "forecast_fetched_at": forecast[0] if forecast else None,
        "alerts_fetched_at": alerts[0] if alerts else None,
        "forecast_start": forecast[1]["rows"][0]["date"] if forecast and forecast[1]["rows"] else None,
    }
    result["data_gaps"] = [
        f"{name} unavailable — near-term score uses the remaining source."
        for name, cached in (("Open-Meteo forecast", forecast), ("NWS alerts", alerts))
        if cached is None
    ]
    return result


def refresh_snapshots(store: Store, today: date | None = None, kinds: tuple[str, ...] = ("exposure", "near_term")) -> None:
    cfg = load_scoring_config()
    for hub in load_hubs():
        if "exposure" in kinds:
            bd = exposure_for_hub(store, hub, cfg)
            if bd["score"] is not None:
                store.save_snapshot(hub.id, "exposure", bd["score"], None, bd)
        if "near_term" in kinds:
            nt = near_term_for_hub(store, hub, cfg)
            store.save_snapshot(hub.id, "near_term", nt["score"], nt["band"], nt)


def _print_table() -> None:
    from weather_risk.settings import get_settings

    store = Store(get_settings().db_path)
    exposure = store.latest_snapshots("exposure")
    near = store.latest_snapshots("near_term")
    families = list(load_scoring_config()["families"])
    header = f"{'hub':13}{'overall':>8}" + "".join(f"{f[:8]:>10}" for f in families) + f"{'top':>14}{'7-day':>10}"
    print(header)
    for hub_id, snap in sorted(exposure.items(), key=lambda kv: -kv[1]["score"]):
        bd = snap["breakdown"]
        fams = "".join(f"{bd['families'][f]['score']:>10.1f}" for f in families)
        nt = near.get(hub_id)
        print(f"{hub_id:13}{snap['score']:>8.1f}{fams}{bd['top_family']:>14}{(nt['band'] if nt else '-'):>10}")


if __name__ == "__main__":
    _print_table()
