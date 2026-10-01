"""Score-change alerts (bonus): tell analysts when a hub's overall Exposure Score moves.

Exposure covers the previous 5 completed years, so scores move only when the window rolls over
(each January), FEMA publishes a new NRI release, the scoring config changes, or a data gap
closes. A scheduled check (every ALERT_CHECK_HOURS, daily by default) or POST /alerts/run compares
current scores with the baseline and posts changes of at least the threshold to a generic JSON
webhook. See docs/DECISIONS.md D22.
"""

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from weather_risk.config import load_hubs
from weather_risk.db import Database, iso, utcnow
from weather_risk.settings import Settings

log = logging.getLogger("weather_risk.alerts")
EVENT = "hub_exposure_score_changed"


@dataclass(frozen=True)
class ScoreChange:
    hub_id: str
    old: float
    new: float
    delta: float


def diff_scores(baseline: dict[str, float], current: dict[str, float | None],
                threshold: float) -> tuple[list[ScoreChange], dict[str, float]]:
    """Changes of at least `threshold` points against the baseline, largest first, and the next baseline.

    The baseline is the score analysts were last told about, so it moves only when a hub alerts:
    small moves accumulate until they cross the threshold. A hub without a current score (data gap)
    keeps its baseline, a new hub joins it silently, and a hub no longer scored is dropped.
    """
    changes, nxt = [], {}
    for hub_id, new in current.items():
        old = baseline.get(hub_id)
        if old is None or new is None:  # new hub → baseline only; data gap → keep the baseline
            if old is not None or new is not None:
                nxt[hub_id] = new if new is not None else old
            continue
        delta = round(new - old, 1)
        if abs(delta) >= threshold:
            changes.append(ScoreChange(hub_id, old, new, delta))
            nxt[hub_id] = new
        else:
            nxt[hub_id] = old
    changes.sort(key=lambda c: -abs(c.delta))
    return changes, nxt


def validate_webhook_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("The webhook URL must be an http(s) URL.")


class AlertService:
    def __init__(self, db: Database, analyzer, settings: Settings, client: httpx.AsyncClient | None):
        self.db, self.analyzer, self.settings, self.client = db, analyzer, settings, client

    async def config(self) -> dict:
        """Admin-saved settings, else the env defaults."""
        saved = await self.db.alert_settings()
        if saved:
            return saved
        url = self.settings.alert_webhook_url
        return {"webhook_url": url, "enabled": bool(url), "threshold": self.settings.alert_threshold,
                "updated_at": None, "updated_by": None}

    async def update_settings(self, *, webhook_url: str | None, enabled: bool, threshold: float,
                              updated_by: str) -> dict:
        webhook_url = (webhook_url or "").strip() or None
        if webhook_url:
            validate_webhook_url(webhook_url)
        if not 0 < threshold <= 100:
            raise ValueError("The threshold must be between 0 and 100 points.")
        await self.db.save_alert_settings(webhook_url=webhook_url, enabled=enabled, threshold=threshold,
                                          updated_by=updated_by)
        return await self.config()

    async def run(self) -> dict:
        """One check: score all hubs, diff against the baseline, record and deliver any alerts."""
        cfg = await self.config()
        result = await self.analyzer.exposure()
        current, skipped = {}, {}
        for hub_id, bd in result.breakdowns.items():
            if bd["data_gaps"] or bd["score"] is None:  # incomplete data must not look like a score change
                current[hub_id], skipped[hub_id] = None, bd["data_gaps"] or ["no score"]
            else:
                current[hub_id] = round(bd["score"], 1)
        previous = await self.db.latest_snapshot()
        outcome = {"checked_at": iso(utcnow()), "threshold": cfg["threshold"], "skipped": skipped,
                   "window": {"start": result.start.isoformat(), "end": result.end.isoformat()}}
        if previous is None:
            await self.db.add_snapshot({h: s for h, s in current.items() if s is not None})
            return outcome | {"status": "baseline", "changes": [], "delivery": None}

        changes, baseline = diff_scores(previous["scores"], current, cfg["threshold"])
        await self.db.add_snapshot(baseline)  # also records when the last check ran
        if not changes:
            return outcome | {"status": "no_changes", "changes": [], "delivery": None}
        names = {h.id: h.name for h in load_hubs()}
        items = [{"hub_id": c.hub_id, "hub": names.get(c.hub_id, c.hub_id), "old_score": c.old, "new_score": c.new,
                  "delta": c.delta, "direction": "up" if c.delta > 0 else "down",
                  "top_hazard": result.breakdowns[c.hub_id].get("top_family")} for c in changes]
        text = (f"Weather risk: {len(items)} hub exposure score change(s) of at least {cfg['threshold']:g} points "
                f"({outcome['window']['start'][:4]}–{outcome['window']['end'][:4]} window) — "
                + "; ".join(f"{i['hub']} {i['old_score']:.1f} → {i['new_score']:.1f} ({i['delta']:+.1f})"
                            for i in items))
        delivery = await self._deliver(cfg, {"event": EVENT, "text": text, **outcome, "changes": items})
        await self.db.add_alerts([{k: i[k] for k in ("hub_id", "old_score", "new_score", "delta")} for i in items],
                                 delivery)
        log.info("alert check: %d change(s), delivery %s", len(items), delivery)
        return outcome | {"status": "changes", "changes": items, "delivery": delivery}

    async def send_test(self) -> dict:
        """Post a sample payload to the saved webhook, even while alerts are disabled."""
        cfg = await self.config()
        delivery = await self._deliver(cfg | {"enabled": True}, {
            "event": "test", "text": "Test alert from the Weather Risk Intelligence Agent — the webhook works.",
            "checked_at": iso(utcnow()), "changes": []})
        return {"ok": delivery == "sent", "delivery": delivery}

    async def _deliver(self, cfg: dict, payload: dict) -> str:
        if not cfg["webhook_url"]:
            return "no webhook URL set"
        if not cfg["enabled"]:
            return "disabled"
        try:
            response = await self.client.post(cfg["webhook_url"], json=payload, timeout=10, follow_redirects=False)
        except httpx.HTTPError as exc:
            return f"failed: {exc.__class__.__name__}"
        return "sent" if response.is_success else f"failed: HTTP {response.status_code}"


async def run_on_schedule(service: AlertService, every_hours: float, first_delay_s: float = 30.0) -> None:
    """In-process scheduler: the first check (shortly after startup) records a baseline and warms the cache."""
    await asyncio.sleep(first_delay_s)
    while True:
        try:
            result = await service.run()
            log.info("scheduled alert check: %s", result["status"])
        except Exception:
            log.exception("scheduled alert check failed")
        await asyncio.sleep(every_hours * 3600)
