"""Score-change alerts (bonus): tell each user when a hub's overall Exposure Score moves.

Every user can set their own generic webhook (URL, on/off, threshold) in the UI; nothing is tied
to a provider, so Make, n8n, Zapier, Slack or Teams workflows, or any HTTP endpoint that accepts
a JSON POST all work. A scheduled check (every ALERT_CHECK_HOURS, daily by default) or
POST /alerts/run scores all hubs once, then compares the scores with each user's own baseline.

Exposure covers the previous 5 completed years, so scores move only when the window rolls over
(each January), FEMA publishes a new NRI release, the scoring config changes, or a data gap
closes. See docs/DECISIONS.md D22.
"""

import asyncio
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from weather_risk.config import load_hubs
from weather_risk.db import Database, iso, utcnow
from weather_risk.settings import Settings

log = logging.getLogger("weather_risk.alerts")
EVENT = "hub_exposure_score_changed"
WEBHOOK_TIMEOUT_S = 10.0


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
    try:
        parsed = urlparse(url)
        parsed.port  # noqa: B018 — raises on a malformed port
    except ValueError as exc:
        raise ValueError("The webhook URL must be a valid http(s) URL.") from exc
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or any(c.isspace() for c in url)
            or len(url) > 2000):
        raise ValueError("The webhook URL must be a valid http(s) URL.")


Resolver = Callable[[str, int], Awaitable[list[str]]]


async def resolve_host(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


def _address_problem(ip: str, allow_private: bool) -> str | None:
    """Any signed-up user chooses the URL the server calls, so internal destinations are refused (SSRF)."""
    addr = ipaddress.ip_address(ip.split("%")[0])
    if addr.version == 6 and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    never = f"{ip} is a link-local, multicast or reserved address (e.g. cloud metadata), which is never allowed"
    if addr.is_link_local or addr.is_multicast or addr.is_unspecified:
        return never
    if addr.is_private or addr.is_loopback:  # before is_reserved: ::1 is both loopback and reserved
        return None if allow_private else (
            f"{ip} is a private or loopback address; set ALLOW_PRIVATE_WEBHOOKS=true if this server "
            "should call internal services such as a self-hosted n8n")
    return never if addr.is_reserved else None


async def destination_problem(url: str, allow_private: bool, resolve: Resolver) -> str | None:
    parsed = urlparse(url)
    try:
        ips = await resolve(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except OSError:
        return f"cannot resolve {parsed.hostname}"
    return next((p for ip in ips if (p := _address_problem(ip, allow_private))), None) if ips else \
        f"cannot resolve {parsed.hostname}"


class AlertService:
    def __init__(self, db: Database, analyzer, settings: Settings, client: httpx.AsyncClient | None,
                 resolve: Resolver = resolve_host):
        self.db, self.analyzer, self.settings, self.client, self.resolve = db, analyzer, settings, client, resolve

    async def config(self, email: str) -> dict:
        """The user's saved settings, else defaults (no URL, off, the default threshold)."""
        saved = await self.db.webhook_settings(email)
        if saved:
            return {k: saved[k] for k in ("webhook_url", "enabled", "threshold", "updated_at")}
        return {"webhook_url": None, "enabled": False, "threshold": self.settings.alert_threshold, "updated_at": None}

    async def update_settings(self, email: str, *, webhook_url: str | None, enabled: bool, threshold: float) -> dict:
        webhook_url = (webhook_url or "").strip() or None
        if webhook_url:
            validate_webhook_url(webhook_url)
            if problem := await destination_problem(webhook_url, self.settings.allow_private_webhooks, self.resolve):
                raise ValueError(f"This webhook URL is not allowed: {problem}.")
        if not 0 < threshold <= 100:
            raise ValueError("The threshold must be between 0 and 100 points.")
        latest = await self.db.latest_snapshot()  # a new subscriber starts from the latest check
        await self.db.save_webhook_settings(email, webhook_url=webhook_url, enabled=enabled, threshold=threshold,
                                            baseline_if_new=latest["scores"] if latest else None)
        return await self.config(email)

    async def run(self) -> dict:
        """One check: score all hubs once, then diff, record and deliver per user."""
        result = await self.analyzer.exposure()
        current, skipped = {}, {}
        for hub_id, bd in result.breakdowns.items():
            if bd["data_gaps"] or bd["score"] is None:  # incomplete data must not look like a score change
                current[hub_id], skipped[hub_id] = None, bd["data_gaps"] or ["no score"]
            else:
                current[hub_id] = round(bd["score"], 1)
        scored = {h: v for h, v in current.items() if v is not None}
        await self.db.add_snapshot(scored)
        window = {"start": result.start.isoformat(), "end": result.end.isoformat()}
        checked_at = iso(utcnow())
        names = {h.id: h.name for h in load_hubs()}
        users = []
        for sub in await self.db.all_webhook_settings():
            if sub["baseline"] is None:
                await self.db.set_baseline(sub["user"], scored)
                users.append({"user": sub["user"], "status": "baseline", "changes": 0, "delivery": None})
                continue
            changes, baseline = diff_scores(sub["baseline"], current, sub["threshold"])
            await self.db.set_baseline(sub["user"], baseline)
            if not changes:
                users.append({"user": sub["user"], "status": "no_changes", "changes": 0, "delivery": None})
                continue
            items = [{"hub_id": c.hub_id, "hub": names.get(c.hub_id, c.hub_id), "old_score": c.old,
                      "new_score": c.new, "delta": c.delta, "direction": "up" if c.delta > 0 else "down",
                      "top_hazard": result.breakdowns[c.hub_id].get("top_family")} for c in changes]
            text = (f"Weather risk: {len(items)} hub exposure score change(s) of at least {sub['threshold']:g} "
                    f"points ({window['start'][:4]}–{window['end'][:4]} window) — "
                    + "; ".join(f"{i['hub']} {i['old_score']:.1f} → {i['new_score']:.1f} ({i['delta']:+.1f})"
                                for i in items))
            delivery = await self._deliver(sub, {"event": EVENT, "text": text, "checked_at": checked_at,
                                                 "threshold": sub["threshold"], "window": window, "changes": items})
            await self.db.add_alerts(sub["user"], [{k: i[k] for k in ("hub_id", "old_score", "new_score", "delta")}
                                                   for i in items], delivery)
            users.append({"user": sub["user"], "status": "changes", "changes": len(items), "delivery": delivery})
        log.info("alert check: %d subscriber(s), %d notified", len(users), sum(u["changes"] > 0 for u in users))
        return {"checked_at": checked_at, "window": window, "skipped": skipped, "users": users}

    async def send_test(self, email: str) -> dict:
        """Post a sample payload to the user's saved webhook, even while alerts are off."""
        cfg = await self.config(email)
        delivery = await self._deliver(cfg | {"enabled": True}, {
            "event": "test", "text": "Test alert from the Weather Risk Intelligence Agent — the webhook works.",
            "checked_at": iso(utcnow()), "threshold": cfg["threshold"], "changes": []})
        return {"ok": delivery == "sent", "delivery": delivery}

    async def _deliver(self, cfg: dict, payload: dict) -> str:
        if not cfg["webhook_url"]:
            return "no webhook URL set"
        if not cfg["enabled"]:
            return "disabled"
        # Re-check at send time: the name may resolve elsewhere now than when it was saved.
        if problem := await destination_problem(cfg["webhook_url"], self.settings.allow_private_webhooks, self.resolve):
            return f"blocked: {problem}"
        try:  # bounded wait, and never follow a redirect to somewhere the user did not enter
            response = await self.client.post(cfg["webhook_url"], json=payload, timeout=WEBHOOK_TIMEOUT_S,
                                              follow_redirects=False)
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
