"""Score-change alerts: the pure diff (hand-computed), the alert run and webhook delivery (respx)."""

import ipaddress
import json
from datetime import date

import httpx
import pytest
import respx

from weather_risk.alerts import AlertService, ScoreChange, diff_scores
from weather_risk.analysis import Exposure
from weather_risk.settings import Settings

HOOK = "https://hooks.example.com/weather"


# -- diff_scores (pure) -------------------------------------------------------------

def test_change_at_or_above_threshold_alerts_and_moves_the_baseline():
    changes, baseline = diff_scores({"miami": 40.0, "houston": 50.0}, {"miami": 46.2, "houston": 47.5}, 5.0)
    assert changes == [ScoreChange("miami", 40.0, 46.2, 6.2)]
    # Houston moved only -2.5: no alert, and its baseline stays at the last value analysts were told about.
    assert baseline == {"miami": 46.2, "houston": 50.0}


def test_small_moves_accumulate_until_they_cross_the_threshold():
    _, baseline = diff_scores({"houston": 50.0}, {"houston": 47.5}, 5.0)
    changes, baseline = diff_scores(baseline, {"houston": 45.0}, 5.0)
    assert changes == [ScoreChange("houston", 50.0, 45.0, -5.0)]  # exactly the threshold counts
    assert baseline == {"houston": 45.0}


def test_delta_is_rounded_before_the_threshold_comparison():
    # 15.2 - 10.2 is 4.999999999999998 in floating point; displayed scores have one decimal.
    changes, _ = diff_scores({"denver": 10.2}, {"denver": 15.2}, 5.0)
    assert changes == [ScoreChange("denver", 10.2, 15.2, 5.0)]


def test_missing_score_keeps_baseline_and_new_hub_is_added_silently():
    changes, baseline = diff_scores({"denver": 30.0, "atlanta": 20.0}, {"denver": None, "dallas": 33.3}, 5.0)
    assert changes == []
    # Denver had a data gap: keep its baseline. Dallas is new: baseline only. Atlanta left the roster: dropped.
    assert baseline == {"denver": 30.0, "dallas": 33.3}


def test_changes_are_ordered_by_size():
    changes, _ = diff_scores({"a": 10.0, "b": 10.0, "c": 10.0}, {"a": 16.0, "b": 2.0, "c": 22.0}, 5.0)
    assert [(c.hub_id, c.delta) for c in changes] == [("c", 12.0), ("b", -8.0), ("a", 6.0)]


# -- AlertService: per-user webhooks (in-memory DB, fake exposure, webhooks mocked) -------

ANA, BEN = "ana@moveo.co.il", "ben@moveo.co.il"
HOOK_BEN = "https://hooks.example.com/ben"


class ScoreAnalyzer:
    """Exposure with hand-set overall scores; `gaps` marks hubs whose data was incomplete."""

    def __init__(self, scores: dict[str, float], gaps: dict[str, list[str]] | None = None):
        self.scores, self.gaps = scores, gaps or {}

    async def exposure(self, hub_ids=None):
        breakdowns = {h: {"score": s, "top_family": "flood", "data_gaps": self.gaps.get(h, [])}
                      for h, s in self.scores.items()}
        return Exposure(date(2021, 1, 1), date(2025, 12, 31), breakdowns, "December 2025")


SETTINGS = Settings(alert_threshold=5.0)
PUBLIC_IP = "93.184.215.14"


def resolver(table: dict[str, list[str]] | None = None):
    """Offline DNS: IP literals resolve to themselves, names in `table` as given, others to a public IP."""
    async def resolve(host: str, port: int) -> list[str]:
        if host in (table or {}):
            return table[host]
        try:
            return [str(ipaddress.ip_address(host))]
        except ValueError:
            return [PUBLIC_IP]
    return resolve


def service(db, scores=None, *, gaps=None, client=None, settings=SETTINGS, resolve=None):
    return AlertService(db, ScoreAnalyzer(scores or {}, gaps), settings, client, resolve=resolve or resolver())


async def run(db, scores, *, gaps=None):
    async with httpx.AsyncClient() as client:
        return await service(db, scores, gaps=gaps, client=client).run()


async def save(db, email, url=HOOK, *, enabled=True, threshold=5.0, settings=SETTINGS, resolve=None):
    return await service(db, settings=settings, resolve=resolve).update_settings(
        email, webhook_url=url, enabled=enabled, threshold=threshold)


def outcome(result, email):
    return next(u for u in result["users"] if u["user"] == email)


async def test_unconfigured_user_gets_defaults(db):
    cfg = await service(db).config(ANA)
    assert (cfg["webhook_url"], cfg["enabled"], cfg["threshold"]) == (None, False, 5.0)


async def test_a_check_without_subscribers_only_records_the_scores(db):
    result = await run(db, {"miami": 41.04, "houston": 50.0})
    assert result["users"] == [] and (await db.latest_snapshot())["scores"] == {"miami": 41.0, "houston": 50.0}


async def test_first_check_after_saving_sets_the_users_baseline_without_alerting(db):
    await save(db, ANA)
    with respx.mock(assert_all_called=False) as router:
        hook = router.post(HOOK).respond(200)
        result = await run(db, {"miami": 41.0})
    assert outcome(result, ANA)["status"] == "baseline" and not hook.called
    assert (await db.webhook_settings(ANA))["baseline"] == {"miami": 41.0}


async def test_saving_takes_the_latest_check_as_the_baseline(db):
    await run(db, {"miami": 41.0})
    await save(db, ANA)
    assert (await db.webhook_settings(ANA))["baseline"] == {"miami": 41.0}
    with respx.mock() as router:
        router.post(HOOK).respond(200)
        result = await run(db, {"miami": 47.3})
    assert outcome(result, ANA)["changes"] == 1


async def test_each_user_is_alerted_on_their_own_webhook_with_their_own_threshold(db):
    await save(db, ANA, HOOK, threshold=5.0)
    await save(db, BEN, HOOK_BEN, threshold=10.0)
    with respx.mock(assert_all_called=False) as router:
        ana_hook, ben_hook = router.post(HOOK).respond(200), router.post(HOOK_BEN).respond(200)
        await run(db, {"miami": 41.0, "houston": 50.0})  # baselines
        result = await run(db, {"miami": 47.3, "houston": 51.0})  # Miami +6.3: ≥ 5 (Ana), < 10 (Ben)
    assert (outcome(result, ANA)["delivery"], outcome(result, BEN)["changes"]) == ("sent", 0)
    assert ana_hook.call_count == 1 and not ben_hook.called
    body = json.loads(ana_hook.calls.last.request.content)
    assert body["event"] == "hub_exposure_score_changed" and "Miami 41.0 → 47.3 (+6.3)" in body["text"]
    assert body["changes"] == [{"hub_id": "miami", "hub": "Miami", "old_score": 41.0, "new_score": 47.3,
                                "delta": 6.3, "direction": "up", "top_hazard": "flood"}]
    assert [(a["hub_id"], a["old_score"], a["new_score"], a["delta"], a["delivery"])
            for a in await db.recent_alerts(ANA)] == [("miami", 41.0, 47.3, 6.3, "sent")]
    assert await db.recent_alerts(BEN) == []
    # Ben's baseline did not move: Miami's change keeps accumulating toward his 10-point threshold.
    assert (await db.webhook_settings(BEN))["baseline"]["miami"] == 41.0


async def test_webhook_failure_is_recorded_not_raised(db):
    await save(db, ANA)
    with respx.mock() as router:
        router.post(HOOK).respond(500)
        await run(db, {"miami": 41.0})
        result = await run(db, {"miami": 30.0})
    assert outcome(result, ANA)["delivery"] == "failed: HTTP 500"
    assert (await db.recent_alerts(ANA))[0]["delivery"] == "failed: HTTP 500"


async def test_redirects_are_not_followed(db):
    await save(db, ANA)
    with respx.mock(assert_all_called=False) as router:
        router.post(HOOK).respond(302, headers={"Location": "https://internal.example.com/admin"})
        elsewhere = router.post("https://internal.example.com/admin").respond(200)
        await run(db, {"miami": 41.0})
        result = await run(db, {"miami": 30.0})
    assert outcome(result, ANA)["delivery"] == "failed: HTTP 302" and not elsewhere.called


async def test_slow_webhook_times_out(db):
    await save(db, ANA)
    with respx.mock() as router:
        hook = router.post(HOOK).mock(side_effect=httpx.ReadTimeout("too slow"))
        await run(db, {"miami": 41.0})
        result = await run(db, {"miami": 30.0})
    assert outcome(result, ANA)["delivery"] == "failed: ReadTimeout"
    assert hook.calls.last.request.extensions["timeout"]["read"] == 10


async def test_hub_with_data_gaps_is_skipped_not_alerted(db):
    await save(db, ANA)
    with respx.mock(assert_all_called=False) as router:
        hook = router.post(HOOK).respond(200)
        await run(db, {"miami": 41.0})
        result = await run(db, {"miami": 12.0}, gaps={"miami": ["Open-Meteo request failed"]})
    assert outcome(result, ANA)["changes"] == 0 and result["skipped"] == {"miami": ["Open-Meteo request failed"]}
    assert not hook.called and (await db.webhook_settings(ANA))["baseline"] == {"miami": 41.0}


async def test_disabled_webhook_still_records_alerts_in_the_feed(db):
    await save(db, ANA, enabled=False)
    with respx.mock(assert_all_called=False) as router:
        hook = router.post(HOOK).respond(200)
        await run(db, {"miami": 41.0})
        result = await run(db, {"miami": 50.0})
    assert outcome(result, ANA)["delivery"] == "disabled" and not hook.called
    assert (await db.recent_alerts(ANA))[0]["delivery"] == "disabled"


async def test_send_test_posts_to_the_users_saved_url(db):
    await save(db, ANA, enabled=False)  # a test is sent even while alerts are off
    with respx.mock() as router:
        hook = router.post(HOOK).respond(204)
        async with httpx.AsyncClient() as client:
            ok = await service(db, client=client).send_test(ANA)
    assert ok == {"ok": True, "delivery": "sent"}
    assert json.loads(hook.calls.last.request.content)["event"] == "test"
    async with httpx.AsyncClient() as client:
        missing = await service(db, client=client).send_test(BEN)
    assert missing == {"ok": False, "delivery": "no webhook URL set"}


@pytest.mark.parametrize("url", ["ftp://example.com/x", "not a url", "file:///etc/passwd", "https://",
                                 "https://exa mple.com/hook", "javascript:alert(1)"])
async def test_only_http_webhook_urls_are_accepted(db, url):
    with pytest.raises(ValueError):
        await save(db, ANA, url)


@pytest.mark.parametrize("threshold", [0, -1, 101])
async def test_threshold_must_be_within_the_score_range(db, threshold):
    with pytest.raises(ValueError):
        await save(db, ANA, threshold=threshold)


async def test_any_http_endpoint_works_make_n8n_zapier_teams(db):
    for url in ("https://hook.eu1.make.com/abc", "http://n8n.local:5678/webhook/weather",
                "https://hooks.zapier.com/hooks/catch/1/2/", "https://prod.westeurope.logic.azure.com/workflows/x"):
        assert (await save(db, ANA, url))["webhook_url"] == url


# -- destination check (self-registered users choose the URL the server calls) -----------

@pytest.mark.parametrize("url", ["http://169.254.169.254/latest/meta-data", "http://[fe80::1]/x", "http://0.0.0.0/x"])
async def test_link_local_and_metadata_addresses_are_always_refused(db, url):
    with pytest.raises(ValueError, match="link-local"):
        await save(db, ANA, url, settings=Settings(allow_private_webhooks=True))


@pytest.mark.parametrize("url", ["http://127.0.0.1:9009/hook", "http://10.0.0.7/hook", "http://[::1]/x",
                                 "http://[::ffff:192.168.1.5]/x"])
async def test_private_addresses_need_an_explicit_opt_in(db, url):
    with pytest.raises(ValueError, match="ALLOW_PRIVATE_WEBHOOKS"):
        await save(db, ANA, url)
    assert (await save(db, ANA, url, settings=Settings(allow_private_webhooks=True)))["webhook_url"] == url


async def test_a_name_that_resolves_to_an_internal_address_is_refused(db):
    with pytest.raises(ValueError, match="private or loopback"):
        await save(db, ANA, "https://n8n.corp.example/webhook", resolve=resolver({"n8n.corp.example": ["10.1.2.3"]}))


async def test_an_unresolvable_host_is_refused(db):
    async def nxdomain(host, port):
        raise OSError("Name or service not known")

    with pytest.raises(ValueError, match="cannot resolve"):
        await save(db, ANA, "https://no-such-host.invalid/hook", resolve=nxdomain)


async def test_delivery_rechecks_the_destination(db):
    # The name resolved to a public address when saved, but to an internal one at send time (DNS rebinding).
    await save(db, ANA)
    rebound = resolver({"hooks.example.com": ["127.0.0.1"]})
    with respx.mock(assert_all_called=False) as router:
        hook = router.post(HOOK).respond(200)
        async with httpx.AsyncClient() as client:
            result = await service(db, client=client, resolve=rebound).send_test(ANA)
    assert result["delivery"].startswith("blocked: 127.0.0.1") and not hook.called

