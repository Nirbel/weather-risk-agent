"""Score-change alerts: the pure diff (hand-computed), the alert run and webhook delivery (respx)."""

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


# -- AlertService (in-memory DB, fake exposure, webhook mocked) -----------------------

class ScoreAnalyzer:
    """Exposure with hand-set overall scores; `gaps` marks hubs whose data was incomplete."""

    def __init__(self, scores: dict[str, float], gaps: dict[str, list[str]] | None = None):
        self.scores, self.gaps = scores, gaps or {}

    async def exposure(self, hub_ids=None):
        breakdowns = {h: {"score": s, "top_family": "flood", "data_gaps": self.gaps.get(h, [])}
                      for h, s in self.scores.items()}
        return Exposure(date(2021, 1, 1), date(2025, 12, 31), breakdowns, "December 2025")


SETTINGS = Settings(alert_webhook_url=HOOK, alert_threshold=5.0)


async def run(db, scores, *, settings=SETTINGS, gaps=None):
    async with httpx.AsyncClient() as client:
        return await AlertService(db, ScoreAnalyzer(scores, gaps), settings, client).run()


async def test_first_run_records_a_baseline_without_alerting(db):
    with respx.mock(assert_all_called=False) as router:
        hook = router.post(HOOK).respond(200)
        result = await run(db, {"miami": 41.04, "houston": 50.0})
    assert result["status"] == "baseline" and result["changes"] == [] and not hook.called
    assert (await db.latest_snapshot())["scores"] == {"miami": 41.0, "houston": 50.0}


async def test_score_change_is_stored_and_posted_to_the_webhook(db):
    with respx.mock() as router:
        hook = router.post(HOOK).respond(200)
        await run(db, {"miami": 41.0, "houston": 50.0})
        result = await run(db, {"miami": 47.3, "houston": 51.0})
    assert result["status"] == "changes" and result["delivery"] == "sent"
    body = json.loads(hook.calls.last.request.content)
    assert body["event"] == "hub_exposure_score_changed" and "Miami 41.0 → 47.3 (+6.3)" in body["text"]
    assert body["changes"] == [{"hub_id": "miami", "hub": "Miami", "old_score": 41.0, "new_score": 47.3,
                                "delta": 6.3, "direction": "up", "top_hazard": "flood"}]
    alerts = await db.recent_alerts()
    assert [(a["hub_id"], a["old_score"], a["new_score"], a["delta"], a["delivery"]) for a in alerts] == [
        ("miami", 41.0, 47.3, 6.3, "sent")]


async def test_webhook_failure_is_recorded_not_raised(db):
    with respx.mock() as router:
        router.post(HOOK).respond(500)
        await run(db, {"miami": 41.0})
        result = await run(db, {"miami": 30.0})
    assert result["delivery"] == "failed: HTTP 500"
    assert (await db.recent_alerts())[0]["delivery"] == "failed: HTTP 500"


async def test_hub_with_data_gaps_is_skipped_not_alerted(db):
    with respx.mock(assert_all_called=False) as router:
        hook = router.post(HOOK).respond(200)
        await run(db, {"miami": 41.0})
        result = await run(db, {"miami": 12.0}, gaps={"miami": ["Open-Meteo request failed"]})
    assert result["changes"] == [] and result["skipped"] == {"miami": ["Open-Meteo request failed"]}
    assert not hook.called and (await db.latest_snapshot())["scores"] == {"miami": 41.0}


async def test_disabled_or_unset_webhook_still_records_alerts(db):
    await db.save_alert_settings(webhook_url=HOOK, enabled=False, threshold=5.0, updated_by="boss@example.com")
    with respx.mock(assert_all_called=False) as router:
        hook = router.post(HOOK).respond(200)
        await run(db, {"miami": 41.0})
        result = await run(db, {"miami": 50.0})
    assert result["delivery"] == "disabled" and not hook.called
    assert (await db.recent_alerts())[0]["delivery"] == "disabled"


async def test_saved_threshold_overrides_the_env_default(db):
    await db.save_alert_settings(webhook_url=None, enabled=True, threshold=10.0, updated_by="boss@example.com")
    await run(db, {"miami": 41.0})
    result = await run(db, {"miami": 50.0})  # +9 < 10
    assert result["changes"] == [] and result["threshold"] == 10.0


async def test_send_test_reports_the_outcome(db):
    with respx.mock() as router:
        router.post(HOOK).respond(204)
        async with httpx.AsyncClient() as client:
            ok = await AlertService(db, ScoreAnalyzer({}), SETTINGS, client).send_test()
    assert ok == {"ok": True, "delivery": "sent"}
    async with httpx.AsyncClient() as client:
        missing = await AlertService(db, ScoreAnalyzer({}), Settings(alert_webhook_url=None),
                                       client).send_test()
    assert missing == {"ok": False, "delivery": "no webhook URL set"}


@pytest.mark.parametrize("url", ["ftp://example.com/x", "not a url", "file:///etc/passwd"])
async def test_only_http_webhooks_are_accepted(db, url):
    with pytest.raises(ValueError):
        await AlertService(db, ScoreAnalyzer({}), SETTINGS, None).update_settings(
            webhook_url=url, enabled=True, threshold=5.0, updated_by="boss@example.com")
