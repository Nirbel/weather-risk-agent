"""API routes and auth with a fake Analyzer and a fake LLM (no network, no keys)."""

import json
from types import SimpleNamespace

import pytest
import respx
from fastapi.testclient import TestClient

from conftest import SYNTHETIC_SCORES, FakeAnalyzer, FakeCompletion
from weather_risk.api.auth import parse_allowlist
from weather_risk.api.main import create_app
from weather_risk.db import Database
from weather_risk.settings import Settings

HOOK = "https://hooks.example.com/weather"
SETTINGS = Settings(auth_allowlist="ana@example.com:analyst, boss@example.com:admin, bob@example.com",
                    jwt_secret="test-secret-that-is-at-least-32-bytes-long", groq_api_key="x",
                    scheduler_enabled=False, alert_webhook_url=HOOK, alert_threshold=5.0,
                    alert_trigger_secret="cron-secret")

PLAN = json.dumps({"intent": "rank", "hubs": None, "region": "midwest", "hazards": ["winter"], "metric": None,
                   "top_k": 3, "time_preset": None, "year": None, "start_date": None, "end_date": None,
                   "weight_overrides": None, "interpretation_notes": [], "clarification_question": None,
                   "out_of_scope_reason": None})
EXPLANATION = json.dumps({"summary": "Minneapolis–St Paul ranks first (90).", "reasoning": ["Winter drives it."],
                          "suggested_follow_ups": []})


class FakeTranscription:
    def __init__(self, text: str = "Why is Dallas high?", error: Exception | None = None):
        self.text, self.error, self.calls = text, error, []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=f"  {self.text} ")


@pytest.fixture
def make_client():
    def factory(*outcomes, settings=SETTINGS, transcription=None):
        app = create_app(settings, db=Database("sqlite+aiosqlite:///:memory:"), analyzer=FakeAnalyzer(),
                         completion=FakeCompletion(*outcomes), transcription=transcription or FakeTranscription())
        return TestClient(app)
    return factory


def auth(client, email="ana@example.com") -> dict:
    response = client.post("/auth/login", json={"email": email})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


# -- auth ------------------------------------------------------------------------

def test_parse_allowlist():
    assert parse_allowlist("A@x.com:admin, b@y.com") == {"a@x.com": "admin", "b@y.com": "analyst"}
    with pytest.raises(ValueError):
        parse_allowlist("a@x.com:root")


def test_login_requires_allow_listed_email(make_client):
    with make_client() as client:
        assert client.post("/auth/login", json={"email": "stranger@example.com"}).status_code == 403
        body = client.post("/auth/login", json={"email": "Boss@Example.com"}).json()
    assert body["role"] == "admin" and body["email"] == "boss@example.com"


def test_empty_allowlist_explains_how_to_fix_it(make_client):
    with make_client(settings=SETTINGS.model_copy(update={"auth_allowlist": ""})) as client:
        response = client.post("/auth/login", json={"email": "ana@example.com"})
    assert response.status_code == 403 and "AUTH_ALLOWLIST" in response.json()["detail"]


def test_protected_routes_need_a_valid_token(make_client):
    with make_client() as client:
        assert client.post("/chat", json={"message": "hi"}).status_code == 401
        assert client.get("/analytics/exposure", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_removed_email_loses_access(make_client):
    with make_client() as client:
        headers = auth(client)
        client.app.state.settings = SETTINGS.model_copy(update={"auth_allowlist": "boss@example.com:admin"})
        assert client.get("/hubs", headers=headers).status_code == 401


def test_admin_routes_reject_analysts(make_client):
    with make_client() as client:
        assert client.post("/admin/warm", headers=auth(client)).status_code == 403
        assert client.post("/admin/warm", headers=auth(client, "boss@example.com")).status_code == 202


# -- chat and conversation history ------------------------------------------------

def test_chat_returns_full_answer_contract(make_client):
    with make_client(PLAN, EXPLANATION) as client:
        answer = client.post("/chat", json={"message": "Midwest winter?"}, headers=auth(client)).json()
    assert answer["table"][0]["hub"] == "Minneapolis–St Paul"
    for key in ("assumptions", "sources", "time_windows"):
        assert answer[key], key
    assert answer["uncertainty"]["level"] in ("low", "medium", "high") and answer["conversation_id"]


def test_conversation_history_can_be_listed_and_reopened(make_client):
    with make_client(PLAN, EXPLANATION, PLAN, EXPLANATION) as client:
        headers = auth(client)
        first = client.post("/chat", json={"message": "Midwest winter?"}, headers=headers).json()
        client.post("/chat", json={"message": "again?", "conversation_id": first["conversation_id"]}, headers=headers)
        listed = client.get("/conversations", headers=headers).json()
        opened = client.get(f"/conversations/{first['conversation_id']}", headers=headers).json()
    assert [(c["title"], c["turns"]) for c in listed] == [("Midwest winter?", 2)]
    assert [t["user_text"] for t in opened["turns"]] == ["Midwest winter?", "again?"]
    assert opened["turns"][0]["answer"]["table"][0]["hub"] == "Minneapolis–St Paul"


def test_other_users_cannot_see_or_continue_a_conversation(make_client):
    with make_client(PLAN, EXPLANATION) as client:
        cid = client.post("/chat", json={"message": "Midwest winter?"}, headers=auth(client)).json()["conversation_id"]
        bob = auth(client, "bob@example.com")
        assert client.get(f"/conversations/{cid}", headers=bob).status_code == 404
        assert client.post("/chat", json={"message": "x", "conversation_id": cid}, headers=bob).status_code == 404
        assert client.get("/conversations", headers=bob).json() == []


# -- analytics: same numbers as chat --------------------------------------------------

def test_analytics_exposure_matches_the_scoring(make_client):
    with make_client() as client:
        body = client.get("/analytics/exposure", headers=auth(client)).json()
    assert body["window"] == {"start": "2021-01-01", "end": "2025-12-31"}
    top = body["hubs"][0]
    assert top["hub"] == "Houston" and top["overall"] == pytest.approx(37.0)  # synthetic: (80+75+10+10+10)/5
    assert top["families"]["flood"]["score"] == 75


def test_analytics_hub_is_deterministic(make_client):
    with make_client() as client:  # FakeCompletion() has no outcomes: any LLM call would fail
        answer = client.get("/analytics/hubs/dallas", headers=auth(client)).json()
        assert client.get("/analytics/hubs/boston", headers=auth(client)).status_code == 404
    assert answer["meta"]["explanation_source"] == "template"
    assert answer["table"][0]["hazard"].startswith("Severe storm")


def test_health(make_client):
    with make_client() as client:
        body = client.get("/health").json()
    assert body["status"] == "ok" and body["hubs"] == 10 and body["llm_configured"]
    assert body["exposure_window"]["start"] == "2021-01-01"
    assert body["voice"] is True


# -- alerts (bonus) ----------------------------------------------------------------

def test_alert_settings_are_admin_only_and_validated(make_client):
    with make_client() as client:
        analyst, admin = auth(client), auth(client, "boss@example.com")
        assert client.get("/alerts/settings", headers=analyst).status_code == 403
        assert client.get("/alerts/settings", headers=admin).json()["webhook_url"] == HOOK  # env default
        bad = client.put("/alerts/settings", headers=admin,
                         json={"webhook_url": "ftp://x", "enabled": True, "threshold": 5})
        assert bad.status_code == 422
        saved = client.put("/alerts/settings", headers=admin,
                           json={"webhook_url": "https://example.org/hook", "enabled": False, "threshold": 2.5}).json()
    assert saved["webhook_url"] == "https://example.org/hook" and saved["threshold"] == 2.5
    assert saved["enabled"] is False and saved["updated_by"] == "boss@example.com"


def test_alert_run_needs_admin_or_the_trigger_secret(make_client):
    with make_client() as client:
        assert client.post("/alerts/run", headers=auth(client)).status_code == 403
        assert client.post("/alerts/run", headers={"X-Alert-Secret": "wrong"}).status_code == 401
        first = client.post("/alerts/run", headers={"X-Alert-Secret": "cron-secret"}).json()
        second = client.post("/alerts/run", headers=auth(client, "boss@example.com")).json()
    assert first["status"] == "baseline" and second["status"] == "no_changes"


def test_score_change_alerts_the_webhook_and_the_feed(make_client, monkeypatch):
    with respx.mock(assert_all_called=False) as router, make_client() as client:
        hook = router.post(HOOK).respond(200)
        admin = auth(client, "boss@example.com")
        client.post("/alerts/run", headers=admin)
        # Miami winter 10 → 60: overall (85+60+10+10+10)/5 = 35.0 → (85+60+60+10+10)/5 = 45.0
        monkeypatch.setitem(SYNTHETIC_SCORES, "miami", {"hurricane": 85, "flood": 60, "winter": 60})
        result = client.post("/alerts/run", headers=admin).json()
        feed = client.get("/alerts", headers=auth(client)).json()
    assert result["delivery"] == "sent" and hook.call_count == 1
    assert [(c["hub"], c["old_score"], c["new_score"], c["delta"]) for c in result["changes"]] == [
        ("Miami", 35.0, 45.0, 10.0)]
    assert feed["alerts"][0]["hub"] == "Miami" and feed["last_check"] and feed["delivering"]


def test_test_webhook_reports_delivery(make_client):
    with respx.mock() as router, make_client() as client:
        router.post(HOOK).respond(500)
        body = client.post("/alerts/test", headers=auth(client, "boss@example.com")).json()
    assert body == {"ok": False, "delivery": "failed: HTTP 500"}


# -- voice (bonus) ---------------------------------------------------------------------

def test_transcribe_returns_text(make_client):
    fake = FakeTranscription()
    with make_client(transcription=fake) as client:
        response = client.post("/transcribe", headers=auth(client),
                               files={"audio": ("question.wav", b"RIFF....WAVE", "audio/wav")})
    assert response.json() == {"text": "Why is Dallas high?"}
    assert fake.calls[0]["model"] == SETTINGS.transcribe_model and fake.calls[0]["file"][0] == "question.wav"


def test_transcribe_errors_are_clear(make_client):
    with make_client(transcription=FakeTranscription(error=RuntimeError("boom"))) as client:
        headers = auth(client)
        assert client.post("/transcribe", headers=headers,
                           files={"audio": ("q.wav", b"", "audio/wav")}).status_code == 400
        failed = client.post("/transcribe", headers=headers, files={"audio": ("q.wav", b"abc", "audio/wav")})
    assert failed.status_code == 502 and "RuntimeError" in failed.json()["detail"]
    no_key = SETTINGS.model_copy(update={"groq_api_key": None})
    with make_client(settings=no_key) as client:
        assert client.get("/health").json()["voice"] is False
        assert client.post("/transcribe", headers=auth(client),
                           files={"audio": ("q.wav", b"abc", "audio/wav")}).status_code == 503

