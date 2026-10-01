"""API routes and auth with a fake Analyzer and a fake LLM (no network, no keys)."""

import json
from types import SimpleNamespace

import pytest
import respx
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from conftest import SYNTHETIC_SCORES, FakeAnalyzer, FakeCompletion
from weather_risk.api.auth import User, issue_token
from weather_risk.api.main import create_app
from weather_risk.db import Database
from weather_risk.settings import Settings

HOOK = "https://hooks.example.com/weather"
SETTINGS = Settings(jwt_secret="test-secret-that-is-at-least-32-bytes-long", groq_api_key="x",
                    scheduler_enabled=False, alert_threshold=5.0, alert_trigger_secret="cron-secret",
                    demo_admin_email="admin@moveo.co.il", demo_admin_password="12345678")
ADMIN = ("admin@moveo.co.il", "12345678")
ANA = ("ana@moveo.co.il", "ana-password-1")
BEN = ("ben@moveo.co.il", "ben-password-1")

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


async def public_dns(host: str, port: int) -> list[str]:
    return ["93.184.215.14"]  # offline: every webhook host resolves to a public address


@pytest.fixture
def make_client():
    def factory(*outcomes, settings=SETTINGS, transcription=None, analyzer=None, db=None):
        app = create_app(settings, db=db or Database("sqlite+aiosqlite:///:memory:"),
                         analyzer=analyzer or FakeAnalyzer(), completion=FakeCompletion(*outcomes),
                         transcription=transcription or FakeTranscription(), resolve=public_dns)
        return TestClient(app)
    return factory


def login(client, email, password):
    return client.post("/auth/login", json={"email": email, "password": password})


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def auth(client, user=ANA) -> dict:
    """Log in (signing up first if needed) and return the Authorization header."""
    response = login(client, *user)
    if response.status_code == 401 and user != ADMIN:
        response = client.post("/auth/signup", json={"email": user[0], "password": user[1]})
    assert response.status_code in (200, 201), response.text
    return bearer(response.json()["token"])


# -- auth: login -------------------------------------------------------------------

def test_demo_admin_can_log_in(make_client):
    with make_client() as client:
        body = login(client, *ADMIN).json()
    assert (body["email"], body["role"]) == ("admin@moveo.co.il", "admin") and body["token"]


def test_wrong_password_and_unknown_email_get_the_same_answer(make_client):
    with make_client() as client:
        wrong = login(client, "admin@moveo.co.il", "123456789")
        unknown = login(client, "nobody@moveo.co.il", "12345678")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json()["detail"] == unknown.json()["detail"] == "Invalid email or password."


def test_login_normalizes_the_email(make_client):
    with make_client() as client:
        assert login(client, "  ADMIN@Moveo.co.IL ", "12345678").status_code == 200


# -- auth: sign-up ------------------------------------------------------------------

def test_signup_creates_an_analyst_and_logs_in(make_client):
    with make_client() as client:
        response = client.post("/auth/signup", json={"email": " Ana@MOVEO.co.il ", "password": ANA[1]})
        assert response.status_code == 201
        assert (response.json()["email"], response.json()["role"]) == ("ana@moveo.co.il", "analyst")
        assert login(client, "ana@moveo.co.il", ANA[1]).json()["role"] == "analyst"
        assert login(client, "ana@moveo.co.il", "wrong-password").status_code == 401


@pytest.mark.parametrize("email", ["ana@moveo.co.il.attacker.com", "ana@gmail.com", "ana@sub.moveo.co.il",
                                   "ana@moveo.co.il@evil.com", "ana"])
def test_signup_is_only_for_moveo_addresses(make_client, email):
    with make_client() as client:
        response = client.post("/auth/signup", json={"email": email, "password": "long-enough-1"})
        assert response.status_code == 422
        assert login(client, email, "long-enough-1").status_code == 401  # nothing was created


def test_signup_cannot_create_or_take_over_an_admin(make_client):
    with make_client() as client:
        with_role = client.post("/auth/signup", json={"email": "eve@moveo.co.il", "password": "long-enough-1",
                                                      "role": "admin"})
        assert with_role.status_code == 422  # no "role" field: sign-up always creates an analyst
        assert login(client, "eve@moveo.co.il", "long-enough-1").status_code == 401
        takeover = client.post("/auth/signup", json={"email": "admin@moveo.co.il", "password": "attacker-pass"})
        assert takeover.status_code == 409
        assert login(client, *ADMIN).json()["role"] == "admin"  # the admin's password did not change
        eve = client.post("/auth/signup", json={"email": "eve@moveo.co.il", "password": "long-enough-1"}).json()
        assert eve["role"] == "analyst"
        assert client.post("/alerts/run", headers=bearer(eve["token"])).status_code == 403


def test_duplicate_signup_and_short_password_are_rejected(make_client):
    with make_client() as client:
        assert client.post("/auth/signup", json={"email": ANA[0], "password": ANA[1]}).status_code == 201
        assert client.post("/auth/signup", json={"email": ANA[0].upper(), "password": ANA[1]}).status_code == 409
        assert client.post("/auth/signup", json={"email": BEN[0], "password": "1234567"}).status_code == 422


def test_passwords_are_stored_hashed(make_client):
    db = Database("sqlite+aiosqlite:///:memory:")
    with make_client(db=db) as client:
        client.post("/auth/signup", json={"email": ANA[0], "password": ANA[1]})
        stored = {email: client.portal.call(db.get_account, email)["password_hash"] for email in (ADMIN[0], ANA[0])}
    for email, password in (ADMIN, ANA):
        assert stored[email].startswith("scrypt$") and password not in stored[email]


def test_demo_admin_password_follows_the_setting(make_client):
    db = Database("sqlite+aiosqlite:///:memory:")
    with make_client(db=db) as client:  # first start: created with 12345678
        assert login(client, *ADMIN).status_code == 200
    rotated = SETTINGS.model_copy(update={"demo_admin_password": "a-new-long-password"})
    with make_client(db=db, settings=rotated) as client:  # restart with DEMO_ADMIN_PASSWORD changed
        assert login(client, *ADMIN).status_code == 401
        assert login(client, ADMIN[0], "a-new-long-password").json()["role"] == "admin"


# -- auth: tokens and protected routes -------------------------------------------------

PUBLIC = {"/auth/login", "/auth/signup", "/health", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def test_every_business_endpoint_needs_a_token(make_client):
    with make_client() as client:
        routes = [r for r in client.app.routes if isinstance(r, APIRoute) and r.path not in PUBLIC]
        assert len(routes) >= 12
        for route in routes:
            path = route.path.replace("{conversation_id}", "x").replace("{hub_id}", "dallas")
            for method in route.methods:
                assert client.request(method, path).status_code == 401, f"{method} {path}"
                assert client.request(method, path, headers=bearer("not-a-token")).status_code == 401


def test_role_comes_from_the_database_not_the_token(make_client):
    with make_client() as client:
        auth(client, ANA)  # Ana exists as an analyst
        forged = issue_token(User(email=ANA[0], role="admin"), SETTINGS)  # validly signed, claims admin
        assert client.post("/alerts/run", headers=bearer(forged)).status_code == 403
        ghost = issue_token(User(email="ghost@moveo.co.il", role="admin"), SETTINGS)
        assert client.get("/hubs", headers=bearer(ghost)).status_code == 401  # no such account


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


def test_users_cannot_read_or_continue_another_users_conversation(make_client):
    with make_client(PLAN, EXPLANATION) as client:
        cid = client.post("/chat", json={"message": "Midwest winter?"}, headers=auth(client, ANA)).json()[
            "conversation_id"]
        ben, admin = auth(client, BEN), auth(client, ADMIN)
        for intruder in (ben, admin):  # even an admin only sees their own history
            assert client.get(f"/conversations/{cid}", headers=intruder).status_code == 404
            assert client.post("/chat", json={"message": "x", "conversation_id": cid},
                               headers=intruder).status_code == 404
            assert client.get("/conversations", headers=intruder).json() == []
        assert client.get("/conversations/does-not-exist", headers=ben).status_code == 404
        assert client.get(f"/conversations/{cid}", headers=auth(client, ANA)).status_code == 200


# -- analytics: same numbers as chat --------------------------------------------------

def test_analytics_exposure_matches_the_scoring(make_client):
    with make_client() as client:
        body = client.get("/analytics/exposure", headers=auth(client)).json()
    assert body["window"] == {"start": "2021-01-01", "end": "2025-12-31"}
    top = body["hubs"][0]
    assert top["hub"] == "Houston" and top["overall"] == pytest.approx(37.0)  # synthetic: (80+75+10+10+10)/5
    assert top["families"]["flood"]["score"] == 75


def test_analytics_never_scores_a_hub_with_incomplete_weather(make_client):
    with make_client(analyzer=FakeAnalyzer(incomplete={"houston"})) as client:
        hubs = {h["hub_id"]: h for h in client.get("/analytics/exposure", headers=auth(client)).json()["hubs"]}
    assert hubs["houston"]["overall"] is None and hubs["houston"]["weather_complete"] is False
    assert hubs["houston"]["families"]["flood"]["score"] is None  # flood uses observed weather
    assert hubs["dallas"]["overall"] == pytest.approx(36.0) and hubs["dallas"]["weather_complete"] is True


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


# -- alerts (bonus): each user's own webhook ---------------------------------------------

def test_any_user_configures_their_own_webhook(make_client):
    with make_client() as client:
        ana, ben = auth(client, ANA), auth(client, BEN)
        assert client.get("/alerts/settings", headers=ana).json() == {
            "webhook_url": None, "enabled": False, "threshold": 5.0, "updated_at": None}
        bad = client.put("/alerts/settings", headers=ana, json={"webhook_url": "ftp://x", "enabled": True,
                                                                "threshold": 5})
        assert bad.status_code == 422
        saved = client.put("/alerts/settings", headers=ana,
                           json={"webhook_url": "https://hook.eu1.make.com/abc", "enabled": True,
                                 "threshold": 2.5}).json()
        assert (saved["webhook_url"], saved["enabled"], saved["threshold"]) == (
            "https://hook.eu1.make.com/abc", True, 2.5)
        assert client.get("/alerts/settings", headers=ben).json()["webhook_url"] is None  # Ben's are separate
        assert client.put("/alerts/settings", headers=ana, json={"webhook_url": None, "enabled": True,
                                                                 "threshold": 0}).status_code == 422


def test_alert_run_needs_admin_or_the_trigger_secret(make_client):
    with make_client() as client:
        assert client.post("/alerts/run", headers=auth(client)).status_code == 403
        assert client.post("/alerts/run", headers={"X-Alert-Secret": "wrong"}).status_code == 401
        assert client.post("/alerts/run", headers={"X-Alert-Secret": "cron-secret"}).status_code == 200
        assert client.post("/alerts/run", headers=auth(client, ADMIN)).status_code == 200


def test_score_change_reaches_the_users_webhook_and_feed(make_client, monkeypatch):
    with respx.mock(assert_all_called=False) as router, make_client() as client:
        hook = router.post(HOOK).respond(200)
        ana, admin = auth(client, ANA), auth(client, ADMIN)
        client.put("/alerts/settings", headers=ana, json={"webhook_url": HOOK, "enabled": True, "threshold": 5})
        client.post("/alerts/run", headers=admin)  # Ana's baseline
        # Miami winter 10 → 60: overall (85+60+10+10+10)/5 = 35.0 → (85+60+60+10+10)/5 = 45.0
        monkeypatch.setitem(SYNTHETIC_SCORES, "miami", {"hurricane": 85, "flood": 60, "winter": 60})
        result = client.post("/alerts/run", headers=admin).json()
        feed = client.get("/alerts", headers=ana).json()
        admin_feed = client.get("/alerts", headers=admin).json()
    assert [(u["user"], u["changes"], u["delivery"]) for u in result["users"]] == [("ana@moveo.co.il", 1, "sent")]
    assert hook.call_count == 1
    assert [(a["hub"], a["old_score"], a["new_score"], a["delta"]) for a in feed["alerts"]] == [
        ("Miami", 35.0, 45.0, 10.0)]
    assert feed["last_check"] and feed["delivering"] and admin_feed["alerts"] == []


def test_test_webhook_uses_the_users_saved_url(make_client):
    with respx.mock() as router, make_client() as client:
        hook = router.post(HOOK).respond(500)
        ana = auth(client)
        assert client.post("/alerts/test", headers=ana).json() == {"ok": False, "delivery": "no webhook URL set"}
        client.put("/alerts/settings", headers=ana, json={"webhook_url": HOOK, "enabled": False, "threshold": 5})
        body = client.post("/alerts/test", headers=ana).json()
    assert body == {"ok": False, "delivery": "failed: HTTP 500"} and hook.call_count == 1


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


# -- clean checkout and startup ------------------------------------------------------------

def test_app_starts_on_a_clean_checkout(tmp_path):
    """The default SQLite path's directory does not exist on a fresh clone; startup must create it."""
    path = tmp_path / "clone" / "data" / "app.db"
    settings = SETTINGS.model_copy(update={"database_url": f"sqlite+aiosqlite:///{path}"})
    with TestClient(create_app(settings, analyzer=FakeAnalyzer())) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert login(client, *ADMIN).status_code == 200  # demo admin seeded on first start
    assert path.is_file()


@pytest.mark.filterwarnings("ignore:The HMAC key is")  # the test forges with a weak key on purpose
def test_weak_jwt_secret_is_replaced_with_a_random_one(make_client):
    with make_client(settings=SETTINGS.model_copy(update={"jwt_secret": "change-me"})) as client:
        token = login(client, *ADMIN).json()["token"]
        assert client.app.state.settings.jwt_secret != "change-me"
        weak = issue_token(User(email=ADMIN[0], role="admin"), SETTINGS.model_copy(update={"jwt_secret": "change-me"}))
        assert client.get("/hubs", headers=bearer(weak)).status_code == 401  # can't forge with the weak secret
        assert client.get("/hubs", headers=bearer(token)).status_code == 200
