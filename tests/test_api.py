"""API routes and auth with a fake Analyzer and a fake LLM (no network, no keys)."""

import json

import pytest
from fastapi.testclient import TestClient

from conftest import FakeAnalyzer, FakeCompletion
from weather_risk.api.auth import parse_allowlist
from weather_risk.api.main import create_app
from weather_risk.db import Database
from weather_risk.settings import Settings

SETTINGS = Settings(auth_allowlist="ana@example.com:analyst, boss@example.com:admin, bob@example.com",
                    jwt_secret="test-secret-that-is-at-least-32-bytes-long", groq_api_key="x")

PLAN = json.dumps({"intent": "rank", "hubs": None, "region": "midwest", "hazards": ["winter"], "metric": None,
                   "top_k": 3, "time_preset": None, "year": None, "start_date": None, "end_date": None,
                   "weight_overrides": None, "interpretation_notes": [], "clarification_question": None,
                   "out_of_scope_reason": None})
EXPLANATION = json.dumps({"summary": "Minneapolis–St Paul ranks first (90).", "reasoning": ["Winter drives it."],
                          "suggested_follow_ups": []})


@pytest.fixture
def make_client():
    def factory(*outcomes, settings=SETTINGS):
        app = create_app(settings, db=Database("sqlite+aiosqlite:///:memory:"), analyzer=FakeAnalyzer(),
                         completion=FakeCompletion(*outcomes))
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
