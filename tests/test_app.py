import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

import app as app_module
from ravo_core import assistant
from ravo_core.embed_cache import set_redis

ACCESS_CODE = "pantry-2026"
SESSION_SECRET = "test-secret"


class FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value

    def incr(self, key):
        value = self.data.get(key, 0) + 1
        self.data[key] = value
        return value

    def expire(self, key, seconds):
        return key in self.data

    def lpush(self, key, value):
        self.data.setdefault(key, []).insert(0, value)
        return 0

    def ltrim(self, key, start, stop):
        self.data[key] = self.data.get(key, [])[start : stop + 1]

    def lrange(self, key, start, stop):
        return self.data.get(key, [])[start : stop + 1]


CANNED_RESULT = {
    "reply": "Almarai Milk is 25.5 EGP.",
    "cards": [],
    "citations": [],
    "trace": {"steps": [], "total_ms": 0.0},
}


@pytest.fixture(autouse=True)
def _app_env(monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", ACCESS_CODE)
    monkeypatch.setenv("SESSION_SECRET", SESSION_SECRET)
    set_redis(FakeRedis())
    yield
    set_redis(None)


@pytest.fixture(autouse=True)
def _canned_run_turn(monkeypatch):
    monkeypatch.setattr(assistant, "run_turn", lambda messages: dict(CANNED_RESULT))


@pytest.fixture
def client(_app_env, _canned_run_turn):
    return TestClient(app_module.app)


def _login(client):
    return client.post("/api/auth", json={"code": ACCESS_CODE})


def _chat_body(message="milk?", conversation_id="c1"):
    return {"messages": [{"role": "user", "content": message}], "conversation_id": conversation_id}


def test_root_redirects_without_cookie(client):
    res = client.get("/", follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/login.html"


def test_wrong_code_rejected(client):
    res = client.post("/api/auth", json={"code": "wrong"})
    assert res.status_code == 401
    assert res.json() == {"error": "That code isn't right."}


def test_right_code_sets_cookie_and_serves_app(client):
    res = client.post("/api/auth", json={"code": ACCESS_CODE})
    assert res.status_code == 200
    assert res.json() == {"ok": True}
    assert client.cookies.get("rx_session")
    res = client.get("/")
    assert res.status_code == 200
    assert "Ravo X-Ray" in res.text
    assert res.headers["cache-control"] == "no-store"


def test_tampered_cookie_rejected(client):
    _login(client)
    cookie = client.cookies.get("rx_session")
    expiry, _, sig = cookie.partition(".")
    client.cookies.set("rx_session", f"{expiry}.{'0' * len(sig)}")
    res = client.get("/", follow_redirects=False)
    assert res.status_code == 303


def test_expired_cookie_rejected(client):
    expiry = int(time.time()) - 10
    sig = hmac.new(SESSION_SECRET.encode(), str(expiry).encode(), hashlib.sha256).hexdigest()
    client.cookies.set("rx_session", f"{expiry}.{sig}")
    res = client.get("/", follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/login.html"


def test_chat_requires_session(client):
    res = client.post("/api/chat", json=_chat_body())
    assert res.status_code == 401
    assert res.json() == {"error": "Enter the access code first."}


def test_valid_chat_returns_turns_left(client):
    _login(client)
    res = client.post("/api/chat", json=_chat_body())
    assert res.status_code == 200
    body = res.json()
    assert body["reply"] == "Almarai Milk is 25.5 EGP."
    assert body["turns_left"] == 19


def test_chat_rejects_message_over_500_chars(client):
    _login(client)
    res = client.post("/api/chat", json=_chat_body(message="a" * 501))
    assert res.status_code == 400
    assert "error" in res.json()


def test_21st_turn_in_conversation_rejected(client):
    _login(client)
    for _ in range(20):
        res = client.post("/api/chat", json=_chat_body(conversation_id="c1"))
        assert res.status_code == 200
    res = client.post("/api/chat", json=_chat_body(conversation_id="c1"))
    assert res.status_code == 429
    assert res.json() == {
        "error": "This conversation has reached its limit. Start a new one to keep going."
    }


def test_per_ip_chat_limit(client):
    _login(client)
    for i in range(app_module.CHAT_PER_IP_LIMIT):
        res = client.post("/api/chat", json=_chat_body(conversation_id=f"c{i}"))
        assert res.status_code == 200
    res = client.post("/api/chat", json=_chat_body(conversation_id="fresh"))
    assert res.status_code == 429
    assert res.json() == {"error": "Too many questions from this network. Try again in an hour."}


def test_auth_attempts_rate_limited(client):
    for _ in range(10):
        client.post("/api/auth", json={"code": "wrong"})
    res = client.post("/api/auth", json={"code": ACCESS_CODE})
    assert res.status_code == 429
    assert res.json() == {"error": "Too many attempts. Wait a few minutes."}


def test_run_turn_error_returns_502(client, monkeypatch):
    _login(client)

    def boom(messages):
        raise RuntimeError("model down")

    monkeypatch.setattr(assistant, "run_turn", boom)
    res = client.post("/api/chat", json=_chat_body())
    assert res.status_code == 502
    assert res.json() == {
        "error": "The assistant couldn't answer just now. Try again in a moment."
    }


def test_eval_missing_returns_404(client, tmp_path, monkeypatch):
    _login(client)
    monkeypatch.setattr(app_module, "EVAL_RESULTS_PATH", tmp_path / "results.json")
    res = client.get("/api/eval")
    assert res.status_code == 404
    assert res.json() == {"error": "No scores yet."}


def test_eval_returns_scores(client, tmp_path, monkeypatch):
    _login(client)
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"accuracy": 0.8, "questions": 48}))
    monkeypatch.setattr(app_module, "EVAL_RESULTS_PATH", results)
    res = client.get("/api/eval")
    assert res.status_code == 200
    assert res.json() == {"accuracy": 0.8, "questions": 48}


def test_health_reports_db_up(client, monkeypatch):
    _login(client)

    class FakeConn:
        def execute(self, sql):
            return self

    monkeypatch.setattr(app_module.tools, "get_conn", lambda: FakeConn())
    res = client.get("/api/health")
    assert res.json() == {"ok": True, "db": True}


def test_health_reports_db_down(client, monkeypatch):
    _login(client)

    class BrokenConn:
        def execute(self, sql):
            raise RuntimeError("db down")

    monkeypatch.setattr(app_module.tools, "get_conn", lambda: BrokenConn())
    res = client.get("/api/health")
    assert res.json() == {"ok": True, "db": False}

def test_chat_rejects_padded_assistant_history(client):
    _login(client)
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x" * 2001},
            {"role": "user", "content": "milk?"},
        ],
        "conversation_id": "c-pad",
    }
    resp = client.post("/api/chat", json=body)
    assert resp.status_code == 400
    assert "2000" in resp.json()["error"]


def test_chat_rejects_non_text_content(client):
    _login(client)
    body = {"messages": [{"role": "user", "content": {"not": "text"}}], "conversation_id": "c-obj"}
    assert client.post("/api/chat", json=body).status_code == 400


def test_root_redirects_when_session_secret_missing(client, monkeypatch):
    monkeypatch.delenv("SESSION_SECRET")
    client.cookies.set("rx_session", "123.abc")
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login.html"


def test_monitor_requires_session(client):
    res = client.get("/api/monitor")
    assert res.status_code == 401
    assert res.json() == {"error": "Enter the access code first."}


def test_monitor_returns_turns_and_summary(client):
    _login(client)
    res = client.get("/api/monitor")
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    body = res.json()
    assert body["turns"] == []
    assert body["summary"]["typed"] == 0
    assert body["summary"]["spoken"] == 0


def test_chat_turn_adds_monitor_entry(client):
    _login(client)
    res = client.post("/api/chat", json=_chat_body())
    assert res.status_code == 200
    assert isinstance(res.json()["monitor_id"], str)

    turns = client.get("/api/monitor").json()["turns"]
    assert len(turns) == 1
    assert turns[0]["total_ms"] == 0.0
    assert turns[0]["cache"] is None
    assert turns[0]["spoken"] is False
