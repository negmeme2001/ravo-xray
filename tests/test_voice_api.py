import base64
import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
from ravo_core.embed_cache import set_redis

ACCESS_CODE = "pantry-2026"
SESSION_SECRET = "test-secret"
LIVEKIT_URL = "wss://example.livekit.cloud"
LIVEKIT_API_KEY = "test-key"
LIVEKIT_API_SECRET = "test-api-secret-for-signing-tokens"


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


@pytest.fixture(autouse=True)
def _app_env(monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", ACCESS_CODE)
    monkeypatch.setenv("SESSION_SECRET", SESSION_SECRET)
    monkeypatch.delenv("LIVEKIT_URL", raising=False)
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    set_redis(FakeRedis())
    yield
    set_redis(None)


@pytest.fixture
def client(_app_env):
    return TestClient(app_module.app)


@pytest.fixture
def voice_env(monkeypatch):
    monkeypatch.setenv("LIVEKIT_URL", LIVEKIT_URL)
    monkeypatch.setenv("LIVEKIT_API_KEY", LIVEKIT_API_KEY)
    monkeypatch.setenv("LIVEKIT_API_SECRET", LIVEKIT_API_SECRET)


def _login(client):
    return client.post("/api/auth", json={"code": ACCESS_CODE})


def _decode_payload(token):
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def test_voice_status_disabled(client):
    _login(client)
    res = client.get("/api/voice/status")
    assert res.status_code == 200
    assert res.json() == {"enabled": False}


def test_voice_status_enabled(client, voice_env):
    _login(client)
    res = client.get("/api/voice/status")
    assert res.status_code == 200
    assert res.json() == {"enabled": True}


def test_voice_token_requires_session(client):
    res = client.post("/api/voice/token")
    assert res.status_code == 401
    assert res.json() == {"error": "Enter the access code first."}


def test_voice_token_unset_returns_503(client):
    _login(client)
    res = client.post("/api/voice/token")
    assert res.status_code == 503
    assert res.json() == {"error": "Voice isn't set up yet."}


def test_voice_token_returns_jwt_with_room_and_agent(client, voice_env):
    _login(client)
    res = client.post("/api/voice/token")
    assert res.status_code == 200
    body = res.json()
    assert body["url"] == LIVEKIT_URL
    assert body["room"].startswith("rx-")
    assert body["token"]

    claims = _decode_payload(body["token"])
    assert claims["sub"].startswith("shopper-")
    assert claims["name"] == "Shopper"
    assert claims["iss"] == LIVEKIT_API_KEY

    video = claims["video"]
    assert video["roomJoin"] is True
    assert video["room"] == body["room"]
    assert video["canPublish"] is True
    assert video["canSubscribe"] is True
    assert video["canPublishData"] is True

    assert claims["roomConfig"]["agents"] == [{"agentName": "ravo-xray"}]


def test_voice_token_rate_limited_per_ip(client, voice_env):
    _login(client)
    for _ in range(10):
        res = client.post("/api/voice/token")
        assert res.status_code == 200
    res = client.post("/api/voice/token")
    assert res.status_code == 429
    assert res.json() == {
        "error": "Too many voice sessions from this network. Try again later."
    }