import math
from array import array

import ravo_core.embed_cache as ec
from ravo_core.db import EMBED_MODEL


class FakeRedis:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value


class ExplodingRedis(FakeRedis):
    def get(self, key):
        raise RuntimeError("boom")

    def set(self, key, value, ex=None):
        raise RuntimeError("boom")


def test_normalize_and_cache_key():
    assert ec.normalize("  Almond   MILK ") == "almond milk"
    assert ec.cache_key("  Almond   MILK ") == ec.cache_key("almond milk")
    assert ec.cache_key("almond milk").startswith(f"emb:{EMBED_MODEL}:")


def test_vector_round_trip():
    vec = [math.pi, 0.1, -1.5, 123456.789, 1.0 / 3.0, 0.0]
    decoded = ec.decode_vector(ec.encode_vector(vec))
    assert len(decoded) == len(vec)
    assert decoded == array("f", vec).tolist()


def test_miss_then_hit_embeds_once(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(ec, "get_redis", lambda: redis)
    calls = []

    def fake_embed(text):
        calls.append(text)
        return [1.0, 2.0, 3.0]

    monkeypatch.setattr(ec, "embed_one", fake_embed)
    v1, info1 = ec.embed_cached("  Almond   MILK ")
    v2, info2 = ec.embed_cached("almond milk")
    assert info1["cache"] == "miss"
    assert info2["cache"] == "hit"
    assert v1 == v2 == [1.0, 2.0, 3.0]
    assert calls == ["  Almond   MILK "]
    assert redis.data.get(ec.cache_key("almond milk")) == ec.encode_vector([1.0, 2.0, 3.0])


def test_no_redis_client_uses_embed(monkeypatch):
    monkeypatch.setattr(ec, "get_redis", lambda: None)
    monkeypatch.setattr(ec, "embed_one", lambda text: [0.5, 0.25])
    vector, info = ec.embed_cached("hello")
    assert vector == [0.5, 0.25]
    assert info["cache"] == "off"
    assert isinstance(info["ms"], float)


def test_redis_error_falls_back_to_off(monkeypatch):
    monkeypatch.setattr(ec, "get_redis", lambda: ExplodingRedis())
    monkeypatch.setattr(ec, "embed_one", lambda text: [7.0])
    vector, info = ec.embed_cached("hello")
    assert vector == [7.0]
    assert info["cache"] == "off"
    assert isinstance(info["ms"], float)

def test_redis_credentials_accept_vercel_marketplace_names(monkeypatch):
    from ravo_core import embed_cache

    for name in ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KV_REST_API_URL", "https://example.upstash.io")
    monkeypatch.setenv("KV_REST_API_TOKEN", "token")
    assert embed_cache._redis_credentials() == ("https://example.upstash.io", "token")
