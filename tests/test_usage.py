from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ravo_core import usage
from ravo_core.embed_cache import set_redis


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.expirations = {}

    def hsetnx(self, key, field, value):
        values = self.hashes.setdefault(key, {})
        if field in values:
            return 0
        values[field] = value
        return 1

    def hincrby(self, key, field, value):
        values = self.hashes.setdefault(key, {})
        values[field] = int(values.get(field, 0)) + value
        return values[field]

    def hincrbyfloat(self, key, field, value):
        values = self.hashes.setdefault(key, {})
        values[field] = float(values.get(field, 0)) + value
        return values[field]

    def expire(self, key, seconds):
        self.expirations[key] = seconds
        return True

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))


class BoomRedis(FakeRedis):
    def hsetnx(self, key, field, value):
        raise RuntimeError("redis down")


@pytest.fixture(autouse=True)
def _clean_redis(monkeypatch):
    monkeypatch.delenv("REDIS_NAMESPACE", raising=False)
    set_redis(None)
    yield
    set_redis(None)


def test_chat_collector_splits_cached_prompt_tokens():
    sdk_usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=25,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
    )

    collected = usage.from_chat_usage("gpt-4.1-mini", sdk_usage)

    assert collected.units_by_model["gpt-4.1-mini"] == {
        "requests": 1,
        "in_tokens": 60,
        "cached_tokens": 40,
        "out_tokens": 25,
    }


def test_collectors_tolerate_missing_and_none_usage_fields():
    chat = usage.from_chat_usage("gpt-4.1-mini", SimpleNamespace(prompt_tokens=None))
    embed = usage.from_embedding_usage("text-embedding-3-small", None)

    assert chat.units_by_model == {"gpt-4.1-mini": {"requests": 1}}
    assert embed.units_by_model == {"text-embedding-3-small": {"requests": 1}}


def test_record_writes_bounded_fields_daily_expiry_and_namespace(monkeypatch):
    fake = FakeRedis()
    set_redis(fake)
    monkeypatch.setenv("REDIS_NAMESPACE", "preview:")
    collected = usage.TurnUsage()
    collected.add("gpt-4.1-mini", {"requests": 2, "in_tokens": 30, "out_tokens": 5})
    collected.record_cache("hit")
    collected.add_flag("relevance_fail_open")

    usage.record("typed", collected, "answered", collected.flags, "prompts123")

    all_key = "preview:usage:v1:all"
    day_key = f"preview:usage:v1:day:{datetime.now(UTC).date().isoformat()}"
    assert fake.hashes[all_key]["turns"] == 1
    assert fake.hashes[all_key]["turns:typed"] == 1
    assert fake.hashes[all_key]["outcome:answered"] == 1
    assert fake.hashes[all_key]["prompt:prompts123:turns"] == 1
    assert fake.hashes[all_key]["prompt:prompts123:outcome:answered"] == 1
    assert fake.hashes[all_key]["unit:typed:gpt-4.1-mini:in_tokens"] == 30
    assert fake.hashes[all_key]["cache:typed:hits"] == 1
    assert fake.hashes[all_key]["flag:relevance_fail_open"] == 1
    assert isinstance(fake.hashes[all_key]["since"], int)
    assert fake.expirations[day_key] == usage.DAY_TTL_SECONDS


def test_record_and_read_fail_open_on_redis_error():
    set_redis(BoomRedis())

    usage.record("typed", usage.TurnUsage(), "chat", [], "p")
    report = usage.read(1)

    assert report["totals"]["total"] == 0
    assert report["since"] is None


def test_read_returns_totals_series_units_cost_and_prompt_sets():
    fake = FakeRedis()
    set_redis(fake)
    collected = usage.TurnUsage()
    collected.add("gpt-4.1-mini", {"in_tokens": 1_000_000})
    collected.record_cache("miss")
    usage.record("typed", collected, "chat", [], "p1")

    report = usage.read(2)

    assert report["totals"] == {
        "total": 1,
        "typed": 1,
        "spoken": 0,
        "cache_hits": 0,
        "cache_misses": 1,
    }
    assert len(report["days"]) == 2
    assert report["days"][-1]["turns"]["typed"] == 1
    assert report["units"]["typed"]["gpt-4.1-mini"]["in_tokens"] == 1_000_000
    assert report["estimated_cost"]["total_usd"] == pytest.approx(0.40)
    assert report["outcomes"]["chat"] == 1
    assert report["prompt_sets"]["p1"]["outcomes"]["chat"] == 1
    assert report["prices"]["as_of"] == "2026-09-29"


class StrictRedis(FakeRedis):
    """Rejects HINCRBY on a field holding a float, as Redis does."""

    def hincrby(self, key, field, value):
        current = self.hashes.get(key, {}).get(field, 0)
        if isinstance(current, float):
            raise RuntimeError("ERR hash value is not an integer")
        return super().hincrby(key, field, value)


def test_fractional_units_always_use_float_counters():
    fake = StrictRedis()
    set_redis(fake)

    usage.record("spoken", {"openai/gpt-live-transcribe": {"stt_seconds": 1.5}}, "answered", [], "abc")
    usage.record("spoken", {"openai/gpt-live-transcribe": {"stt_seconds": 2.0}}, "answered", [], "abc")

    field = "unit:spoken:openai/gpt-live-transcribe:stt_seconds"
    assert fake.hashes["usage:v1:all"][field] == 3.5
    assert fake.hashes["usage:v1:all"]["turns"] == 2


def test_read_splits_estimated_cost_by_channel():
    fake = FakeRedis()
    set_redis(fake)
    usage.record("typed", {"gpt-4.1-mini": {"in_tokens": 1_000_000}}, "answered", [], "abc")
    usage.record("spoken", {"cartesia/sonic-3": {"tts_chars": 1_000_000}}, "answered", [], "abc")

    report = usage.read(1)

    assert report["estimated_cost_by_channel"] == {"typed": 0.4, "spoken": 50.0}
