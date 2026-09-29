import pytest

from ravo_core import monitor
from ravo_core.embed_cache import set_redis

TYPED_TRACE = {
    "steps": [
        {"kind": "llm", "label": "Decide what to look up", "ms": 120.5, "cache": None, "tool": None},
        {"kind": "embed", "label": "Understand the question", "ms": 3.0, "cache": "miss", "tool": "search_products"},
        {"kind": "search", "label": "Search products", "ms": 6.5, "cache": None, "tool": "search_products"},
        {"kind": "embed", "label": "Understand the question", "ms": 2.0, "cache": "hit", "tool": "search_documents"},
        {"kind": "search", "label": "Search documents", "ms": 4.0, "cache": None, "tool": "search_documents"},
        {"kind": "llm", "label": "Write the answer", "ms": 800.0, "cache": None, "tool": None},
    ],
    "total_ms": 950.0,
    "prompt_set": "abc12345",
}

SPOKEN_TRACE = {
    "steps": [
        {"kind": "listen", "label": "Notice you've finished", "ms": 400.0, "cache": None, "tool": None},
        {"kind": "llm", "label": "Write the answer", "ms": 800.0, "cache": None, "tool": None},
        {"kind": "speak", "label": "Start speaking", "ms": 250.0, "cache": None, "tool": None},
    ],
    "total_ms": 2600.0,
    "first_sound_ms": 850.0,
    "filler": True,
    "spoken": True,
    "prompt_set": "abc12345",
}


class FakeRedis:
    def __init__(self):
        self.data = {}

    def lpush(self, key, value):
        self.data.setdefault(key, []).insert(0, value)
        return 0

    def ltrim(self, key, start, stop):
        self.data[key] = self.data.get(key, [])[start : stop + 1]

    def lrange(self, key, start, stop):
        items = self.data.get(key, [])
        return items[start:] if stop == -1 else items[start : stop + 1]

    def lrem(self, key, count, value):
        items = self.data.get(key, [])
        if value in items:
            items.remove(value)
            return 1
        return 0


class BoomRedis:
    def lpush(self, key, value):
        raise RuntimeError("redis down")

    def ltrim(self, key, start, stop):
        raise RuntimeError("redis down")

    def lrange(self, key, start, stop):
        raise RuntimeError("redis down")


@pytest.fixture(autouse=True)
def _no_redis():
    set_redis(None)


def test_summarize_typed_trace():
    entry = monitor.summarize(TYPED_TRACE)

    assert set(entry) == {
        "id", "at", "total_ms", "llm_ms", "embed_ms", "search_ms",
        "listen_ms", "speak_ms", "cache", "spoken",
        "prompt_set", "first_sound_ms", "filler", "est_cost_usd", "outcome",
    }
    assert len(entry["id"]) == 12
    assert entry["total_ms"] == 950.0
    assert entry["llm_ms"] == 920.5
    assert entry["embed_ms"] == 5.0
    assert entry["search_ms"] == 10.5
    assert entry["listen_ms"] == 0.0
    assert entry["speak_ms"] == 0.0
    assert entry["cache"] == "miss"
    assert entry["spoken"] is False
    assert entry["first_sound_ms"] is None
    assert entry["filler"] is False
    assert entry["prompt_set"] == "abc12345"
    assert "label" not in entry


def test_summarize_spoken_trace():
    entry = monitor.summarize(SPOKEN_TRACE)

    assert entry["listen_ms"] == 400.0
    assert entry["speak_ms"] == 250.0
    assert entry["llm_ms"] == 800.0
    assert entry["embed_ms"] == 0.0
    assert entry["search_ms"] == 0.0
    assert entry["cache"] is None
    assert entry["spoken"] is True
    assert entry["first_sound_ms"] == 850.0
    assert entry["filler"] is True


def test_record_and_recent_roundtrip():
    client = FakeRedis()
    set_redis(client)

    first = monitor.record(TYPED_TRACE)
    second = monitor.record(SPOKEN_TRACE)

    assert isinstance(first, str)
    assert isinstance(second, str)
    turns = monitor.recent()
    assert [t["id"] for t in turns] == [first, second]
    assert turns[0]["spoken"] is False
    assert turns[1]["spoken"] is True


def test_recent_trimmed_to_keep():
    client = FakeRedis()
    set_redis(client)

    ids = [monitor.record(TYPED_TRACE) for _ in range(monitor.MONITOR_KEEP + 5)]

    turns = monitor.recent(monitor.MONITOR_KEEP)
    assert len(turns) == monitor.MONITOR_KEEP
    assert [t["id"] for t in turns] == ids[5:]


def test_record_returns_none_without_client():
    assert monitor.record(TYPED_TRACE) is None


def test_recent_empty_without_client():
    assert monitor.recent() == []


def test_record_returns_none_when_redis_raises():
    set_redis(BoomRedis())
    assert monitor.record(TYPED_TRACE) is None


def test_recent_empty_when_redis_raises():
    set_redis(BoomRedis())
    assert monitor.recent() == []


def test_monitor_key_honors_namespace_at_call_time(monkeypatch):
    fake = FakeRedis()
    set_redis(fake)
    monkeypatch.setenv("REDIS_NAMESPACE", "preview:")

    monitor.record(TYPED_TRACE)

    assert "preview:monitor:turns" in fake.data
    assert monitor.MONITOR_KEY not in fake.data


def _typed(entry_id, total, llm, search, embed, cache):
    return {
        "id": entry_id,
        "total_ms": total,
        "llm_ms": llm,
        "embed_ms": embed,
        "search_ms": search,
        "listen_ms": 0.0,
        "speak_ms": 0.0,
        "cache": cache,
        "spoken": False,
    }


def test_summary_excludes_spoken_turns_from_medians():
    turns = [
        _typed("a", 100.0, 50.0, 10.0, 5.0, "miss"),
        _typed("b", 200.0, 80.0, 20.0, 2.0, "hit"),
        {**_typed("c", 999.0, 999.0, 999.0, 999.0, "missing"), "spoken": True},
    ]

    s = monitor.summary(turns)

    assert s["typed"] == 2
    assert s["spoken"] == 1
    assert s["median_total_ms"] == 150.0
    assert s["median_llm_ms"] == 65.0
    assert s["median_search_ms"] == 15.0
    assert s["median_embed_miss_ms"] == 5.0
    assert s["median_embed_hit_ms"] == 2.0
    assert s["cache_hit_rate"] == 0.5


def test_summary_of_empty_list():
    s = monitor.summary([])

    assert s["typed"] == 0
    assert s["spoken"] == 0
    assert s["median_total_ms"] is None
    assert s["median_llm_ms"] is None
    assert s["median_search_ms"] is None
    assert s["median_embed_miss_ms"] is None
    assert s["median_embed_hit_ms"] is None
    assert s["cache_hit_rate"] is None

def test_forget_removes_only_the_given_turns():
    fake = FakeRedis()
    set_redis(fake)
    kept = monitor.record(TYPED_TRACE)
    dropped = [monitor.record(SPOKEN_TRACE), monitor.record(SPOKEN_TRACE)]

    assert monitor.forget(dropped) == 2
    assert [turn["id"] for turn in monitor.recent()] == [kept]


def test_forget_fails_open():
    assert monitor.forget(["abc"]) == 0
    set_redis(BoomRedis())
    assert monitor.forget(["abc"]) == 0
