import asyncio
import types

import pytest

pytest.importorskip("livekit.agents")

import ravo_core.tools as tools
from agent import main as agent
from livekit.agents import llm as agents_llm
from ravo_core import assistant

_PRODUCTS = [
    {"code": "P1", "name": "Almarai Milk", "category": "Dairy", "price_egp": 25.5, "stock": 10, "similarity": 0.912},
    {"code": "P2", "name": "Juhayna Yogurt", "category": "Dairy", "price_egp": 18.0, "stock": 4, "similarity": 0.88},
]


def test_cards_payload_shape():
    timing = {"embed_ms": 12.4, "cache": "hit", "search_ms": 3.1}

    payload = agent.build_cards_payload(_PRODUCTS, timing, "turn-123")

    assert payload == {
        "cards": [
            {"code": "P1", "name": "Almarai Milk", "category": "Dairy", "price_egp": 25.5, "stock": 10},
            {"code": "P2", "name": "Juhayna Yogurt", "category": "Dairy", "price_egp": 18.0, "stock": 4},
        ],
        "timing": {"embed_ms": 12.4, "cache": "hit", "search_ms": 3.1},
        "turn_id": "turn-123",
    }


def test_voice_prompt_contains_base_rules():
    assert assistant.BASE_RULES in agent.VOICE_PROMPT


class _FakeModel:
    def __init__(self, **options):
        self.options = options


class _FakeFallback:
    def __init__(self, models):
        self.models = models


@pytest.fixture
def fake_voice_backends(monkeypatch):
    monkeypatch.setattr(agent.inference, "STT", _FakeModel)
    monkeypatch.setattr(agent.inference, "TTS", _FakeModel)
    monkeypatch.setattr(agent.agents_stt, "FallbackAdapter", _FakeFallback)
    monkeypatch.setattr(agent.agents_tts, "FallbackAdapter", _FakeFallback)
    monkeypatch.setattr(agent.openai, "TTS", _FakeModel)
    monkeypatch.setattr(agent.openai, "LLM", _FakeModel)


def test_legacy_session_options_match_existing_configuration(fake_voice_backends):
    options = agent.session_options("legacy", {})

    assert options["stt"].options["model"] == "deepgram/flux-general-en"
    assert options["stt"].options["language"] == "en"
    assert options["turn_handling"] == {"endpointing": {"max_delay": 1.2}}
    assert "transcription_timeout" not in options
    assert options["tts"].models[0] is options["primary_tts"]


def test_unknown_session_profile_uses_legacy(fake_voice_backends):
    options = agent.session_options("unknown", {})

    assert options["stt"].options["model"] == "deepgram/flux-general-en"
    assert options["turn_handling"] == {"endpointing": {"max_delay": 1.2}}


def test_bilingual_session_options_and_environment_overrides(fake_voice_backends):
    options = agent.session_options(
        "bilingual",
        {
            "VOICE_STT_MODEL": "custom/stt",
            "VOICE_STT_LANGUAGE": "ar_custom",
            "VOICE_TTS_VOICE_EN": "english-voice",
        },
    )

    assert options["stt"].models[0].options["model"] == "custom/stt"
    assert options["stt"].models[0].options["language"] == "ar_custom"
    assert options["stt"].models[1].options["model"] == "deepgram/flux-general-en"
    assert options["primary_tts"].options["language"] == "en"
    assert options["primary_tts"].options["voice"] == "english-voice"
    assert options["turn_handling"] == {
        "endpointing": {"mode": "dynamic", "min_delay": 0.5, "max_delay": 3.0},
        "interruption": {
            "resume_false_interruption": True,
            "false_interruption_timeout": 2.0,
        },
    }
    assert options["transcription_timeout"] == 3.0


def test_bilingual_session_options_use_default_stt(fake_voice_backends):
    options = agent.session_options("bilingual", {})

    assert options["stt"].models[0].options["model"] == "speechmatics/linden-1"
    assert options["stt"].models[0].options["language"] == "ar_en"


class _FakeSwitchableTTS:
    def __init__(self):
        self.calls = []

    def update_options(self, **options):
        self.calls.append(options)


def _store_with_session(profile="bilingual", env=None):
    primary_tts = _FakeSwitchableTTS()
    store = agent.StoreAgent(profile=profile, primary_tts=primary_tts, env=env)
    spoken = []
    session = types.SimpleNamespace(say=spoken.append)
    store._activity = types.SimpleNamespace(session=session)
    return store, primary_tts, spoken


def test_bilingual_filler_asks_for_arabic_clarification():
    store, primary_tts, spoken = _store_with_session()

    with pytest.raises(agent.StopResponse):
        asyncio.run(
            store.on_user_turn_completed(
                agents_llm.ChatContext(),
                types.SimpleNamespace(text_content="umm… يعني…"),
            )
        )

    assert spoken == [agent.understanding.clarification("ar")]
    assert primary_tts.calls == [{"language": "ar"}]
    assert len(store.turn_id) == 12


@pytest.mark.parametrize(
    ("text", "expected_language"),
    [("عايز لبن", "ar"), ("Do you have milk?", "en")],
)
def test_bilingual_real_question_sets_turn_and_tts_language(text, expected_language):
    store, primary_tts, spoken = _store_with_session()

    asyncio.run(
        store.on_user_turn_completed(
            agents_llm.ChatContext(),
            types.SimpleNamespace(text_content=text),
        )
    )

    assert len(store.turn_id) == 12
    assert store.reply_language == expected_language
    assert primary_tts.calls == [{"language": expected_language}]
    assert spoken == []


def test_bilingual_turn_uses_the_arabic_voice_for_arabic_text():
    store, primary_tts, _ = _store_with_session(env={"VOICE_TTS_VOICE_AR": "arabic-voice"})

    asyncio.run(
        store.on_user_turn_completed(
            agents_llm.ChatContext(),
            types.SimpleNamespace(text_content="عندكم لبن شوفان؟"),
        )
    )

    assert primary_tts.calls == [{"language": "ar", "voice": "arabic-voice"}]


def test_bilingual_turn_tells_the_model_the_reply_language():
    store, _, _ = _store_with_session()
    turn_ctx = agents_llm.ChatContext()

    asyncio.run(
        store.on_user_turn_completed(turn_ctx, types.SimpleNamespace(text_content="Do you have the"))
    )

    assert any(
        item.role == "system" and "reply in English" in (item.text_content or "")
        for item in turn_ctx.items
    )


def test_legacy_turn_hook_is_a_noop():
    store, primary_tts, spoken = _store_with_session(profile="legacy")

    asyncio.run(
        store.on_user_turn_completed(
            agents_llm.ChatContext(),
            types.SimpleNamespace(text_content="umm"),
        )
    )

    assert store.turn_id == ""
    assert primary_tts.calls == []
    assert spoken == []


def test_search_products_text_uses_render_products(monkeypatch):
    monkeypatch.setattr(
        tools,
        "search_products",
        lambda query, k=5, in_stock_only=False: {
            "items": _PRODUCTS,
            "timing": {"embed_ms": 1.0, "cache": "miss", "search_ms": 2.0},
        },
    )

    text, items, timing = asyncio.run(agent._search_products("milk"))

    assert text == assistant.render_products(_PRODUCTS)
    assert items == _PRODUCTS
    assert timing == {"embed_ms": 1.0, "cache": "miss", "search_ms": 2.0}


def test_search_products_deduplicates_keeps_five(monkeypatch):
    dup = {"code": "PX", "name": "Almarai Milk", "category": "Dairy", "price_egp": 99.0, "stock": 0, "similarity": 0.5}
    many = _PRODUCTS + [dup]
    monkeypatch.setattr(
        tools,
        "search_products",
        lambda query, k=5, in_stock_only=False: {
            "items": many,
            "timing": {"embed_ms": 0.0, "cache": "off", "search_ms": 0.0},
        },
    )

    text, items, timing = asyncio.run(agent._search_products("milk"))

    assert [item["name"] for item in items] == ["Almarai Milk", "Juhayna Yogurt"]
    assert "99.0" not in text


def test_search_documents_text_uses_render_documents(monkeypatch):
    documents = [
        {"title": "Return Policy", "chunk_index": 2, "content": "14 days", "similarity": 0.9}
    ]
    monkeypatch.setattr(
        tools,
        "search_documents",
        lambda query: {
            "items": documents,
            "timing": {"embed_ms": 0.0, "cache": "off", "search_ms": 0.0},
        },
    )

    text, timing = asyncio.run(agent._search_documents("returns"))

    assert text == assistant.render_documents(documents)
    assert timing == {"embed_ms": 0.0, "cache": "off", "search_ms": 0.0}


_WEAK_ITEMS = [
    {"code": "P1", "name": "Beyti Insulated Cooler Bag", "category": "Household", "price_egp": 400.0, "stock": 6, "similarity": 0.25},
    {"code": "P2", "name": "Generic Headache Relief Gel", "category": "Pain Relief", "price_egp": 88.0, "stock": 12, "similarity": 0.23},
]
_CONFIDENT_ITEMS = [
    {"code": "P1", "name": "Almarai Milk", "category": "Dairy", "price_egp": 25.5, "stock": 10, "similarity": 0.62},
]


def _install_search_tool(monkeypatch, items):
    timing = {"embed_ms": 1.0, "cache": "miss", "search_ms": 2.0}
    monkeypatch.setattr(
        tools,
        "search_products",
        lambda query, k=5, in_stock_only=False: {"items": items, "timing": timing},
    )
    sent = []
    async def fake_send(session, topic, payload):
        sent.append((topic, payload))

    monkeypatch.setattr(agent, "_send_json", fake_send)
    store = agent.StoreAgent()
    store._activity = types.SimpleNamespace(session=None)
    return store, sent, timing


def test_search_products_with_only_weak_matches_sends_no_cards(monkeypatch):
    store, sent, _ = _install_search_tool(monkeypatch, _WEAK_ITEMS)

    text = asyncio.run(store.search_products("laptop"))

    assert text == assistant.render_products(_WEAK_ITEMS)
    assert sent == []
    assert [lookup["tool"] for lookup in store.turn_lookups] == ["search_products"]


def test_search_products_sends_cards_when_a_match_is_confident(monkeypatch):
    store, sent, timing = _install_search_tool(monkeypatch, _CONFIDENT_ITEMS)
    store.turn_id = "turn-456"

    asyncio.run(store.search_products("milk"))

    assert sent == [
        (
            agent.CARDS_TOPIC,
            agent.build_cards_payload(_CONFIDENT_ITEMS, timing, "turn-456"),
        )
    ]

_USER = {"stopped_speaking_at": 100.0, "end_of_turn_delay": 0.4}


def test_voice_trace_for_a_search_turn():
    lookups = [{"tool": "search_products", "started_at": 101.3,
                "timing": {"embed_ms": 3.0, "cache": "hit", "search_ms": 6.5}}]
    reply = {"llm_node_ttft": 0.8, "tts_node_ttfb": 0.25, "e2e_latency": 2.6}

    trace = agent.build_voice_trace(_USER, reply, lookups)

    assert [(s["kind"], s["label"]) for s in trace["steps"]] == [
        ("listen", "Notice you've finished"),
        ("llm", "Decide what to look up"),
        ("embed", "Understand the question"),
        ("search", "Search products"),
        ("llm", "Write the answer"),
        ("speak", "Start speaking"),
    ]
    assert [s["ms"] for s in trace["steps"]] == [400.0, 900.0, 3.0, 6.5, 800.0, 250.0]
    assert trace["steps"][2]["cache"] == "hit"
    assert trace["total_ms"] == 2600.0
    assert trace["spoken"] is True


def test_voice_trace_for_a_direct_answer():
    trace = agent.build_voice_trace(_USER, {"llm_node_ttft": 0.7, "tts_node_ttfb": 0.2, "e2e_latency": 1.4}, [])

    assert [s["label"] for s in trace["steps"]] == ["Notice you've finished", "Write the answer", "Start speaking"]


def test_voice_trace_skips_a_negative_decide_step():
    # Preemptive generation can start the model before the turn is committed.
    lookups = [{"tool": "search_documents", "started_at": 100.2,
                "timing": {"embed_ms": 1.0, "cache": "miss", "search_ms": 2.0}}]

    trace = agent.build_voice_trace(_USER, {"e2e_latency": 1.0}, lookups)

    assert "Decide what to look up" not in [s["label"] for s in trace["steps"]]
    assert "Search documents" in [s["label"] for s in trace["steps"]]


def test_no_voice_trace_without_measured_latency():
    assert agent.build_voice_trace(_USER, {"llm_node_ttft": 0.5}, []) is None


def test_publish_trace_records_monitor_id(monkeypatch):
    sent = {}

    async def fake_send(session, topic, payload):
        sent["payload"] = payload

    monkeypatch.setattr(agent, "_send_json", fake_send)
    monkeypatch.setattr(agent.monitor, "record", lambda trace: "abc")

    trace = {"steps": [], "total_ms": 1.0, "spoken": True}
    asyncio.run(agent._publish_trace(None, trace))

    assert sent["payload"]["monitor_id"] == "abc"


def test_publish_trace_sends_when_monitor_fails(monkeypatch):
    sent = {}

    async def fake_send(session, topic, payload):
        sent["payload"] = payload

    def boom(trace):
        raise RuntimeError("redis down")

    monkeypatch.setattr(agent, "_send_json", fake_send)
    monkeypatch.setattr(agent.monitor, "record", boom)

    trace = {"steps": [], "total_ms": 1.0, "spoken": True}
    asyncio.run(agent._publish_trace(None, trace))

    assert sent["payload"]["monitor_id"] is None
