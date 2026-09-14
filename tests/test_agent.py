import asyncio

import pytest

pytest.importorskip("livekit.agents")

import ravo_core.tools as tools
from agent import main as agent
from ravo_core import assistant

_PRODUCTS = [
    {"code": "P1", "name": "Almarai Milk", "category": "Dairy", "price_egp": 25.5, "stock": 10, "similarity": 0.912},
    {"code": "P2", "name": "Juhayna Yogurt", "category": "Dairy", "price_egp": 18.0, "stock": 4, "similarity": 0.88},
]


def test_cards_payload_shape():
    timing = {"embed_ms": 12.4, "cache": "hit", "search_ms": 3.1}

    payload = agent.build_cards_payload(_PRODUCTS, timing)

    assert payload == {
        "cards": [
            {"code": "P1", "name": "Almarai Milk", "category": "Dairy", "price_egp": 25.5, "stock": 10},
            {"code": "P2", "name": "Juhayna Yogurt", "category": "Dairy", "price_egp": 18.0, "stock": 4},
        ],
        "timing": {"embed_ms": 12.4, "cache": "hit", "search_ms": 3.1},
    }


def test_voice_prompt_contains_base_rules():
    assert assistant.BASE_RULES in agent.VOICE_PROMPT


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
