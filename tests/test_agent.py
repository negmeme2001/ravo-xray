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

    text = asyncio.run(agent._search_documents("returns"))

    assert text == assistant.render_documents(documents)