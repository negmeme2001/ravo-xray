"""Tool-calling store assistant that records an x-ray latency trace per turn."""

import json

from ravo_core import db, tools
from ravo_core.db import get_openai_client

MODEL = "gpt-4.1-mini"
MAX_LLM_CALLS = 3

SYSTEM_PROMPT = (
    "You are the in-store assistant at a supermarket kiosk. Answer in 2 to 4 short "
    "sentences. For every shopping question, including broad ones like 'a healthy "
    "snack for my kids' or 'something for a cold', call search_products first and "
    "recommend only products it returns. Never suggest items from general knowledge "
    "that the search did not return, and never ask permission to search: just search. "
    "Use search_documents for store policies, returns, opening hours, delivery, "
    "parking, payment and staff safety. Never invent products, prices or policies: "
    "if the tools return nothing relevant, say so plainly and offer something else. "
    "Prices are in EGP. Product cards are shown to the shopper separately, so mention "
    "at most the one or two best matches by name and price instead of listing "
    "everything. When answering from a document, say which document it comes from."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search the store catalog for products, prices, stock levels and "
                "what to buy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What the shopper is looking for",
                    },
                    "in_stock_only": {
                        "type": "boolean",
                        "description": "Only return items currently in stock",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search store documents for policies, returns, opening hours, "
                "delivery, parking, payment and staff safety."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The policy or topic to look up",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

SNIPPET_WORDS = 30
MAX_CARDS = 6
MAX_CITATIONS = 3
# Only cite passages nearly as relevant as the best one; weaker ones are noise
# (a staff-safety passage under a delivery answer, for example).
CITATION_SIMILARITY_MARGIN = 0.08


def _llm_step(label: str, ms: float) -> dict:
    return {"kind": "llm", "label": label, "ms": round(ms, 1), "cache": None, "tool": None}


def _tool_step(kind: str, label: str, ms: float, cache, tool: str) -> dict:
    return {"kind": kind, "label": label, "ms": round(ms, 1), "cache": cache, "tool": tool}


def _serialize_tool_call(call) -> dict:
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.function.name, "arguments": call.function.arguments},
    }


def _snippet(text: str, words: int = SNIPPET_WORDS) -> str:
    parts = text.split()
    if len(parts) <= words:
        return text
    return " ".join(parts[:words]) + " …"


def _stock_text(stock: int) -> str:
    return f"in stock ({stock})" if stock > 0 else "out of stock"


def _render_products(items: list[dict]) -> str:
    if not items:
        return "No matching results."
    return "\n".join(
        (
            f"{item['name']} | {item['category']} | {item['price_egp']} EGP | "
            f"{_stock_text(item['stock'])} | {item['similarity']}"
        )
        for item in items
    )


def _render_documents(items: list[dict]) -> str:
    if not items:
        return "No matching results."
    # The model gets the whole passage: an answer often sits past the first
    # few words, and the truncated snippet is only for display.
    return "\n\n".join(
        f"[{item['title']}, part {item['chunk_index'] + 1}] {item['content']}"
        for item in items
    )


def _collect_cards(items: list[dict], cards: dict) -> None:
    for item in items:
        code = item["code"]
        if code in cards:
            continue
        cards[code] = {
            "code": code,
            "name": item["name"],
            "category": item["category"],
            "price_egp": item["price_egp"],
            "stock": item["stock"],
            "similarity": item["similarity"],
        }


def _collect_citations(items: list[dict], citations: list[dict]) -> None:
    for item in items:
        snippet = _snippet(item["content"])
        # Passages after the first begin mid-document, so mark the cut.
        if item["chunk_index"] > 0:
            snippet = "… " + snippet
        citations.append(
            {
                "title": item["title"],
                "chunk_index": item["chunk_index"],
                "snippet": snippet,
                "similarity": item["similarity"],
            }
        )


def _relevant_citations(citations: list[dict]) -> list[dict]:
    if not citations:
        return []
    best = max(c["similarity"] for c in citations)
    kept = [c for c in citations if c["similarity"] >= best - CITATION_SIMILARITY_MARGIN]
    return sorted(kept, key=lambda c: c["similarity"], reverse=True)[:MAX_CITATIONS]


def _execute_tool(call, steps: list[dict], cards: dict, citations: list[dict]) -> str:
    name = call.function.name
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"Tool error: {exc}"
    if not isinstance(args, dict):
        return "Tool error: arguments must be a JSON object"

    if name == "search_products":
        result = tools.search_products(
            str(args.get("query", "")),
            in_stock_only=bool(args.get("in_stock_only", False)),
        )
        steps.append(
            _tool_step("embed", "Understand the question", result["timing"]["embed_ms"], result["timing"]["cache"], name)
        )
        steps.append(_tool_step("search", "Search products", result["timing"]["search_ms"], None, name))
        _collect_cards(result["items"], cards)
        return _render_products(result["items"])

    if name == "search_documents":
        result = tools.search_documents(str(args.get("query", "")))
        steps.append(
            _tool_step("embed", "Understand the question", result["timing"]["embed_ms"], result["timing"]["cache"], name)
        )
        steps.append(_tool_step("search", "Search documents", result["timing"]["search_ms"], None, name))
        _collect_citations(result["items"], citations)
        return _render_documents(result["items"])

    return f"Tool error: unknown tool {name}"


def run_turn(messages: list[dict]) -> dict:
    client = get_openai_client()
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
    steps: list[dict] = []
    cards: dict[str, dict] = {}
    citations: list[dict] = []
    calls_made = 0

    with db.timer() as total:
        while True:
            # The last allowed call must answer, so no paid call is ever discarded
            # and every call made shows up in the trace.
            last_call = calls_made == MAX_LLM_CALLS - 1
            tool_choice = "none" if last_call else "auto"
            with db.timer() as t:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=conversation,
                    tools=TOOLS,
                    tool_choice=tool_choice,
                    temperature=0.3,
                    max_tokens=350,
                )
            llm_ms = round(t.ms, 1)
            calls_made += 1
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []

            if tool_calls and not last_call:
                steps.append(_llm_step("Decide what to look up", llm_ms))
                conversation.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [_serialize_tool_call(call) for call in tool_calls],
                    }
                )
                for call in tool_calls:
                    text = _execute_tool(call, steps, cards, citations)
                    conversation.append({"role": "tool", "tool_call_id": call.id, "content": text})
                continue

            steps.append(_llm_step("Write the answer", llm_ms))
            reply = message.content or ""
            break

    return {
        "reply": reply,
        "cards": list(cards.values())[:MAX_CARDS],
        "citations": _relevant_citations(citations),
        "trace": {"steps": steps, "total_ms": round(total.ms, 1)},
    }