"""Tool-calling store assistant that records an x-ray latency trace per turn."""

import json
import logging

from ravo_core import db, pricing, tools, understanding, usage
from ravo_core.db import get_openai_client
from ravo_core.models import TYPED_CHAT_MODEL
from ravo_core.prompts import active_versions, get, message, prompt_set, render

MODEL = TYPED_CHAT_MODEL
logger = logging.getLogger(__name__)
MAX_LLM_CALLS = 3

BASE_RULES = get("shopper_rules").text

SYSTEM_PROMPT = BASE_RULES

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
                        "description": (
                            "What the shopper is looking for, in English (translate if "
                            "the shopper used another language)"
                        ),
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
                        "description": (
                            "The policy or topic to look up, in English (translate if "
                            "the shopper used another language)"
                        ),
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
# A card at or above this similarity is a sure match; below it, a card shows
# only when the reply names the product.
CONFIDENT_MATCH = 0.45
WEAK_MATCH_NOTE = message("weak_match_note", "en")
# Only cite passages nearly as relevant as the best one; weaker ones are noise
# (a staff-safety passage under a delivery answer, for example).
CITATION_SIMILARITY_MARGIN = 0.08
SEARCH_CANDIDATES = 10
MAX_PRODUCTS_FOR_MODEL = 5


def unique_by_name(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for item in items:  # already ordered best match first
        if item["name"] not in seen:
            seen.add(item["name"])
            unique.append(item)
    return unique


def product_type(name: str) -> str:
    parts = name.split()
    return name if len(parts) == 1 else " ".join(parts[1:])


def cards_to_show(cards: list[dict], reply: str) -> list[dict]:
    reply_lower = reply.lower()
    return [
        card
        for card in cards
        if card["similarity"] >= CONFIDENT_MATCH or product_type(card["name"]).lower() in reply_lower
    ]


def language_instruction(text: str) -> str:
    """The reply language for this turn, decided from the script of the shopper's words.

    Left to the prompt alone, the model answered English and gibberish input in Arabic
    (the store is Egyptian), so the code states the language for every turn.
    """
    return message("language_instruction", understanding.script_language(text))


RELEVANCE_PROMPT = get("relevance_check").text
NOTHING_SOLD = message("nothing_sold", "en")


def keep_relevant(query: str, items: list[dict]) -> tuple[list[dict], float | None]:
    """Drop weak search results that don't meet the shopper's need; returns (items, ms).

    Only runs when every result is a weak match. No similarity cut-off separates real
    substitutes from noise ("water" -> mineral water scores 0.295, "phone charger" ->
    batteries 0.306), and a model shown the noise pitched it (a cooler bag for a laptop)
    despite the prompt, so a short model check removes it before the reply is written.
    It fails open: on any error the items are kept as before.
    """
    if not items or any(item["similarity"] >= CONFIDENT_MATCH for item in items):
        return items, None
    names = "\n".join(f"- {item['name']}" for item in items)
    with db.timer() as t:
        try:
            response = get_openai_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": render("relevance_check", query=query, names=names),
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=150,
            )
            collector = usage.current()
            if collector is not None:
                collector.merge(usage.from_chat_usage(MODEL, getattr(response, "usage", None)))
            wanted = set(json.loads(response.choices[0].message.content or "{}").get("matches", []))
        except Exception:
            logger.warning("relevance check failed; keeping the weak matches", exc_info=True)
            collector = usage.current()
            if collector is not None:
                collector.add_flag("relevance_fail_open")
            wanted = {item["name"] for item in items}
    return [item for item in items if item["name"] in wanted], round(t.ms, 1)


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


def render_products(items: list[dict]) -> str:
    if not items:
        return "No matching results."
    lines = "\n".join(
        (
            f"{item['name']} | {item['category']} | {item['price_egp']} EGP | "
            f"{_stock_text(item['stock'])} | {item['similarity']}"
        )
        for item in items
    )
    # Vector search always returns its nearest products, however far off; say so
    # when nothing is close, or the model pitches the nearest thing (a cooler bag
    # for a laptop).
    if all(item["similarity"] < CONFIDENT_MATCH for item in items):
        return WEAK_MATCH_NOTE + "\n" + lines
    return lines


def render_documents(items: list[dict]) -> str:
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


def _execute_tool(
    call,
    steps: list[dict],
    cards: dict,
    citations: list[dict],
    lookup: dict[str, bool],
) -> str:
    name = call.function.name
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"Tool error: {exc}"
    if not isinstance(args, dict):
        return "Tool error: arguments must be a JSON object"

    if name == "search_products":
        lookup["made"] = True
        result = tools.search_products(
            str(args.get("query", "")),
            k=SEARCH_CANDIDATES,
            in_stock_only=bool(args.get("in_stock_only", False)),
        )
        # Same-named products (one brand's item at several prices) would let the
        # reply quote one price while a card shows another, so keep only the best
        # match per name for both the model and the cards.
        result = {**result, "items": unique_by_name(result["items"])[:MAX_PRODUCTS_FOR_MODEL]}
        steps.append(
            _tool_step("embed", "Understand the question", result["timing"]["embed_ms"], result["timing"]["cache"], name)
        )
        steps.append(_tool_step("search", "Search products", result["timing"]["search_ms"], None, name))
        collector = usage.current()
        if collector is not None:
            collector.record_cache(result["timing"].get("cache"))
        items, check_ms = keep_relevant(str(args.get("query", "")), result["items"])
        if check_ms is not None:
            steps.append(_llm_step("Check the matches", check_ms))
        result = {**result, "items": items}
        if check_ms is not None and not items:
            lookup["not_sold"] = True
            return NOTHING_SOLD
        if items:
            lookup["found"] = True
        _collect_cards(result["items"], cards)
        return render_products(result["items"])

    if name == "search_documents":
        lookup["made"] = True
        result = tools.search_documents(str(args.get("query", "")))
        steps.append(
            _tool_step("embed", "Understand the question", result["timing"]["embed_ms"], result["timing"]["cache"], name)
        )
        steps.append(_tool_step("search", "Search documents", result["timing"]["search_ms"], None, name))
        collector = usage.current()
        if collector is not None:
            collector.record_cache(result["timing"].get("cache"))
        if result["items"]:
            lookup["found"] = True
        _collect_citations(result["items"], citations)
        return render_documents(result["items"])

    return f"Tool error: unknown tool {name}"


def _trace_fields(collector: usage.TurnUsage, outcome: str) -> dict:
    estimated = pricing.estimate(collector.units_by_model)
    return {
        **collector.trace_fields(),
        "est_cost_usd": round(estimated["total_usd"], 6),
        "outcome": outcome,
    }


def _run_turn(messages: list[dict], collector: usage.TurnUsage) -> dict:
    user_text = next(
        (
            str(message.get("content") or "")
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    with db.timer() as check_timer:
        classification = understanding.classify(user_text)
    if classification in {understanding.EMPTY, understanding.FILLER_ONLY}:
        elapsed = round(check_timer.ms, 1)
        return {
            "reply": understanding.clarification(understanding.script_language(user_text)),
            "cards": [],
            "citations": [],
            "trace": {
                "steps": [_llm_step("Check the question", elapsed)],
                "total_ms": elapsed,
                "prompts": active_versions(),
                "prompt_set": prompt_set(),
                **_trace_fields(collector, "clarified"),
            },
        }

    client = get_openai_client()
    system = SYSTEM_PROMPT + " " + language_instruction(user_text)
    conversation = [{"role": "system", "content": system}] + list(messages)
    steps: list[dict] = []
    cards: dict[str, dict] = {}
    citations: list[dict] = []
    lookup = {"made": False, "found": False, "not_sold": False}
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
            collector.merge(usage.from_chat_usage(MODEL, getattr(response, "usage", None)))
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
                    text = _execute_tool(call, steps, cards, citations, lookup)
                    conversation.append({"role": "tool", "tool_call_id": call.id, "content": text})
                continue

            steps.append(_llm_step("Write the answer", llm_ms))
            reply = message.content or ""
            break

    if lookup["found"]:
        outcome = "answered"
    elif lookup["not_sold"]:
        outcome = "not_sold"
    elif lookup["made"]:
        outcome = "answered"
    else:
        outcome = "chat"

    return {
        "reply": reply,
        "cards": cards_to_show(list(cards.values()), reply)[:MAX_CARDS],
        "citations": _relevant_citations(citations),
        "trace": {
            "steps": steps,
            "total_ms": round(total.ms, 1),
            "prompts": active_versions(),
            "prompt_set": prompt_set(),
            **_trace_fields(collector, outcome),
        },
    }


def run_turn(messages: list[dict]) -> dict:
    """Run one typed turn and return its answer, evidence, trace, usage, and outcome."""
    collector = usage.current()
    if collector is not None:
        return _run_turn(messages, collector)
    collector = usage.TurnUsage()
    with usage.collecting(collector):
        return _run_turn(messages, collector)
