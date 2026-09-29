import ravo_core.tools as tools
from ravo_core import assistant


class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [_Choice(_Message(content=content, tool_calls=tool_calls))]


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.create_calls = []

    def create(self, **kwargs):
        self.create_calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, responses):
        self.chat = _Chat(_Completions(responses))


def _search_products(items, monkeypatch, timing=None):
    monkeypatch.setattr(
        tools,
        "search_products",
        lambda query, k=5, in_stock_only=False: {
            "items": items,
            "timing": timing or {"embed_ms": 0.0, "cache": "miss", "search_ms": 0.0},
        },
    )


def _install_client(monkeypatch, responses):
    client = _FakeClient(responses)
    monkeypatch.setattr(assistant, "get_openai_client", lambda: client)
    return client


def _reject_client_call(monkeypatch):
    def reject():
        raise AssertionError("OpenAI client must not be created")

    monkeypatch.setattr(assistant, "get_openai_client", reject)


def _tool_messages(client):
    return [
        message
        for message in client.chat.completions.create_calls[-1]["messages"]
        if message["role"] == "tool"
    ]


def test_tool_path_records_steps_and_collects_cards(monkeypatch):
    items = [
        {"code": "P1", "name": "Almarai Milk", "category": "Dairy", "price_egp": 25.5, "stock": 10, "similarity": 0.912},
        {"code": "P1", "name": "Duplicate", "category": "Dairy", "price_egp": 99.0, "stock": 0, "similarity": 0.5},
        {"code": "P2", "name": "Juhayna Yogurt", "category": "Dairy", "price_egp": 18.0, "stock": 4, "similarity": 0.88},
    ]
    _search_products(items, monkeypatch)
    client = _install_client(
        monkeypatch,
        [
            _Response(tool_calls=[_ToolCall("call-1", "search_products", '{"query": "milk"}')]),
            _Response(content="Almarai Milk is 25.5 EGP."),
        ],
    )

    result = assistant.run_turn([{"role": "user", "content": "what milk do you have?"}])

    assert [step["kind"] for step in result["trace"]["steps"]] == ["llm", "embed", "search", "llm"]
    assert [step["label"] for step in result["trace"]["steps"]] == [
        "Decide what to look up",
        "Understand the question",
        "Search products",
        "Write the answer",
    ]
    assert result["cards"] == [
        {"code": "P1", "name": "Almarai Milk", "category": "Dairy", "price_egp": 25.5, "stock": 10, "similarity": 0.912},
        {"code": "P2", "name": "Juhayna Yogurt", "category": "Dairy", "price_egp": 18.0, "stock": 4, "similarity": 0.88},
    ]
    assert result["citations"] == []
    assert result["reply"] == "Almarai Milk is 25.5 EGP."
    assert result["trace"]["total_ms"] >= sum(step["ms"] for step in result["trace"]["steps"])

    first_call = client.chat.completions.create_calls[0]
    assert first_call["tool_choice"] == "auto"
    assert first_call["messages"][0]["role"] == "system"
    assert first_call["messages"][-1]["role"] == "user"


def test_direct_answer_single_llm_step(monkeypatch):
    client = _install_client(monkeypatch, [_Response(content="We are open until 11 PM.")])

    result = assistant.run_turn([{"role": "user", "content": "how late are you open?"}])

    assert [step["kind"] for step in result["trace"]["steps"]] == ["llm"]
    assert result["trace"]["steps"][0]["label"] == "Write the answer"
    assert result["trace"]["steps"][0]["cache"] is None
    assert result["trace"]["steps"][0]["tool"] is None
    assert result["cards"] == []
    assert result["citations"] == []
    assert result["reply"] == "We are open until 11 PM."
    assert client.chat.completions.create_calls[0]["tool_choice"] == "auto"


def test_cap_forces_final_answer(monkeypatch):
    _search_products([], monkeypatch)
    tool_calls = [
        _Response(tool_calls=[_ToolCall(f"call-{i}", "search_products", '{"query": "milk"}')])
        for i in range(assistant.MAX_LLM_CALLS - 1)
    ]
    client = _install_client(
        monkeypatch,
        tool_calls + [_Response(content="I have limited information.")],
    )

    result = assistant.run_turn([{"role": "user", "content": "milk?"}])

    calls = client.chat.completions.create_calls
    assert len(calls) == assistant.MAX_LLM_CALLS
    assert [call["tool_choice"] for call in calls] == ["auto", "auto", "none"]
    assert result["reply"] == "I have limited information."
    llm_steps = [s for s in result["trace"]["steps"] if s["kind"] == "llm"]
    assert len(llm_steps) == len(calls), "every paid LLM call must appear in the trace"


def test_unknown_tool_records_error_and_still_answers(monkeypatch):
    client = _install_client(
        monkeypatch,
        [
            _Response(tool_calls=[_ToolCall("call-1", "boil_the_sea", "{}")]),
            _Response(content="Sorry, I cannot do that."),
        ],
    )

    result = assistant.run_turn([{"role": "user", "content": "boil the sea"}])

    assert len(_tool_messages(client)) == 1
    assert _tool_messages(client)[0]["content"] == "Tool error: unknown tool boil_the_sea"
    assert result["reply"] == "Sorry, I cannot do that."


def test_bad_arguments_produce_tool_error_message(monkeypatch):
    client = _install_client(
        monkeypatch,
        [
            _Response(tool_calls=[_ToolCall("call-1", "search_products", "{not json")]),
            _Response(content="Sorry."),
        ],
    )

    assistant.run_turn([{"role": "user", "content": "x"}])

    assert len(_tool_messages(client)) == 1
    assert _tool_messages(client)[0]["content"].startswith("Tool error:")


def test_citation_snippet_truncated_to_30_words(monkeypatch):
    content = " ".join(f"word{i}" for i in range(40))
    monkeypatch.setattr(
        tools,
        "search_documents",
        lambda query: {
            "items": [
                {"title": "Return Policy", "chunk_index": 2, "content": content, "similarity": 0.9}
            ],
            "timing": {"embed_ms": 0.0, "cache": "off", "search_ms": 0.0},
        },
    )
    client = _install_client(
        monkeypatch,
        [
            _Response(tool_calls=[_ToolCall("call-1", "search_documents", '{"query": "returns"}')]),
            _Response(content="You can return within 14 days."),
        ],
    )

    result = assistant.run_turn([{"role": "user", "content": "can I return?"}])

    assert result["citations"] == [
        {
            "title": "Return Policy",
            "chunk_index": 2,
            "snippet": "… " + " ".join(f"word{i}" for i in range(30)) + " …",
            "similarity": 0.9,
        }
    ]
    assert result["cards"] == []
    assert [step["label"] for step in result["trace"]["steps"]] == [
        "Decide what to look up",
        "Understand the question",
        "Search documents",
        "Write the answer",
    ]
    # The model reads the whole passage (parts numbered from 1); only the UI snippet is cut.
    assert _tool_messages(client)[0]["content"] == f"[Return Policy, part 3] {content}"

def test_weak_citations_are_dropped():
    citations = [
        {"title": "faq.pdf", "chunk_index": 2, "snippet": "delivery", "similarity": 0.62},
        {"title": "faq.pdf", "chunk_index": 1, "snippet": "ordering", "similarity": 0.58},
        {"title": "safety_manual.pdf", "chunk_index": 3, "snippet": "gloves", "similarity": 0.41},
    ]
    kept = assistant._relevant_citations(citations)
    assert [c["snippet"] for c in kept] == ["delivery", "ordering"]


def test_same_named_products_collapse_to_the_best_match(monkeypatch):
    items = [
        {"code": "P1", "name": "Lamar Popcorn", "category": "Snacks", "price_egp": 213.34, "stock": 200, "similarity": 0.61},
        {"code": "P2", "name": "Panda Trail Mix", "category": "Snacks", "price_egp": 485.69, "stock": 3, "similarity": 0.60},
        {"code": "P3", "name": "Lamar Popcorn", "category": "Snacks", "price_egp": 34.35, "stock": 12, "similarity": 0.58},
    ]
    _search_products(items, monkeypatch)
    client = _install_client(
        monkeypatch,
        [
            _Response(tool_calls=[_ToolCall("call-1", "search_products", '{"query": "healthy snack"}')]),
            _Response(content="Try Lamar Popcorn."),
        ],
    )

    result = assistant.run_turn([{"role": "user", "content": "healthy snack for kids"}])

    assert [c["code"] for c in result["cards"]] == ["P1", "P2"]
    model_view = _tool_messages(client)[0]["content"]
    assert "213.34" in model_view and "34.35" not in model_view


def test_product_type_strips_the_brand():
    assert assistant.product_type("Beyti Insulated Cooler Bag") == "Insulated Cooler Bag"
    assert assistant.product_type("Almarai Milk") == "Milk"
    assert assistant.product_type("Yogurt") == "Yogurt"


def test_cards_to_show_keeps_confident_and_named_cards():
    cards = [
        {"code": "P1", "name": "Almarai Milk", "category": "Dairy", "price_egp": 25.5, "stock": 10, "similarity": 0.62},
        {"code": "P2", "name": "Juhayna Yogurt", "category": "Dairy", "price_egp": 18.0, "stock": 4, "similarity": 0.23},
        {"code": "P3", "name": "Beyti Insulated Cooler Bag", "category": "Household", "price_egp": 400.0, "stock": 6, "similarity": 0.25},
    ]

    assert [c["code"] for c in assistant.cards_to_show(cards, "Try the YOGURT.")] == ["P1", "P2"]
    assert [c["code"] for c in assistant.cards_to_show(cards, "Try the insulated cooler bag.")] == ["P1", "P3"]


def test_prompt_rejects_unrelated_products():
    assert "offer something else" not in assistant.BASE_RULES
    assert "doesn't sell" in assistant.BASE_RULES


def test_prompt_has_language_query_and_unclear_input_rules():
    assert "Reply in the language of the shopper's latest message" in assistant.BASE_RULES
    assert "search queries in English" in assistant.BASE_RULES
    assert "do not search and do not guess" in assistant.BASE_RULES


def test_filler_only_returns_arabic_clarification_without_openai(monkeypatch):
    _reject_client_call(monkeypatch)

    result = assistant.run_turn([{"role": "user", "content": "umm… يعني…"}])

    assert result["reply"] == (
        "معلش، مفهمتش قصدك. بتدور على منتج زي اللبن أو الدوا، ولا عندك سؤال عن "
        "الترجيع أو التوصيل؟"
    )
    assert result["cards"] == []
    assert result["citations"] == []
    assert result["trace"]["steps"] == [
        {
            "kind": "llm",
            "label": "Check the question",
            "ms": result["trace"]["steps"][0]["ms"],
            "cache": None,
            "tool": None,
        }
    ]


def test_empty_returns_english_clarification_without_openai(monkeypatch):
    _reject_client_call(monkeypatch)

    result = assistant.run_turn([{"role": "user", "content": "   ".strip()}])

    assert result["reply"] == (
        "Sorry, I didn't catch that. Are you looking for a product, like milk or medicine, "
        "or asking about returns or delivery?"
    )
    assert result["cards"] == []
    assert result["citations"] == []


def test_weak_search_results_unnamed_in_reply_yield_no_cards(monkeypatch):
    items = [
        {"code": "P1", "name": "Beyti Insulated Cooler Bag", "category": "Household", "price_egp": 400.0, "stock": 6, "similarity": 0.25},
        {"code": "P2", "name": "Generic Headache Relief Gel", "category": "Pain Relief", "price_egp": 88.0, "stock": 12, "similarity": 0.23},
    ]
    _search_products(items, monkeypatch)
    client = _install_client(
        monkeypatch,
        [
            _Response(tool_calls=[_ToolCall("call-1", "search_products", '{"query": "laptop"}')]),
            _Response(content='{"matches": []}'),
            _Response(content="We don't sell laptops."),
        ],
    )

    result = assistant.run_turn([{"role": "user", "content": "do you have laptops?"}])

    assert result["cards"] == []
    assert result["reply"] == "We don't sell laptops."
    # The model writing the reply never sees the loosely linked products.
    tool_result = client.chat.completions.create_calls[2]["messages"][-1]["content"]
    assert "Cooler Bag" not in tool_result
    assert "Check the matches" in [step["label"] for step in result["trace"]["steps"]]


def _product(name, similarity):
    return {"name": name, "category": "Household", "price_egp": 10.0, "stock": 5, "similarity": similarity}


def test_render_products_flags_weak_results():
    text = assistant.render_products([_product("Beyti Insulated Cooler Bag", 0.25)])
    assert text.startswith(assistant.WEAK_MATCH_NOTE)


def test_render_products_leaves_confident_results_alone():
    text = assistant.render_products([_product("Domty Oat Milk", 0.68), _product("Beyti Cooler Bag", 0.2)])
    assert assistant.WEAK_MATCH_NOTE not in text


def _system_prompt_for(question, monkeypatch):
    client = _install_client(monkeypatch, [_Response(content="ok")])
    assistant.run_turn([{"role": "user", "content": question}])
    return client.chat.completions.create_calls[0]["messages"][0]["content"]


def test_english_and_gibberish_input_are_told_to_get_english_replies(monkeypatch):
    assert "reply in English" in _system_prompt_for("Do you have the", monkeypatch)
    assert "reply in English" in _system_prompt_for("asdf", monkeypatch)


def test_arabic_input_is_told_to_get_an_arabic_reply(monkeypatch):
    assert "reply in Egyptian Arabic" in _system_prompt_for("عندكم لبن شوفان؟", monkeypatch)


def test_keep_relevant_skips_the_check_when_a_result_is_confident(monkeypatch):
    _reject_client_call(monkeypatch)
    items = [_product("Domty Oat Milk", 0.68), _product("Beyti Cooler Bag", 0.2)]

    assert assistant.keep_relevant("oat milk", items) == (items, None)


def test_keep_relevant_keeps_only_the_named_matches(monkeypatch):
    _install_client(monkeypatch, [_Response(content='{"matches": ["Almarai Mineral Water"]}')])
    items = [_product("Almarai Mineral Water", 0.295), _product("Panda Lemonade", 0.25)]

    kept, ms = assistant.keep_relevant("water", items)

    assert [item["name"] for item in kept] == ["Almarai Mineral Water"]
    assert ms is not None


def test_keep_relevant_keeps_everything_when_the_check_fails(monkeypatch):
    _install_client(monkeypatch, [_Response(content="not json")])
    items = [_product("Almarai Mineral Water", 0.295)]

    assert assistant.keep_relevant("water", items)[0] == items
