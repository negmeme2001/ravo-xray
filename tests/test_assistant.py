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
        lambda query, in_stock_only=False: {
            "items": items,
            "timing": timing or {"embed_ms": 0.0, "cache": "miss", "search_ms": 0.0},
        },
    )


def _install_client(monkeypatch, responses):
    client = _FakeClient(responses)
    monkeypatch.setattr(assistant, "get_openai_client", lambda: client)
    return client


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
            "snippet": " ".join(f"word{i}" for i in range(30)) + " …",
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