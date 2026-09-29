from ravo_core import assistant, tools, usage
from voicetest.run_typed import run_scenarios


def test_run_scenarios_records_query_language_and_restores_tools(monkeypatch) -> None:
    original_products = lambda query, **kwargs: {"query": query}
    original_documents = lambda query, **kwargs: {"query": query}
    monkeypatch.setattr(tools, "search_products", original_products)
    monkeypatch.setattr(tools, "search_documents", original_documents)

    def fake_run_turn(messages):
        tools.search_products("oat milk", k=2)
        tools.search_documents("سياسة الإرجاع")
        return {
            "reply": "Here is oat milk",
            "cards": [{"name": "Acme Oat Milk"}],
            "trace": {"steps": [{"kind": "search", "tool": "search_products"}]},
        }

    monkeypatch.setattr(assistant, "run_turn", fake_run_turn)
    scenarios = [{
        "id": "01", "language": "en", "kind": "clear", "reference": "oat milk",
        "keywords": [["oat"], ["milk"]],
        "expect": {"should_answer": True, "should_clarify": False,
                   "expected_types": ["Oat Milk"]},
    }]
    rows = run_scenarios(scenarios, 1)
    assert rows[0]["english_queries"] is False
    assert tools.search_products is original_products
    assert tools.search_documents is original_documents


def test_run_scenarios_does_not_record_usage(monkeypatch) -> None:
    monkeypatch.setattr(
        usage,
        "record",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not record")),
    )
    scenario = {
        "id": "01",
        "language": "en",
        "kind": "clear",
        "reference": "hello",
        "keywords": [["hello"]],
        "expect": {"should_answer": True, "should_clarify": False, "expected_types": []},
    }

    rows = run_scenarios(
        [scenario],
        1,
        run_turn=lambda messages: {"reply": "hello", "cards": [], "trace": {"steps": []}},
    )

    assert "error" not in rows[0]
