import pytest

from voicetest.outcome import (
    card_types,
    reply_language_ok,
    score_turn,
    searched,
    summarize_runs,
)
from voicetest.run_voicetest import pcm_rms, voice_timing


def scenario(kind="clear", *, answer=True, clarify=False, keywords=None, expected=None):
    return {
        "id": "01",
        "language": "en",
        "kind": kind,
        "keywords": [["oat"], ["milk"]] if keywords is None else keywords,
        "expect": {
            "should_answer": answer,
            "should_clarify": clarify,
            "expected_types": ["Oat Milk"] if expected is None else expected,
        },
    }


def scored(item, **overrides):
    values = {
        "user_transcript": "oat milk",
        "reply": "Here is oat milk",
        "cards": [{"name": "Acme Oat Milk"}],
        "trace_steps": [{"kind": "search", "tool": "search_products"}],
        "cut_in": False,
        "eot_delay_s": 0.5,
    }
    values.update(overrides)
    return score_turn(item, **values)


def test_reply_language_ok() -> None:
    assert reply_language_ok("ar", "عندنا Acme Oat Milk متاح")
    assert not reply_language_ok("en", "هذا رد عربي")
    assert not reply_language_ok("mixed", "")


def test_searched() -> None:
    steps = [{"kind": "embed", "tool": "search_products"},
             {"kind": "search", "tool": "search_documents"}]
    assert searched(steps)
    assert searched(steps, "search_documents")
    assert not searched(steps, "search_products")


def test_card_types() -> None:
    assert card_types([{"name": "Acme Oat Milk"}, {"name": "Premium Select Whole Milk"}]) == {
        "Oat Milk", "Whole Milk"
    }


def test_score_turn_clarify_pass_and_fail() -> None:
    item = scenario("unfinished", answer=False, clarify=True, expected=[])
    assert scored(item, cards=[], trace_steps=[], reply="Could you clarify?")["clarified"] is True
    assert scored(item, cards=[], reply="Could you clarify?")["clarified"] is False


def test_score_turn_not_sold() -> None:
    item = scenario("not-sold", expected=[])
    assert scored(item, cards=[])["answered_right"] is True
    assert scored(item)["answered_right"] is False


def test_score_turn_policy() -> None:
    item = scenario("policy", expected=[])
    assert scored(item, cards=[], trace_steps=[{"kind": "search", "tool": "search_documents"}])["answered_right"] is True
    assert scored(item, cards=[], trace_steps=[])["answered_right"] is False


def test_score_turn_expected_type_and_heard_none() -> None:
    assert scored(scenario())["answered_right"] is True
    result = scored(scenario(keywords=[]))
    assert result["heard"] is None
    assert result["keyword_recall"] is None


def test_summarize_runs_counts_and_percentiles() -> None:
    rows = [
        {"scenario": "01", "heard": True, "no_cut_in": True, "right_language": True,
         "clarified": None, "answered_right": True, "eot_delay_s": 1.0},
        {"scenario": "01", "heard": False, "no_cut_in": True, "right_language": False,
         "clarified": None, "answered_right": True, "eot_delay_s": 3.0},
        {"scenario": "02", "error": "timeout"},
    ]
    summary = summarize_runs(rows)
    assert summary["overall"]["heard"] == "1/2"
    assert summary["scenarios"]["01"]["answered_right"] == "2/2"
    assert summary["overall"]["p50_eot_delay_s"] == 2.0
    assert summary["overall"]["p95_eot_delay_s"] == pytest.approx(2.9)


def test_audio_timing_helpers() -> None:
    assert pcm_rms([300, -400]) == pytest.approx(353.5533905932738)
    assert voice_timing([0.5, 1.5, 2.4], 1.0, 2.0) == (True, pytest.approx(0.4))
