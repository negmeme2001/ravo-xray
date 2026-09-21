from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from voicetest.scoring import arabic_ratio, keyword_recall

CHECKS = ("heard", "no_cut_in", "right_language", "clarified", "answered_right")


def reply_language_ok(scenario_language: str, reply: str) -> bool:
    if not reply.strip():
        return False
    ratio = arabic_ratio(reply)
    if scenario_language == "ar":
        return ratio >= 0.3
    if scenario_language == "en":
        return ratio <= 0.1
    if scenario_language == "mixed":
        return True
    raise ValueError(f"unknown scenario language: {scenario_language}")


def searched(trace_steps: Sequence[Mapping[str, Any]], tool: str | None = None) -> bool:
    return any(
        step.get("kind") == "search" and (tool is None or step.get("tool") == tool)
        for step in trace_steps
    )


def card_types(cards: Sequence[Mapping[str, Any]]) -> set[str]:
    types: set[str] = set()
    for card in cards:
        name = str(card.get("name", "")).strip()
        if name.startswith("Premium Select "):
            product_type = name[len("Premium Select ") :]
        else:
            _, separator, product_type = name.partition(" ")
            if not separator:
                product_type = name
        if product_type:
            types.add(product_type)
    return types


def score_turn(
    scenario: Mapping[str, Any],
    *,
    user_transcript: str,
    reply: str,
    cards: Sequence[Mapping[str, Any]],
    trace_steps: Sequence[Mapping[str, Any]],
    cut_in: bool | None,
    eot_delay_s: float | None,
) -> dict[str, bool | float | None]:
    recall = keyword_recall(user_transcript, scenario.get("keywords", []))
    expect = scenario["expect"]
    clarified = None
    if expect["should_clarify"]:
        clarified = not searched(trace_steps) and not cards and bool(reply.strip())

    answered_right = None
    if expect["should_answer"]:
        if scenario["kind"] == "not-sold":
            answered_right = not cards
        elif scenario["kind"] == "policy":
            answered_right = searched(trace_steps, "search_documents")
        else:
            answered_right = bool(card_types(cards) & set(expect["expected_types"]))

    return {
        "heard": None if recall is None else recall >= 0.67,
        "no_cut_in": None if cut_in is None else not cut_in,
        "right_language": reply_language_ok(str(scenario["language"]), reply),
        "clarified": clarified,
        "answered_right": answered_right,
        "keyword_recall": recall,
        "eot_delay_s": eot_delay_s,
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary_for(rows: Sequence[Mapping[str, Any]]) -> dict[str, str | float | None]:
    summary: dict[str, str | float | None] = {}
    for check in CHECKS:
        applicable = [row[check] for row in rows if row.get(check) is not None]
        summary[check] = f"{sum(value is True for value in applicable)}/{len(applicable)}"
    delays = [float(row["eot_delay_s"]) for row in rows if row.get("eot_delay_s") is not None]
    summary["p50_eot_delay_s"] = _percentile(delays, 0.50)
    summary["p95_eot_delay_s"] = _percentile(delays, 0.95)
    return summary


def summarize_runs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if "error" not in row]
    scenario_ids = sorted({str(row["scenario"]) for row in rows})
    return {
        "scenarios": {
            scenario_id: _summary_for(
                [row for row in scored if str(row["scenario"]) == scenario_id]
            )
            for scenario_id in scenario_ids
        },
        "overall": _summary_for(scored),
    }
