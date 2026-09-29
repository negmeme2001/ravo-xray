"""Latency trace construction for spoken turns."""

from ravo_core.prompts import active_versions, prompt_set

SEARCH_LABELS = {
    "search_products": "Search products",
    "search_documents": "Search documents",
}


def _step(
    kind: str,
    label: str,
    seconds: float,
    cache: str | None = None,
    tool: str | None = None,
) -> dict:
    return {
        "kind": kind,
        "label": label,
        "ms": round(seconds * 1000, 1),
        "cache": cache,
        "tool": tool,
    }


def build_voice_trace(
    user_metrics: dict,
    reply_metrics: dict,
    lookups: list[dict],
) -> dict | None:
    """Build the typed-trace shape from one spoken turn's LiveKit metrics."""
    e2e = reply_metrics.get("e2e_latency")
    if e2e is None:
        return None
    steps = []
    end_of_turn = user_metrics.get("end_of_turn_delay")
    if end_of_turn is not None:
        steps.append(_step("listen", "Notice you've finished", end_of_turn))
    stopped_at = user_metrics.get("stopped_speaking_at")
    if lookups and stopped_at is not None:
        deciding = lookups[0]["started_at"] - (stopped_at + (end_of_turn or 0.0))
        if deciding > 0:
            steps.append(_step("llm", "Decide what to look up", deciding))
    for lookup in lookups:
        timing = lookup["timing"]
        steps.append(
            _step(
                "embed",
                "Understand the question",
                timing["embed_ms"] / 1000,
                cache=timing["cache"],
                tool=lookup["tool"],
            )
        )
        steps.append(
            _step(
                "search",
                SEARCH_LABELS[lookup["tool"]],
                timing["search_ms"] / 1000,
                tool=lookup["tool"],
            )
        )
    if (first_word := reply_metrics.get("llm_node_ttft")) is not None:
        steps.append(_step("llm", "Write the answer", first_word))
    if (first_sound := reply_metrics.get("tts_node_ttfb")) is not None:
        steps.append(_step("speak", "Start speaking", first_sound))
    return {
        "steps": steps,
        "total_ms": round(e2e * 1000, 1),
        "spoken": True,
        "prompts": active_versions(),
        "prompt_set": prompt_set(),
    }
