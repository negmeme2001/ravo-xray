"""Live monitor: recent answer traces and their summary stats stored in Redis."""

import json
import logging
import statistics
import time
import uuid

from ravo_core.embed_cache import get_redis

MONITOR_KEY = "monitor:turns"
MONITOR_KEEP = 200
MONITOR_SHOWN = 40

logger = logging.getLogger(__name__)


def summarize(trace: dict) -> dict:
    """Reduce a trace to its timing numbers and cache state, dropping all text."""
    totals = {"llm": 0.0, "embed": 0.0, "search": 0.0, "listen": 0.0, "speak": 0.0}
    first_cache = None
    for step in trace.get("steps", []):
        kind = step.get("kind")
        ms = step.get("ms")
        if isinstance(kind, str) and isinstance(ms, (int, float)):
            totals[kind] = totals.get(kind, 0.0) + ms
        if kind == "embed" and first_cache is None:
            first_cache = step.get("cache")
    return {
        "id": uuid.uuid4().hex[:12],
        "at": int(time.time()),
        "total_ms": trace["total_ms"],
        "llm_ms": round(totals["llm"], 1),
        "embed_ms": round(totals["embed"], 1),
        "search_ms": round(totals["search"], 1),
        "listen_ms": round(totals["listen"], 1),
        "speak_ms": round(totals["speak"], 1),
        "cache": first_cache,
        "spoken": bool(trace.get("spoken")),
    }


def record(trace: dict) -> str | None:
    """Store a summarized turn; returns its id, or None if recording is skipped.

    Recording must never take the demo down: without a Redis client or on any
    error it logs a warning and fails open.
    """
    client = get_redis()
    if client is None:
        logger.warning("monitor: no redis client; skipping record")
        return None
    entry = summarize(trace)
    try:
        client.lpush(MONITOR_KEY, json.dumps(entry))
        client.ltrim(MONITOR_KEY, 0, MONITOR_KEEP - 1)
    except Exception:
        logger.warning("monitor: record failed", exc_info=True)
        return None
    return entry["id"]


def forget(ids) -> int:
    """Remove the turns with these ids (a test run's own answers); returns how many.

    Fails open like record: without a Redis client or on any error it removes nothing.
    """
    wanted = set(ids)
    client = get_redis()
    if not wanted or client is None:
        return 0
    removed = 0
    try:
        for item in client.lrange(MONITOR_KEY, 0, -1) or []:
            try:
                entry_id = json.loads(item).get("id")
            except (TypeError, ValueError, AttributeError):
                continue
            if entry_id in wanted:
                removed += client.lrem(MONITOR_KEY, 1, item) or 0
    except Exception:
        logger.warning("monitor: forget failed", exc_info=True)
    return removed


def recent(n: int = MONITOR_SHOWN) -> list[dict]:
    """The newest up to n summarized turns, oldest first."""
    client = get_redis()
    if client is None:
        logger.warning("monitor: no redis client; no recent turns")
        return []
    try:
        raw = client.lrange(MONITOR_KEY, 0, n - 1)
    except Exception:
        logger.warning("monitor: recent failed", exc_info=True)
        return []
    turns = []
    for item in raw or []:
        try:
            turns.append(json.loads(item))
        except (TypeError, ValueError):
            continue
    turns.reverse()
    return turns


def _median_ms(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def summary(turns: list[dict]) -> dict:
    """Headline stats over typed turns, plus the typed and spoken counts."""
    typed = [t for t in turns if not t.get("spoken")]
    spoken = [t for t in turns if t.get("spoken")]
    hits = [t for t in typed if t.get("cache") == "hit"]
    misses = [t for t in typed if t.get("cache") == "miss"]
    if hits or misses:
        cache_hit_rate = round(len(hits) / (len(hits) + len(misses)), 3)
    else:
        cache_hit_rate = None
    return {
        "typed": len(typed),
        "spoken": len(spoken),
        "median_total_ms": _median_ms([t["total_ms"] for t in typed]),
        "median_llm_ms": _median_ms([t["llm_ms"] for t in typed]),
        "median_search_ms": _median_ms([t["search_ms"] for t in typed]),
        "median_embed_miss_ms": _median_ms([t["embed_ms"] for t in misses]),
        "median_embed_hit_ms": _median_ms([t["embed_ms"] for t in hits]),
        "cache_hit_rate": cache_hit_rate,
    }