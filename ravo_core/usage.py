"""Fail-open, text-free turn usage counters stored in Redis hashes."""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Iterator, Mapping

from ravo_core import pricing

USAGE_KEY = "usage:v1:all"
USAGE_DAY_KEY = "usage:v1:day:{date}"
DAY_TTL_SECONDS = 90 * 24 * 3600
CHANNELS = ("typed", "spoken")
OUTCOMES = ("clarified", "chat", "not_sold", "answered", "error")
UNITS = (
    "requests",
    "in_tokens",
    "cached_tokens",
    "out_tokens",
    "embed_tokens",
    "stt_seconds",
    "tts_chars",
    "session_minutes",
)
# Seconds and minutes are fractional, so their fields are always float counters: Redis
# rejects HINCRBY on a field that already holds a float.
FLOAT_UNITS = frozenset({"stt_seconds", "session_minutes"})
FLAGS = ("relevance_fail_open",)

logger = logging.getLogger(__name__)
_active: ContextVar[TurnUsage | None] = ContextVar("ravo_turn_usage", default=None)


def _value(obj: Any, name: str, default: Any = 0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _number(value: Any) -> int | float:
    if value is None or isinstance(value, bool):
        return 0
    return value if isinstance(value, (int, float)) else 0


@dataclass
class TurnUsage:
    """Units and reliability counters collected during one assistant turn."""

    units_by_model: dict[str, dict[str, int | float]] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    flags: set[str] = field(default_factory=set)

    def add(self, model: str, units: Mapping[str, int | float]) -> None:
        model_units = self.units_by_model.setdefault(model, {})
        for unit, value in units.items():
            if unit not in UNITS or not isinstance(value, (int, float)) or value == 0:
                continue
            model_units[unit] = model_units.get(unit, 0) + value

    def merge(self, other: TurnUsage) -> None:
        for model, units in other.units_by_model.items():
            self.add(model, units)
        self.cache_hits += other.cache_hits
        self.cache_misses += other.cache_misses
        self.flags.update(other.flags)

    def record_cache(self, state: str | None) -> None:
        if state == "hit":
            self.cache_hits += 1
        elif state == "miss":
            self.cache_misses += 1

    def add_flag(self, name: str) -> None:
        if name in FLAGS:
            self.flags.add(name)

    def trace_fields(self) -> dict:
        return {
            "usage": {model: dict(units) for model, units in self.units_by_model.items()},
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "flags": sorted(self.flags),
        }

    @classmethod
    def from_trace(cls, trace: Mapping[str, Any]) -> TurnUsage:
        collector = cls()
        raw_units = trace.get("usage", {})
        if isinstance(raw_units, Mapping):
            for model, units in raw_units.items():
                if isinstance(model, str) and isinstance(units, Mapping):
                    collector.add(model, units)
        collector.cache_hits = int(_number(trace.get("cache_hits", 0)))
        collector.cache_misses = int(_number(trace.get("cache_misses", 0)))
        raw_flags = trace.get("flags", [])
        if isinstance(raw_flags, (list, tuple, set)):
            for name in raw_flags:
                if isinstance(name, str):
                    collector.add_flag(name)
        return collector


def from_chat_usage(model: str, sdk_usage: Any) -> TurnUsage:
    """Collect token counts from an OpenAI chat usage object or mapping."""
    prompt_tokens = int(_number(_value(sdk_usage, "prompt_tokens")))
    completion_tokens = int(_number(_value(sdk_usage, "completion_tokens")))
    details = _value(sdk_usage, "prompt_tokens_details", None)
    cached_tokens = min(prompt_tokens, int(_number(_value(details, "cached_tokens"))))
    collector = TurnUsage()
    collector.add(
        model,
        {
            "requests": 1,
            "in_tokens": prompt_tokens - cached_tokens,
            "cached_tokens": cached_tokens,
            "out_tokens": completion_tokens,
        },
    )
    return collector


def from_embedding_usage(model: str, sdk_usage: Any) -> TurnUsage:
    """Collect input tokens from an OpenAI embedding usage object or mapping."""
    collector = TurnUsage()
    collector.add(
        model,
        {
            "requests": 1,
            "embed_tokens": int(_number(_value(sdk_usage, "prompt_tokens"))),
        },
    )
    return collector


def current() -> TurnUsage | None:
    """Return the collector active in this execution context, if any."""
    return _active.get()


@contextmanager
def collecting(collector: TurnUsage) -> Iterator[TurnUsage]:
    """Make a collector available to nested model and embedding helpers."""
    token = _active.set(collector)
    try:
        yield collector
    finally:
        _active.reset(token)


def namespace() -> str:
    """Read the optional Redis namespace at call time."""
    return os.environ.get("REDIS_NAMESPACE", "")


def _key(raw_key: str) -> str:
    return f"{namespace()}{raw_key}"


def _field_increments(
    channel: str,
    collector: TurnUsage,
    outcome: str,
    flags: set[str],
    prompt_set: str,
) -> dict[str, int | float]:
    increments: dict[str, int | float] = {
        "turns": 1,
        f"turns:{channel}": 1,
        f"outcome:{outcome}": 1,
        f"prompt:{prompt_set}:turns": 1,
        f"prompt:{prompt_set}:outcome:{outcome}": 1,
    }
    for model, units in collector.units_by_model.items():
        for unit, value in units.items():
            if unit in UNITS and value:
                increments[f"unit:{channel}:{model}:{unit}"] = value
    for flag in flags & set(FLAGS):
        increments[f"flag:{flag}"] = 1
    if collector.cache_hits:
        increments[f"cache:{channel}:hits"] = collector.cache_hits
    if collector.cache_misses:
        increments[f"cache:{channel}:misses"] = collector.cache_misses
    return increments


def _is_float_field(name: str) -> bool:
    return name.startswith("unit:") and name.rpartition(":")[2] in FLOAT_UNITS


def _queue_increments(target: Any, key: str, increments: Mapping[str, int | float]) -> None:
    for name, value in increments.items():
        if _is_float_field(name):
            target.hincrbyfloat(key, name, float(value))
        else:
            target.hincrby(key, name, int(value))


def _write_without_pipeline(
    client: Any,
    all_key: str,
    day_key: str,
    increments: Mapping[str, int | float],
    now: int,
) -> None:
    client.hsetnx(all_key, "since", now)
    for key in (all_key, day_key):
        _queue_increments(client, key, increments)
    client.expire(day_key, DAY_TTL_SECONDS)


def record(
    channel: str,
    units: TurnUsage | Mapping[str, Mapping[str, int | float]],
    outcome: str,
    flags: set[str] | list[str] | tuple[str, ...],
    prompt_set: str,
) -> None:
    """Increment all-time and UTC-day hashes; skip telemetry on every error."""
    if channel not in CHANNELS or outcome not in OUTCOMES:
        logger.warning("usage: invalid bounded label; skipping record")
        return
    from ravo_core.embed_cache import get_redis

    client = get_redis()
    if client is None:
        logger.warning("usage: no redis client; skipping record")
        return
    collector = units if isinstance(units, TurnUsage) else TurnUsage()
    if not isinstance(units, TurnUsage):
        for model, model_units in units.items():
            collector.add(model, model_units)
    safe_prompt_set = prompt_set if prompt_set and len(prompt_set) <= 64 else "unknown"
    safe_flags = {flag for flag in flags if isinstance(flag, str)} | collector.flags
    increments = _field_increments(channel, collector, outcome, safe_flags, safe_prompt_set)
    now_dt = datetime.now(UTC)
    now = int(now_dt.timestamp())
    all_key = _key(USAGE_KEY)
    day_key = _key(USAGE_DAY_KEY.format(date=now_dt.date().isoformat()))
    try:
        pipeline_factory = getattr(client, "pipeline", None)
        if callable(pipeline_factory):
            pipe = pipeline_factory()
            pipe.hsetnx(all_key, "since", now)
            _queue_increments(pipe, all_key, increments)
            _queue_increments(pipe, day_key, increments)
            pipe.expire(day_key, DAY_TTL_SECONDS)
            pipe.exec()
        else:
            _write_without_pipeline(client, all_key, day_key, increments, now)
    except Exception:
        logger.warning("usage: record failed", exc_info=True)


def _as_number(value: Any) -> int | float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return int(number) if number.is_integer() else number


def _decoded_hash(raw: Any) -> dict[str, int | float]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): _as_number(value) for key, value in raw.items()}


def _empty_outcomes() -> dict[str, int]:
    return {outcome: 0 for outcome in OUTCOMES}


def _turns(data: Mapping[str, int | float]) -> dict[str, int]:
    return {
        "total": int(data.get("turns", 0)),
        "typed": int(data.get("turns:typed", 0)),
        "spoken": int(data.get("turns:spoken", 0)),
    }


def _outcomes(data: Mapping[str, int | float]) -> dict[str, int]:
    return {outcome: int(data.get(f"outcome:{outcome}", 0)) for outcome in OUTCOMES}


def _units(data: Mapping[str, int | float]) -> dict[str, dict[str, dict[str, int | float]]]:
    result: dict[str, dict[str, dict[str, int | float]]] = {channel: {} for channel in CHANNELS}
    for field_name, value in data.items():
        if not field_name.startswith("unit:"):
            continue
        _, channel, remainder = field_name.split(":", 2)
        model, _, unit = remainder.rpartition(":")
        if channel in CHANNELS and model and unit in UNITS:
            result[channel].setdefault(model, {})[unit] = value
    return result


def _combined_units(
    by_channel: Mapping[str, Mapping[str, Mapping[str, int | float]]],
) -> dict[str, dict[str, int | float]]:
    combined: dict[str, dict[str, int | float]] = {}
    for models in by_channel.values():
        for model, units in models.items():
            target = combined.setdefault(model, {})
            for unit, value in units.items():
                target[unit] = target.get(unit, 0) + value
    return combined


def _prompt_sets(data: Mapping[str, int | float]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for field_name, value in data.items():
        if not field_name.startswith("prompt:"):
            continue
        _, prompt_set, suffix = field_name.split(":", 2)
        entry = result.setdefault(prompt_set, {"turns": 0, "outcomes": _empty_outcomes()})
        if suffix == "turns":
            entry["turns"] = int(value)
        elif suffix.startswith("outcome:") and suffix[8:] in OUTCOMES:
            entry["outcomes"][suffix[8:]] = int(value)
    return result


def _read_hashes(client: Any, keys: list[str]) -> list[dict[str, int | float]]:
    pipeline_factory = getattr(client, "pipeline", None)
    if callable(pipeline_factory):
        pipe = pipeline_factory()
        for key in keys:
            pipe.hgetall(key)
        return [_decoded_hash(item) for item in pipe.exec()]
    return [_decoded_hash(client.hgetall(key)) for key in keys]


def read(days: int) -> dict:
    """Read aggregate usage and the last N UTC days, failing open to empty data."""
    if not 1 <= days <= 90:
        raise ValueError("days must be between 1 and 90")
    from ravo_core.embed_cache import get_redis

    today = datetime.now(UTC).date()
    dates = [today - timedelta(days=offset) for offset in range(days - 1, -1, -1)]
    keys = [_key(USAGE_KEY)] + [
        _key(USAGE_DAY_KEY.format(date=date.isoformat())) for date in dates
    ]
    client = get_redis()
    hashes: list[dict[str, int | float]] = [{} for _ in keys]
    if client is None:
        logger.warning("usage: no redis client; returning empty counters")
    else:
        try:
            hashes = _read_hashes(client, keys)
        except Exception:
            logger.warning("usage: read failed", exc_info=True)
    all_time = hashes[0]
    units = _units(all_time)
    estimated_cost = pricing.estimate(_combined_units(units))
    cost_by_channel = {
        channel: round(pricing.estimate(models)["total_usd"], 6) for channel, models in units.items()
    }
    series = [
        {"date": date.isoformat(), "turns": _turns(data), "outcomes": _outcomes(data)}
        for date, data in zip(dates, hashes[1:], strict=True)
    ]
    return {
        "totals": {
            **_turns(all_time),
            "cache_hits": sum(int(all_time.get(f"cache:{c}:hits", 0)) for c in CHANNELS),
            "cache_misses": sum(int(all_time.get(f"cache:{c}:misses", 0)) for c in CHANNELS),
        },
        "days": series,
        "units": units,
        "estimated_cost": estimated_cost,
        "estimated_cost_by_channel": cost_by_channel,
        "outcomes": _outcomes(all_time),
        "flags": {
            flag: int(all_time.get(f"flag:{flag}", 0))
            for flag in FLAGS
            if all_time.get(f"flag:{flag}", 0)
        },
        "prompt_sets": _prompt_sets(all_time),
        "since": int(all_time["since"]) if all_time.get("since") else None,
        "prices": {"as_of": pricing.PRICES_AS_OF, "sources": dict(pricing.SOURCES)},
    }
