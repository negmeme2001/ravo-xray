"""Public list prices used to estimate model costs from recorded units."""

from typing import Final

PRICES_AS_OF: Final[str] = "2026-09-29"

SOURCES: Final[dict[str, str]] = {
    "openai": "https://developers.openai.com/api/docs/pricing",
    "livekit_inference": "https://livekit.com/pricing/inference",
    "livekit": "https://livekit.com/pricing",
}

# Rates are USD per one recorded unit. Keeping the public rate conversion here
# means Redis only stores durable usage counts, never prices.
PRICES: Final[dict[str, dict[str, float]]] = {
    "gpt-4.1-mini": {
        "in_tokens": 0.40 / 1_000_000,
        "cached_tokens": 0.10 / 1_000_000,
        "out_tokens": 1.60 / 1_000_000,
    },
    "text-embedding-3-small": {"embed_tokens": 0.02 / 1_000_000},
    "openai/gpt-live-transcribe": {"stt_seconds": 0.017 / 60},
    "xai/stt-1": {"stt_seconds": 0.0033 / 60},
    "cartesia/sonic-3": {"tts_chars": 50.00 / 1_000_000},
    "livekit agent session": {"session_minutes": 0.01},
}


def estimate(units_by_model: dict[str, dict[str, int | float]]) -> dict:
    """Estimate USD cost for model units, listing models without a price."""
    by_model: dict[str, float] = {}
    unpriced: list[str] = []
    for model, units in units_by_model.items():
        nonzero = any(float(value or 0) != 0 for key, value in units.items() if key != "requests")
        rates = PRICES.get(model)
        if rates is None:
            if nonzero:
                unpriced.append(model)
            continue
        cost = sum(float(units.get(unit, 0) or 0) * rate for unit, rate in rates.items())
        by_model[model] = cost
    return {
        "total_usd": sum(by_model.values()),
        "by_model": by_model,
        "unpriced": sorted(unpriced),
    }
