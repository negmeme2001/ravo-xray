import pytest

from ravo_core import pricing


def test_estimate_prices_cached_chat_embedding_and_voice_units():
    result = pricing.estimate(
        {
            "gpt-4.1-mini": {
                "in_tokens": 1_000_000,
                "cached_tokens": 1_000_000,
                "out_tokens": 1_000_000,
            },
            "text-embedding-3-small": {"embed_tokens": 1_000_000},
            "openai/gpt-live-transcribe": {"stt_seconds": 60},
            "cartesia/sonic-3": {"tts_chars": 1_000_000},
        }
    )

    assert result["by_model"]["gpt-4.1-mini"] == pytest.approx(2.10)
    assert result["by_model"]["text-embedding-3-small"] == pytest.approx(0.02)
    assert result["by_model"]["openai/gpt-live-transcribe"] == pytest.approx(0.017)
    assert result["by_model"]["cartesia/sonic-3"] == pytest.approx(50.0)
    assert result["total_usd"] == pytest.approx(52.137)
    assert result["unpriced"] == []


def test_estimate_lists_unpriced_models_instead_of_calling_them_free():
    result = pricing.estimate({"deepgram/flux-general-en": {"stt_seconds": 60}})

    assert result == {
        "total_usd": 0,
        "by_model": {},
        "unpriced": ["deepgram/flux-general-en"],
    }


def test_price_metadata_is_versioned():
    assert pricing.PRICES_AS_OF == "2026-09-29"
    assert set(pricing.SOURCES) == {"openai", "livekit_inference", "livekit"}
