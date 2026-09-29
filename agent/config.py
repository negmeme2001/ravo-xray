"""Voice profile configuration and LiveKit backend construction."""

import logging
import os
from collections.abc import Mapping

from livekit.agents import inference
from livekit.agents import stt as agents_stt
from livekit.agents import tts as agents_tts
from livekit.plugins import openai

from ravo_core.models import (
    BILINGUAL_STT_BACKUP_MODEL,
    BILINGUAL_STT_MODEL,
    LEGACY_STT_MODEL,
    TTS_MODEL,
    VOICE_LLM_MODEL,
)

AGENT_NAME = os.getenv("VOICE_AGENT_NAME", "ravo-xray")
VOICE_MODEL = VOICE_LLM_MODEL
# The agent runs in eu-central, so the legacy profile uses a model with an EU endpoint.
STT_MODEL = LEGACY_STT_MODEL
FALLBACK_TTS_VOICE = "ash"
# The legacy profile commits a hesitating detector after 1.2 seconds.
MAX_END_OF_TURN_WAIT = 1.2
BILINGUAL_ENDPOINTING_MODE = "dynamic"
# Dynamic endpointing uses a short wait for finished speech and a longer wait mid-thought.
BILINGUAL_MIN_END_OF_TURN_WAIT = 0.5
BILINGUAL_MAX_END_OF_TURN_WAIT = 4.5
# The default threshold accepted half sentences that finished questions consistently exceeded.
BILINGUAL_EOT_THRESHOLD = 0.43
BILINGUAL_RESUME_FALSE_INTERRUPTION = True
BILINGUAL_FALSE_INTERRUPTION_TIMEOUT = 2.0
# This must outlast the longest thinking pause because it starts on every pause.
BILINGUAL_TRANSCRIPTION_TIMEOUT = 8.0
BILINGUAL_STT_LANGUAGES = ["ar", "en"]
# Brand names are English even when the surrounding speech is Arabic.
STT_KEYWORDS = [
    "Beyti",
    "Domty",
    "Juhayna",
    "President",
    "Panda",
    "Nestle",
    "Premium Select",
    "oat milk",
]
# Stalled inference-gateway connections are dropped quickly while audio waits in the stream.
BILINGUAL_STT_ATTEMPT_TIMEOUT = 3.0
BILINGUAL_STT_RETRIES = 3
BILINGUAL_STT_RETRY_INTERVAL = 0.5

logger = logging.getLogger(__name__)


def _profile_name(profile: str) -> str:
    if profile == "bilingual":
        return "bilingual"
    return "legacy"


def _primary_cartesia_tts(profile: str, env: Mapping[str, str]) -> agents_tts.TTS:
    tts_options: dict[str, str] = {"model": TTS_MODEL}
    if profile == "bilingual":
        tts_options["language"] = "en"
        if english_voice := env.get("VOICE_TTS_VOICE_EN"):
            tts_options["voice"] = english_voice
    return inference.TTS(**tts_options)


def _voice_stt(profile: str, env: Mapping[str, str]) -> agents_stt.STT:
    legacy_stt = inference.STT(model=STT_MODEL, language="en")
    if profile == "legacy":
        return legacy_stt
    model = env.get("VOICE_STT_MODEL") or BILINGUAL_STT_MODEL
    language = env.get("VOICE_STT_LANGUAGE")
    if model.startswith("openai/"):
        languages = language.split(",") if language else BILINGUAL_STT_LANGUAGES
        bilingual_stt = openai.STT(
            model=model.removeprefix("openai/"),
            language=languages,
            keywords=STT_KEYWORDS,
        )
    else:
        stt_options = {"model": model}
        # Left unset, the model detects the language itself, which code-switching needs.
        if language:
            stt_options["language"] = language
        bilingual_stt = inference.STT(**stt_options)
    backup_stt = inference.STT(
        model=env.get("VOICE_STT_BACKUP_MODEL") or BILINGUAL_STT_BACKUP_MODEL
    )
    return agents_stt.FallbackAdapter(
        [bilingual_stt, backup_stt, legacy_stt],
        attempt_timeout=BILINGUAL_STT_ATTEMPT_TIMEOUT,
        max_retry_per_stt=BILINGUAL_STT_RETRIES,
        retry_interval=BILINGUAL_STT_RETRY_INTERVAL,
    )


def _env_seconds(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        seconds = float(env.get(key) or default)
    except ValueError:
        logger.warning("ignoring %s=%r; using %s", key, env.get(key), default)
        return default
    return seconds if seconds > 0 else default


def _turn_handling(profile: str, env: Mapping[str, str]) -> dict:
    if profile == "legacy":
        return {"endpointing": {"max_delay": MAX_END_OF_TURN_WAIT}}
    return {
        "turn_detection": inference.TurnDetector(
            unlikely_threshold=_env_seconds(
                env,
                "VOICE_EOT_THRESHOLD",
                BILINGUAL_EOT_THRESHOLD,
            )
        ),
        "endpointing": {
            "mode": BILINGUAL_ENDPOINTING_MODE,
            "min_delay": _env_seconds(
                env,
                "VOICE_MIN_DELAY",
                BILINGUAL_MIN_END_OF_TURN_WAIT,
            ),
            "max_delay": _env_seconds(
                env,
                "VOICE_MAX_DELAY",
                BILINGUAL_MAX_END_OF_TURN_WAIT,
            ),
        },
        "interruption": {
            "resume_false_interruption": BILINGUAL_RESUME_FALSE_INTERRUPTION,
            "false_interruption_timeout": BILINGUAL_FALSE_INTERRUPTION_TIMEOUT,
        },
    }


def session_options(profile: str, env: Mapping[str, str]) -> dict:
    """Build AgentSession options and expose its switchable Cartesia TTS."""
    selected_profile = _profile_name(profile)
    if profile not in {"legacy", "bilingual", ""}:
        logger.warning("unknown voice profile %r; using legacy", profile)
    logger.info("voice profile: %s", selected_profile)

    primary_tts = _primary_cartesia_tts(selected_profile, env)
    options = {
        "stt": _voice_stt(selected_profile, env),
        "tts": agents_tts.FallbackAdapter(
            [
                primary_tts,
                openai.TTS(voice=FALLBACK_TTS_VOICE),
            ]
        ),
        "llm": openai.LLM(model=VOICE_MODEL),
        "turn_handling": _turn_handling(selected_profile, env),
        "primary_tts": primary_tts,
    }
    if selected_profile == "bilingual":
        options["transcription_timeout"] = _env_seconds(
            env,
            "VOICE_TRANSCRIPTION_TIMEOUT",
            BILINGUAL_TRANSCRIPTION_TIMEOUT,
        )
    return options
