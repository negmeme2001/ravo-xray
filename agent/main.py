"""LiveKit voice agent: the spoken version of the Ravo X-Ray store assistant."""

import asyncio
import json
import logging
import os
import time
from collections.abc import Mapping
from uuid import uuid4

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    StopResponse,
    WorkerOptions,
    cli,
    function_tool,
    inference,
)
from livekit.agents import llm as agents_llm
from livekit.agents import stt as agents_stt
from livekit.agents import tts as agents_tts
from livekit.plugins import openai, silero

from ravo_core import assistant, monitor, tools, understanding

AGENT_NAME = os.getenv("VOICE_AGENT_NAME", "ravo-xray")
SESSION_LIMIT_SECONDS = 180
VOICE_MODEL = "gpt-4.1-mini"
# The agent runs in eu-central, where LiveKit Inference serves only models with
# an EU endpoint (Inworld has none, so its sessions fail with REGION_RESTRICTED).
STT_MODEL = "deepgram/flux-general-en"
TTS_MODEL = "cartesia/sonic-3"
FALLBACK_TTS_VOICE = "ash"
# The legacy profile commits a hesitating detector after 1.2 seconds.
MAX_END_OF_TURN_WAIT = 1.2
# Dynamic endpointing adapts its wait to the speaker.
BILINGUAL_ENDPOINTING_MODE = "dynamic"
# A short hesitation should not immediately end a bilingual turn.
BILINGUAL_MIN_END_OF_TURN_WAIT = 0.5
# Three seconds gives a two-second pause margin on the first turn.
BILINGUAL_MAX_END_OF_TURN_WAIT = 3.0
# False interruptions may resume for two seconds before the reply is discarded.
BILINGUAL_RESUME_FALSE_INTERRUPTION = True
BILINGUAL_FALSE_INTERRUPTION_TIMEOUT = 2.0
# Three seconds lets a bilingual transcript arrive after the shopper stops.
BILINGUAL_TRANSCRIPTION_TIMEOUT = 3.0
MAX_PRODUCTS_FOR_VOICE = 5
CARDS_TOPIC = "ravo.cards"
TRACE_TOPIC = "ravo.trace"
# Rooms the voice test suite opens (the site uses "rx-"); their answers stay out of
# the live monitor so a test run never fills the demo's film roll.
TEST_ROOM_PREFIX = "vt-"
# The STT bake-off (voicetest/stt_bench.py) winner: the only candidate that heard English as
# well as flux and also heard Arabic. Speechmatics with ar_en dropped English speech entirely.
BILINGUAL_STT_MODEL = "xai/stt-1"
# A connect to the inference gateway usually takes ~0.5 s but now and then stalls for 10-20 s.
# Drop a stalled one fast and retry (audio waits in the stream meanwhile); the 10 s default
# with one retry lost the shopper's whole first sentence.
BILINGUAL_STT_ATTEMPT_TIMEOUT = 3.0
BILINGUAL_STT_RETRIES = 3
BILINGUAL_STT_RETRY_INTERVAL = 0.5
BASE_RULES = assistant.BASE_RULES
SEARCH_LABELS = {"search_products": "Search products", "search_documents": "Search documents"}

logger = logging.getLogger(__name__)

VOICE_PROMPT = assistant.BASE_RULES + (
    " Your reply is spoken aloud, so say no lists, no markdown and no symbols. "
    "Use one or two short sentences. Say prices naturally and always name the "
    "currency: 'about sixty-five pounds' is wrong, say 'sixty-five Egyptian "
    "pounds'. In Arabic replies, say prices in Egyptian Arabic words with "
    "'جنيه', for example 'خمسة وستين جنيه'."
)


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
    stt_options = {"model": env.get("VOICE_STT_MODEL") or BILINGUAL_STT_MODEL}
    # Left unset, the model detects the language itself, which code-switching needs.
    if language := env.get("VOICE_STT_LANGUAGE"):
        stt_options["language"] = language
    bilingual_stt = inference.STT(**stt_options)
    return agents_stt.FallbackAdapter(
        [bilingual_stt, legacy_stt],
        attempt_timeout=BILINGUAL_STT_ATTEMPT_TIMEOUT,
        max_retry_per_stt=BILINGUAL_STT_RETRIES,
        retry_interval=BILINGUAL_STT_RETRY_INTERVAL,
    )


def _turn_handling(profile: str) -> dict:
    if profile == "legacy":
        return {"endpointing": {"max_delay": MAX_END_OF_TURN_WAIT}}
    return {
        "endpointing": {
            "mode": BILINGUAL_ENDPOINTING_MODE,
            "min_delay": BILINGUAL_MIN_END_OF_TURN_WAIT,
            "max_delay": BILINGUAL_MAX_END_OF_TURN_WAIT,
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
        "turn_handling": _turn_handling(selected_profile),
        "primary_tts": primary_tts,
    }
    if selected_profile == "bilingual":
        options["transcription_timeout"] = BILINGUAL_TRANSCRIPTION_TIMEOUT
    return options


def build_cards_payload(items: list[dict], timing: dict, turn_id: str) -> dict:
    """Build the JSON payload the client uses to render product cards."""
    return {
        "cards": [
            {
                "code": item["code"],
                "name": item["name"],
                "category": item["category"],
                "price_egp": item["price_egp"],
                "stock": item["stock"],
            }
            for item in items
        ],
        "timing": {
            "embed_ms": timing["embed_ms"],
            "cache": timing["cache"],
            "search_ms": timing["search_ms"],
        },
        "turn_id": turn_id,
    }


def _step(kind: str, label: str, seconds: float, cache: str | None = None, tool: str | None = None) -> dict:
    return {"kind": kind, "label": label, "ms": round(seconds * 1000, 1), "cache": cache, "tool": tool}


def build_voice_trace(user_metrics: dict, reply_metrics: dict, lookups: list[dict]) -> dict | None:
    """The x-ray of one spoken turn, in the same shape as a typed turn's trace.

    user_metrics / reply_metrics are the MetricsReport dicts LiveKit attaches to the
    shopper's and the agent's ChatMessage (seconds and epoch timestamps). lookups
    are the searches the agent ran this turn: {"tool", "started_at", "timing"}.
    The total is LiveKit's measured e2e_latency: from the shopper going quiet to the
    agent's first sound. The steps are the sequential pieces of that wait.
    """
    e2e = reply_metrics.get("e2e_latency")
    if e2e is None:
        return None
    steps = []
    end_of_turn = user_metrics.get("end_of_turn_delay")
    if end_of_turn is not None:
        steps.append(_step("listen", "Notice you've finished", end_of_turn))
    stopped_at = user_metrics.get("stopped_speaking_at")
    if lookups and stopped_at is not None:
        # The model's first call runs from the committed turn to the first search.
        deciding = lookups[0]["started_at"] - (stopped_at + (end_of_turn or 0.0))
        if deciding > 0:
            steps.append(_step("llm", "Decide what to look up", deciding))
    for lookup in lookups:
        timing = lookup["timing"]
        steps.append(_step("embed", "Understand the question", timing["embed_ms"] / 1000,
                           cache=timing["cache"], tool=lookup["tool"]))
        steps.append(_step("search", SEARCH_LABELS[lookup["tool"]], timing["search_ms"] / 1000,
                           tool=lookup["tool"]))
    if (first_word := reply_metrics.get("llm_node_ttft")) is not None:
        steps.append(_step("llm", "Write the answer", first_word))
    if (first_sound := reply_metrics.get("tts_node_ttfb")) is not None:
        steps.append(_step("speak", "Start speaking", first_sound))
    return {"steps": steps, "total_ms": round(e2e * 1000, 1), "spoken": True}


async def _search_products(query: str) -> tuple[str, list[dict], dict]:
    result = await asyncio.to_thread(
        tools.search_products, query, k=assistant.SEARCH_CANDIDATES
    )
    items = assistant.unique_by_name(result["items"])[:MAX_PRODUCTS_FOR_VOICE]
    return assistant.render_products(items), items, result["timing"]


async def _search_documents(query: str) -> tuple[str, dict]:
    result = await asyncio.to_thread(tools.search_documents, query)
    return assistant.render_documents(result["items"]), result["timing"]


async def _send_json(session: AgentSession, topic: str, payload: dict) -> None:
    try:
        await session.room_io.room.local_participant.send_text(json.dumps(payload), topic=topic)
    except Exception:
        # The page only loses a visual; the conversation itself must carry on.
        logger.exception("could not send %s to the room", topic)


def _is_test_room(session: AgentSession | None) -> bool:
    try:
        return session.room_io.room.name.startswith(TEST_ROOM_PREFIX)
    except (AttributeError, RuntimeError):
        return False


async def _publish_trace(session: AgentSession, trace: dict) -> None:
    """Record the trace in the live monitor, then send it; never block the turn."""
    if _is_test_room(session):
        trace["monitor_id"] = None
    else:
        try:
            trace["monitor_id"] = await asyncio.to_thread(monitor.record, trace)
        except Exception:
            logger.warning("monitor: record failed for a voice turn", exc_info=True)
            trace["monitor_id"] = None
    await _send_json(session, TRACE_TOPIC, trace)


def _latest_user_text(chat_ctx: agents_llm.ChatContext) -> str:
    for item in reversed(chat_ctx.items):
        if getattr(item, "type", None) == "message" and item.role == "user":
            return item.text_content or ""
    return ""


class StoreAgent(Agent):
    def __init__(
        self,
        profile: str = "legacy",
        primary_tts: agents_tts.TTS | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(instructions=VOICE_PROMPT)
        self.profile = _profile_name(profile)
        self.primary_tts = primary_tts
        self.voice_env = dict(env or {})
        self.reply_language = "en"
        self.turn_id = ""
        self.turn_lookups: list[dict] = []


    async def on_user_turn_completed(
        self,
        turn_ctx: agents_llm.ChatContext,
        new_message: agents_llm.ChatMessage,
    ) -> None:
        if self.profile != "bilingual":
            return

        self.turn_id = uuid4().hex[:12]
        text = new_message.text_content or ""
        # The transcript's script decides the language: the bilingual STT reports "ar_en"
        # for every turn, which would pin English speakers to Arabic replies.
        self.reply_language = understanding.script_language(text)
        if self.primary_tts is not None:
            voice_key = f"VOICE_TTS_VOICE_{self.reply_language.upper()}"
            tts_options = {"language": self.reply_language}
            if voice := self.voice_env.get(voice_key):
                tts_options["voice"] = voice
            self.primary_tts.update_options(**tts_options)

        if understanding.classify(text) in {understanding.EMPTY, understanding.FILLER_ONLY}:
            self.session.say(understanding.clarification(self.reply_language))
            raise StopResponse

    async def llm_node(
        self,
        chat_ctx: agents_llm.ChatContext,
        tools: list,
        model_settings,
    ):
        if self.profile == "bilingual":
            # Left to the prompt, the model answers English in Arabic (the store is Egyptian),
            # so each generation states the reply language, as the typed chat does. It goes on
            # a copy here, not into the turn in on_user_turn_completed: editing the turn there
            # discards the reply the agent starts drafting while the shopper finishes.
            chat_ctx = chat_ctx.copy()
            chat_ctx.add_message(
                role="system", content=assistant.language_instruction(_latest_user_text(chat_ctx))
            )
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    @function_tool
    async def search_products(self, query: str) -> str:
        """Search the store catalog for products, prices and stock levels."""
        started_at = time.time()
        text, items, timing = await _search_products(query)
        self.turn_lookups.append({"tool": "search_products", "started_at": started_at, "timing": timing})
        if any(item["similarity"] >= assistant.CONFIDENT_MATCH for item in items):
            await _send_json(
                self.session,
                CARDS_TOPIC,
                build_cards_payload(items, timing, self.turn_id),
            )
        return text

    @function_tool
    async def search_documents(self, query: str) -> str:
        """Search store documents for policies, returns, hours and delivery."""
        started_at = time.time()
        text, timing = await _search_documents(query)
        self.turn_lookups.append({"tool": "search_documents", "started_at": started_at, "timing": timing})
        return text

    async def on_enter(self) -> None:
        self.session.say("Hi, I'm the store assistant. What are you looking for?")


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()
    # Loading the HTTP parser lazily blocked the first turn's event loop for ~600 ms.
    import h11._receivebuffer  # noqa: F401


async def _say_goodbye_after_limit(session: AgentSession, ctx: JobContext) -> None:
    await asyncio.sleep(SESSION_LIMIT_SECONDS)
    try:
        await session.say("That's all the time I have today. Goodbye!")
    except Exception:
        logger.warning("could not say the goodbye message", exc_info=True)
    finally:
        ctx.shutdown(reason="session time limit reached")


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    profile = _profile_name(os.environ.get("VOICE_PROFILE", ""))
    options = session_options(os.environ.get("VOICE_PROFILE", ""), os.environ)
    primary_tts = options.pop("primary_tts")
    session = AgentSession(vad=ctx.proc.userdata["vad"], **options)
    agent = StoreAgent(profile=profile, primary_tts=primary_tts, env=os.environ)
    turn: dict = {"user_metrics": None}


    if profile == "bilingual":

        @session.on("user_transcription_timeout")
        def _on_transcription_timeout(_event) -> None:
            try:
                if agent.reply_language == "ar":
                    session.say("معلش، مسمعتكش كويس. ممكن تقول تاني؟")
                else:
                    session.say("Sorry, I didn't catch that. Could you say it again?")
            except Exception:
                logger.exception("could not ask the shopper to repeat after transcription timeout")

    @session.on("conversation_item_added")
    def _on_item(event) -> None:
        try:
            item = event.item
            if getattr(item, "type", None) != "message":
                return
            if item.role == "user":
                turn["user_metrics"] = dict(item.metrics)
                agent.turn_lookups = []
            elif item.role == "assistant" and turn["user_metrics"] is not None:
                # Replies with no shopper turn before them (the greeting) get no x-ray.
                trace = build_voice_trace(turn["user_metrics"], dict(item.metrics), agent.turn_lookups)
                turn["user_metrics"] = None
                agent.turn_lookups = []
                if trace is not None:
                    if profile == "bilingual":
                        trace["turn_id"] = agent.turn_id
                        trace["language"] = agent.reply_language
                    asyncio.create_task(_publish_trace(session, trace))
        except Exception:
            logger.exception("could not process a conversation item")

    await session.start(agent=agent, room=ctx.room)
    asyncio.create_task(_say_goodbye_after_limit(session, ctx))


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name=AGENT_NAME,
        )
    )
