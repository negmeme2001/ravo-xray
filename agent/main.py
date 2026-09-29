"""LiveKit voice agent: the spoken version of the Ravo X-Ray store assistant."""

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from uuid import uuid4

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    StopResponse,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.agents import llm as agents_llm
from livekit.agents import tts as agents_tts
from livekit.agents.voice import SpeechHandle
from livekit.plugins import silero

from agent import config as voice_config
from agent.trace import build_voice_trace
from ravo_core import assistant, monitor, tools, understanding
from ravo_core.prompts import get, message

AGENT_NAME = voice_config.AGENT_NAME
SESSION_LIMIT_SECONDS = 180
MAX_PRODUCTS_FOR_VOICE = 5
CARDS_TOPIC = "ravo.cards"
TRACE_TOPIC = "ravo.trace"
# Rooms the voice test suite opens (the site uses "rx-"); their answers stay out of
# the live monitor so a test run never fills the demo's film roll.
TEST_ROOM_PREFIX = "vt-"
BASE_RULES = assistant.BASE_RULES

logger = logging.getLogger(__name__)

VOICE_PROMPT = assistant.BASE_RULES + get("voice_style").text


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


async def _search_products(query: str) -> tuple[str, list[dict], dict]:
    result = await asyncio.to_thread(
        tools.search_products, query, k=assistant.SEARCH_CANDIDATES
    )
    items = assistant.unique_by_name(result["items"])[:MAX_PRODUCTS_FOR_VOICE]
    items, check_ms = await asyncio.to_thread(assistant.keep_relevant, query, items)
    if check_ms is not None and not items:
        return assistant.NOTHING_SOLD, items, result["timing"]
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


def _finish_turn_trace(
    agent: "StoreAgent",
    user_metrics: dict,
    reply_metrics: dict,
) -> dict | None:
    trace = build_voice_trace(
        user_metrics,
        reply_metrics,
        agent.turn_lookups,
        first_sound_at=agent.first_sound_at,
        filler=agent.filler_played,
    )
    agent._clear_turn()
    return trace


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
        self.profile = voice_config._profile_name(profile)
        self.primary_tts = primary_tts
        self.voice_env = dict(env or {})
        self.reply_language = "en"
        self.turn_id = ""
        self.turn_lookups: list[dict] = []
        self.turn_committed = False
        self.first_sound_at: float | None = None
        self.filler_played = False
        self.filler_enabled = voice_config.filler_enabled(self.profile, self.voice_env)
        self.filler_delay = voice_config.filler_delay(self.voice_env)

    def _search_filler(self, _step: int) -> SpeechHandle | None:
        if not self.filler_enabled or not self.turn_committed or self.filler_played:
            return None
        self.filler_played = True
        return self.session.say(
            message("filler", self.reply_language),
            add_to_chat_ctx=False,
        )

    @asynccontextmanager
    async def _search_filler_context(self, context: RunContext) -> AsyncIterator[None]:
        if not self.filler_enabled:
            yield
            return
        async with context.with_filler(
            self._search_filler,
            delay=self.filler_delay,
            interval=voice_config.FILLER_RETRY_INTERVAL,
            max_steps=1,
        ):
            yield

    def _record_first_sound(self, created_at: float) -> None:
        if self.turn_committed and self.first_sound_at is None:
            self.first_sound_at = created_at

    def _clear_turn(self) -> None:
        self.turn_lookups = []
        self.turn_committed = False
        self.first_sound_at = None
        self.filler_played = False

    async def on_user_turn_completed(
        self,
        turn_ctx: agents_llm.ChatContext,
        new_message: agents_llm.ChatMessage,
    ) -> None:
        self.turn_committed = True
        self.first_sound_at = None
        self.filler_played = False
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
    async def search_products(self, context: RunContext, query: str) -> str:
        """Search the store catalog for products, prices and stock levels."""
        started_at = time.time()
        async with self._search_filler_context(context):
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
    async def search_documents(self, context: RunContext, query: str) -> str:
        """Search store documents for policies, returns, hours and delivery."""
        started_at = time.time()
        async with self._search_filler_context(context):
            text, timing = await _search_documents(query)
        self.turn_lookups.append({"tool": "search_documents", "started_at": started_at, "timing": timing})
        return text

    async def on_enter(self) -> None:
        self.session.say(message("greeting", "en"))


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()
    # Loading the HTTP parser lazily blocked the first turn's event loop for ~600 ms.
    import h11._receivebuffer  # noqa: F401


async def _say_goodbye_after_limit(session: AgentSession, ctx: JobContext) -> None:
    await asyncio.sleep(SESSION_LIMIT_SECONDS)
    try:
        await session.say(message("goodbye", "en"))
    except Exception:
        logger.warning("could not say the goodbye message", exc_info=True)
    finally:
        ctx.shutdown(reason="session time limit reached")


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    profile = voice_config._profile_name(os.environ.get("VOICE_PROFILE", ""))
    options = voice_config.session_options(os.environ.get("VOICE_PROFILE", ""), os.environ)
    primary_tts = options.pop("primary_tts")
    session = AgentSession(vad=ctx.proc.userdata["vad"], **options)
    agent = StoreAgent(profile=profile, primary_tts=primary_tts, env=os.environ)
    turn: dict = {"user_metrics": None}


    if profile == "bilingual":

        @session.on("user_transcription_timeout")
        def _on_transcription_timeout(_event) -> None:
            try:
                if agent.reply_language == "ar":
                    session.say(message("didnt_catch", "ar"))
                else:
                    session.say(message("didnt_catch", "en"))
            except Exception:
                logger.exception("could not ask the shopper to repeat after transcription timeout")

    @session.on("agent_state_changed")
    def _on_agent_state_changed(event) -> None:
        if event.new_state == "speaking":
            agent._record_first_sound(event.created_at)

    @session.on("conversation_item_added")
    def _on_item(event) -> None:
        try:
            item = event.item
            if getattr(item, "type", None) != "message":
                return
            if item.role == "user":
                # Lookups are not cleared here: the reply is drafted (and its searches run)
                # before the shopper's turn is committed, so clearing now dropped them.
                turn["user_metrics"] = dict(item.metrics)
            elif item.role == "assistant" and turn["user_metrics"] is not None:
                # Replies with no shopper turn before them (the greeting) get no x-ray.
                trace = _finish_turn_trace(
                    agent,
                    turn["user_metrics"],
                    dict(item.metrics),
                )
                turn["user_metrics"] = None
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
