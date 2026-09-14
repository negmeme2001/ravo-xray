"""LiveKit voice agent: the spoken version of the Ravo X-Ray store assistant."""

import asyncio
import json
import logging
import time

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    function_tool,
    inference,
)
from livekit.agents import tts as agents_tts
from livekit.plugins import openai, silero

from ravo_core import assistant, tools

AGENT_NAME = "ravo-xray"
SESSION_LIMIT_SECONDS = 180
VOICE_MODEL = "gpt-4.1-mini"
# The agent runs in eu-central, where LiveKit Inference serves only models with
# an EU endpoint (Inworld has none, so its sessions fail with REGION_RESTRICTED).
STT_MODEL = "deepgram/flux-general-en"
TTS_MODEL = "cartesia/sonic-3"
FALLBACK_TTS_VOICE = "ash"
MAX_PRODUCTS_FOR_VOICE = 5
CARDS_TOPIC = "ravo.cards"
TRACE_TOPIC = "ravo.trace"
BASE_RULES = assistant.BASE_RULES
SEARCH_LABELS = {"search_products": "Search products", "search_documents": "Search documents"}

logger = logging.getLogger(__name__)

VOICE_PROMPT = assistant.BASE_RULES + (
    " Your reply is spoken aloud, so say no lists, no markdown and no symbols. "
    "Use one or two short sentences. Say prices naturally and always name the "
    "currency: 'about sixty-five pounds' is wrong, say 'sixty-five Egyptian "
    "pounds'."
)


def build_cards_payload(items: list[dict], timing: dict) -> dict:
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


class StoreAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=VOICE_PROMPT)
        self.turn_lookups: list[dict] = []

    @function_tool
    async def search_products(self, query: str) -> str:
        """Search the store catalog for products, prices and stock levels."""
        started_at = time.time()
        text, items, timing = await _search_products(query)
        self.turn_lookups.append({"tool": "search_products", "started_at": started_at, "timing": timing})
        await _send_json(self.session, CARDS_TOPIC, build_cards_payload(items, timing))
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
    session = AgentSession(
        stt=inference.STT(model=STT_MODEL, language="en"),
        # A failing voice provider would leave the shopper in silence, so OpenAI's
        # TTS takes over automatically if the primary errors.
        tts=agents_tts.FallbackAdapter(
            [inference.TTS(model=TTS_MODEL), openai.TTS(voice=FALLBACK_TTS_VOICE)]
        ),
        llm=openai.LLM(model=VOICE_MODEL),
        vad=ctx.proc.userdata["vad"],
    )
    agent = StoreAgent()
    turn: dict = {"user_metrics": None}

    @session.on("conversation_item_added")
    def _on_item(event) -> None:
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
                asyncio.create_task(_send_json(session, TRACE_TOPIC, trace))

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